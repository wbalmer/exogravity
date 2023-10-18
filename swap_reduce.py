#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Extract astrometry from a SWAP observation of a reference binary

This script is part of the exoGravity data reduction package.
The swap_reduce script is used to extract the astrometry of a reference binary, which is used to
calculate the metrology zero point (phase reference) when observing in dual-field/off-axis

Authors:
  M. Nowak, and the exoGravity team.
"""

# standard imports
import sys, os
import numpy as np
import astropy.io.fits as fits
# cleanGravity imports
import cleanGravity as gravity
from cleanGravity import complexstats as cs
# local package functions
from exogravity.utils import *
from exogravity.common import *
# ruamel to read config yml file
try:
    import ruamel.yaml
    RUAMEL = True
except: # if ruamel not available, switch back to pyyaml, which does not handle comments properly
    import yaml
    RUAMEL = False
# other stuff
import scipy.optimize
import itertools
# argparse for command line arguments
import argparse

# create the parser for command lines arguments
parser = argparse.ArgumentParser(description=
"""
Extract astrometry from a GRAVITY SWAP observation 
""")

# required arguments are the path to the folder containing the data, and the path to the config yml file to write 
parser.add_argument('config_file', type=str, help="the path the to YAML configuration file.")

# some elements from the config file can be overridden with command line arguments
parser.add_argument("--figdir", metavar="DIR", type=str, default=argparse.SUPPRESS,
                    help="name of a directory where to store the output PDF files [overrides value from yml file]")   

parser.add_argument("--go_fast", metavar="EMPTY or TRUE/FALSE", type=bool, nargs="?", default=argparse.SUPPRESS, const = True,
                    help="if set, average over DITs to accelerate calculations. [overrides value from yml file]")   

parser.add_argument("--gradient", metavar="EMPTY or TRUE/FALSE", type=bool, default=argparse.SUPPRESS, nargs="?", const = True,
                     help="if set, improves the estimate of the location of chi2 minima by performing a gradient descent from the position on the map. [overrides value from yml file]")   

parser.add_argument("--use_local", metavar="EMPTY or TRUE/FALSE", type=bool, default=argparse.SUPPRESS, nargs="?", const = True,
                     help="if set, uses the local minima will be instead of global ones. Useful when dealing with multiple close minimums. [overrides value from yml file]")  
 
parser.add_argument("--save_residuals", metavar="EMPTY or TRUE/FALSE", type=bool, default=argparse.SUPPRESS, nargs="?", const = True,
                     help="if set, saves fit residuals as npy files for further inspection. mainly a DEBUG option. [overrides value from yml file]")

# for the astrometry map
parser.add_argument("--ralim_swap", metavar=('MIN', 'MAX'), type=float, nargs=2, default=argparse.SUPPRESS,
                    help="specify the RA range (in mas) over which to search for the astrometry of the swap. [overrides value from yml file]") 
parser.add_argument("--declim_swap", metavar=('MIN', 'MAX'), type=float, nargs=2, default=argparse.SUPPRESS,
                    help="specify the DEC range (in mas) over which to search for the astrometry of the swap. [overrides value from yml file]")  
parser.add_argument("--nra_swap", metavar="N", type=int, default=argparse.SUPPRESS,
                    help="number of points over the RA range of the swap. [overrides value from yml file]")  
parser.add_argument("--ndec_swap", metavar="N", type=int, default=argparse.SUPPRESS,
                    help="number of points over the DEC range of the swap. [overrides value from yml file]")    

# whether to zoom
parser.add_argument("--zoom", metavar="ZOOM_FACTOR", type=int, default=argparse.SUPPRESS,
                    help="create an additional zoomed in version of the chi2 map around the best guess from the initial map.[overrides value from yml file]")


# IF BEING RUN AS A SCRIPT, LOAD COMMAND LINE ARGUMENTS
if __name__ == "__main__":    
    # load arguments into a dictionnary
    args = parser.parse_args()
    dargs = vars(args) # to treat as a dictionnary

    CONFIG_FILE = dargs["config_file"]
    if not(os.path.isfile(CONFIG_FILE)):
        raise Exception("Error: argument {} is not a file".format(CONFIG_FILE))
    
    # READ THE CONFIGURATION FILE
    if RUAMEL:
        cfg = ruamel.yaml.load(open(CONFIG_FILE, "r"), Loader=ruamel.yaml.RoundTripLoader)
    else:
        cfg = yaml.safe_load(open(CONFIG_FILE, "r"))

    # override some values with command line arguments
    for key in dargs:
        if key in cfg["general"]:
            cfg["general"][key] = dargs[key]

# IF THIS FILE IS RUNNING AS A MODULE, WE WILL TAKE CONFIGURATION FILE FROM THE PARENT MODULE
if __name__ != "__main__":
    import exogravity
    cfg = exogravity.cfg

    
#######################
# START OF THE SCRIPT #
#######################        
        
N_RA = cfg["general"]["n_ra_swap"]
N_DEC = cfg["general"]["n_dec_swap"]
RA_LIM = cfg["general"]["ralim_swap"]
DEC_LIM = cfg["general"]["declim_swap"]
EXTENSION = cfg["general"]["extension"]
REDUCTION = cfg["general"]["reduction"]
GRADIENT = cfg["general"]["gradient"]
ZOOM = cfg["general"]["zoom"]
FIGDIR = cfg["general"]["figdir"]
GO_FAST = cfg["general"]["go_fast"]

printinf("RA grid set to [{:.2f}, {:.2f}] with {:d} points".format(RA_LIM[0], RA_LIM[1], N_RA))
printinf("DEC grid set to [{:.2f}, {:.2f}] with {:d} points".format(DEC_LIM[0], DEC_LIM[1], N_DEC))

# GET FILES FOR THE SWAP 
if not("swap_ois" in cfg.keys()):
    printerr("No SWAP files given!")
DATA_DIR = cfg['general']['datadir']    
SWAP_FILES = [DATA_DIR+cfg["swap_ois"][preduce[list(preduce.keys())[0]]["swap_oi"]]["filename"] for preduce in cfg["general"]["reduce_swaps"]]
SWAP_REJECT_DITS = [preduce[list(preduce.keys())[0]]["reject_dits"] for preduce in cfg["general"]["reduce_swaps"]]
SWAP_REJECT_BASELINES = [preduce[list(preduce.keys())[0]]["reject_baselines"] for preduce in cfg["general"]["reduce_swaps"]]

# LOAD GRAVITY PLOT is savefig requested
if not(FIGDIR is None):
    import matplotlib
#    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages        
    from cleanGravity import gravityPlot as gPlot
    if not(os.path.isdir(FIGDIR)):
        os.makedirs(FIGDIR)
        printinf("Directory {} was not found and has been created".format(FIGDIR))
    
# retrieve all astrored files from the datadir
printinf("Found a total of {:d} astroreduced files in {}".format(len(SWAP_FILES), DATA_DIR))

objOis = []
ois1 = []
ois2 = []

# load OBs and put them in two groups based on the two swap pos
for k in range(0, len(SWAP_FILES)):
    filename = SWAP_FILES[k]
    printinf("Loading file "+filename)    
    oi = gravity.GravityDualfieldAstrored(filename, corrMet = cfg["general"]["corr_met"], extension = EXTENSION, corrDisp = cfg["general"]["corr_disp"])
    # note: in case go_fast is set, files should not be averaged over DITs at this step, but later
    if oi.swap:
        printinf("Adding OB {} to group 1 (swap=True)".format(filename.split('/')[-1]))
        ois1.append(oi)
    else:
        printinf("Adding OB {} to group 2 (swap=False)".format(filename.split('/')[-1]))
        ois2.append(oi)        
    objOis.append(oi)    
printinf("Found {:d} OBs for Group 1 and {:d} for Group 2".format(len(ois1), len(ois2)))

# check if no group is empty
if len(ois1) == 0:
    printerr("Group 1 does not contain any OB")
if len(ois2) == 0:
    printerr("Group 2 does not contain any OB")

# filter data
if REDUCTION == "astrored":
    for k in range(len(SWAP_FILES)):
        oi = objOis[k]
#        oi.calibrateWithFlux()
        # explicitly set ignored dits to NAN
        if not(SWAP_REJECT_DITS[k] is None):
            if len(SWAP_REJECT_DITS[k]) > 0:
                a = SWAP_REJECT_DITS[k]
                b = range(oi.visOi.nchannel)
                (a, b, c) = np.meshgrid(a, b, range(oi.nwav))
                oi.visOi.flagPoints((a, b, c))
                printinf("Ignoring some dits in file {}".format(oi.filename))           
        # explicitly set ignored baselines to NAN
        if not(SWAP_REJECT_BASELINES[k] is None):
            if len(SWAP_REJECT_BASELINES[k]) > 0:            
                a = range(oi.visOi.ndit)
                b = SWAP_REJECT_BASELINES[k]
                (a, b, c) = np.meshgrid(a, b, range(oi.nwav))
                oi.visOi.flagPoints((a, b, c))
                printinf("Ignoring some baselines in file {}".format(oi.filename))
    
if (GO_FAST):
    printinf("gofast flag is set. Averaging over DITs")
    for oi in objOis:
        printinf("Averaging file {}".format(oi.filename))
        oi.visOi.recenterPhase(oi.sObjX, oi.sObjY) # mean should be calculated in the frame of the SC
        oi.computeMean()
        oi.visOi.recenterPhase(-oi.sObjX, -oi.sObjY) # back to original (FT) frame        

# calculate the useful w for plotting
w = np.zeros([6, oi.nwav])
for k in range(6):
    w[k, :] = oi.wav*1e6

# A function to calculate the chi2 for a given oi and ra/dec
def compute_chi2(ois1, ois2, ra, dec):
    # reference visibility for group 1        
    visRefS1 = np.zeros([ois1[0].visOi.nchannel, ois1[0].nwav], "complex")
    nGoodPointsS1 = np.zeros([ois1[0].visOi.nchannel, ois1[0].nwav]) # to keep track of the number of good points to calc the mean
    # reference visibility for group 2
    visRefS2 = np.zeros([ois2[0].visOi.nchannel, ois2[0].nwav], "complex")
    nGoodPointsS2 = np.zeros([ois2[0].visOi.nchannel, ois2[0].nwav]) # to keep track of the number of good points to calc the mean                
    for oi in ois1+ois2:
        this_u = (ra*oi.visOi.uCoord + dec*oi.visOi.vCoord)/1e7
        phase = 2*np.pi*this_u*1e7/3600.0/360.0/1000.0*2*np.pi
        phi = np.exp(1j*phase)
        if oi.swap:
            visRefS1 = visRefS1+np.nansum(np.conj(phi)*(oi.visOi.visRef)*(1-oi.visOi.flag), axis = 0)
            nGoodPointsS1 = nGoodPointsS1+(1-oi.visOi.flag).sum(axis = 0)
        else:
            visRefS2 = visRefS2+np.nansum(phi*oi.visOi.visRef*(1-oi.visOi.flag), axis = 0)
            nGoodPointsS2 = nGoodPointsS2+(1-oi.visOi.flag).sum(axis = 0)
    visRefS1 = visRefS1/nGoodPointsS1
    visRefS2 = visRefS2/nGoodPointsS2
    visRefSwap = np.sqrt(visRefS1*np.conj(visRefS2))                
    chi2 = np.nansum(np.imag(visRefSwap)**2)
    chi2_baselines = np.nansum(np.imag(visRefSwap)**2, axis = 1)
    return chi2, chi2_baselines, visRefSwap, np.abs(visRefS1)/np.abs(visRefS2)

# use the above function to fill out the chi2 maps
# if zoom if requested, then we start by the large map to get our first guess
if ZOOM > 0:
    RA_LIM_INITGUESS = [RA_LIM[0], RA_LIM[1]]
    DEC_LIM_INITGUESS = [DEC_LIM[0], DEC_LIM[1]]
    printinf("RA grid: [{:.2f}, {:.2f}] with {:d} points".format(RA_LIM_INITGUESS[0], RA_LIM_INITGUESS[1], N_RA))
    printinf("DEC grid: [{:.2f}, {:.2f}] with {:d} points".format(DEC_LIM_INITGUESS[0], DEC_LIM_INITGUESS[1], N_DEC))
    raValues_initguess = np.linspace(RA_LIM_INITGUESS[0], RA_LIM_INITGUESS[1], N_RA)
    decValues_initguess = np.linspace(DEC_LIM_INITGUESS[0], DEC_LIM_INITGUESS[1], N_DEC)
    chi2Best = np.inf
    chi2Map_initguess = np.zeros([N_RA, N_DEC])
    chi2Map_baselines_initguess = np.zeros([oi.visOi.nchannel, N_RA, N_DEC])
    raBests_initguess = np.zeros(len(objOis))
    decBests_initguess = np.zeros(len(objOis)) 
    # start the fit
    for i in range(N_RA):
        ra = raValues_initguess[i]    
        printinf("Calculating chi2 map for ra value {} ({}/{})".format(ra, i+1, N_RA))
        for j in range(N_DEC):
            dec = decValues_initguess[j]
            chi2, chi2_baselines, visRefSwap, v1_v2_ratio = compute_chi2(ois1, ois2, ra, dec)
            chi2Map_initguess[i, j] = chi2
            chi2Map_baselines_initguess[:, i, j] = chi2_baselines
            if chi2Map_initguess[i, j] < chi2Best:
                chi2Best = chi2Map_initguess[i, j]
                ra_initguess = ra
                dec_initguess = dec
    RA_LIM = [ra_initguess-(RA_LIM[1]-RA_LIM[0])/(2*ZOOM), ra_initguess+(RA_LIM[1]-RA_LIM[0])/(2*ZOOM)]
    DEC_LIM = [dec_initguess-(DEC_LIM[1]-DEC_LIM[0])/(2*ZOOM), dec_initguess+(DEC_LIM[1]-DEC_LIM[0])/(2*ZOOM)]
     
# prepare zoomed chi2Maps
raValues = np.linspace(RA_LIM[0], RA_LIM[1], N_RA)
decValues = np.linspace(DEC_LIM[0], DEC_LIM[1], N_DEC)
chi2Map = np.zeros([N_RA, N_DEC])
chi2Map_baselines = np.zeros([oi.visOi.nchannel, N_RA, N_DEC])

# to keep track of the best fit
chi2Best = np.inf
bestFit = np.zeros([oi.visOi.nchannel, oi.visOi.nwav], "complex")
raGuess = 0
decGuess = 0
raBest = 0
decBest = 0

# start the fit
for i in range(N_RA):
    ra = raValues[i]    
    printinf("Calculating chi2 map for ra value {} ({}/{})".format(ra, i+1, N_RA))
    for j in range(N_DEC):
        dec = decValues[j]
        chi2, chi2_baselines, visRefSwap, v1_v2_ratio = compute_chi2(ois1, ois2, ra, dec)
        chi2Map[i, j] = chi2
        chi2Map_baselines[:, i, j] = chi2_baselines
        if chi2Map[i, j] < chi2Best:
            chi2Best = chi2Map[i, j]
            raGuess = ra
            decGuess = dec
            bestFit = np.real(visRefSwap)+0j

printinf("Best astrometric solution on map: RA={}, DEC={}".format(raGuess, decGuess))

if GRADIENT:
    printinf("Performing gradient-descent")
    chi2norm = compute_chi2(ois1, ois2, 0, 0)[0] # to avoid large values
    chi2 = lambda astro : compute_chi2(ois1, ois2, astro[0], astro[1])[0]/chi2norm # only chi2, not per baseline
    opt = scipy.optimize.minimize(chi2, x0=[raGuess, decGuess])
    raBest = opt["x"][0]
    decBest = opt["x"][1]
    chi2, chi2_baselines, visRefSwap, v1_v2_ratio = compute_chi2(ois1, ois2, raBest, decBest)
    bestFit = np.real(visRefSwap)+0j
    bestFitS1 = bestFit*np.sqrt(v1_v2_ratio)
    bestFitS2 = bestFit/np.sqrt(v1_v2_ratio)
    printinf("Best astrometric solution after gradient descent: RA={}, DEC={}".format(raBest, decBest))
    # generate error bars from random trials
    printinf("Generating error bars from random trials.")
    n_trials = 20 
    astrometry_trials = np.zeros([n_trials, 2])   
    # make a copy of the visRef for future use
    visRefs1_noerr = [np.copy(oi.visOi.visRef) for oi in ois1]
    visRefs2_noerr = [np.copy(oi.visOi.visRef) for oi in ois2]
    for r in range(n_trials):
        for k in range(len(ois1)):
            oi = ois1[k]
            for dit in range(oi.visOi.ndit):
                for c in range(oi.visOi.nchannel):
#                    err = (np.random.multivariate_normal(np.zeros(oi.nwav), np.real(oi.visOi.visRefCov[dit][c].todense()+oi.visOi.visRefPcov[dit][c].todense())/2) + 1j*np.random.multivariate_normal(np.zeros(oi.nwav), np.real(oi.visOi.visRefCov[dit][c].todense()-oi.visOi.visRefPcov[dit][c].todense())/2))
                    err = (np.random.randn(oi.nwav)*np.diag( np.real(oi.visOi.visRefCov[dit][c].todense()+oi.visOi.visRefPcov[dit][c].todense())/2 )**0.5 + 1j*np.random.randn(oi.nwav) * np.diag( np.real(oi.visOi.visRefCov[dit][c].todense()-oi.visOi.visRefPcov[dit][c].todense())/2)**0.5)
                    oi.visOi.visRef[dit, c, :] = visRefs1_noerr[k][dit, c, :] + err
        for k in range(len(ois2)):
            oi = ois2[k]
            for dit in range(oi.visOi.ndit):            
                for c in range(oi.visOi.nchannel):
#                    err = (np.random.multivariate_normal(np.zeros(oi.nwav), np.real(oi.visOi.visRefCov[dit][c].todense()+oi.visOi.visRefPcov[dit][c].todense())/2) + 1j*np.random.multivariate_normal(np.zeros(oi.nwav), np.real(oi.visOi.visRefCov[dit][c].todense()-oi.visOi.visRefPcov[dit][c].todense())/2))
                    err = (np.random.randn(oi.nwav)*np.diag( np.real(oi.visOi.visRefCov[dit][c].todense()+oi.visOi.visRefPcov[dit][c].todense())/2)**0.5 + 1j*np.random.randn(oi.nwav) * np.diag( np.real(oi.visOi.visRefCov[dit][c].todense()-oi.visOi.visRefPcov[dit][c].todense())/2)**0.5)
                    oi.visOi.visRef[dit, c, :] = visRefs2_noerr[k][dit, c, :] + err
        chi2 = lambda dastro : compute_chi2(ois1, ois2, raBest+dastro[0]/1000, decBest+dastro[1]/1000)[0]/chi2norm # only chi2, not per baseline                
        opt = scipy.optimize.minimize(chi2, x0=[10, 10])
        astrometry_trials[r, :] = opt["x"]/1000
        cov_mat = np.cov(astrometry_trials.T) 
        ra_std = cov_mat[0, 0]**0.5
        dec_std = cov_mat[1, 1]**0.5
        rho = cov_mat[0, 1]/(ra_std*dec_std)       
else:
    raBest = raGuess
    decBest = decGuess
    ra_std = float("nan")
    dec_std = float("nan")
    rho = float("nan")
    
# get the astrometric values and add it to the YML file (same value for all swap Ois)
for key in cfg['swap_ois'].keys():
    cfg["swap_ois"][key]["astrometric_solution"] = [float(raBest), float(decBest)] # YAML cannot convert numpy types
    cfg["swap_ois"][key]["astrometric_guess"] = [float(raGuess), float(decGuess)] # YAML cannot convert numpy types 
    cfg["swap_ois"][key]["errors"] = [float(ra_std), float(dec_std), float(rho)] 

# if this script is used as a standalone script, save the updated yml file
if __name__ == "__main__":
    f = open(CONFIG_FILE, "w")
    if RUAMEL:
        f.write(ruamel.yaml.dump(cfg, Dumper=ruamel.yaml.RoundTripDumper))
    else:
        f.write(yaml.safe_dump(cfg, default_flow_style = False)) 
    f.close()

# otherwise, store the updated cfg in the parent package
if __name__ != "__main__":
    exogravity.cfg = cfg    
    
# we'll need the phase reference for the plots
for oi in ois1+ois2:
    if oi.swap:
        oi.visOi.recenterPhase(-raBest, -decBest)        
    else:
        oi.visOi.recenterPhase(raBest, decBest)

# coadd elements in each group to obtain a reference visibility at each position of the swap            
visRefS1 = np.zeros([oi.visOi.nchannel, oi.nwav], "complex")
visRefS2 = np.zeros([oi.visOi.nchannel, oi.nwav], "complex")
printinf("Calculating reference visibility for group 1")
for k in range(len(ois1)):
    oi = ois1[k]
    visRefS1 = visRefS1+np.nanmean(oi.visOi.visRef, axis = 0)
visRefS1 = visRefS1/len(ois1)
printinf("Calculating reference visibility for group 2")
for k in range(len(ois2)):
    oi = ois2[k]
    visRefS2 = visRefS2+np.nanmean(oi.visOi.visRef, axis = 0)
visRefS2 = visRefS2/len(ois2)
printinf("Recentering OBs on initial positions")
for oi in ois1+ois2:
    if oi.swap:
        oi.visOi.recenterPhase(raBest, decBest)
    else:
        oi.visOi.recenterPhase(-raBest, -decBest)        

# the phase ref is the average of the phase of the two groups
printinf("Calculating phase ref")
phaseRef = np.angle(0.5*(visRefS1+visRefS2))

# remove phaseRef from OBs
for oi in objOis:
    oi.visOi.addPhase(-phaseRef)

# now its time to plot the chi2Maps in FIGDIR if given
if not(FIGDIR is None):
    hdu = fits.PrimaryHDU(chi2Map.transpose(1, 0))
    hdu.header["CRPIX1"] = 0.0
    hdu.header["CRVAL1"] = raValues[0]
    hdu.header["CDELT1"] = raValues[1] - raValues[0]
    hdu.header["CRPIX2"] = 0.0
    hdu.header["CRVAL2"] = decValues[0]
    hdu.header["CDELT2"] = decValues[1] - decValues[0]
    hdul = fits.HDUList([hdu])
    hdul.writeto(FIGDIR+"/chi2Map_swap.fits", overwrite = True)
    
    with PdfPages(FIGDIR+"/swap_fit_results.pdf") as pdf:
                
        if ZOOM > 0:   
            fig = plt.figure(figsize = (10, 10))
            ax = fig.add_subplot(1, 1, 1)
            oi = objOis[k]
            name = oi.filename.split('/')[-1]
            im = ax.imshow(chi2Map_initguess.T, origin = "lower", extent = [np.min(raValues_initguess), np.max(raValues_initguess), np.min(decValues_initguess), np.max(decValues_initguess)])
            ax.plot([np.min(raValues), np.max(raValues), np.max(raValues), np.min(raValues), np.min(raValues)], [np.min(decValues), np.min(decValues), np.max(decValues), np.max(decValues), np.min(decValues)], "C3")
            ax.set_xlabel("$\Delta{}\mathrm{RA}$ (mas)")
            ax.set_ylabel("$\Delta{}\mathrm{DEC}$ (mas)")
            ax.set_title(name)
            plt.colorbar(im)
            plt.tight_layout()
            pdf.savefig()

            fig = plt.figure(figsize = (10, 10))
            for c in range(np.shape(chi2Map_baselines_initguess)[0]):
                ax = fig.add_subplot(3, 2, c+1)
                im = ax.imshow(chi2Map_baselines_initguess[c, :, :].T, origin = "lower", extent = [np.min(raValues_initguess), np.max(raValues_initguess), np.min(decValues_initguess), np.max(decValues_initguess)])
                ax.set_xlabel("$\Delta{}\mathrm{RA}$ (mas)")
                ax.set_ylabel("$\Delta{}\mathrm{DEC}$ (mas)")
                ax.set_title(oi.basenames[c])
                plt.colorbar(im)            
            plt.tight_layout()
            pdf.savefig()


        fig = plt.figure(figsize = (10, 10))
        ax = fig.add_subplot(1, 1, 1)
        oi = objOis[k]
        name = oi.filename.split('/')[-1]
        im = ax.imshow(chi2Map.T, origin = "lower", extent = [np.min(raValues), np.max(raValues), np.min(decValues), np.max(decValues)])
        ax.plot(raBest, decBest, "*r")
        ax.set_xlabel("$\Delta{}\mathrm{RA}$ (mas)")
        ax.set_ylabel("$\Delta{}\mathrm{DEC}$ (mas)")
        ax.set_title(name)
        plt.colorbar(im)
        plt.tight_layout()
        pdf.savefig()

        fig = plt.figure(figsize = (10, 10))
        for c in range(np.shape(chi2Map_baselines)[0]):
            ax = fig.add_subplot(3, 2, c+1)
            im = ax.imshow(chi2Map_baselines[c, :, :].T, origin = "lower", extent = [np.min(raValues), np.max(raValues), np.min(decValues), np.max(decValues)])
            ax.set_xlabel("$\Delta{}\mathrm{RA}$ (mas)")
            ax.set_ylabel("$\Delta{}\mathrm{DEC}$ (mas)")
            ax.set_title(oi.basenames[c])
            plt.colorbar(im)            
        plt.tight_layout()
        pdf.savefig()

        for k in range(len(objOis)):
            oi = objOis[k]
            fig = plt.figure(figsize=(10, 8))
            gPlot.reImPlot(w, np.ma.masked_array(oi.visOi.visRef, oi.visOi.flag).mean(axis = 0), subtitles = oi.basenames, fig = fig)
            # move the best fit back to star frame to overplot it
            this_u = (raBest*oi.visOi.uCoord + decBest*oi.visOi.vCoord)/1e7
            phase = 2*np.pi*this_u*1e7/3600.0/360.0/1000.0*2*np.pi
            phi = np.exp(1j*phase)
            if oi.swap:
                gPlot.reImPlot(w, np.ma.masked_array(phi*bestFitS1, oi.visOi.flag).mean(axis = 0), fig = fig)                
            else:
                gPlot.reImPlot(w, np.ma.masked_array(np.conj(phi)*bestFitS2, oi.visOi.flag).mean(axis = 0), fig = fig)
            plt.legend([oi.filename.split("/")[-1], "Swap fit (from combined files)"])
            pdf.savefig()
            


        
