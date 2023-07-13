#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Extract astrometry from a SWAP observation of a reference binary

This script is part of the exoGravity data reduction package.
The swap_reduce script is used to extract the astrometry of a reference binary, which is used to
calculate the metrology zero point (phase reference) when observing in dual-field/off-axis

Args:
  config_file (str): path to the YAML configuration file. Only used to retrieve the path to the SWAP files of the observation. 
  ralim (range): specify the RA range (in mas) over which to search for the astrometry: Example: ralim=[142,146]
  declim (range): same as ralim, for declination
  nra, ndec (int, optional): number of points over the RA and DEC range (the more points, the longer the calculation). Default is 100.
  gofast (bool, optional): if set, average over DITs to accelerate calculations (Usage: --gofast, or gofast=True)

Example:
  python swap_reduce config_file=full/path/to/yourconfig.yml ralim=[800,850] declim=[432,482] --gofast --noinv

Authors:
  M. Nowak, and the exoGravity team.

Version:
  xx.xx
"""

# standard imports
import numpy as np
import scipy.optimize
import sys, os
import glob
import itertools
import astropy.io.fits as fits

# cleanGravity imports
import cleanGravity as gravity
from cleanGravity import complexstats as cs

# local stuff
from utils import *

# ruamel to read config yml file
try:
    import ruamel.yaml
    RUAMEL = True
except: # if ruamel not available, switch back to pyyaml, which does not handle comments properly
    import yaml
    RUAMEL = False

# arg should be the path to the data directory
dargs = args_to_dict(sys.argv)

if "help" in dargs.keys():
    print(__doc__)
    stop()
    
REQUIRED_ARGS = ["config_file"]
for req in REQUIRED_ARGS:
    if not(req in dargs.keys()):
        printerr("Argument '"+req+"' is not optional for this script. Required args are: "+', '.join(REQUIRED_ARGS))
        stop()

# READ THE CONFIGURATION FILE
CONFIG_FILE = dargs["config_file"]
if not(os.path.isfile(CONFIG_FILE)):
    raise Exception("Error: argument {} is not a file".format(CONFIG_FILE))
if RUAMEL:
    cfg = ruamel.yaml.load(open(CONFIG_FILE, "r"), Loader=ruamel.yaml.RoundTripLoader)
else:
    cfg = yaml.safe_load(open(CONFIG_FILE, "r"))

N_RA = cfg["general"]["n_ra_swap"]
N_DEC = cfg["general"]["n_dec_swap"]
RA_LIM = cfg["general"]["ralim_swap"]
DEC_LIM = cfg["general"]["declim_swap"]
EXTENSION = cfg["general"]["extension"]
REDUCTION = cfg["general"]["reduction"]
GRADIENT = cfg["general"]["gradient"]
printinf("RA grid set to [{:.2f}, {:.2f}] with {:d} points".format(RA_LIM[0], RA_LIM[1], N_RA))
printinf("DEC grid set to [{:.2f}, {:.2f}] with {:d} points".format(DEC_LIM[0], DEC_LIM[1], N_DEC))

# GET FILES FOR THE SWAP 
if not("swap_ois" in cfg.keys()):
    printerr("No SWAP files given in {}!".format(CONFIG_FILE))
DATA_DIR = cfg['general']['datadir']    
SWAP_FILES = [DATA_DIR+cfg["swap_ois"][preduce[list(preduce.keys())[0]]["swap_oi"]]["filename"] for preduce in cfg["general"]["reduce_swaps"]]
SWAP_REJECT_DITS = [preduce[list(preduce.keys())[0]]["reject_dits"] for preduce in cfg["general"]["reduce_swaps"]]
SWAP_REJECT_BASELINES = [preduce[list(preduce.keys())[0]]["reject_baselines"] for preduce in cfg["general"]["reduce_swaps"]]

# load other parameters
if "gofast" in dargs.keys():
    GO_FAST = dargs["gofast"].lower()=="true" # bypass value from config file
else: # default is False
    GO_FAST = False
if "gradient" in dargs.keys():
    GRADIENT = dargs["gradient"].lower()=="true" # bypass value from config file
    
# figdir to know where to put figures
FIGDIR = cfg["general"]["figdir"]
if "figdir" in dargs.keys(): # overwrite
    FIGDIR = dargs["figdir"]

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
    stop()
if len(ois2) == 0:
    printerr("Group 2 does not contain any OB")
    stop()

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
            visRefS1 = visRefS1+np.nansum(np.conj(phi)*oi.visOi.visRef*(1-oi.visOi.flag), axis = 0)
            nGoodPointsS1 = nGoodPointsS1+(1-oi.visOi.flag).sum(axis = 0)
        else:
            visRefS2 = visRefS2+np.nansum(phi*oi.visOi.visRef*(1-oi.visOi.flag), axis = 0)
            nGoodPointsS2 = nGoodPointsS2+(1-oi.visOi.flag).sum(axis = 0)
    visRefS1 = visRefS1/nGoodPointsS1
    visRefS2 = visRefS2/nGoodPointsS2
    visRefSwap = np.sqrt(visRefS1*np.conj(visRefS2))
    chi2 = np.nansum(np.imag(visRefSwap)**2)
    chi2_baselines = np.nansum(np.imag(visRefSwap)**2, axis = 1)
    return chi2, chi2_baselines, visRefSwap


# prepare chi2Maps
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
        chi2, chi2_baselines, visRefSwap = compute_chi2(ois1, ois2, ra, dec)
        chi2Map[i, j] = chi2
        chi2Map_baselines[:, i, j] = chi2_baselines
        if chi2Map[i, j] < chi2Best:
            chi2Best = chi2Map[i, j]
            raGuess = ra
            decGuess = dec
            bestFit = np.real(visRefSwap)+0j
            visRefSwapBest = visRefSwap

printinf("Best astrometric solution on map: RA={}, DEC={}".format(raGuess, decGuess))

if GRADIENT:
    printinf("Performing gradient-descent")
    chi2norm = compute_chi2(ois1, ois2, 0, 0)[0] # to avoid large values
    chi2 = lambda astro : compute_chi2(ois1, ois2, astro[0], astro[1])[0]/chi2norm # only chi2, not per baseline
    opt = scipy.optimize.minimize(chi2, x0=[raGuess, decGuess])
    raBest = opt["x"][0]
    decBest = opt["x"][1]
    printinf("Best astrometric solution after gradient descent: RA={}, DEC={}".format(raBest, decBest))    
else:
    raBest = raGuess
    decBest = decGuess
    
# get the astrometric values and add it to the YML file (same value for all swap Ois)
for key in cfg['swap_ois'].keys():
    cfg["swap_ois"][key]["astrometric_solution"] = [float(raBest), float(decBest)] # YAML cannot convert numpy types

# rewrite the YML
f = open(CONFIG_FILE, "w")
if RUAMEL:
    f.write(ruamel.yaml.dump(cfg, Dumper=ruamel.yaml.RoundTripDumper))
else:
    f.write(yaml.safe_dump(cfg, default_flow_style = False)) 
f.close()
    
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
        
        for k in range(len(objOis)):
            oi = objOis[k]
            fig = plt.figure(figsize=(10, 8))
            gPlot.reImPlot(w, np.ma.masked_array(oi.visOi.visRef, oi.visOi.flag).mean(axis = 0), subtitles = oi.basenames, fig = fig)
            # move the best fit back to star frame to overplot it
            this_u = (raBest*oi.visOi.uCoord + decBest*oi.visOi.vCoord)/1e7
            phase = 2*np.pi*this_u*1e7/3600.0/360.0/1000.0*2*np.pi
            phi = np.exp(1j*phase)
            if oi.swap:
                gPlot.reImPlot(w, np.ma.masked_array(phi*bestFit, oi.visOi.flag).mean(axis = 0), fig = fig)                
            else:
                gPlot.reImPlot(w, np.ma.masked_array(np.conj(phi)*bestFit, oi.visOi.flag).mean(axis = 0), fig = fig)
            plt.legend([oi.filename.split("/")[-1], "Swap fit (from combined files)"])
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

        
