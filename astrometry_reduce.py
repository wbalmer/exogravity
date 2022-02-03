#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Extract astrometry from an exoGravity observation

This script is part of the exoGravity data reduction package.
The astrometry_reduce script is used to extract the astrometry from an exoplanet observation made with GRAVITY, in dual-field mode.
To use this script, you need to call it with a configuration file, see example below.

Args:
  config_file (str): the path the to YAML configuration file.
  gofast (bool, optional): if set, average over DITs to accelerate calculations (Usage: --gofast, or gofast=True, or goFast=False). This will bypass
                           the value contained in the YAML file.
  noinv (bool, optional): if set, avoid inversion of the covariance matrix and replace it with a gaussian elimination approach. 
                          Bypass the value in the YAML file. (Usage: --noinv, or noinv=True, or noinv=False)
  save_residuals (bool, optional): if set, save the residuals in npy files. figdir must also be provided for this option to be used

Example:
  python astrometry_reduce config_file=full/path/to/yourconfig.yml

Authors:
  M. Nowak, and the exoGravity team.

Version:
  xx.xx
"""

# BASIC IMPORTS
import numpy as np
import scipy.sparse, scipy.sparse.linalg
from scipy.linalg import lapack
import astropy.io.fits as fits
# cleanGravity IMPORTS
import cleanGravity as gravity
from cleanGravity import complexstats as cs
from cleanGravity.utils import loadFitsSpectrum, saveFitsSpectrum
from utils import * # utils from this exooGravity package
from common import *
# other random stuffs
import glob
import itertools
# ruamel to read config yml file
try:
    import ruamel.yaml
    RUAMEL = True
except: # if ruamel not available, switch back to pyyaml, which does not handle comments properly
    import yaml
    RUAMEL = False
import sys, os

# load aguments into a dictionnary
dargs = args_to_dict(sys.argv)

if "help" in dargs.keys():
    print(__doc__)
    stop()

# arg should be the path to the config yal file    
REQUIRED_ARGS = ["config_file"]
for req in REQUIRED_ARGS:
    if not(req in dargs.keys()):
        printerr("Argument '"+req+"' is not optional for this script. Required args are: "+', '.join(REQUIRED_ARGS))

CONFIG_FILE = dargs["config_file"]
if not(os.path.isfile(CONFIG_FILE)):
    raise Exception("Error: argument {} is not a file".format(CONFIG_FILE))
    
# READ THE CONFIGURATION FILE
if RUAMEL:
    cfg = ruamel.yaml.load(open(CONFIG_FILE, "r"), Loader=ruamel.yaml.RoundTripLoader)
else:
    cfg = yaml.safe_load(open(CONFIG_FILE, "r"))
DATA_DIR = cfg["general"]["datadir"]
PHASEREF_MODE = cfg["general"]["phaseref_mode"]
CONTRAST_FILE = cfg["general"]["contrast_file"]
NO_INV = cfg["general"]["noinv"]
GO_FAST = cfg["general"]["gofast"]
FIGDIR = cfg["general"]["figdir"]
SAVE_RESIDUALS = cfg["general"]["save_residuals"]
PLANET_FILES = [DATA_DIR+cfg["planet_ois"][k]["filename"] for k in cfg["general"]["reduce"]] # list of files corresponding to planet exposures
if not("swap_ois" in cfg.keys()):
    SWAP_FILES = []
elif cfg["swap_ois"] is None:
    SWAP_FILES = []
else:
    SWAP_FILES = [DATA_DIR+cfg["swap_ois"][k]["filename"] for k in cfg["swap_ois"].keys()]
STAR_ORDER = cfg["general"]["star_order"]
N_OPD = cfg["general"]["n_opd"]
N_RA = cfg["general"]["n_ra"]
N_DEC = cfg["general"]["n_dec"]
RA_LIM = cfg["general"]["ralim"]
DEC_LIM = cfg["general"]["declim"]
EXTENSION = cfg["general"]["extension"]
REDUCTION = cfg["general"]["reduction"]

# FILTER DATA
PHASEREF_ARCLENGTH_THRESHOLD = cfg["general"]["phaseref_arclength_threshold"]
FT_FLUX_THRESHOLD = cfg["general"]["ft_flux_threshold"]
IGNORE_BASELINES = cfg["general"]["ignore_baselines"]

# OVERWRITE SOME OF THE CONFIGURATION VALUES WITH ARGUMENTS FROM COMMAND LINE
if "gofast" in dargs.keys():
    GO_FAST = dargs["gofast"].lower()=="true" # bypass value from config file
if "noinv" in dargs.keys():
    NO_INV = dargs["noinv"] # bypass value from config file
if "figdir" in dargs.keys():
    FIGDIR = dargs["figdir"] # bypass value from config file
if "save_residuals" in dargs.keys():
    SAVE_RESIDUALS = True # bypass value from config file

# LOAD GRAVITY PLOT is savefig requested
if not(FIGDIR is None):
    from cleanGravity import gravityPlot as gPlot
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib.backends.backend_pdf import PdfPages    
    if not(os.path.isdir(FIGDIR)):
        os.makedirs(FIGDIR)
        printinf("Directory {} was not found and has been created".format(FIGDIR))

# THROW AN ERROR IF save_residuals requested, but no FIGDIR provided
if SAVE_RESIDUALS and (FIGDIR is None):
    printerr("save_residuals option requested, but no figdir provided.")
        
# extract list of useful star ois from the list indicated in the star_indices fields of the config file:
star_indices = []
for k in cfg["general"]["reduce"]:
    star_indices = star_indices+cfg["planet_ois"][k]["star_indices"]
star_indices = list(set(star_indices)) # remove duplicates
STAR_FILES = [DATA_DIR+cfg["star_ois"][k]["filename"] for k in star_indices]


# read the contrast file if given, otherwise simply set it to 1
if CONTRAST_FILE is None:
    contrast_data = None
elif CONTRAST_FILE.split('.')[-1].lower() in ["fit", "fits"]:
    contrast_wav, dummy, dummy2, contrast_data, dummy3 = loadFitsSpectrum(CONTRAST_FILE)
else:
    printerr("The given contrast file ({}) should have a fits extension".format(fits))

# lists to contain the planet, star, and potentially swap OIs
objOis = [] # contains the OIs on the planet itself

# LOAD DATA
for filename in PLANET_FILES:
    printinf("Loading file "+filename)
    if REDUCTION == "astrored":
        oi = gravity.GravityDualfieldAstrored(filename, corrMet = cfg["general"]["corr_met"], extension = EXTENSION, corrDisp = cfg["general"]["corr_disp"])
        printinf("File is on planet. FT coherent flux: {:.2e}".format(np.mean(np.abs(oi.visOi.visDataFt))))                
    elif REDUCTION == "dualscivis":
        oi = gravity.GravityDualfieldScivis(filename, extension = EXTENSION)
        printinf("File is on planet")
    else:
        printerr("Unknonwn reduction type '{}'.".format(REDUCTION))        
    objOis.append(oi)

# filter data
if REDUCTION == "astrored":
    # flag points based on FT value and phaseRef arclength
    ftThresholdPlanet = cfg["general"]["ftOnPlanetMeanFlux"]*FT_FLUX_THRESHOLD
    for oi in objOis:
        filter_ftflux(oi, ftThresholdPlanet)
        if PHASEREF_ARCLENGTH_THRESHOLD > 0:
            filter_phaseref_arclength(oi, PHASEREF_ARCLENGTH_THRESHOLD)
        # explicitly set ignored baselines to NAN
        if len(IGNORE_BASELINES)>0:
            a = range(oi.visOi.ndit)
            b = IGNORE_BASELINES
            (a, b, c) = np.meshgrid(a, b, range(oi.nwav))
            oi.visOi.flagPoints((a, b, c))
            printinf("Ignoring some baselines in file {}".format(oi.filename))

# replace data by the mean over all DITs if go_fast has been requested in the config file
printinf("gofast flag is set. Averaging over DITs")
for oi in objOis: # mean should not be calculated on swap before phase correction
    if (GO_FAST):
        printinf("Averaging file {}".format(oi.filename))
        oi.visOi.recenterPhase(oi.sObjX, oi.sObjY)
        oi.computeMean()
        oi.visOi.recenterPhase(-oi.sObjX, -oi.sObjY)

# calculate the very useful w for plotting
oi = objOis[0]
w = np.zeros([oi.visOi.nchannel, oi.nwav])
for c in range(oi.visOi.nchannel):
    w[c, :] = oi.wav*1e6
    
if REDUCTION == "astrored":
    wFt = np.zeros([oi.visOi.nchannel, oi.visOi.nwavFt])
    for c in range(oi.visOi.nchannel):
        wFt[c, :] = range(oi.visOi.nwavFt)

# retrieve the reference visibility for each OB
printinf("Retrieving visibility references from fits files")
visRefs = [oi.visOi.visRef.mean(axis=0)*0 for oi in objOis]
for k in range(len(objOis)):
    oi = objOis[k]
    try:
        visRefs[k] = fits.getdata(oi.filename, "EXOGRAV_VISREF").field("EXOGRAV_VISREF")
    except:
        printerr("Cannot find visibility reference (EXOGRAV_VISREF) in file {}.".format(oi.filename))

# subtract the reference phase to each OB
printinf("Subtracting phase reference to each planet OI.")
for k in range(len(objOis)):
    oi = objOis[k]
    oi.visOi.addPhase(-np.angle(visRefs[k]))

# create projector matrices
for k in range(len(objOis)):
    oi = objOis[k]
    printinf("Create projector matrices (p_matrices) ({}/{})".format(k+1, len(objOis)))
    # calculate the projector
    wav = oi.wav*1e6
    vectors = np.zeros([STAR_ORDER+1, oi.nwav], 'complex64')
    thisVisRef = visRefs[k]
    thisAmpRef = np.abs(thisVisRef)
    oi.visOi.p_matrices = np.zeros([oi.visOi.ndit, oi.visOi.nchannel, oi.nwav, oi.nwav], 'complex64')
    P = np.zeros([oi.visOi.nchannel, oi.nwav, oi.nwav], 'complex64')
    for dit in range(oi.visOi.ndit): # the loop of DIT number is necessary because of bad points mgmt (bad_indices depends on dit)
        for c in range(oi.visOi.nchannel):
            for j in range(STAR_ORDER+1):
                vectors[j, :] = np.abs(thisAmpRef[c, :])*(wav-np.mean(wav))**j # pourquoi ampRef et pas visRef ?
            bad_indices = np.where(oi.visOi.flag[dit, c, :])
            vectors[:, bad_indices] = 0
            for l in range(oi.nwav):
                x = np.zeros(oi.nwav)
                x[l] = 1
                coeffs = np.linalg.lstsq(vectors.T, x, rcond=-1)[0]
                P[c, :, l] = x - np.dot(vectors.T, coeffs)
        oi.visOi.p_matrices[dit, :, :, :] = P
        
printinf("Starting calculation of H matrices")
for k in range(len(objOis)):
    printinf("Calculating H ({:d}/{:d})".format(k+1, len(objOis)))
    oi = objOis[k]
    oi.visOi.h_matrices = np.zeros([oi.visOi.ndit, oi.visOi.nchannel, oi.visOi.nwav, oi.visOi.nwav], 'complex64')
    oi.visOi.m = np.zeros([oi.visOi.ndit, oi.visOi.nchannel])
    for dit in range(oi.visOi.ndit):
        for c in range(oi.visOi.nchannel):
            (G, d, H) = np.linalg.svd(oi.visOi.p_matrices[dit, c, :, :]) # scipy faster but different result?
            D = np.diag(d)
            oi.visOi.h_matrices[dit, c, :, :] = H
            m = int(np.sum(d))
            oi.visOi.m[dit, c] = m
        
# project visibilities
for k in range(len(objOis)):
    oi = objOis[k]
    printinf("Projecting visibilities ({}/{})".format(k+1, len(objOis)))    
    oi.visOi.visProj = [[[] for c in range(oi.visOi.nchannel)] for dit in range(oi.visOi.ndit)]
    for dit in range(oi.visOi.ndit):
        for c in range(oi.visOi.nchannel):
            m = int(oi.visOi.m[dit, c])
            oi.visOi.visProj[dit][c] = np.dot(oi.visOi.h_matrices[dit, c, 0:m, :], oi.visOi.visRef[dit, c, :])

# invert covariance matrices
for k in range(len(objOis)):
    oi = objOis[k]
    printinf("Inverting covariance matrices ({}/{})".format(k+1, len(objOis)))        
    oi.visOi.W2inv = [[[] for c in range(oi.visOi.nchannel)] for dit in range(oi.visOi.ndit)]
    for dit in range(oi.visOi.ndit): 
        for c in range(oi.visOi.nchannel):
            m = int(oi.visOi.m[dit, c])                   
            # propagate projection of visibilities on errors
            H_sp = scipy.sparse.csc_matrix(oi.visOi.h_matrices[dit, c, 0:m, :])
            W_sp = scipy.sparse.csr_matrix(oi.visOi.visRefCov[dit, c].todense())
            Z_sp = scipy.sparse.csr_matrix(oi.visOi.visRefPcov[dit, c].todense())
            W = np.dot((H_sp.dot(W_sp)).todense(), cs.adj(H_sp).todense())
            Z = np.dot((H_sp.dot(Z_sp)).todense(), H_sp.T.todense())
            W2 = cs.extended_covariance(W, Z)#.real
            oi.visOi.W2inv[dit][c] = np.linalg.inv(W2)
            # invert the covariance matrix
            #ZZ, _ = lapack.dpotrf(W2)
            #T, info = lapack.dpotri(ZZ)
            #oi.visOi.W2inv[dit][c] = np.triu(T) + np.triu(T, k=1).T

# prepare chi2Maps
printinf("RA grid: [{:.2f}, {:.2f}] with {:d} points".format(RA_LIM[0], RA_LIM[1], N_RA))
printinf("DEC grid: [{:.2f}, {:.2f}] with {:d} points".format(DEC_LIM[0], DEC_LIM[1], N_DEC))
raValues = np.linspace(RA_LIM[0], RA_LIM[1], N_RA)
decValues = np.linspace(DEC_LIM[0], DEC_LIM[1], N_DEC)
chi2Maps = np.zeros([len(objOis), N_RA, N_DEC])
# to store best fits values on each file
bestFits = []
kappas = np.zeros(len(objOis))
raBests = np.zeros(len(objOis))
decBests = np.zeros(len(objOis))

for k in range(len(objOis)):
    oi = objOis[k]
    chi2Best = np.inf
    # prepare array
    phi_values = np.zeros([oi.visOi.ndit, oi.visOi.nchannel, oi.nwav], 'complex')
    # calculate As and Bs
    phiBest = np.zeros([oi.visOi.ndit, oi.visOi.nchannel, oi.nwav], "complex")
    printinf("Calculating chi2Map for file {}".format(oi.filename))
    for i in range(N_RA):
        for j in range(N_DEC):
            ra = raValues[i]
            dec = decValues[j]
            A, B = 0., 0.
            kappa2 = 0
            this_u = (ra*oi.visOi.uCoord + dec*oi.visOi.vCoord)/1e7
            phase = 2*np.pi*this_u*1e7/3600.0/360.0/1000.0*2*np.pi
            phi = np.exp(-1j*phase)*np.abs(visRefs[k])            
            if not(contrast_data is None):
                phi = phi*contrast_data
            phiProj = [[[] for c in range(oi.visOi.nchannel)] for dit in range(oi.visOi.ndit)]
            for dit in range(oi.visOi.ndit):
                for c in range(oi.visOi.nchannel):
                    m = int(oi.visOi.m[dit, c])
                    bad_indices = np.where(oi.visOi.flag[dit, c, :])
                    phi[dit, c, bad_indices] = 0
                    phiProj[dit][c] = np.dot(oi.visOi.h_matrices[dit, c, 0:m, :], phi[dit, c, :])
            for dit in range(oi.visOi.ndit):
                for c in range(oi.visOi.nchannel):
                    phiProj2 = cs.conj_extended(phiProj[dit][c])
                    PV2 = cs.conj_extended(oi.visOi.visProj[dit][c])            
                    Q = np.dot(cs.adj(phiProj2), oi.visOi.W2inv[dit][c])
                    A = A+np.real(np.dot(Q, PV2))
                    B = B+np.real(np.dot(Q, phiProj2))
            kappa = A/B
            chi2Maps[k, i, j] = -A**2/B # constant term was removed in this formula
            if kappa < 0: # negative contrast if forbidden
                kappa = 0
                chi2Maps[k, i, j] = 0 # 0 is the reference chi2 in the calculations, since constant term of the likelihood was removed
            if chi2Maps[k, i, j] < chi2Best:
                phiBest = kappa*phi
                chi2Best = chi2Maps[k, i, j]
                raBests[k] = ra
                decBests[k] = dec
                kappas[k] = kappa
    bestFits.append(phiBest)

# calculate best fit with only a star model by projetting visibilities                
bestFitStars = []
for k in range(len(objOis)):
    oi = objOis[k]
    bestFitStar = np.zeros([oi.visOi.ndit, oi.visOi.nchannel, oi.nwav], "complex")
    for dit in range(oi.visOi.ndit):
        for c in range(oi.visOi.nchannel):
            bestFitStar[dit, c, :] = oi.visOi.visRef[dit, c, :] - np.dot(oi.visOi.p_matrices[dit, c, :, :], oi.visOi.visRef[dit, c, :])
    bestFitStars.append(bestFitStar)

# also project the best fit in order to be able to compare it with data
for k in range(len(objOis)):
    oi = objOis[k]
    for dit in range(oi.visOi.ndit):
        for c in range(oi.visOi.nchannel):
            bestFits[k][dit, c, :] = np.dot(oi.visOi.p_matrices[dit, c, :, :], bestFits[k][dit, c, :])
    
# combine chi2Maps to get best astrometric solution
chi2Map = np.sum(chi2Maps, axis = 0)
ind = np.where(chi2Map == np.min(chi2Map))
raBest = raValues[ind[0][0]]
decBest = decValues[ind[1][0]]

# for each map, look for local minimum close to the best solution
raBests_local = np.zeros(len(objOis))
decBests_local = np.zeros(len(objOis))
for k in range(len(objOis)):
    printinf("Looking for local chi2 minimun for file {}".format(objOis[k].filename))
    i, j = np.where(raValues == raBest)[0][0], np.where(decValues == decBest)[0][0]    
    i, j = findLocalMinimum(chi2Maps[k, :, :], i, j, jump_size=5)
    raBests_local[k] = raValues[i]
    decBests_local[k] = decValues[j]
    if (raBests_local[k] != raBests[k]) or (decBests_local[k] != decBests[k]):
        printwar("Local minimum for file {} is different from global minimum".format(objOis[k].filename))


# look for best opd on each baseline
bestOpds = np.zeros([len(objOis), objOis[0].visOi.nchannel])
for k in range(len(objOis)):
    oi = objOis[k]
    printwar("Looking for best OPD on each baseline for file {} is different from global minimum".format(objOis[k].filename))    
    opdLimits = np.zeros([oi.visOi.nchannel, 2]) # min and max for each channel
    opdLimits[:, 0] = +np.inf
    opdLimits[:, 1] = -np.inf
    # we'll go tgrough all the values and find the largets and smallest opd
    for ra in raValues:
        for dec in decValues:
            opd = oi.visOi.getOpd(ra, dec) # opd[dit, c] is opd on channel c for dit dit corresponding to the ra/dec
            for c in range(oi.visOi.nchannel):
                if np.min(opd[:, c]) < opdLimits[c, 0]: # if bigger than current biggest we keep it
                    opdLimits[c, 0] = np.min(opd[:, c])
                if np.max(opd[:, c]) > opdLimits[c, 1]: # if smaller than current smallest we keep it
                    opdLimits[c, 1] = np.max(opd[:, c])
    # now we can create the array using request nopd
    opdRanges = np.zeros([oi.visOi.nchannel, N_OPD])
    for c in range(oi.visOi.nchannel):
        opdRanges[c, :] = np.linspace(opdLimits[c, 0], opdLimits[c, 1], N_OPD)
    # calculate the opd map
    opdFit = oi.visOi.fitVisRefOpd(opdRanges, target_spectrum = np.abs(visRefs[k]), poly_spectrum = np.abs(visRefs[k]), poly_order = STAR_ORDER, no_inv = False)
    bestOpds[k, :] = opdFit["best"].mean(axis = 0)

decOpdBaselines = np.zeros([len(objOis), objOis[0].visOi.nchannel, 3, N_RA])
for k in range(len(objOis)):
    oi = objOis[k]
#    wavref = 1e6*np.mean(oi.fluxOi.flux.mean(axis = 0).mean(axis = 0)*oi.wav)/np.mean(oi.fluxOi.flux) # flux weighted average wavelength in microns
    wavref = 1e6*np.mean(oi.wav)
    for c in range(oi.visOi.nchannel):
        for j in range(np.shape(decOpdBaselines)[2]):
            decOpdBaselines[k, c, j, :] = ((bestOpds[k, c]+(j-np.shape(decOpdBaselines)[2]//2)*wavref)/1e6*1000.0*3600.0*360.0/2.0/np.pi - raValues*oi.visOi.u[:, c].mean(axis = 0))/oi.visOi.v[:, c].mean(axis = 0)
        
cov = np.cov(raBests, decBests)
printinf("RA: {:.2f}+-{:.3f} mas".format(np.mean(raBests), cov[0, 0]**0.5))
printinf("DEC: {:.2f}+-{:.3f} mas".format(np.mean(decBests), cov[1, 1]**0.5))
printinf("COV: {:.2f}".format(cov[0, 1]/np.sqrt(cov[0, 0]*cov[1, 1])))

cov_local = np.cov(raBests_local, decBests_local)
printinf("RA (from combined map): {:.2f}+-{:.3f} mas".format(raBest, cov_local[0, 0]**0.5))
printinf("DEC (from combined map): {:.2f}+-{:.3f} mas".format(decBest, cov_local[1, 1]**0.5))

printinf("Contrast obtained (mean, min, max): {:.2e}, {:.2e}, {:.2e}".format(np.mean(kappas), np.min(kappas), np.max(kappas)))

for k in range(len(PLANET_FILES)):
    ind = cfg["general"]["reduce"][k]
    cfg["planet_ois"][ind]["astrometric_solution"] = [float(raBests[k]), float(decBests[k])] # YAML cannot convert numpy types
            
f = open(CONFIG_FILE, "w")
if RUAMEL:
    f.write(ruamel.yaml.dump(cfg, Dumper=ruamel.yaml.RoundTripDumper))
else:
    f.write(yaml.safe_dump(cfg, default_flow_style = False)) 
f.close()


if not(FIGDIR is None):
    hdu = fits.PrimaryHDU(chi2Maps.transpose(0, 2, 1))
    hdu.header["CRPIX1"] = 0.0
    hdu.header["CRVAL1"] = raValues[0]
    hdu.header["CDELT1"] = raValues[1] - raValues[0]
    hdu.header["CRPIX2"] = 0.0
    hdu.header["CRVAL2"] = decValues[0]
    hdu.header["CDELT2"] = decValues[1] - decValues[0]
    hdul = fits.HDUList([hdu])
    hdul.writeto(FIGDIR+"/chi2Maps.fits", overwrite = True)
    
    with PdfPages(FIGDIR+"/astrometry_fit_results.pdf") as pdf:
        dRa = raValues[1] - raValues[0]
        dDec = decValues[1] - decValues[0]    
        mapExtent = [np.min(raValues), np.max(raValues), np.min(decValues), np.max(decValues)]
    
        plt.figure()
        plt.imshow(chi2Map.T, origin = "lower", extent = mapExtent)
        plt.xlabel("$\Delta{}\mathrm{RA}$ (mas)")
        plt.ylabel("$\Delta{}\mathrm{DEC}$ (mas)")
        pdf.savefig()

        # UV plot
        plt.figure()
        gPlot.uvMap(objOis[0].visOi.uCoord[0, :, :], objOis[0].visOi.vCoord[0, :, :], targetCoord=(raBest, decBest), legend = objOis[0].basenames, symmetric = True)
        pdf.savefig()

        for k in range(len(objOis)):
            oi = objOis[k]
            fig = plt.figure(figsize=(10, 8))
            gPlot.reImPlot(w, np.ma.masked_array(oi.visOi.visRef-bestFitStars[k], oi.visOi.flag).mean(axis = 0), subtitles = oi.basenames, fig = fig)
            gPlot.reImPlot(w, np.ma.masked_array(bestFits[k], oi.visOi.flag).mean(axis = 0), fig = fig)
            plt.legend([oi.filename.split("/")[-1], "Astrometry fit"])
            pdf.savefig()
            fig = plt.figure(figsize=(10, 8))        
            gPlot.reImPlot(w, np.ma.masked_array(oi.visOi.visRef, oi.visOi.flag).mean(axis = 0), subtitles = oi.basenames, fig = fig)
            gPlot.reImPlot(w, np.ma.masked_array(bestFitStars[k], oi.visOi.flag).mean(axis = 0), fig = fig)
            plt.legend([oi.filename.split("/")[-1], "Star fit"])
            pdf.savefig()

            fig = plt.figure(figsize = (10, 10))
            n = int(np.ceil(np.sqrt(len(objOis)+1)))
            ax = fig.add_subplot(n, n, 1)
            gPlot.uvMap(objOis[0].visOi.uCoord[0, :, :], objOis[0].visOi.vCoord[0, :, :], targetCoord=(raBest, decBest), legend = objOis[0].basenames, symmetric = True, ax = ax, colors = ["C"+str(k) for k in range(6)])
        for k in range(len(objOis)):
            ax = fig.add_subplot(n, n, k+2)
            oi = objOis[k]
            name = oi.filename.split('/')[-1]
            ax.imshow(chi2Maps[k, :, :].T, origin = "lower", extent = mapExtent)
            ax.plot(raBests[k], decBests[k], "+C1")
            ax.plot(raBests_local[k], decBests_local[k], "+C2")                 
            ax.plot(raBest, decBest, "+C3")
            for c in range(oi.visOi.nchannel):
                if c not in IGNORE_BASELINES:
                    for j in range(np.shape(decOpdBaselines)[2]):
                        if j == np.shape(decOpdBaselines)[2]//2:
                            ax.plot(raValues, decOpdBaselines[k, c, j, :], 'C'+str(c)+'-', linewidth=2)
                        else:
                            ax.plot(raValues, decOpdBaselines[k, c, j, :], 'C'+str(c)+'--', linewidth=2)                        
            plt.xlim(mapExtent[0], mapExtent[1])
            plt.ylim(mapExtent[2], mapExtent[3])
            ax.set_xlabel("$\Delta{}\mathrm{RA}$ (mas)")
            ax.set_ylabel("$\Delta{}\mathrm{DEC}$ (mas)")
            ax.set_title(name)
        plt.tight_layout()
        pdf.savefig()

        fig = plt.figure(figsize = (6, 6))
        ax = fig.add_subplot(111)
        ax.plot(raBests, decBests, "oC0")
        ax.plot(raBests_local, decBests_local, "oC1")    
        val, vec = np.linalg.eig(cov)
        e1=matplotlib.patches.Ellipse((raBests.mean(),decBests.mean()), 2*val[0]**0.5, 2*val[1]**0.5, angle=np.arctan2(vec[0,1],-vec[1,1])/np.pi*180, fill=False, color='C0', linewidth=2, linestyle='--')
        e2=matplotlib.patches.Ellipse((np.mean(raBests), np.mean(decBests)), 3*2*val[0]**0.5, 3*2*val[1]**0.5, angle=np.arctan2(vec[0, 1], -vec[1, 1])/np.pi*180.0, fill=False, color='C0', linewidth=2)
        ax.add_patch(e1)
        ax.add_patch(e2)    
        ax.plot([np.mean(raBests)], [np.mean(decBests)], '+C0')
        ax.text(raBests.mean()+val[0]**0.5, decBests.mean()+val[1]**0.5, "RA={:.2f}+-{:.3f}\nDEC={:.2f}+-{:.3f}\nCOV={:.2f}".format(np.mean(raBests), cov[0, 0]**0.5, np.mean(decBests), cov[1, 1]**0.5, cov[0, 1]/np.sqrt(cov[0, 0]*cov[1, 1])))
    
        val, vec = np.linalg.eig(cov_local)
        e1=matplotlib.patches.Ellipse((raBests_local.mean(), decBests_local.mean()), 2*val[0]**0.5, 2*val[1]**0.5, angle=np.arctan2(vec[0,1],-vec[1,1])/np.pi*180, fill=False, color='C1', linewidth=2, linestyle='--')
        e2=matplotlib.patches.Ellipse((np.mean(raBests_local), np.mean(decBests_local)), 3*2*val[0]**0.5, 3*2*val[1]**0.5, angle=np.arctan2(vec[0, 1], -vec[1, 1])/np.pi*180.0, fill=False, color='C1', linewidth=2)
        ax.add_patch(e1)
        ax.add_patch(e2)    
        ax.plot([np.mean(raBests_local)], [np.mean(decBests_local)], '+C1')
        ax.text(raBests_local.mean()-val[0]**0.5, decBests_local.mean()-val[1]**0.5, "RA={:.2f}+-{:.3f}\nDEC={:.2f}+-{:.3f}\nCOV={:.2f}".format(np.mean(raBests_local), cov_local[0, 0]**0.5, np.mean(decBests_local), cov_local[1, 1]**0.5, cov_local[0, 1]/np.sqrt(cov_local[0, 0]*cov_local[1, 1])))    
        ax.set_xlabel("$\Delta{}\mathrm{RA}$ (mas)")
        ax.set_ylabel("$\Delta{}\mathrm{DEC}$ (mas)")
        plt.axis("equal")
        pdf.savefig()
    
if SAVE_RESIDUALS:
    np.save(FIGDIR+"/"+"wav.npy", w[0, :])    
    for k in range(len(objOis)):
        oi = objOis[k]
        name = oi.filename.split('/')[-1].split('.fits')[0]
        visRef = oi.visOi.visRef
        visFitStar = bestFitStars[k]
        visFit = bestFits[k]
        flags = oi.visOi.flag
        visFt = oi.visOi.visDataFt        
        np.save(FIGDIR+"/"+name+"_ref.npy", visRef)
        np.save(FIGDIR+"/"+name+"_fit.npy", visFit)
        np.save(FIGDIR+"/"+name+"_starfit.npy", visFitStar)
        np.save(FIGDIR+"/"+name+"_flags.npy", flags)
        np.save(FIGDIR+"/"+name+"_ft.npy", visFt)                                

        


