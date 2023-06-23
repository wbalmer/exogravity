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
GRADIENT = cfg["general"]["gradient"]
USE_LOCAL = cfg["general"]["use_local"]
FIGDIR = cfg["general"]["figdir"]
SAVE_RESIDUALS = cfg["general"]["save_residuals"]
PLANET_FILES = [DATA_DIR+cfg["planet_ois"][preduce[list(preduce.keys())[0]]["planet_oi"]]["filename"] for preduce in cfg["general"]["reduce_planets"]]
PLANET_REJECT_DITS = [preduce[list(preduce.keys())[0]]["reject_dits"] for preduce in cfg["general"]["reduce_planets"]]
PLANET_REJECT_BASELINES = [preduce[list(preduce.keys())[0]]["reject_baselines"] for preduce in cfg["general"]["reduce_planets"]]
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

# OVERWRITE SOME CONFIGURATION VALUES WITH ARGUMENTS FROM COMMAND LINE
if "gofast" in dargs.keys():
    GO_FAST = dargs["gofast"].lower()=="true" # bypass value from config file
if "noinv" in dargs.keys():
    NO_INV = dargs["noinv"] # bypass value from config file
if "figdir" in dargs.keys():
    FIGDIR = dargs["figdir"] # bypass value from config file
if "save_residuals" in dargs.keys():
    SAVE_RESIDUALS = True # bypass value from config file
if "ralim" in dargs.keys():
    RA_LIM = [float(dummy) for dummy in dargs["ralim"].replace("[", "").replace("]", "").split(",")]
if "n_ra" in dargs.keys():
    N_RA = int(dargs["n_ra"])
if "declim" in dargs.keys():
    DEC_LIM = [float(dummy) for dummy in dargs["declim"].replace("[", "").replace("]", "").split(",")]    
if "n_dec" in dargs.keys():
    N_DEC = int(dargs["n_dec"])
if "gradient" in dargs.keys():
    GRADIENT = dargs["gradient"].lower()=="true" # bypass value from config file
if "use_local" in dargs.keys():
    USE_LOCAL = dargs["use_local"].lower()=="true" # bypass value from config file    
    
# LOAD GRAVITY PLOT is savefig requested
if not(FIGDIR is None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from cleanGravity import gravityPlot as gPlot
    if not(os.path.isdir(FIGDIR)):
        os.makedirs(FIGDIR)
        printinf("Directory {} was not found and has been created".format(FIGDIR))

# THROW AN ERROR IF save_residuals requested, but no FIGDIR provided
if SAVE_RESIDUALS and (FIGDIR is None):
    printerr("save_residuals option requested, but no figdir provided.")
        
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
        printerr("Unknown reduction type '{}'.".format(REDUCTION))        
    objOis.append(oi)

# filter data
if REDUCTION == "astrored":
    # flag points based on FT value and phaseRef arclength
    ftThresholdPlanet = cfg["general"]["ftOnPlanetMeanFlux"]*FT_FLUX_THRESHOLD
    for k in range(len(PLANET_FILES)):
        oi = objOis[k]
        filter_ftflux(oi, ftThresholdPlanet)
        if PHASEREF_ARCLENGTH_THRESHOLD > 0:
            filter_phaseref_arclength(oi, PHASEREF_ARCLENGTH_THRESHOLD) 
        # explicitly set ignored dits to NAN
        if not(PLANET_REJECT_DITS[k] is None):
            if len(PLANET_REJECT_DITS[k]) > 0:
                a = PLANET_REJECT_DITS[k]
                b = range(oi.visOi.nchannel)
                (a, b, c) = np.meshgrid(a, b, range(oi.nwav))
                oi.visOi.flagPoints((a, b, c))
                printinf("Ignoring some dits in file {}".format(oi.filename))           
        # explicitly set ignored baselines to NAN
        if not(PLANET_REJECT_BASELINES[k] is None):
            if len(PLANET_REJECT_BASELINES[k]) > 0:            
                a = range(oi.visOi.ndit)
                b = PLANET_REJECT_BASELINES[k]
                (a, b, c) = np.meshgrid(a, b, range(oi.nwav))
                oi.visOi.flagPoints((a, b, c))
                printinf("Ignoring some baselines in file {}".format(oi.filename))

# replace data by the mean over all DITs if go_fast has been requested in the config file
if (GO_FAST):
    printinf("gofast flag is set. Averaging over DITs")            
    for oi in objOis: # mean should not be calculated on swap before phase correction
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

# A function to calculate the chi2 for a given oi and ra/dec
def compute_chi2(oi, ra, dec):
    # prepare array
    phi_values = np.zeros([oi.visOi.ndit, oi.visOi.nchannel, oi.nwav], 'complex')
    # calculate As and Bs
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
    if kappa < 0: # negative contrast if forbidden
        kappa = 0
        return 0, 0
    else:
        return -A**2/B, kappa

# use the above function to fill out the chi2 maps
# prepare chi2Maps
printinf("RA grid: [{:.2f}, {:.2f}] with {:d} points".format(RA_LIM[0], RA_LIM[1], N_RA))
printinf("DEC grid: [{:.2f}, {:.2f}] with {:d} points".format(DEC_LIM[0], DEC_LIM[1], N_DEC))
raValues = np.linspace(RA_LIM[0], RA_LIM[1], N_RA)
decValues = np.linspace(DEC_LIM[0], DEC_LIM[1], N_DEC)
chi2Maps = np.zeros([len(objOis), N_RA, N_DEC])
# to store best fits values on each file
bestFits = []
kappas = np.zeros(len(objOis))
raBests_global = np.zeros(len(objOis))
decBests_global = np.zeros(len(objOis))
bestFitStars = []    

for k in range(len(objOis)):        
    printinf("Calculating chi2Map for file {}".format(objOis[k].filename))
    chi2Best = np.inf
    for i in range(N_RA):
        for j in range(N_DEC):
            ra = raValues[i]
            dec = decValues[j]
            chi2, kappa = compute_chi2(objOis[k], ra, dec)
            chi2Maps[k, i, j] = chi2
            if chi2Maps[k, i, j] < chi2Best:
                chi2Best = chi2Maps[k, i, j]
                raBests_global[k] = ra
                decBests_global[k] = dec
                kappas[k] = kappa                
                this_u = (ra*objOis[k].visOi.uCoord + dec*objOis[k].visOi.vCoord)/1e7 
                phase = 2*np.pi*this_u*1e7/3600.0/360.0/1000.0*2*np.pi 
                phi = np.exp(-1j*phase)*np.abs(visRefs[k])                            
                phiBest = kappa*phi
    bestFits.append(phiBest)            

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
    
# either the global (default) or local minimum can be used as the best initial guess (our simply best astrometry)
if USE_LOCAL:
    printinf("Using local minima instead of global minima for each DIT")
    raGuesses = np.copy(raBests_local)
    decGuesses = np.copy(decBests_local)
else:
    raGuesses = np.copy(raBests_global)
    decGuesses = np.copy(decBests_global)
    
raBests = np.copy(raGuesses)
decBests = np.copy(decGuesses)    

# if gradient descent is requested, we use the best chi2 to start a gradient search for true minimum
formal_errors = []
if GRADIENT:
    for k in range(len(objOis)):
        printinf("Performing gradient-descent for file {}".format(oi.filename)) 
        ndof = 2*objOis[k].visOi.ndit*objOis[k].visOi.nchannel*(objOis[k].visOi.nwav - 6) - 1 # number of dof (*2 because these are complex numbers       
        chi2 = lambda astro : compute_chi2(objOis[k], astro[0], astro[1])[0]/ndof # only chi2, not kappa. 
        opt = scipy.optimize.minimize(chi2, x0=[raGuesses[k], decGuesses[k]])
        raBests[k] = opt["x"][0]
        decBests[k] = opt["x"][1]
        # from the Hessian matrix, we can also look for the fiducial error bars
        ra_err = np.sqrt(opt["hess_inv"][0, 0])
        dec_err = np.sqrt(opt["hess_inv"][1, 1])        
        rho = opt["hess_inv"][0, 1]/np.sqrt(ra_err**2*dec_err**2)
        formal_errors.append([ra_err, dec_err, rho])

# calculate the best fits
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
    

# look for best opd on each baseline
bestOpds = np.zeros([len(objOis), objOis[0].visOi.nchannel])
for k in range(len(objOis)):
    oi = objOis[k]
    printinf("Looking for best OPD on each baseline for file {}".format(objOis[k].filename))    
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
            u = np.ma.masked_array(oi.visOi.uCoord[:, c, :]*oi.visOi.wav, oi.visOi.flag[:, c, :])
            v = np.ma.masked_array(oi.visOi.vCoord[:, c, :]*oi.visOi.wav, oi.visOi.flag[:, c, :])
            decOpdBaselines[k, c, j, :] = ((bestOpds[k, c]+(j-np.shape(decOpdBaselines)[2]//2)*wavref)/1e6*1000.0*3600.0*360.0/2.0/np.pi - raValues*u.mean())/v.mean()
            
cov = np.cov(raBests, decBests)
printinf("RA: {:.2f}+-{:.3f} mas".format(np.mean(raBests), cov[0, 0]**0.5))
printinf("DEC: {:.2f}+-{:.3f} mas".format(np.mean(decBests), cov[1, 1]**0.5))
printinf("COV: {:.2f}".format(cov[0, 1]/np.sqrt(cov[0, 0]*cov[1, 1])))

#cov_local = np.cov(raBests_local, decBests_local)
#printinf("RA (from combined map): {:.2f}+-{:.3f} mas".format(raBest, cov_local[0, 0]**0.5))
#printinf("DEC (from combined map): {:.2f}+-{:.3f} mas".format(decBest, cov_local[1, 1]**0.5))
#printinf("COV (from combined map): {:.2f} mas".format(cov_local[0, 1]/np.sqrt(cov_local[0, 0]*cov_local[1, 1])))

printinf("Contrast obtained (mean, min, max): {:.2e}, {:.2e}, {:.2e}".format(np.mean(kappas), np.min(kappas), np.max(kappas)))

for k in range(len(PLANET_FILES)):
    preduce = cfg["general"]["reduce_planets"][k]
    preduce[list(preduce.keys())[0]]["astrometric_guess"] = [float(raGuesses[k]), float(decGuesses[k])] # YAML cannot convert numpy types    
    preduce[list(preduce.keys())[0]]["astrometric_solution"] = [float(raBests[k]), float(decBests[k])] # YAML cannot convert numpy types
    ra_err, dec_err, rho = formal_errors[k]
    preduce[list(preduce.keys())[0]]["formal_errors"] = [float(ra_err), float(dec_err), float(rho)] # YAML cannot convert numpy types
            
f = open(CONFIG_FILE, "w")
if RUAMEL:
    f.write(ruamel.yaml.dump(cfg, Dumper=ruamel.yaml.RoundTripDumper))
else:
    f.write(yaml.safe_dump(cfg, default_flow_style = False)) 
f.close()

# get max distance in UV plane for plotting UV maps
coords = np.concatenate([np.sqrt(oi.visOi.uCoord**2+oi.visOi.vCoord**2) for oi in objOis])
r = np.nanmax(coords)
        
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
    
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        im = ax.imshow(chi2Map.T, origin = "lower", extent = mapExtent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_xlabel("$\Delta{}\mathrm{RA}$ (mas)")
        ax.set_ylabel("$\Delta{}\mathrm{DEC}$ (mas)")
        pdf.savefig()
        plt.close(fig)

        fig = plt.figure(figsize = (10, 10))
        n = int(np.ceil(np.sqrt(len(objOis)+1)))
        # plot all baselines. For this, fo through all objOis and keep track of which ones are plotted already
        ax = fig.add_subplot(n, n, 1)
        plotted = []
        for k in range(len(objOis)):
            oi = objOis[k]
            uCoord = oi.visOi.uCoord
            vCoord = oi.visOi.vCoord
            uCoord[np.where(uCoord == 0)] = float("nan")
            vCoord[np.where(vCoord == 0)] = float("nan")
            if k == 0:
                gPlot.uvMap(np.nanmean(uCoord, axis = 0), np.nanmean(vCoord, axis = 0), targetCoord=(raBest, decBest), symmetric = True, ax = ax, colors = ["C"+str(c) for c in range(oi.visOi.nchannel)], lim = 2*r)
            else:
                gPlot.uvMap(np.nanmean(uCoord, axis=0), np.nanmean(vCoord, axis = 0), targetCoord=None, symmetric = True, ax = ax, colors = ["C"+str(c) for c in range(oi.visOi.nchannel)], lim=2*r)
        for k in range(len(objOis)):
            ax = fig.add_subplot(n, n, k+2)
            oi = objOis[k]
            name = oi.filename.split('/')[-1]
            im = ax.imshow(chi2Maps[k, :, :].T, origin = "lower", extent = mapExtent)
            ax.plot(raBests[k], decBests[k], "+C1")
            ax.plot(raBests_local[k], decBests_local[k], "+C2")                 
            ax.plot(raBest, decBest, "+C3")
            reject_baselines = PLANET_REJECT_BASELINES[k]
            if reject_baselines is None:
                reject_baselines = []
            for c in range(oi.visOi.nchannel):
                if not(c in reject_baselines):
                    for j in range(np.shape(decOpdBaselines)[2]):
                        if j == np.shape(decOpdBaselines)[2]//2:
                            ax.plot(raValues, decOpdBaselines[k, c, j, :], 'C'+str(c)+'-', linewidth=2)
                        else:
                            ax.plot(raValues, decOpdBaselines[k, c, j, :], 'C'+str(c)+'--', linewidth=2)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            ax.set_xlim(mapExtent[0], mapExtent[1])
            ax.set_ylim(mapExtent[2], mapExtent[3])
            ax.set_xlabel("$\Delta{}\mathrm{RA}$ (mas)")
            ax.set_ylabel("$\Delta{}\mathrm{DEC}$ (mas)")
            ax.set_title(name)
        plt.tight_layout()
        pdf.savefig()
        plt.close(fig)
        
        fig = plt.figure(figsize = (10, 10))
        ax = fig.add_subplot(111)
        ax.plot(raGuesses, decGuesses, "oC4", label = "Global min per file")
        ax.plot(raBests_local, decBests_local, "+C1", label = "Local min per file")
        ax.plot(raBests, decBests, "+C2", alpha = 0.6)
        if GRADIENT:
            for k in range(len(objOis)):
                # a line to connect the initial guess to the optimized solution
                if k==0:
                    l1, l2 = "Gradient descent per file", "Formal $\chi^2$ error"
                else:
                    l1, l2 = None, None
                ax.plot([raGuesses[k], raBests[k]], [decGuesses[k], decBests[k]], "C2", linestyle = "dotted", label = l1, alpha = 0.6)
                # the individual error ellipse derived from chi2
                ra_err, dec_err, rho = formal_errors[k]
                cov = np.array([[ra_err**2, rho*ra_err*dec_err], [rho*ra_err*dec_err, dec_err**2]])# reconstruct covariance                
                val, vec = np.linalg.eig(cov) 
                e1=matplotlib.patches.Ellipse((raBests[k], decBests[k]), 2*val[0]**0.5, 2*val[1]**0.5, angle=np.arctan2(vec[0,1],-vec[1,1])/np.pi*180, fill=False, color='C2', linewidth=2, alpha = 0.6, linestyle='--', label = l2)       
                ax.add_patch(e1)

        try:
            cov = np.cov(raBests, decBests)
            val, vec = np.linalg.eig(cov)
            e1=matplotlib.patches.Ellipse((raBests.mean(),decBests.mean()), val[0]**0.5, val[1]**0.5, angle=np.arctan2(vec[0,1],-vec[1,1])/np.pi*180, fill=False, color='C0', linewidth=2, linestyle='-', label = "Dispersion on gradient descent")
            ax.add_patch(e1)
            ax.plot([np.mean(raBests)], [np.mean(decBests)], '+C0', label = "Mean of gradient descent")
            ax.text(raBests.mean()+val[0]**0.5, decBests.mean()+val[1]**0.5, "RA={:.2f}+-{:.3f}\nDEC={:.2f}+-{:.3f}\nCOV={:.2f}".format(np.mean(raBests), cov[0, 0]**0.5, np.mean(decBests), cov[1, 1]**0.5, cov[0, 1]/np.sqrt(cov[0, 0]*cov[1, 1])), color="C0")
        except np.linalg.LinAlgError:
            printwar("infs or Nans when calculating covariance on astrometry solutions")
    
        ax.legend(loc=4)
        ax.set_xlabel("$\Delta{}\mathrm{RA}$ (mas)")
        ax.set_ylabel("$\Delta{}\mathrm{DEC}$ (mas)")
        plt.axis("equal")       
        pdf.savefig()
        plt.close(fig)

        
        """
        # UV plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for k in range(len(objOis)):
            oi = objOis[k]
            uCoord = oi.visOi.uCoord
            vCoord = oi.visOi.vCoord
            uCoord[np.where(uCoord == 0)] = float("nan")
            vCoord[np.where(vCoord == 0)] = float("nan")
            if k == 0:
                gPlot.uvMap(np.nanmean(uCoord, axis = 0), np.nanmean(vCoord, axis = 0), targetCoord=(raBest, decBest), symmetric = True, ax = ax, colors = ["C"+str(c) for c in range(oi.visOi.nchannel)], lim = 2*r)
            else:
                gPlot.uvMap(np.nanmean(uCoord, axis=0), np.nanmean(vCoord, axis = 0), targetCoord=None, symmetric = True, ax = ax, colors = ["C"+str(c) for c in range(oi.visOi.nchannel)], lim=2*r) 
        pdf.savefig()
        plt.close(fig)
        """
        
        for k in range(len(objOis)):
            oi = objOis[k]

            if not(GO_FAST):
                fig = plt.figure(figsize=(10, 8))
                for dit in range(oi.visOi.ndit):
                    if dit == 0:
                        gPlot.reImPlot(w, np.ma.masked_array((oi.visOi.visRef-bestFitStars[k])[dit, :, :], oi.visOi.flag[dit, :, :]), subtitles = oi.basenames, fig = fig, xlabel = "Wavelength ($\mu\mathrm{m}$)", color="C"+str(dit), alpha = 0.5)
                    else:
                        gPlot.reImPlot(w, np.ma.masked_array((oi.visOi.visRef-bestFitStars[k])[dit, :, :], oi.visOi.flag[dit, :, :]), fig = fig, color="C"+str(dit), alpha = 0.5)
                    gPlot.reImPlot(w, np.ma.masked_array(bestFits[k], oi.visOi.flag)[dit, :, :], fig = fig, color="C"+str(dit), linestyle="--", alpha = 0.5)
                plt.legend([oi.filename.split("/")[-1], "Astrometry fit"])
                pdf.savefig()
                plt.close(fig)

                # combine fit and data or fit and res in single array
                dataFit = np.ma.masked_array(np.zeros([oi.visOi.ndit, 2*oi.visOi.nchannel, oi.visOi.nwav], "complex"), np.zeros([oi.visOi.ndit, 2*oi.visOi.nchannel, oi.visOi.nwav]))
                dataRes = np.ma.masked_array(np.zeros([oi.visOi.ndit, 2*oi.visOi.nchannel, oi.visOi.nwav], "complex"), np.zeros([oi.visOi.ndit, 2*oi.visOi.nchannel, oi.visOi.nwav]))                
                for c in range(oi.visOi.nchannel):
                    dataFit[:, 2*c, :] = np.ma.masked_array(oi.visOi.visRef[:, c, :] - bestFitStars[k][:, c, :], oi.visOi.flag[:, c, :])
                    dataFit[:, 2*c+1, :] = np.ma.masked_array(bestFits[k][:, c, :], oi.visOi.flag[:, c, :])
                    dataRes[:, 2*c, :] = np.ma.masked_array(oi.visOi.visRef[:, c, :] - bestFitStars[k][:, c, :], oi.visOi.flag[:, c, :])                    
                    dataRes[:, 2*c+1, :] = np.ma.masked_array(oi.visOi.visRef[:, c, :] - bestFitStars[k][:, c, :] - bestFits[k][:, c, :], oi.visOi.flag[:, c, :])
                vmin = np.min([np.min(np.real(dataFit)), np.min(np.imag(dataFit))])
                vmax = np.max([np.max(np.real(dataFit)), np.max(np.imag(dataFit))])     
                fig = plt.figure(figsize=(10, 8))
                subtitles = [item for sublist in [[b+"\nDATA", b+"\nFIT"] for b in oi.basenames] for item in sublist]
                gPlot.reImWaterfall(dataFit, subtitles = subtitles, fig = fig, vmin = vmin, vmax = vmax)
                pdf.savefig()
                plt.close(fig)
                fig = plt.figure(figsize=(10, 8))
                subtitles = [item for sublist in [[b+"\nDATA", b+"\nRES"] for b in oi.basenames] for item in sublist]
                gPlot.reImWaterfall(dataRes, subtitles = subtitles, fig = fig, vmin = vmin, vmax = vmax)
                pdf.savefig()
                plt.close(fig)                
                
            fig = plt.figure(figsize=(10, 8))
            # to plot the average planet in the star frame, we cannot naively average because it would blur the fringes.
            # we need to convert into the planet frame, calculate the mean here, and then move back to the star frame
            wavelet = oi.visOi.getWavelet(raBests[k], decBests[k]) # we'll use the conj to move to planet frame
            waveletMean = np.exp(-1j*2*np.pi/w*np.tile(oi.visOi.getOpd(raBests[k], decBests[k]).mean(axis = 0), [oi.nwav, 1]).T) # to move the mean back to star frame
            gPlot.reImPlot(w, np.ma.masked_array(np.conj(wavelet)*(oi.visOi.visRef-bestFitStars[k]), oi.visOi.flag).mean(axis = 0)*waveletMean, subtitles = oi.basenames, fig = fig, xlabel = "Wavelength ($\mu\mathrm{m}$)")
            gPlot.reImPlot(w, np.ma.masked_array(np.conj(wavelet)*bestFits[k], oi.visOi.flag).mean(axis = 0)*waveletMean, fig = fig)
            plt.legend([oi.filename.split("/")[-1], "Astrometry fit"])
            pdf.savefig()
            
            fig = plt.figure(figsize=(10, 8))
            gPlot.reImPlot(w, np.ma.masked_array(oi.visOi.visRef, oi.visOi.flag).mean(axis = 0), subtitles = oi.basenames, fig = fig, xlabel = "Wavelength ($\mu\mathrm{m}$)")
            gPlot.reImPlot(w, np.ma.masked_array(bestFitStars[k], oi.visOi.flag).mean(axis = 0), fig = fig)
            plt.legend([oi.filename.split("/")[-1], "Star fit"])
            pdf.savefig()
            plt.close(fig)


        
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

        


