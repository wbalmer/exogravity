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

# IMPORTS
import numpy as np
import scipy.sparse, scipy.sparse.linalg
from scipy.linalg import lapack
import astropy.io.fits as fits
import cleanGravity as gravity
from cleanGravity import complexstats as cs
from cleanGravity.utils import loadFitsSpectrum, saveFitsSpectrum
from utils import *
import glob
import itertools
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
        stop()

CONFIG_FILE = dargs["config_file"]
if not(os.path.isfile(CONFIG_FILE)):
    raise Exception("Error: argument {} is not a file".format(CONFIG_FILE))

if not("save_residuals") in dargs.keys():
    dargs['save_residuals'] = False
    
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

# OVERWRITE SOME OF THE CONFIGURATION VALUES WITH ARGUMENTS FROM COMMAND LINE
if "gofast" in dargs.keys():
    GO_FAST = dargs["gofast"] # bypass value from config file    
if "noinv" in dargs.keys():
    NO_INV = dargs["noinv"] # bypass value from config file
if "figdir" in dargs.keys():
    FIGDIR = dargs["figdir"] # bypass value from config file

# LOAD GRAVITY PLOT is savefig requested
if not(FIGDIR is None):
    from cleanGravity import gravityPlot as gPlot
    import matplotlib.pyplot as plt
    import matplotlib
    if not(os.path.isdir(FIGDIR)):
        os.makedirs(FIGDIR)
        printinf("Directory {} was not found and has been created".format(FIGDIR))

# THROW AN ERROR IF save_residuals requested, but not FIGDIR provided
if dargs['save_residuals'] and (FIGDIR is None):
    printerr("save_residuals option requested, but not figdir provided.")
        
# extract list of useful star ois from the list indicated in the star_indices fields of the config file:
star_indices = []
for k in cfg["general"]["reduce"]:
    star_indices = star_indices+cfg["planet_ois"][k]["star_indices"]
star_indices = list(set(star_indices)) # remove duplicates
STAR_FILES = [DATA_DIR+cfg["star_ois"][k]["filename"] for k in star_indices]


# read the contrast file if given, otherwise simply set it to 1
if CONTRAST_FILE is None:
    contrast = 1
elif CONTRAST_FILE.split('.')[-1].lower() in ["fit", "fits"]:
    contrast_wav, dummy, dummy2, contrast_data, dummy3 = loadFitsSpectrum(CONTRAST_FILE)
else:
    printerr("The given contrast file ({}) should have a fits extension".format(fits))

# two lists to contain the planet and star OIs
starOis = [] # will contain OIs on the central star
objOis = [] # contains the OIs on the planet itself
swapOis = [] # first position of the swap (only in DF_SWAP mode)


# LOAD DATA
for filename in PLANET_FILES:
    printinf("Loading file "+filename)
    oi = gravity.GravityDualfieldAstrored(filename, corrMet = cfg["general"]["corr_met"], extension = EXTENSION, corrDisp = cfg["general"]["corr_disp"])
    objOis.append(oi)
    printinf("File is on planet. FT coherent flux: {:.2e}".format(np.mean(np.abs(oi.visOi.visDataFt))))        

for filename in STAR_FILES:
    printinf("Loading file "+filename)
    if (PHASEREF_MODE == "DF_SWAP"):
        oi = gravity.GravityDualfieldAstrored(filename, corrMet = "drs", extension = EXTENSION, corrDisp = "drs")
    else:
        oi = gravity.GravityDualfieldAstrored(filename, corrMet = cfg["general"]["corr_met"], extension = EXTENSION, corrDisp = cfg["general"]["corr_disp"])
    starOis.append(oi)
    printinf("File is on star")
                                              
for filename in SWAP_FILES:
    printinf("Loading file "+filename)
    oi = gravity.GravityDualfieldAstrored(filename, corrMet = cfg["general"]["corr_met"], extension = EXTENSION, corrDisp = cfg["general"]["corr_disp"])
    swapOis.append(oi)
    printinf("File is from a SWAP")

        
# flag points based on FT value
ftThreshold = np.array([np.abs(oi.visOi.visDataFt).mean() for oi in objOis]).mean()/10.0
printinf("Flag data below FT threshold of {:.2e}".format(ftThreshold))
for oi in objOis:
    oi.visOi.visDataFt[:, 0, :] = ftThreshold/100
    oi.visOi.visDataFt[:, 1, :] = ftThreshold/100
    oi.visOi.visDataFt[:, 2, :] = ftThreshold/100
#    oi.visOi.visDataFt[:, 3, :] = ftThreshold/100
#    oi.visOi.visDataFt[:, 4, :] = ftThreshold/100
#    oi.visOi.visDataFt[:, 5, :] = ftThreshold/100
    a, b = np.where(np.abs(oi.visOi.visDataFt).mean(axis = -1) < ftThreshold)
    (a, b, c) = np.meshgrid(a, b, range(oi.nwav))
    oi.visOi.flagPoints((a, b, c))

    
# replace data by the mean over all DITs if go_fast has been requested in the config file
printinf("gofast flag is set. Averaging over DITs")
for oi in objOis+starOis: # mean should not be calculated on swap before phase correction
    if (GO_FAST):
        printinf("Averaging file {}".format(oi.filename))
        oi.computeMean()
        oi.visOi.recenterPhase(oi.sObjX, oi.sObjY)
        u = np.copy(oi.visOi.u)
        oi.visOi.u = np.zeros([1, oi.visOi.nchannel])
        oi.visOi.u[0, :] = np.mean(u, axis = 0)
        v = np.copy(oi.visOi.v)
        oi.visOi.v = np.zeros([1, oi.visOi.nchannel])        
        oi.visOi.v[0, :] = np.mean(v, axis = 0)                
        oi.visOi.visData = np.tile(np.mean(oi.visOi.visData*~oi.visOi.flag, axis = 0), [1, 1, 1])#/np.sum(~oi.visOi.flag, axis = 0), [1, 1, 1])
        oi.visOi.visRef = np.tile(np.mean(oi.visOi.visRef*~oi.visOi.flag, axis = 0), [1, 1, 1])#/np.sum(~oi.visOi.flag, axis = 0), [1, 1, 1])
        oi.visOi.uCoord = np.tile(np.mean(oi.visOi.uCoord*~oi.visOi.flag, axis = 0), [1, 1, 1])#/np.sum(~oi.visOi.flag, axis = 0), [1, 1, 1])
        oi.visOi.vCoord = np.tile(np.mean(oi.visOi.vCoord*~oi.visOi.flag, axis = 0), [1, 1, 1])#/np.sum(~oi.visOi.flag, axis = 0), [1, 1, 1])
#        oi.visOi.visCov = np.tile(np.mean(oi.visOi.visCov*~oi.visOi.flagCov, axis = 0), [1, 1, 1, 1])#/np.sum(~oi.visOi.flagCov, axis = 0), [1, 1, 1, 1])
#        oi.visOi.visPcov = np.tile(np.mean(oi.visOi.visPcov*~oi.visOi.flagCov, axis = 0), [1, 1, 1, 1])#/np.sum(~oi.visOi.flagCov, axis = 0), [1, 1, 1, 1])
#        oi.visOi.visRefCov = np.tile(np.mean(oi.visOi.visRefCov*~oi.visOi.flagCov, axis = 0), [1, 1, 1, 1])#/np.sum(~oi.visOi.flagCov, axis = 0), [1, 1, 1, 1])
#        oi.visOi.visRefPcov = np.tile(np.mean(oi.visOi.visRefPcov*~oi.visOi.flagCov, axis = 0), [1, 1, 1, 1])#/np.sum(~oi.visOi.flagCov, axis = 0), [1, 1, 1, 1])
        for dit, c in itertools.product(range(1, oi.visOi.ndit), range(oi.visOi.nchannel)):
            oi.visOi.visRefCov[0, c] = oi.visOi.visRefCov[0, c] + oi.visOi.visRefCov[dit, c]
            oi.visOi.visRefPcov[0, c] = oi.visOi.visRefPcov[0, c] + oi.visOi.visRefPcov[dit, c]
        oi.visOi.visRefCov = oi.visOi.visRefCov[0:1, :]/oi.visOi.ndit**2
        oi.visOi.visRefPcov = oi.visOi.visRefPcov[0:1, :]/oi.visOi.ndit**2       
        oi.visOi.flag = np.tile(np.any(oi.visOi.flag, axis = 0), [1, 1, 1])
        oi.visOi.flagCov = np.tile(np.any(oi.visOi.flagCov, axis = 0), [1, 1, 1, 1])        
        oi.visOi.ndit = 1
        oi.ndit = 1        
        oi.visOi.recenterPhase(-oi.sObjX, -oi.sObjY)        
        oi.computeMean()
        oi.visOi.visErr = np.zeros([1, oi.visOi.nchannel, oi.nwav], 'complex')    
        oi.visOi.visErr[0, :, :] = np.copy(oi.visOi.visErrMean)


# calculate the very useful w for plotting
oi = objOis[0]
w = np.zeros([oi.visOi.nchannel, oi.nwav])
wFt = np.zeros([oi.visOi.nchannel, oi.visOi.nwavFt])
for c in range(oi.visOi.nchannel):
    w[c, :] = oi.wav*1e6
    wFt[c, :] = range(oi.visOi.nwavFt)

    
# create the visibility reference. This step depends on PHASEREF_MODE (DF_STAR or DF_SWAP)
printinf("Creating the visibility reference from {:d} star observations.".format(len(starOis)))
visRefs = [oi.visOi.visRef.mean(axis=0)*0 for oi in objOis]
for k in range(len(objOis)):
    planet_ind = cfg["general"]["reduce"][k]
    ampRef = np.zeros([oi.visOi.nchannel, oi.nwav])
    visRef = np.zeros([oi.visOi.nchannel, oi.nwav])
    for ind in cfg["planet_ois"][planet_ind]["star_indices"]:
        # not all star files have been loaded, so we cannot trust the order and we need to explicitly look for the correct one
        starOis_ind = [soi.filename for soi in starOis].index(DATA_DIR+cfg["star_ois"][ind]["filename"]) 
        soi = starOis[starOis_ind]
        visRef = visRef+soi.visOi.visRef.mean(axis = 0)
        ampRef = ampRef+np.abs(soi.visOi.visRef.mean(axis = 0))
    ###===### 
#    visRef = np.zeros([oi.visOi.nchannel, oi.nwav])
#    for ind in range(len(starOis)):
#        soi = starOis[ind]
#        visRef = visRef+soi.visOi.visRef.mean(axis = 0)                                                         
    ###===###
#    visRefs[k] = ampRef/len(cfg["planet_ois"][planet_ind]["star_indices"])*np.exp(1j*np.angle(visRef/len(objOis)))#/len(cfg["planet_ois"][planet_ind]["star_indices"])))#/len(starOis))) ###===###
    visRefs[k] = ampRef/len(cfg["planet_ois"][planet_ind]["star_indices"])*np.exp(1j*np.angle(visRef/len(cfg["planet_ois"][planet_ind]["star_indices"])))#/len(starOis))) ###===###     

# in DF_SWAP mode, thephase reference of the star cannot be used. We need to extract the phase ref from the SWAP observations
if PHASEREF_MODE == "DF_SWAP":
    printinf("DF_SWAP mode set.")
    printinf("Calculating the reference phase from {:d} swap observation".format(len(swapOis)))
    # first we need to shift all of the visibilities to the 0 OPD position, using the separation of the SWAP binary
    # if swap ra and dec values are provided (from the swapReduce script), we can use them
    # otherwise we can default to the fiber separation value
    for k in range(len(swapOis)):
        oi = swapOis[k]
        key = cfg["swap_ois"].keys()[k] # the corresponding key in the config file
        if (not("astrometric_solution" in cfg["swap_ois"][key])):
            printwar("astrometric_solution not provided for swap {:d}. Defaulting to fiber position RA={:.2f}, DEC={:.2f}".format(k, oi.sObjX, oi.sObjY))
            swap_ra = oi.sObjX
            swap_dec = oi.sObjY
        else:
            if oi.swap:
                swap_ra = -cfg["swap_ois"][key]["astrometric_solution"][0]
                swap_dec = -cfg["swap_ois"][key]["astrometric_solution"][1]                
            else:
                swap_ra = cfg["swap_ois"][key]["astrometric_solution"][0]
                swap_dec = cfg["swap_ois"][key]["astrometric_solution"][1]                
        oi.visOi.recenterPhase(swap_ra, swap_dec)
    # now that the visibilities are centered, we can take the mean of the visibilities and 
    # extract the phase. But we need to separate the two positions of the swap
    phaseRef1 = np.zeros([oi.visOi.nchannel, oi.nwav])
    phaseRef2 = np.zeros([oi.visOi.nchannel, oi.nwav])
    for k in range(len(swapOis)):
        if swapOis[k].swap:
            phaseRef2 = phaseRef2+np.mean(swapOis[k].visOi.visRef, axis = 0)
        else:
            phaseRef1 = phaseRef1+np.mean(swapOis[k].visOi.visRef, axis = 0)
    # now we can the phaseref
    phaseRef = 0.5*(np.angle(phaseRef2)+np.angle(phaseRef1))
    # because the phase are defined mod 2pi, phaseref can have pi offsets. We need to unwrap that
    # For this we test the phase continuity of a phase-referenced swap obs
    testCase = np.angle(swapOis[0].visOi.visRef.mean(axis = 0))-phaseRef
    unwrapped = 0.5*np.unwrap(2*testCase)
    correction = unwrapped - testCase
    phaseRef = phaseRef - correction
    # for convenience, we store this ref in visRef angle, getting rid of the useless values from the star
    visRefs = [np.abs(visRef)*np.exp(1j*phaseRef) for visRef in visRefs]


# subtract the reference phase to each OB
printinf("Subtracting phase reference to each planet OI.")
for k in range(len(objOis)):
    oi = objOis[k]
    oi.visOi.addPhase(-np.angle(visRefs[k]))


# prepare chi2Maps
printinf("RA grid: [{:.2f}, {:.2f}] with {:d} points".format(RA_LIM[0], RA_LIM[1], N_RA))
printinf("DEC grid: [{:.2f}, {:.2f}] with {:d} points".format(DEC_LIM[0], DEC_LIM[1], N_DEC))
raValues = np.linspace(RA_LIM[0], RA_LIM[1], N_RA)
decValues = np.linspace(DEC_LIM[0], DEC_LIM[1], N_DEC)
chi2Maps = np.zeros([len(objOis), N_RA, N_DEC])

# calculate limits in OPD
opdLimits = np.zeros([len(objOis), objOis[0].visOi.nchannel, 2])
opdLimits[:, :, 0] = +np.inf
opdLimits[:, :, 1] = -np.inf

for k in range(len(objOis)):
    oi = objOis[k]
    for ra in raValues:
        for dec in decValues:
            opd = oi.visOi.getOpd(ra, dec)
            for c in range(oi.visOi.nchannel):
                if np.min(opd[:, c]) < opdLimits[k, c, 0]:
                    opdLimits[k, c, 0] = np.min(opd[:, c])
                if np.max(opd[:, c]) > opdLimits[k, c, 1]:
                    opdLimits[k, c, 1] = np.max(opd[:, c])

opdRanges = np.zeros([len(objOis), oi.visOi.nchannel, N_OPD])
for k in range(len(objOis)):
    oi = objOis[k]
    for c in range(oi.visOi.nchannel):
        opdRanges[k, c, :] = np.linspace(opdLimits[k, c, 0], opdLimits[k, c, 1], N_OPD)

# calculate maps in OPD
opdChi2Maps = []
bestOpdFits = []
nditsTot = np.sum(np.array([oi.visOi.ndit for oi in objOis]))
for k in range(len(objOis)):
    oi = objOis[k]
    opdChi2Map = np.zeros([oi.visOi.ndit, oi.visOi.nchannel, N_OPD]) 
    bestOpdFit = np.zeros([oi.visOi.ndit, oi.visOi.nchannel, oi.nwav], 'complex')   
    oi = objOis[k]
    visRef = visRefs[k]
#    visRef = totalVisRef
    for dit in range(oi.visOi.ndit):
        printinf("Calculating chi2 map in OPD space for planet OI {:d} of {:d} (dit {:d}/{:d}).".format(k+1, len(objOis), dit+1, oi.visOi.ndit))
        for c in range(oi.visOi.nchannel):
            # filter data points based on flag status
            bad_indices = np.where(oi.visOi.flag[dit, c, :])
            visRefNoBad = np.copy(oi.visOi.visRef[dit, c, :])
            visRefNoBad[bad_indices] = 0            
            # cov and pcov
            W = oi.visOi.visRefCov[dit, c].todense() #np.diag((np.real(oi.visOi.visErr[dit, c, :])**2+np.imag(oi.visOi.visErr[dit, c, :])**2)) 
            Z = oi.visOi.visRefPcov[dit, c].todense() #np.diag((np.real(oi.visOi.visErr[dit, c, :])**2-np.imag(oi.visOi.visErr[dit, c, :])**2))
 #           W = np.diag((np.real(oi.visOi.visErr[dit, c, :])**2+np.imag(oi.visOi.visErr[dit, c, :])**2)) 
 #           Z = np.diag((np.real(oi.visOi.visErr[dit, c, :])**2-np.imag(oi.visOi.visErr[dit, c, :])**2))            
            W2 = cs.extended_covariance(W, Z).real
            if NO_INV:
                W2sp = scipy.sparse.csr_matrix(W2)
                W2invA2 = np.zeros([2*oi.nwav, 2*STAR_ORDER+1], 'complex')
            else:
                ZZ, _ = lapack.dpotrf(W2)
                T, info = lapack.dpotri(ZZ)
                W2inv = np.triu(T) + np.triu(T, k=1).T
            for j in range(N_OPD):
                opd = opdRanges[k, c, j]
                # calculate A matrix
                A = np.zeros([oi.nwav, STAR_ORDER*2+1], 'complex')
                for p in range(STAR_ORDER):
                    A[:, 2*p] = np.abs(visRef[c, :])*((oi.wav-np.mean(oi.wav))*1e6)**p
                    A[:, 2*p+1] = 1j*np.abs(visRef[c, :])*((oi.wav-np.mean(oi.wav))*1e6)**p                    
                # last column requires phase and contrast
                phase = 2*np.pi*opd/(oi.wav*1e6)
                if not(CONTRAST_FILE is None):
                    A[:, -1] = np.exp(-1j*phase)*np.abs(visRef[c, :])*(np.interp(oi.wav, contrast_wav*1e-6, contrast_data)/np.max(contrast_data))
                else:
                    A[:, -1] = np.exp(-1j*phase)*np.abs(visRef[c, :])
                # filter nan points in the model
                A[bad_indices, :] = 0
                # maximum likelihood explicit solution
                A2 = cs.conj_extended(A)
                if NO_INV:
                    for p in range(2*STAR_ORDER+1):
                        W2invA2[:, p] = scipy.sparse.linalg.spsolve(W2sp, A2[:, p])
                else:
                    W2invA2 = np.dot(W2inv, A2)
                left = np.real( np.dot(cs.adj(cs.conj_extended(visRefNoBad)), W2invA2) )
                right = np.real( np.dot(cs.adj(A2), W2invA2) )
                dzeta = np.transpose(np.dot(left, np.linalg.pinv(right)))
                if (dzeta[-1] < 0) or (j==0):
                    dzeta[-1] = 0
                    A[:, -1] = 0
                    A2 = cs.conj_extended(A)
                    if NO_INV:
                        W2invA2[:, -1] = scipy.sparse.linalg.spsolve(W2sp, A2[:, -1])
                    else:
                        W2invA2 = np.dot(W2inv, A2)
                    left = np.real( np.dot(cs.adj(cs.conj_extended(visRefNoBad)), W2invA2) )
                    right = np.real( np.dot(cs.adj(A2), W2invA2) )
                    dzeta = np.transpose(np.dot(left, np.linalg.pinv(right)))
                    dzeta[-1] = 0
                # calculate the corresponding chi2
                vec = cs.conj_extended(visRefNoBad) - np.dot(A2, dzeta)
                if NO_INV:
                    rightVec = scipy.sparse.linalg.spsolve(W2sp, vec)
                    opdChi2Map[dit, c, j] = np.real(np.dot(cs.adj(vec), rightVec))
                else:
                    opdChi2Map[dit, c, j] = np.real(np.dot(np.dot(cs.adj(vec), W2inv), vec))
                """
                print("1.00 -> "+str(np.real(np.dot(np.dot(cs.adj(vec), W2inv), vec))))
                vec = cs.conj_extended(visRefNoBad) - np.dot(A2, 0.99*dzeta)
                print("0.99 -> "+str(np.real(np.dot(np.dot(cs.adj(vec), W2inv), vec))))                
                vec = cs.conj_extended(visRefNoBad) - np.dot(A2, 1.01*dzeta)
                print("1.01 -> "+str(np.real(np.dot(np.dot(cs.adj(vec), W2inv), vec))))                
                stop()
                """
    opdChi2Maps.append(opdChi2Map)

    
# calculate the chi2Maps from the OPD maps
chi2Maps = []
for k in range(len(objOis)):
    printinf("Calculating chi2 map in RA/DEC from OPD maps for planet OI {:d} of {:d}.".format(k+1, len(objOis)))
    oi = objOis[k]
    chi2Map = np.zeros([oi.visOi.ndit, N_RA, N_DEC])
    for i in range(N_RA):
        for j in range(N_DEC):
            ra = raValues[i]
            dec = decValues[j]
            opd = oi.visOi.getOpd(ra, dec)
            for dit in range(oi.visOi.ndit):
                for c in range(oi.visOi.nchannel):
                    chi2Map[dit, i, j] += np.interp(opd[dit, c], opdRanges[k, c, :], opdChi2Maps[k][dit, c, :])
    chi2Maps.append(chi2Map)

    
# calculate the combined chi2Map and best parameters for each OB
chi2Map = np.zeros([len(objOis), N_RA, N_DEC])
raBest = np.zeros(len(objOis))
decBest = np.zeros(len(objOis))
for k in range(len(objOis)):
    chi2Map[k, :, :] = np.sum(chi2Maps[k], axis = 0)
    (a, b) = np.where(chi2Map[k, :, :] == np.min(chi2Map[k, :, :]))
    raBest[k] = raValues[a[0]]
    decBest[k] = decValues[b[0]]

cov = np.cov(raBest, decBest)
printinf("RA: {:.2f}+-{:.3f} mas".format(np.mean(raBest), cov[0, 0]**0.5))
printinf("DEC: {:.2f}+-{:.3f} mas".format(np.mean(decBest), cov[1, 1]**0.5))
printinf("COV: {:.2f}".format(cov[0, 1]/np.sqrt(cov[0, 0]*cov[1, 1])))


for k in range(len(objOis)):
    oi = objOis[k]
    bestFitStar = np.zeros([oi.visOi.ndit, oi.visOi.nchannel, oi.nwav], "complex")
    bestFit = np.zeros([oi.visOi.ndit, oi.visOi.nchannel, oi.nwav], "complex")    
    thisraBest = raBest[k]
    thisdecBest = decBest[k]
    visRef = visRefs[k]    
    opdBest = oi.visOi.getOpd(thisraBest, thisdecBest)
    printinf("Recaculating best fit for planet OI {:d} of {:d}).".format(k+1, len(objOis)))
    for dit in range(oi.visOi.ndit):
        for c in range(0, oi.visOi.nchannel):
            # filter data points based on flag status
            bad_indices = np.where(oi.visOi.flag[dit, c, :])
            visRefNoBad = np.copy(oi.visOi.visRef[dit, c, :])
            visRefNoBad[bad_indices] = 0                        
            # cov and pcov
            W = oi.visOi.visRefCov[dit, c].todense() #np.diag((np.real(oi.visOi.visErr[dit, c, :])**2+np.imag(oi.visOi.visErr[dit, c, :])**2)) 
            Z = oi.visOi.visRefPcov[dit, c].todense() #np.diag((np.real(oi.visOi.visErr[dit, c, :])**2-np.imag(oi.visOi.visErr[dit, c, :])**2))
            W2 = cs.extended_covariance(W, Z).real
            if NO_INV:
                W2sp = scipy.sparse.csr_matrix(W2)
                W2invA2 = np.zeros([2*oi.nwav, 2*STAR_ORDER+1], 'complex')
            else:
                ZZ, _ = lapack.dpotrf(W2)
                T, info = lapack.dpotri(ZZ)
                W2inv = np.triu(T) + np.triu(T, k=1).T
            opd = opdBest[dit, c]
            # calculate A matrix
            A = np.zeros([oi.nwav, STAR_ORDER*2+1], 'complex')
            for p in range(STAR_ORDER):
                A[:, 2*p] = np.abs(visRef[c, :])*((oi.wav-np.mean(oi.wav))*1e6)**p
                A[:, 2*p+1] = 1j*np.abs(visRef[c, :])*((oi.wav-np.mean(oi.wav))*1e6)**p
            # last column requires phase and contrast
            phase = 2*np.pi*opd/(oi.wav*1e6)
            if not(CONTRAST_FILE is None):
                A[:, -1] = np.exp(-1j*phase)*np.abs(visRef[c, :])*(np.interp(oi.wav, contrast_wav*1e-6, contrast_data)/np.max(contrast_data))
            else:
                A[:, -1] = np.exp(-1j*phase)*np.abs(visRef[c, :])
            # filter nan points in the model
            A[bad_indices, :] = 0                
            # maximum likelihood explicit solution
            A2 = cs.conj_extended(A)
            if NO_INV:
                for p in range(2*STAR_ORDER+1):
                    W2invA2[:, p] = scipy.sparse.linalg.spsolve(W2sp, A2[:, p])
            else:
                W2invA2 = np.dot(W2inv, A2)
            left = np.real( np.dot( cs.adj(cs.conj_extended(visRefNoBad)), W2invA2 ))
            right = np.real( np.dot( cs.adj(A2), W2invA2 ) )
            dzeta = np.transpose(np.dot(left, np.linalg.pinv(right)))
            # calculate the corresponding fits
            if dzeta[-1] < 0:
                dzeta[-1] = 0
                A[:, -1] = 0
                A2 = cs.conj_extended(A)
                if NO_INV:
                    W2invA2[:, -1] = scipy.sparse.linalg.spsolve(W2sp, A2[:, -1])
                else:
                    W2invA2 = np.dot(W2inv, A2)
                left = np.real( np.dot(cs.adj(cs.conj_extended(visRefNoBad)), W2invA2) )
                right = np.real( np.dot(cs.adj(A2), W2invA2) )
                dzeta = np.transpose(np.dot(left, np.linalg.pinv(right)))
            bestFit[dit, c, :] = np.dot(A2, dzeta)[0:oi.nwav]
            dzeta[-1] = 0
            bestFitStar[dit, c, :] = np.dot(A2, dzeta)[0:oi.nwav] # star only
    oi.visOi.bestFit = bestFit
    oi.visOi.bestFitStar = bestFitStar

for k in range(len(PLANET_FILES)):
    ind = cfg["general"]["reduce"][k]
    cfg["planet_ois"][ind]["astrometric_solution"] = [float(raBest[k]), float(decBest[k])] # YAML cannot convert numpy types
            
f = open(CONFIG_FILE, "w")
if RUAMEL:
    f.write(ruamel.yaml.dump(cfg, Dumper=ruamel.yaml.RoundTripDumper))
else:
    f.write(yaml.safe_dump(cfg, default_flow_style = False)) 
f.close()



    
if not(FIGDIR is None):
    for k in range(len(objOis)):
        oi = objOis[k]
        fig = plt.figure(figsize=(10, 8))
        gPlot.reImPlot(w, np.ma.masked_array(oi.visOi.visRef-oi.visOi.bestFitStar, oi.visOi.flag).mean(axis = 0), subtitles = oi.basenames, fig = fig)
        gPlot.reImPlot(w, np.ma.masked_array(oi.visOi.bestFit-oi.visOi.bestFitStar, oi.visOi.flag).mean(axis = 0), fig = fig)
        plt.legend([oi.filename.split("/")[-1], "Astrometry fit"])
        plt.savefig(FIGDIR+"/astrometry_fit_"+str(k)+".pdf")

    fig = plt.figure(figsize = (10, 10))
    n = int(np.ceil(np.sqrt(len(objOis))))
    for k in range(len(objOis)):
        ax = fig.add_subplot(n, n, k+1)
        oi = objOis[k]
        name = oi.filename.split('/')[-1]
        ax.imshow(chi2Maps[k].sum(axis = 0).T, origin = "lower", extent = [np.min(raValues), np.max(raValues), np.min(decValues), np.max(decValues)])
        ax.set_xlabel("$\Delta{}\mathrm{RA}$ (mas)")
        ax.set_ylabel("$\Delta{}\mathrm{DEC}$ (mas)")
        ax.set_title(name)
    plt.tight_layout()
    plt.savefig(FIGDIR+"/astrometry_chi2Maps.pdf")

    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(111)
    ax.plot(raBest, decBest, "o")
    val, vec = np.linalg.eig(cov)
    e1 = matplotlib.patches.Ellipse((np.mean(raBest), np.mean(decBest)), 2*val[0]**0.5, 2*val[1]**0.5, angle = np.arctan2(vec[0, 1], -vec[1, 1])/np.pi*180.0, fill=False, color = 'k', linewidth=2, linestyle = '--')
    e2 = matplotlib.patches.Ellipse((np.mean(raBest), np.mean(decBest)), 3*2*val[0]**0.5, 3*2*val[1]**0.5, angle = np.arctan2(vec[0, 1], -vec[1, 1])/np.pi*180.0, fill=False, color = 'k', linewidth=2)    
    ax.add_patch(e1)
    ax.add_patch(e2)    
    ax.plot([np.mean(raBest)], [np.mean(decBest)], '+k')
    ax.text(raBest.mean()+val[0]**0.5, decBest.mean()+val[1]**0.5, "RA={:.2f}+-{:.3f}\nDEC={:.2f}+-{:.3f}\nCOV={:.2f}".format(np.mean(raBest), cov[0, 0]**0.5, np.mean(decBest), cov[1, 1]**0.5, cov[0, 1]/np.sqrt(cov[0, 0]*cov[1, 1])))
    ax.set_xlabel("$\Delta{}\mathrm{RA}$ (mas)")
    ax.set_ylabel("$\Delta{}\mathrm{DEC}$ (mas)")
    plt.axis("equal")
    plt.savefig(FIGDIR+"/solution.pdf")

    
if dargs['save_residuals']:
    for k in range(len(objOis)):
        oi = objOis[k]
        name = oi.filename.split('/')[-1].split('.fits')[0]
        visPla = oi.visOi.visRef - oi.visOi.bestFitStar
        visRes = np.ma.masked_array(oi.visOi.visRef - oi.visOi.bestFit, oi.visOi.flag).data
        visResMask = np.ma.masked_array(oi.visOi.visRef - oi.visOi.bestFit, oi.visOi.flag).mask
        np.save(FIGDIR+"/"+name+"_residuals.npy", visRes)
        np.save(FIGDIR+"/"+name+"_planet.npy", visPla)        
        np.save(FIGDIR+"/"+name+"_mask.npy", visResMask)    

stop()


# ACTUAL CODE ENDS HERE!!!


"""
# estimate the distribution of likelihood
all_opts = []
all_smallMaps = []
all_fits = []
for k in range(len(objOis)-1):
    # convert chi2 to likelihoo
    likelihood = np.exp(-0.5*(chi2Map[k, :, :]-np.min(chi2Map[k, :, :])))

    # get indice of max and trim the map
    n = 7
    ra_ind = np.where(raValues == raBest[k])[0][0]
    dec_ind = np.where(decValues == decBest[k])[0][0]
    ra = raValues[ra_ind-n:ra_ind+n]
    dec = decValues[dec_ind-n:dec_ind+n]

    X = np.zeros([2*n, 2])
    X[:, 0] = ra
    X[:, 1] = dec

    smallMap = likelihood[ra_ind-n:ra_ind+n, dec_ind-n:dec_ind+n]

    def gaussian(x):
        W = np.zeros([2, 2])
        W[0, 0] = np.sqrt(1+x[0]**2+x[1]**2)+x[0]
        W[1, 1] = np.sqrt(1+x[0]**2+x[1]**2)-x[0]
        W[0, 1] = x[1]
        W[1, 0] = x[1]
        W = np.exp(x[2])*W
        model = np.zeros([2*n, 2*n])
        for k in range(2*n):
            for i in range(2*n):
                thisX = np.array([[X[k, 0]-x[3], X[i, 1]-x[4]]])
                model[k, i] = np.exp(-0.5*np.dot(np.dot(thisX, np.linalg.inv(W)), thisX.T))[0, 0]
        return model

    def sqrsum(x):
        return np.sum((smallMap - gaussian(x))**2)

    opt = scipy.optimize.minimize(sqrsum, [0.25, -1.8, -9.5, raBest[k], decBest[k]])
    xopt = opt['x']
    
    all_opts.append(opt)
    all_smallMaps.append(smallMap)
    all_fits.append(gaussian(xopt))    

np.savetxt("astrometry.txt", np.array([opt['x'] for opt in all_opts]))

for k in range(len(objOis)):
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    p = ax.imshow(all_smallMaps[k].T, origin = 'lower')
    plt.colorbar(p)
    ax = fig.add_subplot(1, 2, 2)    
    p = ax.imshow(all_fits[k].T, origin = 'lower')
    plt.colorbar(p)
    plt.savefig("./likelihood_fits/fit_"+str(k)+".png")
    plt.close()

# save results
astrometry = np.zeros([len(objOis), 2])
astrometry[:, 0] = raBest
astrometry[:, 1] = decBest
np.save('chi2Map', chi2Map)
np.save('astrometry', astrometry)

# plot maps
for k in range(len(objOis)):
    plt.figure()
    plt.imshow(chi2Map[k, :, :].T/(objOis[k].visOi.ndit*objOis[k].visOi.nchannel*objOis[k].nwav),  origin = "lower", extent = [ra_lim[0], ra_lim[1], dec_lim[0], dec_lim[1]])
    plt.colorbar()
"""

# recalculate best fits
pBar = patiencebar.Patiencebar(valmax = len(objOis), title = "Recalculating best Fits...")



fig = plt.figure(figsize = (6, 6))
plt.plot(raBest, decBest, "o")
plt.xlabel("$\Delta{}\mathrm{RA}$ (mas)")
plt.ylabel("$\Delta{}\mathrm{DEC}$ (mas)")
plt.axis('equal')
plt.tight_layout()
if SAVEFIG:
    plt.savefig("astrometryReport/astrometry.pdf")
    plt.close(fig)

fig = plt.figure(figsize = (12, 6))
ax = fig.add_subplot(211)
ax.plot([oi.time for oi in objOis], raBest, "o")
ax.set_xlabel("Time (s)")
ax.set_ylabel("$\Delta{}\mathrm{RA}$ (mas)")
ax = fig.add_subplot(212)
ax.plot([oi.time for oi in objOis], decBest, "o")
ax.set_xlabel("Time (s)")
ax.set_ylabel("$\Delta{}\mathrm{DEC}$ (mas)")
plt.tight_layout()
if SAVEFIG:
    plt.savefig("astrometryReport/astrometry_time.pdf")
    plt.close(fig)
                     

for k in range(len(objOis)):
    fig = plt.figure(figsize = (12, 6))
    ylim = np.percentile(np.abs(objOis[k].visOi.visRef), 99)
    gravityPlot.reImPlot(w, objOis[k].visOi.visRef.mean(axis = 0), subtitles = oi.basenames, fig = fig, ylim = [-ylim, ylim])
    gravityPlot.reImPlot(w, objOis[k].visOi.bestFitStar.mean(axis = 0), fig = fig)
    if SAVEFIG:
        plt.savefig("astrometryReport/starfit_"+str(k)+".pdf")
        plt.close(fig)

    fig = plt.figure(figsize = (12, 6))
    ylim = np.percentile(np.abs(objOis[k].visOi.visRef-objOis[k].visOi.bestFitStar), 99)
    gravityPlot.reImPlot(w, objOis[k].visOi.visRef.mean(axis = 0) - objOis[k].visOi.bestFitStar.mean(axis = 0), subtitles = oi.basenames, fig = fig, ylim = [-ylim, ylim])
    gravityPlot.reImPlot(w, objOis[k].visOi.bestFit.mean(axis = 0) - objOis[k].visOi.bestFitStar.mean(axis = 0), fig = fig)
    if SAVEFIG:
        plt.savefig("astrometryReport/planetfit_"+str(k)+".pdf")
        plt.close(fig)

    fig = plt.figure(figsize = (6, 6))
    gravityPlot.uvMap(objOis[k].visOi.uCoordMean, objOis[k].visOi.vCoordMean, targetCoord=[objOis[k].sObjX, objOis[k].sObjY], fig = fig, legend = objOis[k].basenames)
    plt.tight_layout()
    if SAVEFIG:
        plt.savefig("astrometryReport/uvmap_"+str(k)+".pdf")
        plt.close(fig)

    fig = plt.figure(figsize = (6, 6))
    plt.imshow(chi2Maps[k][0, :, :].T, origin = "lower", extent = [np.min(raValues), np.max(raValues), np.min(decValues), np.max(decValues)])
    plt.text(np.min(raValues), np.min(decValues)+0.1*(np.max(decValues)-np.min(decValues)), "RA: {:.2f}".format(raBest[k]), color = "w", weight = "bold")
    plt.text(np.min(raValues), np.min(decValues)+0.05*(np.max(decValues)-np.min(decValues)), "DEC: {:.2f}".format(decBest[k]), color = "w", weight = "bold")    
    plt.xlabel("$\Delta{}\mathrm{RA}$ (mas)")
    plt.ylabel("$\Delta{}\mathrm{DEC}$ (mas)")
    plt.tight_layout()
    if SAVEFIG:
        plt.savefig("astrometryReport/chi2Map_"+str(k)+".pdf")
        plt.close(fig)

stop()

f = open("astrometryReport/report.tex", "w")
f.write("\\documentclass[a4paper, landscape]{article}\n")
f.write("\\usepackage{graphicx}\n")
f.write("\\usepackage[left=1cm, right=1cm, top = 1cm, bottom = 1cm]{geometry}\n")
f.write("\\begin{document}\n")
f.write("\\includegraphics[width=0.3\linewidth]{astrometry.pdf}\n")
f.write("\\includegraphics[width=0.60\linewidth]{astrometry_time.pdf}\n")
f.write("\\clearpage\n")
for k in range(len(objOis)):
    f.write("\\includegraphics[width=0.3\linewidth]{uvmap_"+str(k)+".pdf}\n")
    f.write("\\includegraphics[width=0.60\linewidth]{starfit_"+str(k)+".pdf}\n")
    f.write("\n")
    f.write("\\includegraphics[width=0.3\linewidth]{chi2Map_"+str(k)+".pdf}\n")
    f.write("\\includegraphics[width=0.60\linewidth]{planetfit_"+str(k)+".pdf}\n")
    f.write("\n")
f.write("\\end{document}\n")
f.close()


plt.figure()
plt.plot(raBest, decBest, 'o')
for k in range(len(objOis)):
    plt.text(raBest[k], decBest[k], str(k))

stop

"""
w = np.zeros([oi.visOi.nchannel, oi.nwav])
for c in range(oi.visOi.nchannel):
    w[c, :] = oi.wav*1e6
    
m=1
fig = plt.figure()
gravityPlot.reImPlot(w, objOis[0].visOi.visRef[m, :, :], fig = fig, subtitles = objOis[0].basenames)
gravityPlot.reImPlot(w, bestFitStar[m, :, :], fig = fig)

fig = plt.figure()
gravityPlot.reImPlot(w, objOis[0].visOi.visRef[m, :, :]-bestFitStar[m, :, :], fig = fig, subtitles = objOis[0].basenames)
gravityPlot.reImPlot(w, (bestFit[m, :, :]-bestFitStar[m, :, :]), fig = fig)

"""
m=0
oi = objOis[m]
maps = np.copy(opdChi2Maps[m])
for dit in range(oi.visOi.ndit):
    for c in range(oi.visOi.nchannel):
        maps[dit, c, :] = (maps[dit, c, :] - np.min(maps[dit, c, :]))/(np.max(maps[dit, c, :]) - np.min(maps[dit, c, :]))
        
fig = plt.figure()
gravityPlot.waterfall(opdRanges[m, :, :], maps, fig = fig, subtitles = oi.basenames, xlabel = "OPD ($\mu\mathrm{m}$)")

plt.figure()
plt.imshow(chi2Maps[m].sum(axis = 0).T/(oi.visOi.ndit*oi.visOi.nchannel*oi.nwav), origin = "lower", extent = [np.min(raValues), np.max(raValues), np.min(decValues), np.max(decValues)])
plt.colorbar()



oinoref = gravity.GravityDualfieldAstrored(oi.filename, corrMet = "sylvestre", opdDispCorr = True, extension = 10)

fig = plt.figure(figsize = (11.5, 9))
gravityPlot.reImPlot(w, oinoref.visOi.visData[0, :, :], fig = fig, subtitles = oi.basenames, xlabel = "Wavelength ($\mu\mathrm{m}$)")
gravityPlot.reImPlot(w, oinoref.visOi.visRef[0, :, :], fig = fig)

fig = plt.figure(figsize = (11.5, 9))
gravityPlot.reImPlot(w, oinoref.visOi.visRef[0, :, :]+np.float('nan')+1j*np.float('nan'), fig = fig, subtitles = oi.basenames, xlabel = "Wavelength ($\mu\mathrm{m}$)")
gravityPlot.reImPlot(w, oinoref.visOi.visRef[0, :, :], fig = fig)
gravityPlot.reImPlot(w, oi.visOi.visRef[0, :, :], fig = fig)

fig = plt.figure(figsize = (11.5, 9))
gravityPlot.reImPlot(w, oi.visOi.visData.mean(axis = 0)+np.float('nan')+1j*np.float('nan'), fig = fig, subtitles = oi.basenames, xlabel = "Wavelength ($\mu\mathrm{m}$)")
gravityPlot.reImPlot(w, oi.visOi.visData.mean(axis = 0)+np.float('nan')+1j*np.float('nan'), fig = fig)
gravityPlot.reImPlot(w, oi.visOi.visRef.mean(axis = 0), fig = fig)
gravityPlot.reImPlot(w, oi.visOi.bestFitStar.mean(axis = 0), fig = fig)

fig = plt.figure(figsize = (11.5, 9))
gravityPlot.reImPlot(w, oi.visOi.visData.mean(axis = 0)+np.float('nan')+1j*np.float('nan'), fig = fig, subtitles = oi.basenames, xlabel = "Wavelength ($\mu\mathrm{m}$)")
gravityPlot.reImPlot(w, oi.visOi.visData.mean(axis = 0)+np.float('nan')+1j*np.float('nan'), fig = fig)
gravityPlot.reImPlot(w, oi.visOi.visRef.mean(axis = 0)-oi.visOi.bestFitStar.mean(axis = 0), fig = fig)
gravityPlot.reImPlot(w, oi.visOi.bestFit.mean(axis = 0) - oi.visOi.bestFitStar.mean(axis = 0), fig = fig)


w = np.zeros([oi.visOi.nchannel, oi.nwav])
for c in range(oi.visOi.nchannel):
    w[c, :] = oi.wav*1e6

pdf = PdfPages('starfits.pdf')

for k in range(len(objOis)):
    fig = plt.figure()
    oi = objOis[k]
    gravityPlot.reImPlot(w, oi.visOi.visRef.mean(axis = 0), fig = fig, subtitles = oi.basenames)
    oi.visOi.fitStarDitByDit(visRefs[0], order = STAR_ORDER)
    gravityPlot.reImPlot(w, oi.visOi.visStar.mean(axis = 0), fig = fig)
    pdf.savefig()
    plt.close(fig)

for k in range(len(objOis)):
    fig = plt.figure()
    oi = objOis[k]
    gravityPlot.reImPlot(w, oi.visOi.visRef.mean(axis = 0), fig = fig, subtitles = oi.basenames)
    pdf.savefig()
    plt.close(fig)
    
pdf.close()



fig = plt.figure(figsize = (12, 10))
gravityPlot.reImPlot(w, starOis[0].visOi.visRef.mean(axis = 0), fig = fig, subtitles = starOis[0].basenames)
for k in range(1, len(starOis)):
    gravityPlot.reImPlot(w, starOis[k].visOi.visRef.mean(axis = 0), fig = fig)


fig = plt.figure(figsize = (12, 10))
gravityPlot.modPhasePlot(w, starOis[0].visOi.visRef.mean(axis = 0), fig = fig, subtitles = starOis[0].basenames)
for k in range(1, len(starOis)):
    gravityPlot.modPhasePlot(w, starOis[k].visOi.visRef.mean(axis = 0), fig = fig)


###
#wav, flux, fluxCov, contrast, contrastCov = loadFitsSpectrum("/home/mnowak/Documents/exoGravity/spectra/BetaPictorisc_2020-03-08_ADI.fits")
wav, flux, fluxCov, contrast, contrastCov = loadFitsSpectrum("/home/mnowak/Documents/exoGravity/spectra/BetaPictorisb_2018-09-22.fits")
contrast = np.interp(oi.wav*1e6, wav, contrast)
oi = objOis[0]
plt.figure()
for c in range(oi.visOi.nchannel):
    freq = oi.visOi.freq[:, c, :].mean(axis = 0)/1e6
    amp = oi.dit/soi.dit*np.abs(soi.visOi.visRef[:, c, :]).mean()
    plt.semilogy(freq, np.abs(oi.dit/soi.dit*soi.visOi.visRef[:, c, :]).mean(axis = 0)/amp, 'C'+str(c))    
    plt.semilogy(freq, np.abs(oi.dit*oi.visOi.visRef[:, c, :]).mean(axis = 0)/amp, '-.C'+str(c))
#    plt.semilogy(freq, (np.abs(oi.dit*oi.visOi.visRef[:, c, :].mean(axis = 0) - oi.dit*oi.visOi.bestFitStar[:, c, :].mean(axis = 0)))/amp, '--C'+str(c))
    plt.semilogy(freq, np.abs(oi.dit/soi.dit*soi.visOi.visRef[:, c, :]).mean(axis = 0)/amp*contrast, 'C'+str(c))    
    plt.semilogy(freq, np.abs(oi.dit*oi.visOi.visRef[:, c, :]).mean(axis = 0)**0.5/(len(objOis))**0.5/amp, 'k--')

plt.legend(["On-star", "On-planet stellar residual", "On-planet planet flux", "Shot noise on stellar residual"])
    
plt.xlabel("Frequency (Mrad/s)")
plt.ylabel("Contrast")

plt.grid(which = "both")
plt.xlim(15, 65)
plt.ylim(1e-6, 2)
###

    

for k in cfg['planet_ois']:
    print(cfg['planet_ois'][k]['filename'])

chiMin = 0

superChi2Map = np.zeros([len(raValues), len(decValues)])
for k in range(len(objOis)):
    superChi2Map = superChi2Map + chi2Maps[k].sum(axis = 0)
    chiMin = chiMin+opdChi2Maps[k][:, :, 0].sum()
    
#superChi2Map[:, 0] = np.float('nan')
#superChi2Map[:, -1] = np.float('nan')
#superChi2Map[0, :] = np.float('nan')
#superChi2Map[-1, :] = np.float('nan')
#superChi2Map[-2, :] = np.float('nan')

#zmap = (233*12*5-(6*12*5+6+2))*
zmap = (chiMin-superChi2Map)/2
xfib = objOis[0].sObjX
yfib = objOis[0].sObjY

import matplotlib
axcol = 'lightgray'
matplotlib.rc('axes',edgecolor=axcol)
fig = plt.figure()
ax = fig.add_subplot(111)
p = ax.imshow(zmap.T, origin = 'lower', extent = [np.min(raValues), np.max(raValues), np.min(decValues), np.max(decValues)], aspect = 'equal', vmin = 0)
cbar = fig.colorbar(p)
cbar.set_label("Periodogram power")
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')    
ax.yaxis.label.set_color(axcol)
ax.xaxis.label.set_color(axcol)
ax.tick_params(axis='x', colors=axcol)
ax.tick_params(axis='y', colors=axcol)
r = 30
plt.xlim([xfib-r-1, xfib+r+1])
plt.ylim([yfib-r-1, yfib+r+1])

theta = np.linspace(0,2*np.pi,100);      # A vector of 100 angles from 0 to 2*pi
xCircle = xfib+r*1.0*np.cos(theta);              # x coordinates for circle
yCircle = yfib+r*1.0*np.sin(theta);              # y coordinates for circle
s = 1
xSquare = [xfib+r+s, xfib+r+s, xfib-r-s, xfib-r-s, xfib+r+s, xfib+r+s];         # x coordinates for square
ySquare = [yfib, yfib-r-s, yfib-r-s, yfib+r+s, yfib+r+s, yfib];         # y coordinates for square
hp = plt.fill(list(xCircle)+xSquare, list(yCircle)+ySquare, 'w');

ax.set_xlabel('$\Delta\mathrm{RA}$ (mas)')
ax.set_ylabel('$\Delta\mathrm{DEC}$ (mas)')
ax.xaxis.set_label_coords(0.75, 0.55)
ax.yaxis.set_label_coords(0.55, 0.75)

plt.plot(xCircle, yCircle, '--', color='gray')
#plt.plot(xfib+r*np.cos(theta)/2, yfib+r*np.sin(theta)/2, '--', color=axcol, alpha=0.5)

plt.text(xfib+r/2**0.5+1, yfib-r/2**0.5-1, "Fiber Field-of-view", ha='center', va='center', rotation = 45,color = 'gray')
#plt.text(xfib+r/2**0.5/2+1, yfib-r/2**0.5/2-1, "Half-injection", ha='center', va='center', rotation = 45, color = axcol)
#lt.xlabel("U coordinate ($\\times{}10^7 \mathrm{ rad}^{-1}$)")
#lt.ylabel("V coordinate ($\\times{}10^7 \mathrm{ rad}^{-1}$)")

# Clip
