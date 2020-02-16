#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Extract contrast spectrum from an exoGravity observation

This script is part of the exoGravity data reduction package.
The spectrum_reduce script is used to extract the contrast spectrum from an exoplanet observation made with GRAVITY, in dual-field mode.
To use this script, you need to call it with a configuration file, and an output directory, see example below.

Args:
  config_file (str): the path the to YAML configuration file.
  outputdit (str): /path/to/outputdir in which the script will create a "contrast.txt" file and a "covariance.txt" file, to contain the result
  gofast (bool, optional): if set, average over DITs to accelerate calculations (Usage: --gofast, or gofast=True, or goFast=False). This will bypass
                           the value contained in the YAML file.

Example:
  python astrometry_reduce config_file=full/path/to/yourconfig.yml outputdir=/path/to/outputdir

Authors:
  M. Nowak, and the exoGravity team.

Version:
  xx.xx
"""

import numpy as np
import scipy.special
import scipy.linalg
from datetime import datetime
import cleanGravity as gravity
from cleanGravity import complexstats as cs
import glob
from cleanGravity.utils import loadFitsSpectrum, saveFitsSpectrum
from utils import *
import itertools
from astropy.io import fits
try:
    import ruamel.yaml
    RUAMEL = True
except: # if ruamel not available, switch back to pyyaml, which does not handle comments properly
    import yaml
    RUAMEL = False
import sys, os


#import matplotlib.pyplot as plt
#plt.ion()
#from cleanGravity import gravityPlot

### GLOBALS ###
SPECTRUM_FILENAME = "spectrum.fits"
###

# load aguments into a dictionnary
dargs = args_to_dict(sys.argv)

if "help" in dargs.keys():
    print(__doc__)
    stop()

# arg should be the path to the config yal file    
REQUIRED_ARGS = ["config_file", "outputdir"]
for req in REQUIRED_ARGS:
    if not(req in dargs.keys()):
        printerr("Argument '"+req+"' is not optional for this script. Required args are: "+', '.join(REQUIRED_ARGS))
        stop()

OUTPUT_DIR = dargs["outputdir"]
# Test if output dir exists
if not(os.path.isdir(OUTPUT_DIR)):
    printwar("Creating directory {}.".format(OUTPUT_DIR))    
#    r = input("[INPUT]: Create it? (y/n).")
#    if not(r.lower() in ['y', 'yes']):
#        printerr("Interrupted by user")
    os.mkdir(OUTPUT_DIR)
    
#else: #if directory aready exits, check for files
#    if os.path.isfile(OUTPUT_DIR+SPECTRUM_FILENAME):
#        printwar("A file {} has been found in specified output directory {}, and will be overwritten by this script.".format(SPECTRUM_FILENAME, OUTPUT_DIR))
#        r = input("[INPUT]: Do you want to continue? (y/n)")
#        if not(r.lower() in ['y', 'yes']):
#            printerr("Interrupted by user")
#    if os.path.isfile(OUTPUT_DIR+COVARIANCE_FILENAME):
#        printwar("A file {} has been found in specified output directory {}, and will be overwritten by this script.".format(COVARIANCE_FILENAME, OUTPUT_DIR))
#        r = input("[INPUT]: Do you want to continue? (y/n)")
#        if not(r.lower() in ['y', 'yes']):
#            printerr("Interrupted by user")               

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
REFLAG = cfg['general']['reflag']
FIGDIR = cfg["general"]["figdir"]
PLANET_FILES = [DATA_DIR+cfg["planet_ois"][k]["filename"] for k in cfg["general"]["reduce"]] # list of files corresponding to planet exposures
if not("swap_ois" in cfg.keys()):
    SWAP_FILES = []
elif cfg["swap_ois"] is None:
    SWAP_FILES = []
else:
    SWAP_FILES = [DATA_DIR+cfg["swap_ois"][k]["filename"] for k in cfg["swap_ois"].keys()]
STAR_ORDER = cfg["general"]["star_order"]
STAR_DIAMETER = cfg["general"]["star_diameter"]

# OVERWRITE SOME OF THE CONFIGURATION VALUES WITH ARGUMENTS FROM COMMAND LINE
if "gofast" in dargs.keys():
    GO_FAST = dargs["gofast"].lower()=="true" # bypass value from config file
if "reflag" in dargs.keys():
    REFLAG = dargs["reflag"].lower()=="true" # bypass value from config file
if "noinv" in dargs.keys():
    NO_INV = dargs["noinv"].lower()=="true" # bypass value from config file    
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

# extract list of useful star ois from the list indicated in the star_indices fields of the config file:
star_indices = []
for k in cfg["general"]["reduce"]:
    star_indices = star_indices+cfg["planet_ois"][k]["star_indices"]
star_indices = list(set(star_indices)) # remove duplicates
STAR_FILES = [DATA_DIR+cfg["star_ois"][k]["filename"] for k in star_indices]

# extract the astrometric solutions
try:
    RA_SOLUTIONS = [cfg["planet_ois"][k]["astrometric_solution"][0] for k in cfg["general"]["reduce"]]
    DEC_SOLUTIONS = [cfg["planet_ois"][k]["astrometric_solution"][1] for k in cfg["general"]["reduce"]]
    printinf("Astrometry extracted from config file {}".format(CONFIG_FILE))
except:
    printerr("Could not extract the astrometry from the config file {}".format(CONFIG_FILE))
    printinf("Please run the script 'astrometryReduce' first, or provide RA DEC values as arguments:")
    printinf("spectrumReduce.py config.yal ra_value_in_mas dec_value_in_mas")
    stop()


#### deprecated part of the code #####
# HD206893
#datadir = "/home/mcn35/2018-09-21_allwavelengths/"
#datadir = "/data1/gravity/PDS70/"
#datafiles = glob.glob(datadir+'/GRAVI*astrored*.fits')
#datafiles.sort()
#ra = np.array([130.85151515, 131.05353535, 131.25555556, 130.85151515])
#dec = np.array([197.82747475, 197.72646465, 197.32242424, 198.02949495])

#ra = np.array([127.31616162, 127.31616162, 127.01313131, 126.81111111, 127.01313131])
#dec = np.array([199.03959596, 198.73656566, 199.34262626, 199.54464646, 199.54464646])

#ra = np.array([130.85151515, 131.05353535, 131.25555556, 130.85151515])*0+131.00
#dec = np.array([197.82747475, 197.72646465, 197.32242424, 198.02949495])*0+197.72


#astrometry = np.loadtxt("/home/mcn35/astrometry.txt")
#ra = astrometry[:, 0]
#dec = astrometry[:, 1]

#############

starOis = []
objOis = []
swapOis = [] # first position of the swap (only in DF_SWAP mode)

sFluxOis = []
oFluxOis = []


# LOAD DATA
t = time.time()
for filename in PLANET_FILES+STAR_FILES+SWAP_FILES:
    printinf("Loading file "+filename)
    if (PHASEREF_MODE == "DF_SWAP") and (filename in STAR_FILES):
        oi = gravity.GravityDualfieldAstrored(filename, corrMet = "drs", extension = 10, corrDisp = "drs")
    elif filename in PLANET_FILES:
        oi = gravity.GravityDualfieldAstrored(filename, corrMet = cfg["general"]["corr_met"], extension = 10, corrDisp = cfg["general"]["corr_disp"], reflag = REFLAG)
    else:
        oi = gravity.GravityDualfieldAstrored(filename, corrMet = cfg["general"]["corr_met"], extension = 10, corrDisp = cfg["general"]["corr_disp"])
    if filename in PLANET_FILES:
        objOis.append(oi)
        printinf("File is on planet. FT coherent flux: {:.2e}".format(np.mean(np.abs(oi.visOi.visDataFt))))        
    elif filename in SWAP_FILES:
        swapOis.append(oi)
        printinf("File is from a SWAP")
    else:
        starOis.append(oi)
        printinf("File is on star")


# flag points based on FT value
ftThreshold = np.array([np.abs(oi.visOi.visDataFt).mean() for oi in objOis]).mean()/10.0
printinf("Flag data below FT threshold of {:.2e}".format(ftThreshold))
for oi in objOis:
#    oi.visOi.visDataFt[:, 0, :] = ftThreshold/100
#    oi.visOi.visDataFt[:, 1, :] = ftThreshold/100
#    oi.visOi.visDataFt[:, 2, :] = ftThreshold/100    
#    oi.visOi.visDataFt[:, 3, :] = ftThreshold/100        
    a, b = np.where(np.abs(oi.visOi.visDataFt).mean(axis = -1) < ftThreshold)
    (a, b, c) = np.meshgrid(a, b, range(oi.nwav))
    oi.visOi.flagPoints((a, b, c))

# normalize by FT flux to get rid of atmospheric transmission variations
printinf("Normalizing visibilities to FT coherent flux.")
for oi in objOis+starOis:
    oi.visOi.scaleVisibilities(1.0/np.abs(oi.visOi.visDataFt).mean(axis = -1))


# replace data by the mean over all DITs if go_fast has been requested in the config file
printinf("gofast flag is set. Averaging over DITs")
for oi in objOis+starOis: # mean should not be calculated on swap before phase correction
    if (GO_FAST):
        printinf("Averaging file {}".format(oi.filename))        
        oi.computeMean()
        oi.visOi.recenterPhase(oi.sObjX, oi.sObjY, radec = True)
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
        oi.visOi.visRefCov = oi.visOi.visRefCov[0:1, :]/oi.visOi.ndit
        oi.visOi.visRefPcov = oi.visOi.visRefPcov[0:1, :]/oi.visOi.ndit        
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
for c in range(oi.visOi.nchannel):
    w[c, :] = oi.wav*1e6

# create visRefs fromn star_indices indicated in the config file
printinf("Creating the visibility reference from {:d} star observations.".format(len(starOis)))
visRefs = [oi.visOi.visRef.mean(axis = 0)*0 for oi in objOis]
for k in range(len(objOis)):
    planet_ind = cfg["general"]["reduce"][k]
    ampRef = np.zeros([oi.visOi.nchannel, oi.nwav])
    visRef = np.zeros([oi.visOi.nchannel, oi.nwav])    
    for ind in cfg["planet_ois"][planet_ind]["star_indices"]:
        # not all star files have ben loaded, so we cannot trust the order and we need to explicitly look for the correct one
        starOis_ind = [soi.filename for soi in starOis].index(DATA_DIR+cfg["star_ois"][ind]["filename"]) 
        soi = starOis[starOis_ind]
        visRef = visRef+soi.visOi.visRef.mean(axis = 0)
        ampRef = ampRef+np.abs(soi.visOi.visRef.mean(axis = 0))
    ###===### 
    visRef = np.zeros([oi.visOi.nchannel, oi.nwav])
    for ind in range(len(starOis)):
        soi = starOis[ind]
        visRef = visRef+soi.visOi.visRef.mean(axis = 0)                                                         
    ###===###
    visRefs[k] = ampRef/len(cfg["planet_ois"][planet_ind]["star_indices"])*np.exp(1j*np.angle(visRef/len(objOis)))#/len(cfg["planet_ois"][planet_ind]["star_indices"])))#/len(starOis))) ###===###

    
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

# calculate the phi values
printinf("Calculating the phi values")
for k in range(len(objOis)):
    oi = objOis[k]
    oi.visOi.phi_values = np.zeros([oi.visOi.ndit, oi.visOi.nchannel, oi.nwav], 'complex64')
    this_u = (RA_SOLUTIONS[k]*oi.visOi.uCoord + DEC_SOLUTIONS[k]*oi.visOi.vCoord)/1e7
    phase = 2*np.pi*this_u*1e7/3600.0/360.0/1000.0*2*np.pi
    for dit in range(oi.visOi.ndit):
        for c in range(oi.visOi.nchannel):
            oi.visOi.phi_values[dit, c, :] = np.exp(1j*phase[dit, c, :])

# create projector matrices
printinf("Create projector matrices (p_matrices)")
for k in range(len(objOis)):
    oi = objOis[k]
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
            oi.visOi.p_matrices[dit, c, :, :] = np.dot(np.diag(oi.visOi.phi_values[dit, c, :]), np.dot(P[c, :, :], np.diag(oi.visOi.phi_values[dit, c, :].conj())))

# change frame
printinf("Switching on-planet visibilities to planet reference frame")
for k in range(len(objOis)):
    oi = objOis[k]
    oi.visOi.recenterPhase(RA_SOLUTIONS[k], DEC_SOLUTIONS[k], radec = True)

# project visibilities
printinf("Projecting visibilities orthogonally to speckle noise")
for k in range(len(objOis)):
    oi = objOis[k]
    for dit in range(oi.visOi.ndit):
        for c in range(oi.visOi.nchannel):
            bad_indices = np.where(oi.visOi.flag[dit, c, :])
            oi.visOi.visRef[dit, c, bad_indices] = 0        
            oi.visOi.visRef[dit, c, :] = np.dot(oi.visOi.p_matrices[dit, c, :, :], oi.visOi.visRef[dit, c, :])
    oi.visOi.visRefMean = oi.visOi.visRef.mean(axis = 0)

# estimate the covariance matrices
#W_obs = []
#Z_obs = []
#for k in range(len(objOis)):
#    oi = objOis[k]
#    visRef = oi.visOi.visRef
#    s = np.shape(visRef)
#    visRef = np.reshape(visRef, [s[0], s[1]*s[2]])
#    W_obs.append(cs.cov(visRef.T))
#    Z_obs.append(cs.pcov(visRef.T))

# calculate all H matrices
printinf("Starting calculation of H matrices")
counter = 1
for k in range(len(objOis)):
    printinf("Calculating H ({:d}/{:d})".format(counter, len(objOis)))
    counter = counter+1
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

# TODO: is it useful?
"""
# if injection data available, load them
if os.path.isfile('injection.npy'):
    inj_coeffs = np.load("injection.npy")
else:
    inj_coeffs = None    
inj_coeffs = None

# calculate chromatic coupling from coefficients and fiber offset
sObjX = np.array([oi.sObjX for oi in objOis])
sObjY = np.array([oi.sObjY for oi in objOis])
offsets = np.sqrt((ra - sObjX)**2+(dec - sObjY)**2)
"""
chromatic_coupling = np.zeros([len(objOis), objOis[0].nwav])
ref_coupling = 1#np.polyval(chromatic_coefficients[0, ::-1], 0)
for k in range(len(objOis)):
   # offset = offsets[k]
   # a0 = np.polyval(chromatic_coefficients[0, ::-1], offset)
   # a1 = np.polyval(chromatic_coefficients[1, ::-1], offset)
   # a2 = np.polyval(chromatic_coefficients[2, ::-1], offset)
    chromatic_coupling[k, :] = 1#np.polyval([a2/ref_coupling, a1/ref_coupling, a0/ref_coupling], objOis[k].wav*1e6 - 2.2)
### END OF TODO
    
# calculate the resolved stellar visibility function
resolved_star_models = []
for k in range(len(objOis)):
    oi = objOis[k]
    model = np.zeros([oi.visOi.ndit, oi.visOi.nchannel, oi.nwav])
    if STAR_DIAMETER == 0:
        model = model+1
    else:
        model = 2*scipy.special.j1(np.pi*STAR_DIAMETER*oi.visOi.freq)/(np.pi*STAR_DIAMETER*oi.visOi.freq)
    resolved_star_models.append(model)
    
# big H and big U
nwav = objOis[0].nwav
nb = objOis[0].visOi.nchannel
nt = len(objOis)
mtot = int(np.sum([np.sum(oi.visOi.m) for oi in objOis]))
H = np.zeros([mtot, nwav], 'complex64')
Y = np.zeros([mtot], 'complex64')
r = 0
for k in range(len(objOis)):
    oi = objOis[k]
    for dit in range(oi.visOi.ndit):
        for c in range(oi.visOi.nchannel):
            m = int(oi.visOi.m[dit, c])
            # TODO: useful?
#            if (inj_coeffs is None):
            G = np.diag(np.abs(visRefs[k][c, :])/resolved_star_models[k][dit, c, :]*chromatic_coupling[k, :])
#            else:
#                G = np.diag((inj_coeffs[k][dit, c, 0]*np.abs(ampRefs[k][c, :])+inj_coeffs[k][dit, c, 1]*(oi.wav - np.min(oi.wav))*1e6*np.abs(ampRefs[k][c, :]))/resolved_star_models[k][dit, c, :]*chromatic_coupling[k, :])
            H[r:(r+m), :] = np.dot(oi.visOi.h_matrices[dit, c, 0:m, :], G)
            # filter bad datapoints by setting the corresponding columns of H to 0
            indx = np.where(oi.visOi.flag[dit, c, :])
            H[r:(r+m), indx] = 0
            # Y is different from the paper, as we don't really use "U" here. So we don't divide by abs(visRefs[k]), and we add G in the definition of H
            Y[r:r+m] = np.dot(oi.visOi.h_matrices[dit, c, 0:m, :], oi.visOi.visRef[dit, c, :])
            r = r+m

            
Y2 = cs.conj_extended(Y)
H2 = cs.conj_extended(H)

# block calculate W2inv*H2, cov(U2)*W2inv*H2, and pcov(U2)*W2inv*H2
nb = objOis[0].visOi.nchannel
nwav = objOis[0].nwav
W2invH2 = np.zeros([2*mtot, nwav], 'complex64')
covY2W2invH2 = np.zeros([2*mtot, nwav], 'complex64')
pcovY2W2invH2 = np.zeros([2*mtot, nwav], 'complex64')
r = 0
nwav = objOis[0].nwav
nb = objOis[0].visOi.nchannel
#Ginv = np.zeros([nb*nwav, nb*nwav], 'complex64')    

printinf("Starting Calculation of W2invH2")
counter = 1
for k in range(len(objOis)):
    oi = objOis[k]
    nchannel = oi.visOi.nchannel
    for dit in range(oi.visOi.ndit):
        printinf("Calculating W2invH2 ({:d}/{:d})".format(counter, np.sum(np.array([oi.visOi.ndit for oi in objOis]))))
        counter = counter+1
        this_dit_mtot = 0
        for c in range(nchannel):        
            m = int(oi.visOi.m[dit, c])
            this_dit_mtot = this_dit_mtot + m
#        W_elem = W_obs[k][0:this_dit_mtot, 0:this_dit_mtot]
#        Z_elem = Z_obs[k][0:this_dit_mtot, 0:this_dit_mtot]
        this_r = 0
        H_elem = np.zeros([this_dit_mtot, oi.visOi.nchannel*oi.nwav], 'complex64')        
        W_elem = np.zeros([this_dit_mtot, this_dit_mtot], 'complex64')
        W2_elem_inv = np.zeros([2*this_dit_mtot, 2*this_dit_mtot], 'complex64')
        Z_elem = np.zeros([this_dit_mtot, this_dit_mtot], 'complex64')
        for c in range(nchannel):
#            Omega_c_sp = scipy.sparse.csc_matrix(np.diag(oi.visOi.phi_values[dit, c, :]))
#            Omega[c*oi.nwav:(c+1)*oi.nwav, c*oi.nwav:(c+1)*oi.nwav] = Omega_c_sp.todense()
#            Ginv_c_sp = scipy.sparse.csc_matrix(np.diag(1.0/np.abs(visRefs[k][c, :])))
 #           Ginv[c*oi.nwav:(c+1)*oi.nwav, c*oi.nwav:(c+1)*oi.nwav] = Ginv_c_sp.todense()
            m = int(oi.visOi.m[dit, c])            
            H_elem_c_sp = scipy.sparse.csc_matrix(oi.visOi.h_matrices[dit, c, 0:m, :])
            H_elem[this_r:this_r+m, c*oi.nwav:(c+1)*nwav] = H_elem_c_sp.todense()
#            Wc_sp = scipy.sparse.csr_matrix(np.diag(oi.visOi.visErr[dit, c, :].real**2+oi.visOi.visErr[dit, c, :].imag**2))
#            Zc_sp = scipy.sparse.csr_matrix(np.diag(oi.visOi.visErr[dit, c, :].real**2-oi.visOi.visErr[dit, c, :].imag**2))
            Wc_sp = scipy.sparse.csr_matrix(oi.visOi.visRefCov[dit, c].todense())
            Zc_sp = scipy.sparse.csr_matrix(oi.visOi.visRefPcov[dit, c].todense())
#            Wc_sp = (Ginv_c_sp.dot(Wc_sp)).dot(cs.adj(Ginv_c_sp))
#            Zc_sp = (Ginv_c_sp.dot(Zc_sp)).dot(Ginv_c_sp.T)
            W_elem_c = np.dot((H_elem_c_sp.dot(Wc_sp)).todense(), cs.adj(H_elem_c_sp).todense())
            Z_elem_c = np.dot((H_elem_c_sp.dot(Zc_sp)).todense(), H_elem_c_sp.T.todense())
            W_elem[this_r:this_r+m, this_r:this_r+m] = W_elem_c
            Z_elem[this_r:this_r+m, this_r:this_r+m] = Z_elem_c
            W2_elem_c = cs.extended_covariance(W_elem_c, Z_elem_c)
            W2_elem_c_inv = np.linalg.inv(W2_elem_c)
            W2_elem_inv[this_r:this_r+m, this_r:this_r+m] = W2_elem_c_inv[0:m, 0:m]
            W2_elem_inv[this_r:this_r+m, this_dit_mtot+this_r:this_dit_mtot+this_r+m] = W2_elem_c_inv[0:m, m:2*m]
            W2_elem_inv[this_dit_mtot+this_r:this_dit_mtot+this_r+m, this_r:this_r+m] = W2_elem_c_inv[m:2*m, 0:m]
            W2_elem_inv[this_dit_mtot+this_r:this_dit_mtot+this_r+m, this_dit_mtot+this_r:this_dit_mtot+this_r+m] = W2_elem_c_inv[m:2*m, m:2*m]
            this_r = this_r+m
#        Omega_sp = scipy.sparse.csc_matrix(Omega)
#        H_elem_sp = scipy.sparse.csc_matrix(H_elem)
#        Rcov = np.diag(oi.visOi.visErr[dit, :, :].real.reshape(oi.nwav*oi.visOi.nchannel)**2)
#        Icov = np.diag(oi.visOi.visErr[dit, :, :].imag.reshape(oi.nwav*oi.visOi.nchannel)**2)  
#        W_dit = np.dot(np.dot(Omega, Rcov+Icov), cs.adj(Omega))
#        Z_dit = np.dot(np.dot(Omega, Rcov-Icov), Omega.T)      
#        W_dit = np.dot(np.dot(Omega, W_dit), cs.adj(Omega))
#        Z_dit = np.dot(np.dot(Omega, Z_dit), cs.adj(Omega))
#        W_elem = np.dot(np.dot(H_elem, W_dit), cs.adj(H_elem))
#        Z_elem = np.dot(np.dot(H_elem, Z_dit), H_elem.T)
#        W2_elem = cs.extended_covariance(W_elem, Z_elem)                         
#        W2_elem_sp = scipy.sparse.csc_matrix(W2_elem)
#        W2invA2 = np.zeros([2*oi.nwav, 2*STAR_ORDER+1], 'complex')
#        ZZ, _ = lapack.dpotrf(W2)
#        T, info = lapack.dpotri(ZZ)
#        W2_elem_inv = np.triu(T) + np.triu(T, k=1).T
#        W2_elem_inv = np.linalg.inv(W2_elem)
        [A, B, C, D] = [W2_elem_inv[0:this_dit_mtot, 0:this_dit_mtot], W2_elem_inv[0:this_dit_mtot, this_dit_mtot:], W2_elem_inv[this_dit_mtot:, 0:this_dit_mtot], W2_elem_inv[this_dit_mtot:, this_dit_mtot:]]
        W2invH2[r:r+this_dit_mtot, :] = np.dot(A, H[r:r+this_dit_mtot])+np.dot(B, H[r:r+this_dit_mtot].conj())
        W2invH2[mtot+r:mtot+r+this_dit_mtot, :] = np.dot(C, H[r:r+this_dit_mtot])+np.dot(D, H[r:r+this_dit_mtot].conj())
        covY2W2invH2[r:r+this_dit_mtot, :] = np.dot(W_elem, W2invH2[r:r+this_dit_mtot, :]) + np.dot(Z_elem, W2invH2[mtot+r:r+mtot+this_dit_mtot, :])
        covY2W2invH2[mtot+r:mtot+r+this_dit_mtot, :] = np.dot(cs.adj(Z_elem), W2invH2[r:r+this_dit_mtot, :]) + np.dot(np.conj(W_elem), W2invH2[mtot+r:r+mtot+this_dit_mtot, :])
        pcovY2W2invH2[r:r+this_dit_mtot, :] = np.dot(Z_elem, W2invH2[r:r+this_dit_mtot, :]) + np.dot(W_elem, W2invH2[mtot+r:r+mtot+this_dit_mtot, :])
        pcovY2W2invH2[mtot+r:mtot+r+this_dit_mtot, :] = np.dot(cs.adj(W_elem), W2invH2[r:r+this_dit_mtot, :]) + np.dot(np.conj(Z_elem), W2invH2[mtot+r:r+mtot+this_dit_mtot, :]) 
        r = r+this_dit_mtot

# calculate spectrum
printinf("Calculating contrast spectrum")
A = np.real(np.dot(cs.adj(Y2), W2invH2))
try:
    B = np.linalg.inv(np.real(np.dot(cs.adj(H2), W2invH2)))
except:
    printwar("Problem with the inversion of model. Is the matrix singular? Trying a pseudo inverse instead.")
    B = np.linalg.pinv(np.real(np.dot(cs.adj(H2), W2invH2)))
C = np.dot(A, B)

# calculate errors
printinf("Calculating covariance matrix")
covY2 = np.dot(cs.adj(W2invH2), covY2W2invH2)
pcovY2 = np.dot(W2invH2.T, pcovY2W2invH2)
Cerr = np.dot(B, np.dot(0.5*np.real(covY2+pcovY2), B.T))

# save raw contrast spectrum and diagonal error terms in a file
printinf("Writting spectrum in FITS file")
saveFitsSpectrum(OUTPUT_DIR+"/"+SPECTRUM_FILENAME, wav, C, Cerr, C, Cerr)


# now we want to recontruct the planet visibilities and the best fit obtained, in the original
# pipeline reference frame. The idea is that we want to compare the residuals to the pipeline
# original errors to check from problems and for cosmic rays
for k in range(len(objOis)):
    oi = objOis[k]
    # we'll store the recontructed planet visibilities and the fit directly in the oi object
    oi.visOi.visPlanet = np.zeros([oi.visOi.ndit, oi.visOi.nchannel, oi.nwav], "complex")
    oi.visOi.visPlanetFit = np.zeros([oi.visOi.ndit, oi.visOi.nchannel, oi.nwav], "complex")    
    for dit, c in itertools.product(range(oi.visOi.ndit), range(oi.visOi.nchannel)):
        oi.visOi.visPlanet[dit, c, :] = np.dot(oi.visOi.p_matrices[dit, c, :, :], oi.visOi.visRef[dit, c, :])
#        oi.visOi.visPlanet[dit, c, -STAR_ORDER-1] = 0 # this projects orthogonally to the speckle noise
#        oi.visOi.visPlanet[dit, c, :] = np.dot(cs.adj(oi.visOi.h_matrices[dit, c, :, :]), oi.visOi.visPlanet[dit, c, :])        
        oi.visOi.visPlanet[dit, c, :] = oi.visOi.visPlanet[dit, c, :]*np.exp(1j*np.angle(visRefs[k][c, :]))        
        oi.visOi.visPlanetFit[dit, c, :] = C*np.abs(visRefs[k])[c, :]
        bad_indices = np.where(oi.visOi.flag[dit, c, :])
        oi.visOi.visPlanetFit[dit, c, bad_indices] = 0
        oi.visOi.visPlanetFit[dit, c, :] = np.dot((oi.visOi.p_matrices[dit, c, :, :]), oi.visOi.visPlanetFit[dit, c, :])                
        oi.visOi.visPlanetFit[dit, c, :] = oi.visOi.visPlanetFit[dit, c, :]*np.exp(1j*np.angle(visRefs[k][c, :]))

# now we are going to calculate the residuals, and the distances in units of error bars
for k in range(len(objOis)):
    oi = objOis[k]
    oi.visOi.residuals = np.zeros([oi.visOi.ndit, oi.visOi.nchannel, oi.nwav], "complex")
    oi.visOi.fitDistance = np.zeros([oi.visOi.ndit, oi.visOi.nchannel, oi.nwav])
    for dit, c in itertools.product(range(oi.visOi.ndit), range(oi.visOi.nchannel)):
        oi.visOi.residuals[dit, c, :] = oi.visOi.visPlanet[dit, c, :]-oi.visOi.visPlanetFit[dit, c, :]
        realErr = np.real(0.5*(np.diag(oi.visOi.visRefCov[dit, c].todense())+np.diag(oi.visOi.visRefPcov[dit, c].todense())))**0.5
        imagErr = np.real(0.5*(np.diag(oi.visOi.visRefCov[dit, c].todense())-np.diag(oi.visOi.visRefPcov[dit, c].todense())))**0.5   
        oi.visOi.fitDistance[dit, c, :] = np.abs(np.real(oi.visOi.residuals[dit, c])/realErr+1j*np.imag(oi.visOi.residuals[dit, c])/imagErr)

        
# now we want to save a new flag to indicate which points are badly fitted. But ONLY if gofast is set to False (we can't flag point if the individual DITs have been binned)
if (GO_FAST==False):
    printinf("Reflagging bad datapoints based on fit results (5 sigmas)")
    for k in range(len(objOis)):
        oi = objOis[k]
        hdul = fits.open(oi.filename, mode = "update")
        if "REFLAG" in [hdu.name for hdu in hdul]:
            hdul.pop([hdu.name for hdu in hdul].index("REFLAG"))
        reflag = np.zeros([oi.visOi.ndit, oi.visOi.nchannel, oi.nwav], 'bool')
        indx = np.where(oi.visOi.fitDistance > 5)
        reflag[indx] = True
        reflag = reflag | oi.visOi.flag
        hdul.append(fits.BinTableHDU.from_columns([fits.Column(name="REFLAG", format = str(oi.nwav)+"L", array = reflag.reshape([oi.visOi.nchannel*oi.visOi.ndit, oi.visOi.nwav]))], name = "REFLAG"))
        hdul.writeto(oi.filename, overwrite = "True")
        hdul.close()
        printinf("A total of {:d} bad points have be reflagged for file {}".format(len(indx[0]), oi.filename))

if not(FIGDIR is None):
    for k in range(len(objOis)*0+4):
        oi = objOis[k]
        fig = plt.figure(figsize = (10, 8))
        gPlot.reImPlot(w, (oi.visOi.visPlanet*(1-oi.visOi.flag)).mean(axis = 0)*np.exp(-1j*np.angle(visRefs[k])), subtitles = oi.basenames, fig = fig)
        gPlot.reImPlot(w, oi.visOi.visPlanetFit.mean(axis = 0)*np.exp(-1j*np.angle(visRefs[k])), fig = fig)
        plt.savefig(FIGDIR+"/spectrum_fit_"+str(k)+".pdf")
    fig = plt.figure(figsize=(10, 4))
    plt.plot(wav, C, '.-')
    plt.xlabel("Wavelength ($\mu\mathrm{m}$)")
    plt.ylabel("Contrast")
    plt.savefig(FIGDIR+"/contrast.pdf")


    
stop()


# ACTUAL CODE ENDS HERE!!!





# spectrum calibration for plot
wav_grav = oi.wav*1e6
#(starWav, starFlux) = exosed.utilities.convertSpectrum(starSpectrum, instrument = 'gravity-midres', resolution = 200000)
#star = np.interp(wav_grav, starWav, starFlux)

diagErrors = np.array([Cerr[k, k] for k in range(len(C))])**0.5

fig = plt.figure(figsize = (10, 6))

ax1 = fig.add_subplot(211)
ax1.errorbar(wav_grav, C, yerr = diagErrors, fmt = '.')

ax2 = fig.add_subplot(212, sharex = ax1)
#ax2.errorbar(wav_grav, C*star, yerr = diagErrors*star, fmt = '.')

ax1.set_xlabel("Wavelength ($\mu\mathrm{m}$)")
ax2.set_xlabel("Wavelength ($\mu\mathrm{m}$)")
ax1.set_ylabel("Contrast")
ax2.set_ylabel("Flux")

fig.tight_layout()

# save the data
data = np.zeros([len(wav), 2])
data[:, 0] = wav
data[:, 1] = np.dot(A, B)
#np.savetxt("betapicb_contrast.txt", data)
#np.savetxt("betapicb_errors.txt", Cerr)




stop

"""
data = np.loadtxt('results_5648/betapicb_contrast.txt')
C = data[:, 1]

all_coeffs = []
ind = np.zeros([6, 400])
for c in range(objOis[0].visOi.nchannel):
    ind[c, :] = range(400)
w = np.zeros([6, objOis[0].nwav])
for c in range(objOis[0].visOi.nchannel):
    w[c, :] = objOis[0].wav
# fit the slope
pbar = patiencebar.Patiencebar(valmax = len(objOis))
for k in range(len(objOis)):
    pbar.update()
    oi = objOis[k]
    coeffs = np.zeros([oi.visOi.ndit, 6, 2])    
    Y = np.zeros([oi.visOi.nchannel, oi.nwav], 'complex')    
    Yfit = np.zeros([oi.visOi.nchannel, oi.nwav], 'complex')
    for dit in range(oi.visOi.ndit):
        for c in range(oi.visOi.nchannel):
            m = int(oi.visOi.m[dit, c])                                    
            X = np.zeros([m, 2], 'complex')
            H = oi.visOi.h_matrices[dit, c, 0:m, :]
            U = np.dot(H, oi.visOi.visRef[dit, c, :])
            U2 = gravity.conj_extended(U)            
            G = np.diag(np.abs(totalVisRef[c, :]))
            Lambda = np.diag((oi.wav - np.mean(oi.wav))*1e6)
            X[:, 0] = np.dot(np.dot(H, G), C)
            X[:, 1] = np.dot(np.dot(H, G), np.dot(Lambda, C))
            X2 = gravity.conj_extended(X)
            Rcov = np.diag(oi.visOi.visErr[dit, c, :].real)**2
            Icov = np.diag(oi.visOi.visErr[dit, c, :].imag)**2
            Omega = np.diag(oi.visOi.phi_values[dit, c, :])
            W = np.dot(np.dot(Omega, Rcov+Icov), gravity.adj(Omega))
            Z = np.dot(np.dot(Omega, Rcov-Icov), Omega.T)        
            W = np.dot(np.dot(H, W), gravity.adj(H))
            Z = np.dot(np.dot(H, Z), H.T)        
            W2 = gravity.extended_covariance(W, Z)                         
            W2 = gravity.extended_covariance(W, Z)
            W2inv = np.linalg.inv(W2)
            K = np.real(np.dot(np.dot(gravity.adj(X2), W2inv), X2))
            A = np.dot(np.real(np.dot(np.dot(gravity.adj(U2), W2inv), X2)), np.linalg.pinv(K))
            coeffs[dit, c, :] = np.real(A)
            Xp = np.zeros([oi.nwav, 2], 'complex')
            Xp[:, 0] = np.dot(G, C)
            Xp[:, 1] = np.dot(G, np.dot(Lambda, C))
            Yfit[c, :] = Yfit[c, :] + np.dot(A, Xp.T)
    all_coeffs.append(coeffs)            
    fig = plt.figure()
    gravity.reImPlot(w, oi.visOi.visRef.sum(axis = 0), fig = fig, subtitles = oi.basenames)
    gravity.reImPlot(w, Yfit, fig = fig)

coeffs = np.concatenate(all_coeffs)

fig = plt.figure()
gravity.baselinePlot(ind, coeffs[:, :, 0].T, fig = fig, subtitles = oi.basenames)

fig = plt.figure()
gravity.baselinePlot(ind, coeffs[:, :, 1].T/coeffs[:, :, 0].T, fig = fig, subtitles = oi.basenames)


slopes = np.zeros([objOis[0].visOi.nchannel, objOis[0].nwav])
for c in range(objOis[0].visOi.nchannel):
    slopes[c, :] = coeffs[:, :, 1].mean(axis = 0)[c]*(objOis[0].wav - np.mean(objOis[0].wav))*1e6+coeffs[:, :, 0].mean(axis = 0)[c]
fig = plt.figure()
gravity.baselinePlot(w, slopes, fig = fig, subtitles = oi.basenames)
"""


### NORMALIZE ###

# ESO K filter
data = np.loadtxt("eso_filter_K.txt", skiprows = 4)
eso_wav = data[:, 0]
eso_filt = data[:, 1]
wav = eso_wav

# star spectrum
starWav = val[:, 0]
starFlux = val[:, 1]

# ESO K mag calib
eso_filt_interp = np.interp(starWav, eso_wav, eso_filt)
flux = np.trapz(eso_filt_interp*starFlux, starWav)/0.33
eso_zp = 4.12e-10
eso_mag = 5.687
eso_flux = eso_zp*10**(-eso_mag/2.5)

plt.figure()
plt.plot(starWav, starFlux)
plt.plot(starWav, starFlux*eso_filt_interp)

norm = eso_flux / flux

"""
stop

# conversion to gaia mag
starPhotons = starFlux  * (starWav)/(10**6*6.626e-34*299792458.0) # planck constant and speed of light. photon flux phot/s/m^2
g_interp = np.interp(starWav, wav, g)
betapic_gaia_flux = np.trapz(g_interp*starPhotons, starWav)*0.7278 # telescope surface
betapic_gaia_mag = gaia_zeropoint - 2.5*np.log(betapic_gaia_flux)/np.log(10)
print(betapic_gaia_mag)
norm = 10**((betapic_gaia_mag - 3.72)/2.5)
"""

"""
# GRAVITY
hdu = fits.open("spectra/Mickael/Synthetic_spectrum_BpicA.fits")[0]
data = hdu.data
ind = np.where((data[:, 0] < 3.0) & (data[:, 0] > 1.5))[0]
starSpectrum = exosed.Spectrum(data[ind, 0], data[ind, 1])
(starWav, starFlux) = exosed.utilities.convertSpectrum(starSpectrum, instrument = 'gravity-midres', resolution = 20000)
starWav = starSpectrum.wav
starFlux = starSpectrum.flux
data = np.loadtxt(GRAVITY_SPECTRUM)
raw_wav = data[:, 0]/10.0
star = np.interp(raw_wav, starWav, starFlux)
raw_flux = data[:, 1]*star#*norm
W = np.loadtxt(GRAVITY_SPECTRUM_ERR)
raw_flux_err = np.array([W[k, k]**0.5 for k in range(len(raw_wav))])*star#*norm

# GPI
data = np.loadtxt("spectra/gpi_spectrum.txt")
wav_gpi = data[:, 0]
flux_gpi = data[:, 1]
flux_err_gpi = data[:, 2]
flux_gpi[-44:-33] = float('nan')

# photometry
zero_point = 3.961e-11*1e4*1e-7*1e4 # W/s/cm2/micron 
bpic_mag = 3.48
mag = np.array([9.2, 9.02, 8.92, 8.8])
mag_err = np.array([0.1, 0.13, 0.13, 0.6])
phot_wav = np.array([2.159, 2.174, 2.174, 2.174])
phot_wav_err = np.array([0.324, 0.269, 0.269, 0.269])/2.0
d_bpic = 19.44
d_vega = 7.68

phot = (10**(-(bpic_mag+mag)/2.5)*zero_point)#*(d_vega/d_bpic)**2
phot_err = ( (10**(-(bpic_mag+mag+mag_err)/2.5) - 10**(-(bpic_mag+mag)/2.5))*zero_point )#*(d_vega/d_bpic)**2

plt.figure(figsize=(10,6));
plt.errorbar(raw_wav, 1e14*raw_flux, yerr = 1e14*raw_flux_err, fmt = 'o')
plt.errorbar(raw_wav, 1e14*raw_flux*norm, yerr = 1e14*raw_flux_err*norm, fmt = 'o')
plt.errorbar(wav_gpi, 1e14*flux_gpi, yerr = 1e14*flux_err_gpi, fmt = 'o')
plt.errorbar(phot_wav[1:], 1e14*phot[1:], xerr = phot_wav_err[1:], yerr = 1e14*phot_err[1:], fmt = 's', capsize=8)
plt.errorbar(phot_wav[0:1], 1e14*phot[0:1], xerr = phot_wav_err[0:1], yerr = 1e14*phot_err[0:1], fmt = 's', capsize=8)
plt.legend(["Gravity calib Mickael", "Gravity ESO K calib", "GPI", "GPI NICI photometry", "VLT NACO photometry"])
plt.xlim(1.8, 2.6)
plt.ylim(0.2, 0.9)
plt.xlabel("Wavelength ($\mu\mathrm{m}$)")
plt.ylabel("Flux ($\\times{}10^{-14}\mathrm{W}\,\mathrm{m}^{-2}\mu\mathrm{m}^{-1}$)")
"""


norm = 1.54e-6

C08 = np.loadtxt("contrast_08.txt")
diagErrors08 = np.loadtxt("contrast_err_08.txt")

C07 = np.loadtxt("contrast_07.txt")
diagErrors07 = np.loadtxt("contrast_err_07.txt")

data = np.loadtxt("hd206893.csv", delimiter = ";")
flux_h = data[:-6, 1]
wav_h = data[:-6, 0]

wav_mag = data[-6:-4, 0]
mag = data[-6:-4, 1]
mag_err = mag - data[-4::2, 1]

plt.figure()
plt.errorbar(wav_grav, C08*star*norm*1e15, yerr = diagErrors08*star*norm*1e15, fmt = '.', capsize=2, color = "gray", alpha = 0.7)
plt.errorbar(wav_grav, C07*star*norm*1e15, yerr = diagErrors07*star*norm*1e15, fmt = '.', capsize=2, color = "orange", alpha = 0.7)
#plt.plot(wav_h, flux_h*1e15, 'o')
plt.errorbar(wav_mag, mag*1e15, yerr = mag_err*1e15, fmt = ".k", markersize = 15, linewidth = 2, capsize = 10)

plt.plot(wav_grav, (C08+C07)/2*1e15*star*norm, 'o--')

plt.xlabel("Wavelength ($\mu$m)")
plt.ylabel("Flux ($\\times{}10^{-15}\mathrm{W}\mathrm{m}^{-2}\mathrm{s}^{-1}\mu{}\mathrm{m}^{-1}$)")

plt.legend(["IRDIS $K1$ and $K_2$ mag", "GRAVITY July 2019", "GRAVITY Aug 2019"])
