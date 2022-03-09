#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Calculate the phase reference which needs to be subtracted to the phase of the visibilities in dual field

This script is part of the exoGravity data reduction package.
The create_phase_reference script is used to extract the phase reference from the swap files or the on-star files in 
an exoplanet observation made with GRAVITY, in dual-field mode.
To use this script, you need to call it with a configuration file, see example below.

Args:
  config_file (str): the path the to YAML configuration file.

Example:
  python create_phase_reference config_file=full/path/to/yourconfig.yml

Authors:
  M. Nowak, and the exoGravity team.

Version:
  xx.xx
"""

# BASIC IMPORTS
import numpy as np
import astropy.io.fits as fits
import cleanGravity as gravity
from utils import * # utils from this exoGravity package
from common import *
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
if not("save_residuals") in dargs.keys():
    dargs['save_residuals'] = False
    
# READ THE CONFIGURATION FILE
if RUAMEL:
    cfg = ruamel.yaml.load(open(CONFIG_FILE, "r"), Loader=ruamel.yaml.RoundTripLoader)
else:
    cfg = yaml.safe_load(open(CONFIG_FILE, "r"))
    
DATA_DIR = cfg["general"]["datadir"]
PHASEREF_MODE = cfg["general"]["phaseref_mode"]
FIGDIR = cfg["general"]["figdir"]
PLANET_FILES = [DATA_DIR+cfg["planet_ois"][preduce[list(preduce.keys())[0]]["planet_oi"]]["filename"] for preduce in cfg["general"]["reduce_planets"]]
if not("swap_ois" in cfg.keys()):
    SWAP_FILES = []
elif cfg["swap_ois"] is None:
    SWAP_FILES = []
else:
    SWAP_FILES = [DATA_DIR+cfg["swap_ois"][preduce[list(preduce.keys())[0]]["swap_oi"]]["filename"] for preduce in cfg["general"]["reduce_swaps"]]

EXTENSION = cfg["general"]["extension"]
REDUCTION = cfg["general"]["reduction"]

PHASEREF_ARCLENGTH_THRESHOLD = cfg["general"]["phaseref_arclength_threshold"]
FT_FLUX_THRESHOLD = cfg["general"]["ft_flux_threshold"]

# OVERWRITE SOME OF THE CONFIGURATION VALUES WITH ARGUMENTS FROM COMMAND LINE
if "figdir" in dargs.keys():
    FIGDIR = dargs["figdir"] # bypass value from config file

# LOAD GRAVITY PLOT is savefig requested
if not(FIGDIR is None):
    from cleanGravity import gravityPlot as gPlot
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib.backends.backend_pdf import PdfPages        
    if not(os.path.isdir(FIGDIR)):
        os.makedirs(FIGDIR)
        printinf("Directory {} was not found and has been created".format(FIGDIR))

# extract list of useful star ois from the list indicated in the star_indices fields of the config file:
star_indices = []
for preduce in cfg["general"]["reduce_planets"]:
    pkey = preduce[list(preduce.keys())[0]]["planet_oi"]
    star_indices = star_indices+cfg["planet_ois"][pkey]["star_indices"]
star_indices = list(set(star_indices)) # remove duplicates
STAR_FILES = [DATA_DIR+cfg["star_ois"][k]["filename"] for k in star_indices]

# lists to contain the planet, star, and potentially swap OIs
starOis = [] # will contain OIs on the central star
objOis = [] # contains the OIs on the planet itself
swapOis = [] # first position of the swap (only in DF_SWAP mode)

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

for filename in STAR_FILES:
    printinf("Loading file "+filename)
    if (PHASEREF_MODE == "DF_SWAP"):
        if REDUCTION == "astrored":        
            oi = gravity.GravityDualfieldAstrored(filename, corrMet = cfg["general"]["corr_met"], extension = EXTENSION, corrDisp = cfg["general"]["corr_disp"])
        elif REDUCTION == "dualscivis":
            oi = gravity.GravityDualfieldScivis(filename, extension = EXTENSION)
        else:
            printerr("Unknonwn reduction type '{}'.".format(REDUCTION))            
    else:
        if REDUCTION == "astrored":                
            oi = gravity.GravityDualfieldAstrored(filename, corrMet = cfg["general"]["corr_met"], extension = EXTENSION, corrDisp = cfg["general"]["corr_disp"])
        elif REDUCTION == "dualscivis":
            oi = gravity.GravityDualfieldScivis(filename, extension = EXTENSION)
        else:
            printerr("Unknonwn reduction type '{}'.".format(REDUCTION))                        
    starOis.append(oi)
    printinf("File is on star")
                                              
for filename in SWAP_FILES:
    printinf("Loading file "+filename)
    if REDUCTION == "astrored":                
        oi = gravity.GravityDualfieldAstrored(filename, corrMet = cfg["general"]["corr_met"], extension = EXTENSION, corrDisp = cfg["general"]["corr_disp"])
    elif REDUCTION == "dualscivis":    
        oi = gravity.GravityDualfieldScivis(filename, extension = EXTENSION)
    else:
        printerr("Unknonwn reduction type '{}'.".format(REDUCTION))                                                                        
    swapOis.append(oi)
    printinf("File is from a SWAP")

if REDUCTION == "astrored":
    # flag points based on FT value and phaseRef arclength
    ftThresholdStar = cfg["general"]["ftOnStarMeanFlux"]*FT_FLUX_THRESHOLD
    for oi in starOis:
        filter_ftflux(oi, ftThresholdStar)             

# calculate the very useful w for plotting
if len(objOis) > 0:
    oi = objOis[0]
else:
    oi = swapOis[0]    
w = np.zeros([oi.visOi.nchannel, oi.nwav])
for c in range(oi.visOi.nchannel):
    w[c, :] = oi.wav*1e6
        
# create the visibility reference. This step depends on PHASEREF_MODE (DF_STAR or DF_SWAP)
printinf("Creating the visibility reference from {:d} star observations.".format(len(starOis)))
visRefs = [oi.visOi.visRef.mean(axis=0)*0 for oi in objOis]
for k in range(len(objOis)):
    preduce = cfg["general"]["reduce_planets"][k]
    planet_ind = preduce[list(preduce.keys())[0]]["planet_oi"]
    ampRef = np.zeros([oi.visOi.nchannel, oi.nwav])
    visRef = np.zeros([oi.visOi.nchannel, oi.nwav], "complex")
    if cfg["general"]["calib_strategy"].lower()=="none":
        ampRef = ampRef+1 # put the amplitude reference to one if no calibration strategy is used
        visRefs[k] = ampRef
    else: # otherwise, create the amplitude reference from the proper on-star observations
        for ind in cfg["planet_ois"][planet_ind]["star_indices"]:
            # not all star files have been loaded, so we cannot trust the order and we need to explicitly look for the correct one
            starOis_ind = [soi.filename for soi in starOis].index(DATA_DIR+cfg["star_ois"][ind]["filename"]) 
            soi = starOis[starOis_ind]
            visRef = visRef+soi.visOi.visRef.mean(axis = 0)
            #        ampRef = ampRef+soi.fluxOi.flux.mean(axis = 0).mean(axis = 0)#np.abs(soi.visOi.visRef.mean(axis = 0))
            ampRef = ampRef+np.abs(soi.visOi.visRef.mean(axis = 0))
        ampRef=ampRef/len(cfg["planet_ois"][planet_ind]["star_indices"])
        visRefs[k] = ampRef*np.exp(1j*np.angle(visRef/len(cfg["planet_ois"][planet_ind]["star_indices"])))#/len(starOis))) ###===###
    
# in DF_SWAP mode, the phase reference of the star cannot be used. We need to extract the phase ref from the SWAP observations
if PHASEREF_MODE == "DF_SWAP":
    printinf("DF_SWAP mode set.")
    printinf("Calculating the reference phase from {:d} swap observation".format(len(swapOis)))
    # first we need to shift all of the visibilities to the 0 OPD position, using the separation of the SWAP binary
    # if swap ra and dec values are provided (from the swapReduce script), we can use them
    # otherwise we can default to the fiber separation value
    for k in range(len(swapOis)):
        oi = swapOis[k]
        key = list(cfg["swap_ois"].keys())[k] # the corresponding key in the config file
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
    # now we can take the phaseref
    phaseRef = np.angle(0.5*(phaseRef2+phaseRef1))
    # for convenience, we store this ref in visRef angle, getting rid of the useless values from the star
    visRefs = [2*np.abs(visRef)*np.exp(1j*phaseRef) for visRef in visRefs] # factor 2 because the beamspliter is used for on-star observations

# SAVE VISREF IN THE FITS FILE
for k in range(len(objOis)):
    oi = objOis[k]
    printinf("Saving reference visibility in {}".format(oi.filename))
    hdul = fits.open(oi.filename, mode = "update")
    if "EXOGRAV_VISREF" in [hdu.name for hdu in hdul]:
        hdul.pop([hdu.name for hdu in hdul].index("EXOGRAV_VISREF"))
    hdul.append(fits.BinTableHDU.from_columns([fits.Column(name="EXOGRAV_VISREF", format = str(oi.nwav)+"C32", array = visRefs[k].reshape([oi.visOi.nchannel, oi.visOi.nwav]))], name = "EXOGRAV_VISREF"))
    hdul.writeto(oi.filename, overwrite = "True")
    hdul.close()
    
# IN DF_SWAP mode, we also save the visref in the swap files to allow the user to treat these files with atrometry_reduce
#if PHASEREF_MODE == "DF_SWAP":
#    for k in range(len(swapOis)):
#        oi = swapOis[k]
#        printinf("Saving reference visibility in {}".format(oi.filename))
#        hdul = fits.open(oi.filename, mode = "update")
#        if "EXOGRAV_VISREF" in [hdu.name for hdu in hdul]:
#            hdul.pop([hdu.name for hdu in hdul].index("EXOGRAV_VISREF"))
#        hdul.append(fits.BinTableHDU.from_columns([fits.Column(name="EXOGRAV_VISREF", format = str(oi.nwav)+"C32", array = visRefsForSwaps[k].reshape([oi.visOi.nchannel, oi.visOi.nwav]))], name = "EXOGRAV_VISREF"))
#        hdul.writeto(oi.filename, overwrite = "True")
#        hdul.close()

if not(FIGDIR is None):
    with PdfPages(FIGDIR+"/phase_reference.pdf") as pdf:
        for k in range(len(objOis)):
            oi = objOis[k]
            fig = plt.figure(figsize=(10, 8))
            gPlot.modPhasePlot(w, visRefs[k], subtitles = oi.basenames, fig = fig)
            plt.legend(["PhaseRef "+oi.filename.split("/")[-1]])            
            pdf.savefig()
            plt.close(fig)
        for k in range(len(objOis)):
            oi = objOis[k]
            fig = plt.figure(figsize=(10, 8))
            gPlot.modPhasePlot(w, np.ma.masked_array(oi.visOi.visRef/visRefs[k], oi.visOi.flag).mean(axis = 0), subtitles = oi.basenames, fig = fig)
            plt.legend([oi.filename.split("/")[-1]+"/Ref"])
            pdf.savefig()
            plt.close(fig)        
