#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Calculate the phase reference which needs to be subtracted to the phase of the visibilities in dual field

This script is part of the exoGravity data reduction package.
The create_phase_reference script is used to extract the phase reference from the swap files or the on-star files in 
an exoplanet observation made with GRAVITY, in dual-field mode.
To use this script, you need to call it with a configuration file, see example below.

Authors:
  M. Nowak, and the exoGravity team.
"""

# BASIC IMPORTS
import sys, os
import numpy as np
import astropy.io.fits as fits
# cleanGravity package
import cleanGravity as gravity
# package related functions
from exogravity.utils import * # utils from this exoGravity package
from exogravity.common import *
# ruamel to read config yml file
try:
    import ruamel.yaml
    RUAMEL = True
except: # if ruamel not available, switch back to pyyaml, which does not handle comments properly
    import yaml
    RUAMEL = False
# argparse for command line arguments
import argparse

# create the parser for command lines arguments
parser = argparse.ArgumentParser(description=
"""
Calculate the phase reference which needs to be subtracted to the phase of the visibilities in dual field
""")

# required arguments are the path to the folder containing the data, and the path to the config yml file to write 
parser.add_argument('config_file', type=str, help="the path the to YAML configuration file.")

# some elements from the config file can be overridden with command line arguments
parser.add_argument("--figdir", metavar="DIR", type=str, default=argparse.SUPPRESS,
                    help="name of a directory where to store the output PDF files [overrides value from yml file]")   

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

DATA_DIR = cfg["general"]["datadir"]
PHASEREF_MODE = cfg["general"]["phaseref_mode"]
FIGDIR = cfg["general"]["figdir"]
EXTENSION = cfg["general"]["extension"]
REDUCTION = cfg["general"]["reduction"]
PHASEREF_ARCLENGTH_THRESHOLD = cfg["general"]["phaseref_arclength_threshold"]
FT_FLUX_THRESHOLD = cfg["general"]["ft_flux_threshold"]

PLANET_FILES = [DATA_DIR+cfg["planet_ois"][preduce[list(preduce.keys())[0]]["planet_oi"]]["filename"] for preduce in cfg["general"]["reduce_planets"]]

if not("swap_ois" in cfg.keys()):
    SWAP_FILES = []
elif cfg["swap_ois"] is None:
    SWAP_FILES = []
else:
    SWAP_FILES = [DATA_DIR+cfg["swap_ois"][preduce[list(preduce.keys())[0]]["swap_oi"]]["filename"] for preduce in cfg["general"]["reduce_swaps"]]


        
# LOAD GRAVITY PLOT is savefig requested
if not(FIGDIR is None):
    import matplotlib
    matplotlib.use('Agg')        
    import matplotlib.pyplot as plt    
    from cleanGravity import gravityPlot as gPlot
    from matplotlib.backends.backend_pdf import PdfPages        
    if not(os.path.isdir(FIGDIR)):
        os.makedirs(FIGDIR)
        printinf("Directory {} was not found and has been created".format(FIGDIR))

# extract list of useful star ois from the list indicated in the star_indices fields of the config file:
star_indices = []
if cfg["general"]["calib_strategy"].lower() == "all": # in this case we need everything
    star_indices = list(cfg["star_ois"].keys())
else:
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
            oi = gravity.GravitySinglefieldAstrored(filename, corrMet = cfg["general"]["corr_met"], extension = EXTENSION, corrDisp = cfg["general"]["corr_disp"])
        elif REDUCTION == "dualscivis":
            oi = gravity.GravitySinglefieldScivis(filename, extension = EXTENSION)
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

printinf("Normalizing on-star visibilities to FT coherent flux.")
for oi in starOis+objOis:
    oi.visOi.scaleVisibilities(1.0/np.abs(oi.visOi.visDataFt).mean(axis = -1))
        
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


# in DF_STAR mode, the phase reference of the star is used
# IN DF_SWAP, we also need to go through this step to retrieve amplitude of the star for reference
if (PHASEREF_MODE == "DF_STAR") or (PHASEREF_MODE == "DF_SWAP"):
    for k in range(len(objOis)):
        preduce = cfg["general"]["reduce_planets"][k]
        planet_ind = preduce[list(preduce.keys())[0]]["planet_oi"]
        ampRef = np.zeros([oi.visOi.nchannel, oi.nwav])
        visRef = np.zeros([oi.visOi.nchannel, oi.nwav], "complex")
        if cfg["general"]["calib_strategy"].lower()=="none":
            ampRef = ampRef+1 # put the amplitude reference to one if no calibration strategy is used
            visRefs[k] = ampRef # no vis reference in this cas
        elif cfg["general"]["calib_strategy"].lower()=="self":
            ampRef = np.abs(objOis[k].visOi.visRef).mean(axis=0)
            visRefs[k] = ampRef # no vis reference in this case
        else: # otherwise, create the amplitude reference from the proper on-star observations
            for ind in cfg["planet_ois"][planet_ind]["star_indices"]: 
                # not all star files may have been loaded, so we cannot trust the order and we need to explicitly look for the correct one
                starOis_ind = [soi.filename for soi in starOis].index(DATA_DIR+cfg["star_ois"][ind]["filename"]) 
                soi = starOis[starOis_ind]
                visRef = visRef+soi.visOi.visRef.mean(axis = 0)
                ampRef = ampRef+np.abs(soi.visOi.visRef.mean(axis = 0))
            ampRef = ampRef/len(cfg["planet_ois"][planet_ind]["star_indices"])
            visRefs[k] = ampRef*np.exp(1j*np.angle(visRef/len(cfg["planet_ois"][planet_ind]["star_indices"])))#/len(starOis))) ###===###
    
# in DF_SWAP mode, the phase reference of the star cannot be used. We need to extract the phase ref from the SWAP observations
if PHASEREF_MODE == "DF_SWAP":
    printinf("DF_SWAP mode set.")
    # if the swap strategy is requested, we also store the swap amplitude as the amplitude reference
    if cfg["general"]["calib_strategy"].lower()=="swap":
        printinf("Calculating the amplitude reference from {:d} swap observations".format(len([oi for oi in swapOis if oi.swap])))        
        ampRef = np.zeros([swapOis[0].visOi.nchannel, oi.nwav])        
        for k in range(len(swapOis)):
            oi = swapOis[k]
            if oi.swap:
                ampRef = ampRef+np.abs(swapOis[k].visOi.visRef).mean(axis = 0)
        ampRef = ampRef/len([oi for oi in swapOis if oi.swap])
        visRefs = [0.5*ampRef for visRef in visRefs] # phaseRef will be added after. factor 1/2 because in swap the flux is two times higher than with the usual on-axis    
    printinf("Calculating the reference phase from {:d} swap observations".format(len(swapOis)))
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
    
if not(FIGDIR is None):
    with PdfPages(FIGDIR+"/phase_reference.pdf") as pdf:
        for k in range(len(objOis)):
            oi = objOis[k]
            fig = plt.figure(figsize=(10, 8))
            gPlot.modPhasePlot(w, visRefs[k], subtitles = oi.basenames, fig = fig, xlabel="Wavelength ($\mu\mathrm{m}$)")
            plt.legend(["PhaseRef "+oi.filename.split("/")[-1]])            
            pdf.savefig()
            plt.close(fig)
        for k in range(len(objOis)):
            oi = objOis[k]
            fig = plt.figure(figsize=(10, 8))
            gPlot.modPhasePlot(w, np.ma.masked_array(oi.visOi.visRef/visRefs[k], oi.visOi.flag).mean(axis = 0), subtitles = oi.basenames, fig = fig, xlabel="Wavelength ($\mu\mathrm{m}$)")
            plt.legend([oi.filename.split("/")[-1]+"/Ref"])
            pdf.savefig()
            plt.close(fig)        
