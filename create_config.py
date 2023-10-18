#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create the YAML configuration script

This script is part of the exoGravity data reduction package.
The create_config script is used to create a basic YAML configuration file for an observation, which contains data
used by the other scripts, formatted in YAML to be easily loaded as a Python dict.

Configuration:
  The configuration file must contain all necessary information as to where the data are located on your disk, which mode
  of observation was used, as well as some parameters for the data reduction. It is strongly advised to use the
  create_config.py script (also provided with the exoGravity data reduction package) to create your configuration file, 
  which you can then tweek to fit your need. Please refer to the documentation of create_config.py.

Authors:
  M. Nowak, and the exoGravity team.
"""

# basic imports
import sys, os
import numpy as np
import astropy.io.fits as fits
from datetime import datetime
# cleanGravity 
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
# other random stuff
import glob

# create the parser for command lines arguments
parser = argparse.ArgumentParser(description=
"""
create a basic YAML configuration file for an observation, which contains data
used by the other scripts, formatted in YAML to be easily loaded as a Python dict.
""")

# required arguments are the path to the folder containing the data, and the path to the config yml file to write 
parser.add_argument('datadir', type=str, help="the directory in which the fits files are located")
parser.add_argument('output', type=str, help="name (with full path) of the yml file in which to store the resulting configuration")

# optional arguments
parser.add_argument("--figdir", metavar="DIR", type=str, default=None,
                    help="name of a directory where to store the output PDF files (not created if no figdir given)")   

# for the astrometry map
parser.add_argument("--ralim", metavar=('MIN', 'MAX'), type=float, nargs=2, default=None,
                    help="specify the RA range (in mas) over which to search for the astrometry. Default: fiber position +/- 30 mas (UTs) or +/- 120 mas (ATs).") 
parser.add_argument("--declim", metavar=('MIN', 'MAX'), type=float, nargs=2, default=None, 
                    help="specify the DEC range (in mas) over which to search for the astrometry. Default: fiber position +/- 30 mas (UTs) or +/- 120 mas (ATs).")  
parser.add_argument("--nra", metavar="N", type=int, default=50,
                    help="number of points over the RA range. Default: 50")  
parser.add_argument("--ndec", metavar="N", type=int, default=50,
                    help="number of points over the DEC range. Default: 50")    

# for the astrometry map on the swap
parser.add_argument("--ralim_swap", metavar=('MIN', 'MAX'), type=float, nargs=2, default=None,
                    help="specify the RA range (in mas) over which to search for the astrometry of the swap. Default: fiber position +/- 30 mas (UTs) or +/- 120 mas (ATs).")     
parser.add_argument("--declim_swap", metavar=('MIN', 'MAX'), type=float, nargs=2, default=None,
                    help="specify the DEC range (in mas) over which to search for the astrometry. Default: fiber position +/- 30 mas (UTs) or +/- 120 mas (ATs).")    
parser.add_argument("--nra_swap", metavar="N", type=int, default=50,
                    help="number of points over the RA range for the swap. Default: 50")    
parser.add_argument("--ndec_swap", metavar="N", type=int, default=50,
                    help="number of points over the DEC range for the swap. Default: 50")    

# whether to zoom
parser.add_argument("--zoom", metavar="ZOOM_FACTOR", type=int, help="Create an additional zoomed in version of the chi2 map around the best guess from the initial map. Default: No zoom", default = 0)

# some parameters related to how files are loaded
parser.add_argument("--target", metavar="NAME", type=str, default = None,
                    help="restrict the reduction to a particular target (from FITS header). Default: None")
parser.add_argument("--swap_target", metavar="NAME", type=str, default = None,
                    help="target name (from FITS header) for the binary swap calibration observation if off-axis mode. Default: None")
parser.add_argument("--fiber_pos", metavar=("RA", "DEC"), type=int, nargs=2, default = None,
                    help="restrict the reduction to a single fiber position (in mas). Default: all files are taken")

parser.add_argument("--reduction", metavar="LEVEL", type=str, default = "astrored", choices=["astrored", "dualscivis"],
                    help="the data reduction level to use. Default: astrored")
parser.add_argument("--extension", metavar="EXT", type=int, default = 10, choices=[10, 11, 12],
                    help="Which extension to use for the polarimetry. Default: 10 (combined)")
parser.add_argument("--corr_met", metavar="METHOD", type=str, default = "drs", choices = ["drs", "sylvestre"],
                    help="how to calculate the metrology correction for astrored files. can be can be 'sylvestre' or 'drs', or possibly 'none'. Default: drs")
parser.add_argument("--corr_disp", metavar="METHOD", type=str, default = "drs", choices = ["drs", "sylvestre"],
                    help="how to calculate the dispersion correction for astrored files. can be can be 'sylvestre' or 'drs', or possibly 'none'. Default: drs")

# data filtering
parser.add_argument("--phaseref_arclength_threshold", metavar="THRESHOLD", type=float, default = 5,
                    help="remove all dit/baseline with an arclength of the phaseref polyfit larger this threshold. Default: 5")
parser.add_argument("--ft_flux_threshold", metavar="THRESHOLD", type=float, default = 0.2,
                    help="remove all dits/baselines with an ft flux below THRESH*MEAN(FT_FLUX_ON_STAR) (default 0.2)")
parser.add_argument("--reflag", metavar="TRUE/FALSE", type=bool, default=False,
                     help="if set, use the REFLAG table to filter bad datapoints. Requires running the scriot twice! Default: false")  # deprecated?

# for the OPD calculations on baselines
parser.add_argument("--nopd", type=int, default=100,
                    help="DEPRECATED. Default: 100")    

# how to deal with detrending and calibrations
parser.add_argument("--poly_order", metavar="ORDER", type=int, default=4,
                    help="order of the polynomial used to detrend the data (remove the stellar component in exoplanet observations). Default: 4")   
parser.add_argument("--contrast_file", metavar="FILE", type=str, default=None,
                    help="path to the contrast file (planet/star) to use as a model for the planet visibility amplitude. Default: Constant contrast")
parser.add_argument("--star_diameter", metavar="DIAMETER", type=float, default = 0,
                    help="the diameter of the central star (in mas, to scale the visibility amplitude). Default: 0 mas")
parser.add_argument("--calib_strategy", metavar="STRATEGY", type=str, default = "nearest", choices = ["nearest", "self", "all", "none"],
                    help="""
                    how to calculate the reference the coherent flux. 'all' for using all file. 'nearest' for using the two nearest available files.
                    erence. 'none' to skip the calibration. 'self' to calibrate each file by itself. Default: 'nearest'
                    """)

# switches for different behaviors
parser.add_argument("--go_fast", metavar="EMPTY or TRUE/FALSE", type=bool, default=False, nargs="?", const=True,
                    help="if set, average over DITs to accelerate calculations. Default: false")   
parser.add_argument("--gradient", metavar="EMPTY or TRUE/FALSE", type=bool, default=True,  nargs="?", const=True,
                     help="if set, improves the estimate of the location of chi2 minima by performing a gradient descent from the position on the map. Default: true")   
parser.add_argument("--use_local", metavar="EMPTY or TRUE/FALSE", type=bool, default=False,  nargs="?", const=True,
                     help="if set, uses the local minima will be instead of global ones. Useful when dealing with multiple close minimums. Default: false")   
parser.add_argument("--noinv", metavar="EMPTY or TRUE/FALSE", type=bool, default=False,  nargs="?", const=True,
                     help="if set, avoid inversion of the covariance matrix and replace it with a gaussian elimination approach. DEPRECATED. Default: false")  # deprecated?
parser.add_argument("--save_residuals", metavar="EMPTY or TRUE/FALSE", type=bool, default=False,  nargs="?", const=True,
                     help="if set, saves fit residuals as npy files for further inspection. mainly a DEBUG option. Default: false")


# IF BEING RUN AS A SCRIPT, LOAD COMMAND LINE ARGUMENTS
if __name__ == "__main__":    

    args = parser.parse_args()
    dargs = vars(args) # to treat as a dictionnary

    if not(os.path.isdir(dargs["datadir"])):
        printerr("Data directory {} not found".format(dargs["datadir"]))

    if os.path.isfile(dargs["output"]):
        printinf("File {} already exists.".format(dargs["output"]))
        r = printinp("Overwrite it? (y/n)")
        if not(r.lower() in ["y", "yes"]):
            printerr("User abort")

# if this file is being used as a module, load the dargs dict from parent package
if __name__ != "__main__":
    import exogravity
    dargs = exogravity.dargs

#######################
# START OF THE SCRIPT #
#######################       
 
# load the datafiles
datafiles = glob.glob(dargs["datadir"]+'/GRAVI*'+dargs["reduction"]+'*.fits')
# remove duplicates with _s extension
to_keep = []
for k in range(len(datafiles)):
    filename = datafiles[k]
    filename = filename.replace(".fits", "_s.fits")
    if not(filename in datafiles):
        to_keep.append(k)
datafiles = [datafiles[k] for k in to_keep]
datafiles.sort()
printinf("{:d} files found".format(len(datafiles)))

starOis = []
objOis = []
swapOis = []

for k in range(len(datafiles)):
    filename = datafiles[k]
    printinf("Loading "+filename)
    oi = gravity.GravityOiFits(filename)
    oi.visOi = gravity.VisOiFits(filename, reduction = "astrored", mode = "dualfield", extension = int(dargs["extension"]))
    oi.ndit = oi.visOi.ndit
    oi.nwav = oi.visOi.nwav        
    oi.visOi.scaleVisibilities(1.0/oi.dit)
    d = (oi.sObjX**2+oi.sObjY**2 )**0.5
    msg = "Target is {}; Fiber position is [{:.2f}, {:.2f}] mas. Distance is {:.2f} mas. ".format(oi.target, oi.sObjX, oi.sObjY, d)    
    if d < 10:
        printinf(msg+"Assuming file to be on star.")
        starOis.append(oi)
    elif oi.target == dargs['swap_target']:
        printinf(msg+"Target is {}. This is a SWAP!".format(oi.target))
        swapOis.append(oi)
    else:
        if not(dargs['target'] is None):
            if dargs['target'] != oi.target:
                printinf("Target does not match the provided target name. Skipping this file.")
                continue                        
        printinf(msg+"Assuming file to be on planet.")
        if not(dargs["fiber_pos"] is None):
            if ((oi.sObjX - dargs["fiber_pos"][0])**2 + (oi.sObjY - dargs["fiber_pos"][1])**2)**0.5>1:
                printwar("File at is too far for the requested fiber position. This file will be ignored")
            else:
                objOis.append(oi)
        else:
            objOis.append(oi)

# select the field of the default chi2Maps depending if observations are from UTs or ATs
if len(objOis) != 0:
    if (objOis[0].stationNames[0] == "U1"):
        FIELD = 60 # UT field
    else:
        FIELD = 240 # UT field
else:
    if len(swapOis) != 0:
        if (swapOis[0].stationNames[0] == "U1"):
            FIELD = 60 # UT field
        else:
            FIELD = 240 # UT field    

if ((dargs["ralim"] is None) or (dargs["ralim"] is None)):
    ra = np.mean(np.array([oi.sObjX for oi in objOis]))
    ralim = [float(ra)-FIELD/2, float(ra)+FIELD/2] # get rid of numpy type so that yaml conversion works
    dec = np.mean(np.array([oi.sObjY for oi in objOis]))
    declim = [float(dec)-FIELD/2, float(dec)+FIELD/2]
else:
    ralim = [float(r) for r in dargs["ralim"].split(']')[0].split('[')[1].split(',')]
    declim = [float(r) for r in dargs["declim"].split(']')[0].split('[')[1].split(',')]
printinf("RA grid set to [{:.2f}, {:.2f}] with {:d} points".format(ralim[0], ralim[1], dargs["nra"]))
printinf("DEC grid set to [{:.2f}, {:.2f}] with {:d} points".format(declim[0], declim[1], dargs["ndec"]))

if len(swapOis) > 0:
    printinf("At least one SWAP file detected. Setting phaseref_mode to SWAP.")
    phaseref_mode = "DF_SWAP"
else:
    printinf("No SWAP file detected. Setting phaseref_mode to STAR.")
    phaseref_mode = "DF_STAR"

if phaseref_mode == "DF_SWAP":
    oi = swapOis[0]
    oi = gravity.GravityDualfieldAstrored(oi.filename, extension = dargs["extension"], corrMet = "None", corrDisp = "None")
    if oi.swap:
        sObjX, sObjY = -oi.sObjX, -oi.sObjY
    else:
        sObjX, sObjY = oi.sObjX, oi.sObjY        
    if ((dargs["ralim_swap"] is None) or (dargs["ralim_swap"] is None)):
        ralim_swap = [sObjX-FIELD/2, sObjX+FIELD/2] # get rid of numpy type so that yaml conversion works
        declim_swap = [sObjY-FIELD/2, sObjY+FIELD/2]
    else:
        ralim_swap = [float(r) for r in dargs["ralim_swap"].split(']')[0].split('[')[1].split(',')]
        declim_swap = [float(r) for r in dargs["declim_swap"].split(']')[0].split('[')[1].split(',')]
    printinf("RA grid set to [{:.2f}, {:.2f}] with {:d} points".format(ralim_swap[0], ralim_swap[1], dargs["nra_swap"]))
    printinf("DEC grid set to [{:.2f}, {:.2f}] with {:d} points".format(declim_swap[0], declim_swap[1], dargs["ndec_swap"]))
else:
    ralim_swap = None
    declim_swap = None    
    
preduces = []
for k in range(len(objOis)):
    preduces.append({"p"+str(k): {"planet_oi": "p"+str(k), "reject_baselines": None, "reject_dits": None}})

wreduces = []
for k in range(len(swapOis)):
    wreduces.append({"w"+str(k): {"swap_oi": "w"+str(k), "reject_baselines": None, "reject_dits": None}})    
    
general = {"datadir": dargs["datadir"],
           "phaseref_mode": phaseref_mode,
           "calib_strategy": dargs["calib_strategy"],           
           "corr_met": dargs['corr_met'],
           "corr_disp": dargs['corr_disp'],
           "extension": int(dargs["extension"]),
           "reduction": dargs["reduction"],                      
           "go_fast": dargs['go_fast'],
           "gradient": dargs['gradient'],
           "zoom": dargs['zoom'],           
           "use_local": dargs['use_local'],                      
           "noinv": dargs['noinv'],
           "reflag": dargs['reflag'],           
           "contrast_file": dargs['contrast_file'],
           "figdir": dargs["figdir"],
           "save_residuals": dargs["save_residuals"],           
           "n_opd": int(dargs["nopd"]),
           "n_ra": int(dargs["nra"]),
           "n_dec": int(dargs["ndec"]),
           "n_ra_swap": int(dargs["nra_swap"]),
           "n_dec_swap": int(dargs["ndec_swap"]),           
           "ralim": ralim,
           "declim": declim,
           "ralim_swap": ralim_swap,
           "declim_swap": declim_swap,           
           "poly_order": int(dargs["poly_order"]),
           "star_diameter": float(dargs["star_diameter"]),
           "phaseref_arclength_threshold": float(dargs["phaseref_arclength_threshold"]),
           "ft_flux_threshold": float(dargs["ft_flux_threshold"]),
           "reduce_planets": preduces,
           "reduce_swaps": wreduces           
           }

planet_files = {}
for k in range(len(objOis)):
    oi = objOis[k]
    d = {"filename": oi.filename.split(dargs["datadir"])[1],
         "dit": oi.dit,
         "ndit": oi.ndit,
         "nwav": oi.nwav,         
         "mjd": oi.mjd,
         "sObjX": oi.sObjX,
         "sObjY": oi.sObjY,
         "ftMeanFlux": float(np.abs(oi.visOi.visDataFt).mean())
     }
    if dargs["calib_strategy"].lower() == "nearest":
        oiAfter = oi.getClosestOi(starOis, forceAfter = True)
        oiBefore = oi.getClosestOi(starOis, forceBefore = True)
        d["star_indices"] = ["s"+str(j) for j in [starOis.index(oiBefore), starOis.index(oiAfter)]]
    elif dargs["calib_strategy"].lower() == "all":
        d["star_indices"] = ["s"+str(j) for j in range(len(starOis))]
    elif dargs["calib_strategy"].lower() == "none":
        d["star_indices"] = []
    else:
        printerr("Unknown calibration strategy: {}".format(dargs["calib_strategy"]))
    planet_files["p"+str(k)] = d

star_files = {}
for k in range(len(starOis)):
    oi = starOis[k]
    d = {"filename": oi.filename.split(dargs["datadir"])[1],
         "dit": oi.dit,
         "ndit": oi.ndit,
         "nwav": oi.nwav,         
         "mjd": oi.mjd,
         "sObjX": oi.sObjX,
         "sObjY": oi.sObjY,
         "ftMeanFlux": float(np.abs(oi.visOi.visDataFt).mean())
     }
    star_files["s"+str(k)] = d

swap_files = {}
for k in range(len(swapOis)):
    oi = swapOis[k]
    if oi.sObjX < 0:
        pos = -1
    elif oi.sObjX > 0:
        pos = 1
    elif oi.sObjY < 0:
        pos = -1
    else:
        pos = 1
    if (oi.sObjX*sObjX > 0):
        swap = False
    else:
        swap = True
    d = {"filename": oi.filename.split(dargs["datadir"])[1],
         "dit": oi.dit,
         "ndit": oi.ndit,
         "nwav": oi.nwav,
         "mjd": oi.mjd,
         "sObjX": oi.sObjX,
         "sObjY": oi.sObjY,
         "swap": swap,         
         "ftMeanFlux": float(np.abs(oi.visOi.visDataFt).mean())/oi.dit         
     }         
    swap_files["w"+str(k)] = d

general["ftOnStarMeanFlux"] = float(np.mean(np.array([star_files[key]["ftMeanFlux"] for key in star_files.keys()])))
general["ftOnPlanetMeanFlux"] = float(np.mean(np.array([planet_files[key]["ftMeanFlux"] for key in planet_files.keys()])))

# dictionnary for keywords used when calibrating the spectrum
calib_dict = {"star_name": None,
              "magnitudes": {"2MASS/2MASS.Ks": [0, 0]},
              "parallax": [0, 0],
              "model": "bt-nextgen",
              "bounds": {"teff": [0, 0],
                         "logg": [0, 0],
                         "feh": [0., 0.],
                         "radius": [1., 100]}
              }

cfg = {"general": general,
       "planet_ois": planet_files,
       "star_ois": star_files,
       "swap_ois": swap_files,       
       "spectral_calibration": calib_dict 
   }


# is used as a self-contained script, write the yml
if __name__ == "__main__":
    f = open(dargs["output"], "w")
    if RUAMEL:
        f.write(ruamel.yaml.dump(cfg, default_flow_style = False))
    else:
        f.write(yaml.safe_dump(cfg, default_flow_style = False)) 
    f.close()
    printinf("Saved config for {:d} files to {}".format(len(datafiles), dargs["output"]))

# otherwise, just store it in parent exogravity package
if __name__ != "__main__":
    exogravity.cfg = cfg
    
