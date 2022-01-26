#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create the YAML configuration script

This script is part of the exoGravity data reduction package.
The create_config script is used to create a basic YAML configuration file for an observation, which contains data
used by the other scripts, formatted in YAML to be easily loaded as a Python dict.

Args:
  datadir (str): the directory in which the fits files are located
  output (str): name (with full path) of the yml file in which to store the resulting configuration
  figdir (str, optional): name of a directory where to store the output PDF files (not created if no figdir given)
  ralim (range, optional): specify the RA range (in mas) over which to search for the astrometry: Example: ralim=[142,146]
  declim (range, optional): same as ralim, for declination
  nra, ndec (int, optional): number of points over the RA and DEC range (the more points, the longer the calculation)
  nopd (int, optional): number of points to use when creating the OPD chi2 maps in the astrometry reduction. 100 is the default and is usually fine.
  gofast (bool, optional): if set, average over DITs to accelerate calculations (Usage: --gofast, or gofast=True)
  reflag (str, optional): can be set to True in order to use the REFLAG table to filter bad datapoints
  noinv (bool, optional): if set, avoid inversion of the covariance matrix and replace it with a gaussian elimination approach. 
                          This can sometimes speed up calculations, but it depends. As a general rule, the more DITs you have, 
                          the less likely it is that noinv will be beneficial. It also depends on resolution, with higher resolution
                          making noinv more likely to be useful.
  contrast_file (str, optional): path to the contrast file (planet/star) to use as a model for the planet visibility amplitude.
                                 Default is None, which is a constant contrast (as a function of wavelength).
  star_order (int, optional): the order of the polynomial used to fit the stellar component of the visibility (default is 4)
  star_diameter (float, optional): the diameter of the central star (in mas, to scale the visibility amplitude). Default is 0. 
  corr_met (str, optional): can be "sylvestre" or "drs", or possibly "none" depending on which formula to use for the metrology correction. Default is sylvestre
  corr_disp (str, optional): can be "sylvestre" or "drs", or possibly "none" depending on which formula to use for the dispersion correction. Default is sylvestre
  extension (int, optional): the FITS extension to use, which depends on the polarization in the data. Deafult to 10 (combined).
  swap_target (str, optional): the name of the target for the swap observations if off-axis mode
  calib_strategy (str, optional): "all" to use all star files to calibrate visibility reference. "Nearest" to use the two nearest. On-axis only. "None" to skip calibration
  target (str, optional): if you want to restrict to a particular target name, you can specify one
  reduction (str, optional, indev): default is "astrored". You can select "dualscivis".
  ft_flux_threshold (float, optional): if positive, all data with ft flux below this number times the mean ft flux for on star observation are removed (default 0.2)
  phaseref_arclength_threshold (float, optional): if positive, the arclength of the polyfit to phaseref is calculated for each dit and baseline, and the data with an arclength 
                                                  greater that this threshold are removed. Default: 5
 
Example:
  Minimal
    python create_config datadir=/path/to/your/data/directory output=/path/to/result.yml
  To set the RA/DEC range to explore to find the astrometry, and the resolution:
    python create_config datadir=/data/directory output=result.yml ralim=[142,146] declim=[73,78] nra=200 ndec=200
  Speed up calculation to test an astrometry range
    python create_config datadir=/data/directory output=result.yml ralim=[140,150] declim=[70,80] nra=50 ndec=50 --gofast

Configuration:
  The configuration file must contain all necessary information as to where the data are located on your disk, which mode
  of observation was used, as well as some parameters for the data reduction. It is strongly advised to use the
  create_config.py script (also provided with the exoGravity data reduction package) to create your configuration file, 
  which you can then tweek to fit your need. Please refer to the documentation of create_config.py.

Authors:
  M. Nowak, and the exoGravity team.

Version:
  xx.xx
"""

import astropy.io.fits as fits
from datetime import datetime
import numpy as np
import cleanGravity as gravity
import glob
import sys
import os
try:
    import ruamel.yaml
    RUAMEL = True
except: # if ruamel not available, switch back to pyyaml, which does not handle comments properly
    import yaml
    RUAMEL = False
from utils import * 

REQUIRED_ARGS = ["datadir", "output"]

dargs = args_to_dict(sys.argv)

if "help" in dargs.keys():
    print(__doc__)
    stop()

for req in REQUIRED_ARGS:
    if not(req in dargs.keys()):
        printerr("Argument '"+req+"' is not optional for this script. Required args are: "+', '.join(REQUIRED_ARGS))
        stop()

if not(os.path.isdir(dargs["datadir"])):
    printerr("Data directory {} not found".format(dargs["datadir"]))
    stop()

if os.path.isfile(dargs["output"]):
    printinf("File {} already exists.".format(dargs["output"]))
    r = printinp("Overwrite it? (y/n)")
    if not(r.lower() in ["y", "yes"]):
        printerr("User abort")
        stop()

if not("ralim" in dargs.keys()) or not("declim" in dargs.keys()):
    printwar("ralim or declim not provided in args. Default: fiber position +/- 5 mas.")
    dargs["ralim"] = None
    dargs["declim"] = None

if not("nra" in dargs.keys()):
    printwar("nra not provided in args. Default: nra=100")
    dargs["nra"] = 100
if not("ndec" in dargs.keys()):
    printwar("ndec not provided in args. Default: ndec=100")
    dargs["ndec"] = 100
if not("nopd" in dargs.keys()):
    printwar("nopd not provided in args. Default: to nopd=100")
    dargs["nopd"] = 100
if not("star_order" in dargs.keys()):
    printwar("star_order not provided in args. Default: star_order=4")
    dargs["star_order"] = 4
if not("gofast" in dargs.keys()):
    printwar("Value for gofast option not set. Defaut: gofast=False")
    dargs['gofast'] = False    
if not("noinv" in dargs.keys()):
    printwar("Value for noinv option not set. Default: noinv=False")
    dargs['noinv'] = False
if not("reflag" in dargs.keys()):
    printwar("Value for reflag not given. Default: reflag = True")
    dargs['reflag'] = True
if not("contrast_file" in dargs.keys()):
    printwar("Contrast file not given. Constant contrast will be used")
    dargs['contrast_file'] = None
if not("extension" in dargs.keys()):
    printwar("extension not given. Using basic value '10'.")
    dargs['extension'] = 10    

if not("star_diameter" in dargs.keys()):
    printwar("star_diameter not provided in args. Default: star_diameter=0 (point source)")
    dargs["star_diameter"] = 0

if not("corr_met" in dargs.keys()):
    printwar("corr_met not specified. Using 'sylvestre'")
    dargs["corr_met"] = "sylvestre"
if not("corr_disp" in dargs.keys()):
    printwar("corr_disp not specified. Using 'sylvestre'")
    dargs["corr_disp"] = "sylvestre"

if not("swap_target" in dargs.keys()):
    printwar("SWAP target name not given. Assuming the observation to be on-axis (no swap)")
    dargs["swap_target"] = "%%"

if not("target" in dargs.keys()):
    dargs["target"] = None

if not("calib_strategy" in dargs.keys()):
    printwar("calib strategy not given. Using default 'nearest'")
    dargs["calib_strategy"] = "nearest"

if not("reduction" in dargs.keys()):
    printwar("reduction not given. Using default 'astrored'")
    dargs["reduction"] = "astrored"

if not("phaseref_arclength_threshold" in dargs.keys()):
    printwar("phaseref_arclength_threshold not given. Using default value of 5")
    dargs["phaseref_arclength_threshold"] = 5
if not("ft_flux_threshold" in dargs.keys()):
    printwar("ft_flux_threshold not given. Using default value of 0.2")
    dargs["ft_flux_threshold"] = 0.2
if not("ignore_baselines" in dargs.keys()):
    printwar("By default, no baseline will be ignored. Add baseline indices to 'ignore_baselines' in the yml to ignore some baselines.")
    dargs["ignore_baselines"] = []

if not("figdir" in dargs.keys()):
    printwar("No figdir given. The output PDF will not be created, and only the terminal output will be available.")
    dargs["figdir"] = None


    
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
    oi.visOi.scaleVisibilities(1.0/oi.dit)
    d = (oi.sObjX**2+oi.sObjY**2 )**0.5
    msg = "Target is {}; Fiber distance is {:.2f} mas. ".format(oi.target, d)
    if not(dargs['target'] is None):
        if dargs['target'] != oi.target:
            printinf("Target does not match the provided target name. Skipping this file.")
            continue
    if d < 10:
        printinf(msg+"Assuming file to be on star.")
        starOis.append(oi)
    elif oi.target == dargs['swap_target']:
        printinf(msg+"Target is {}. This is a SWAP!".format(oi.target))
        swapOis.append(oi)
    else:
        printinf(msg+"Assuming file to be on planet.")
        objOis.append(oi)

# select the field of the default chi2Maps depending if observations are from UTs or ATs
if (objOis[0].stationNames[0] == "U1"):
    FIELD = 30 # UT field
else:
    FIELD = 120 # UT field

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

general = {"datadir": dargs["datadir"],
           "phaseref_mode": phaseref_mode,
           "calib_strategy": dargs["calib_strategy"],           
           "corr_met": dargs['corr_met'],
           "corr_disp": dargs['corr_disp'],
           "extension": int(dargs["extension"]),
           "reduction": dargs["reduction"],                      
           "gofast": dargs['gofast'],
           "noinv": dargs['noinv'],
           "reflag": dargs['reflag'],           
           "contrast_file": dargs['contrast_file'],
           "figdir": dargs["figdir"],
           "save_residuals": False,           
           "n_opd": int(dargs["nopd"]),
           "n_ra": int(dargs["nra"]),
           "n_dec": int(dargs["ndec"]),
           "ralim": ralim,
           "declim": declim,
           "star_order": int(dargs["star_order"]),
           "star_diameter": float(dargs["star_diameter"]),
           "phaseref_arclength_threshold": float(dargs["phaseref_arclength_threshold"]),
           "ft_flux_threshold": float(dargs["ft_flux_threshold"]),                                 
           "reduce": ["p"+str(j) for j in range(len(objOis))],
           "ignore_baselines": dargs["ignore_baselines"],
           }

planet_files = {}
for k in range(len(objOis)):
    oi = objOis[k]
    d = {"filename": oi.filename.split(dargs["datadir"])[1],
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
    d = {"filename": oi.filename.split(dargs["datadir"])[1],
         "mjd": oi.mjd,
         "sObjX": oi.sObjX,
         "sObjY": oi.sObjY,
         "ftMeanFlux": float(np.abs(oi.visOi.visDataFt).mean())/oi.dit         
     }         
    swap_files["w"+str(k)] = d

general["ftOnStarMeanFlux"] = float(np.mean(np.array([star_files[key]["ftMeanFlux"] for key in star_files.keys()])))
general["ftOnPlanetMeanFlux"] = float(np.mean(np.array([planet_files[key]["ftMeanFlux"] for key in planet_files.keys()])))
    
cfg = {"general": general,
       "planet_ois": planet_files,
       "star_ois": star_files,
       "swap_ois": swap_files 
   }

f = open(dargs["output"], "w")
if RUAMEL:
    f.write(ruamel.yaml.dump(cfg, default_flow_style = False))
else:
    f.write(yaml.safe_dump(cfg, default_flow_style = False)) 
f.close()

printinf("Saved config for {:d} files to {}".format(len(datafiles), dargs["output"]))
