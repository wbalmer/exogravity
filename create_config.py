#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create the YAML configuration script

This script is part of the exoGravity data reduction package.
The create_config script is used to create a basic YAML configuration file for an observation, which contains data
used by the other scripts, formatted in YAML to be easily loaded as a Python dict.

Args:
  datadir (str): the directory in which the fits files are located
  output (str): name (with full path) of the yml file in which to store the resulting configuration
  ralim (range, optional): specify the RA range (in mas) over which to search for the astrometry: Example: ralim=[142,146]
  declim (range, optional): same as ralim, for declination
  nra, ndec (int, optional): number of points over the RA and DEC range (the more points, the longer the calculation)
  nopd (int, optional): number of points to use when creating the OPD chi2 maps in the astrometry reduction. 100 is the default and is usually fine.
  gofast (bool, optional): if set, average over DITs to accelerate calculations (Usage: --gofast, or gofast=True)
  noinv (bool, optional): if set, avoid inversion of the covariance matrix and replace it with a gaussian elimination approach. 
                          This can sometimes speed up calculations, but it depends. As a general rule, the more DITs you have, 
                          the less likely it is that noinv will be beneficial. It also depends on resolution, with higher resolution
                          making noinv more likely to be useful.
  contrast_file (str, optional): path to the contrast file (planet/star) to use as a model for the planet visibility amplitude.
                                 Default is None, which is a constant contrast (as a function of wavelength).
  star_order (int, optional): the order of the polynomial used to fit the stellar component of the visibility (default is 3)
  star_diameter (float, optional): the diameter of the central star (in mas, to scale the visibility amplitude). Default is 0. 
  corr_met (str, optional): can be "sylvestre" or "drs", or possibly "none" depending on which formula to use for the metrology correction
  corr_disp (str, optional): can be "sylvestre" or "drs", or possibly "none" depending on which formula to use for the dispersion correction


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

Todo:
  *identify which values/plots should be extracted along the reduction and saved from checking data quality
  *take into account proper complex errors
  *Exception management?

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
    printwar("ralim or declim not provided in args. Defaulting to fiber position +/- 5 mas.")
    dargs["ralim"] = None
    dargs["declim"] = None

if not("nra" in dargs.keys()):
    printwar("nra not provided in args. Defaulting to nra=100")
    dargs["nra"] = 100
if not("ndec" in dargs.keys()):
    printwar("ndec not provided in args. Defaulting to ndec=100")
    dargs["ndec"] = 100
if not("nopd" in dargs.keys()):
    printwar("nopd not provided in args. Defaulting to nopd=100")
    dargs["nopd"] = 100
if not("star_order" in dargs.keys()):
    printwar("star_order not provided in args. Defaulting to star_order=3")
    dargs["star_order"] = 3    
if not("gofast" in dargs.keys()):
    printwar("Value for gofast option not set. Defaulting to gofast=False")
    dargs['gofast'] = False    
if not("noinv" in dargs.keys()):
    printwar("Value for noinv option not set. Defaulting to noinv=False")
    dargs['noinv'] = False
if not("contrast_file" in dargs.keys()):
    printwar("Contrast file not given. Constant contrast will be used")
    dargs['contrast_file'] = None

if not("star_diameter" in dargs.keys()):
    printwar("star_diameter not provided in args. Defaulting to star_diameter=0 (point source)")
    dargs["star_diameter"] = 0

if not("corr_met" in dargs.keys()):
    printwar("corr_met not specified. Using 'sylvestre'")
    dargs["corr_met"] = "sylvestre"
if not("corr_disp" in dargs.keys()):
    printwar("corr_disp not specified. Using 'sylvestre'")
    dargs["corr_disp"] = "sylvestre"        

    
# load the datafiles
datafiles = glob.glob(dargs["datadir"]+'/GRAVI*astrored*.fits')
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
    d = (oi.sObjX**2+oi.sObjY**2 )**0.5
    msg = "Fiber distance is {:.2f} mas. ".format(d)
    if d < 10:
        printinf(msg+"Assuming file to be on star.")
        starOis.append(oi)
    elif d < 500:
        printinf(msg+"Assuming file to be on planet.")
        objOis.append(oi)
    else:
        printinf(msg+"Assuming file to be part of a swap.")
        swapOis.append(oi)

if ((dargs["ralim"] is None) or (dargs["ralim"] is None)):
    ra = np.mean(np.array([oi.sObjX for oi in objOis]))
    ralim = [float(ra)-5, float(ra)+5] # get rid of numpy type so that yaml conversion works
    dec = np.mean(np.array([oi.sObjY for oi in objOis]))
    declim = [float(dec)-5, float(dec)+5]
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
           "corr_met": dargs['corr_met'],
           "corr_disp": dargs['corr_disp'],                      
           "gofast": dargs['gofast'],
           "noinv": dargs['noinv'],
           "contrast_file": dargs['contrast_file'],
           "fig_dir": None
           "n_opd": int(dargs["nopd"]),
           "n_ra": int(dargs["nra"]),
           "n_dec": int(dargs["ndec"]),
           "ralim": ralim,
           "declim": declim,
           "star_order": int(dargs["star_order"]),
           "star_diameter": float(dargs["star_diameter"]),           
           "reduce": ["p"+str(j) for j in range(len(objOis))]}

planet_files = {}
for k in range(len(objOis)):
    oi = objOis[k]
    oiAfter = oi.getClosestOi(starOis, forceAfter = True)
    oiBefore = oi.getClosestOi(starOis, forceBefore = True)
    d = {"filename": oi.filename.split(dargs["datadir"])[1],
         "mjd": oi.mjd,
         "sObjX": oi.sObjX,
         "sObjY": oi.sObjY,
         "star_indices": ["s"+str(j) for j in [starOis.index(oiBefore), starOis.index(oiAfter)]]
     }
    planet_files["p"+str(k)] = d

star_files = {}
for k in range(len(starOis)):
    oi = starOis[k]
    d = {"filename": oi.filename.split(dargs["datadir"])[1],
         "mjd": oi.mjd,
         "sObjX": oi.sObjX,
         "sObjY": oi.sObjY
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
     }         
    swap_files["w"+str(k)] = d


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
