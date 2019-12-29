#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A script to create the index.csv file used to identify star and planet exposures
@author: mnowak
"""

import astropy.io.fits as fits
from datetime import datetime
import numpy as np
import cleanGravity as gravity
import glob
import sys
import os
from config import dictToYaml
from utils import * 

REQUIRED_ARGS = ["datadir", "output"]

dargs = args_to_dict(sys.argv)

for req in REQUIRED_ARGS:
    if not(req in dargs.keys()):
        printerr("Argument '"+req+"' is not optional for this script. Required args are: "+', '.join(REQUIRED_ARGS))
        stop()

if not(os.path.isdir(dargs["datadir"])):
    printerr("Data directory {} not found".format(dargs["datadir"]))
    stop()

if os.path.isfile(dargs["output"]):
    printinf("File {} already exists. Overwrite it? (y/n)".format(dargs["output"]))
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


# load the datafiles
datafiles = glob.glob(dargs["datadir"]+'/GRAVI*astrored*.fits')
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

if dargs["ralim"] is None:
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
           "go_fast": True,
           "no_inv": True,
           "contrast_file": None,
           "save_fig": True,
           "n_opd": dargs["nopd"],
           "n_ra": dargs["nra"],
           "n_dec": dargs["ndec"],
           "ralim": ralim,
           "declim": declim,
           "star_order": dargs["star_order"],
           "reduce": list(range(len(objOis)))}

planet_files = {}
for k in range(len(objOis)):
    oi = objOis[k]
    oiAfter = oi.getClosestOi(starOis, forceAfter = True)
    oiBefore = oi.getClosestOi(starOis, forceBefore = True)
    d = {"filename": oi.filename.split(dargs["datadir"])[1],
         "mjd": oi.mjd,
         "sObjX": oi.sObjX,
         "sObjY": oi.sObjY,
         "star_indices": [starOis.index(oiBefore), starOis.index(oiAfter)]
     }
    planet_files[str(k)] = d

star_files = {}
for k in range(len(starOis)):
    oi = starOis[k]
    d = {"filename": oi.filename.split(dargs["datadir"])[1],
         "mjd": oi.mjd,
         "sObjX": oi.sObjX,
         "sObjY": oi.sObjY
     }
    star_files[str(k)] = d

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
         "position": pos,
         "swap_ra": 824.76*pos,
         "swap_dec": 748.84*pos
     }
    swap_files[str(k)] = d


cfg = {"general": general,
       "planet_ois": planet_files,
       "star_ois": star_files,
       "swap_ois": swap_files
   }

f = open(dargs["output"], "w")
f.write(dictToYaml(cfg))
f.close()

printinf("Saved config for {:d} files to {}".format(len(datafiles), dargs["output"]))

