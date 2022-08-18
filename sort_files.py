#!/usr/bin/env python
# -*- coding: utf-8 -*-

import astropy.io.fits as fits
import glob
import sys
import os
import shutil
from utils import *

REQUIRED_ARGS = ["datadir"]

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

# load the datafiles
datafiles = glob.glob(dargs["datadir"]+'/GRAVI*'+'*astroreduced.fits')
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

targets_dict = {}

# find all targets and separate filenames in target filenames
for filename in datafiles:
    hdul = fits.open(filename)
    header = hdul[0].header
    target = header["OBJECT"]
    if not(target in list(targets_dict.keys())):
        targets_dict[target] = []
    targets_dict[target].append(filename)

# list files for the user
for target in list(targets_dict.keys()):
    print("{}:".format(target))
    filenames = targets_dict[target]
    for filename in filenames:
        hdul = fits.open(filename)
        header = hdul[0].header        
        sObjX, sObjY = header["HIERARCH ESO INS SOBJ OFFX"], header["HIERARCH ESO INS SOBJ OFFY"]
        distance = (sObjX**2+sObjY**2)**0.5
        if (distance < 10):
            print("  ON STAR: {} mas, {} mas".format(sObjX, sObjY))
        else:
            print("  ON PLANET: {} mas, {} mas".format(sObjX, sObjY))            

# move each target to its own subdir
for target in list(targets_dict.keys()):
    if not(os.path.isdir(dargs["datadir"]+"/"+target)):
        os.mkdir(dargs["datadir"]+"/"+target)
    filenames = targets_dict[target]
    for filename in filenames:
        if not(os.path.isfile(dargs["datadir"]+"/"+target+"/"+filename.split("/")[-1])):
            shutil.copy(filename, dargs["datadir"]+"/"+target+"/"+filename.split("/")[-1])
