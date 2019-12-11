#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A script to create the index.csv file used to identify star and planet exposures
@author: mnowak
"""

import astropy.io.fits as fits
from datetime import datetime
import cleanGravity as gravity
import glob
import sys
import os
from config import dictToYaml#

#PDS70
#nRa = 400
#nDec = 400
#ra_lim = [-215-1.5, -215+1.5]
#dec_lim = [32.3-1.5, 32.3+1.5]

# HD206893
nRa = 100
nDec = 100
#ra_lim = [126.0-3, 126.0+3]
#dec_lim = [200-3, 200+3]
ra_lim = [130.7-5, 130.7+5]
dec_lim = [198.08-5, 198.08+5]

# Beta Pic 2019
#nRa = 100
#nDec = 100
#ra_lim = [146.5-2, 146.5+2]
#dec_lim = [248.57-2, 248.57+2]

# Beta Pic 2018
#nRa = 100
#nDec = 100
#ra_lim = [68.48-1.5, 68.48+1.5]
#dec_lim = [126.2-1.5, 126.2+1.5]

args = sys.argv
if len(args) != 3:
    raise Exception("This script takes exactly 2 arguments: path_to_datadir and path_to_output_config.yml")

datadir = args[1]
output = args[2]

if not(os.path.isdir(datadir)):
    raise Exception("Data directory {} not found".format(datadir))

if os.path.isfile(output):
    inp = input("File {} already exists. Overwrite it? (y/n)".format(output))
    if inp.upper() != "Y":
        raise Exception("User abort")

datafiles = glob.glob(datadir+'/GRAVI*astrored*.fits')
datafiles.sort()

starOis = []
objOis = []

for k in range(len(datafiles)):
    filename = datafiles[k]
    print("Loading "+filename)
    oi = gravity.GravityOiFits(filename)
    if oi.sObjX**2+oi.sObjY**2 < 10**2:
        starOis.append(oi)
    else:
        objOis.append(oi)

general = {"datadir": datadir,
           "go_fast": True,
           "no_inv": True,
           "contrast_file": None,
           "save_fig": True,
           "n_opd": 100,
           "n_ra": 100,
           "n_dec": 100,
           "ra_lim": ra_lim,
           "dec_lim": dec_lim,
           "star_order": 3,
           "reduce": list(range(len(objOis)))}

planet_files = {}
for k in range(len(objOis)):
    oi = objOis[k]
    oiAfter = oi.getClosestOi(starOis, forceAfter = True)
    oiBefore = oi.getClosestOi(starOis, forceBefore = True)
    d = {"filename": oi.filename.split(datadir)[1],
         "mjd": oi.mjd,
         "sObjX": oi.sObjX,
         "sObjY": oi.sObjY,
         "star_indices": [starOis.index(oiBefore), starOis.index(oiAfter)]
     }
    planet_files[str(k)] = d

star_files = {}
for k in range(len(starOis)):
    oi = starOis[k]
    d = {"filename": oi.filename.split(datadir)[1],
         "mjd": oi.mjd,
         "sObjX": oi.sObjX,
         "sObjY": oi.sObjY
     }
    star_files[str(k)] = d


cfg = {"general": general,
       "planet_ois": planet_files,
       "star_ois": star_files
   }

f = open(output, "w")
f.write(dictToYaml(cfg))
f.close()

print(dictToYaml(cfg))
print("Saved config for {:d} files to {}".format(len(datafiles), output))

