#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Extract astrometry from a SWAP observation of a reference binary

This script is part of the exoGravity data reduction package.
The swap_reduce script is used to extract the astrometry of a reference binary, which is used to
calculate the metrology zero point (phase reference) when observing in dual-field/off-axis

Args:
  config_file (str): path to the YAML configuration file. Only used to retrieve the path to the SWAP files of the observation. 
  ralim (range): specify the RA range (in mas) over which to search for the astrometry: Example: ralim=[142,146]
  declim (range): same as ralim, for declination
  raguess, decguess (float): your best guess for the RA and DEC of the binary (in mas). Used to recenter the phases at the beginning of the reduction.
  nra, ndec (int, optional): number of points over the RA and DEC range (the more points, the longer the calculation). Default is 100.
  gofast (bool, optional): if set, average over DITs to accelerate calculations (Usage: --gofast, or gofast=True)

Example:
  python swap_reduce config_file=full/path/to/yourconfig.yml ralim=[800,850] declim=[432,482] --gofast --noinv

Authors:
  M. Nowak, and the exoGravity team.

Version:
  xx.xx
"""

import numpy as np
from cleanGravity import complexstats as cs
from cleanGravity import gravityPlot
import cleanGravity as gravity
import glob
import sys, os
from utils import *
import ruamel.yaml

# again, need to clean the imports
#import matplotlib.pyplot as plt
#import cleanGravity as gravity


# arg should be the path to the data directory
dargs = args_to_dict(sys.argv)

if "help" in dargs.keys():
    print(__doc__)
    stop()
    

REQUIRED_ARGS = ["config_file", "raguess", "decguess", "ralim", "declim"]
for req in REQUIRED_ARGS:
    if not(req in dargs.keys()):
        printerr("Argument '"+req+"' is not optional for this script. Required args are: "+', '.join(REQUIRED_ARGS))
        stop()

# READ THE CONFIGURATION FILE
CONFIG_FILE = dargs["config_file"]
if not(os.path.isfile(CONFIG_FILE)):
    raise Exception("Error: argument {} is not a file".format(CONFIG_FILE))
cfg = ruamel.yaml.load(open(CONFIG_FILE, "r"), Loader=ruamel.yaml.RoundTripLoader)

# GET FILES FOR THE SWAP 
if not("swap_ois" in cfg.keys()):
    printerr("No SWAP files given in {}!".format(CONFIG_FILE))
DATA_DIR = cfg['general']['datadir']    
SWAP_FILES = [DATA_DIR+cfg["swap_ois"][k]["filename"] for k in cfg["swap_ois"].keys()]

# load other parameters
if "gofast" in dargs.keys():
    GO_FAST = dargs["gofast"]
else: # default is False
    GO_FAST = False
if "unwrap" in dargs.keys(): # experimental
    UNWRAP = dargs["unwrap"]
else: # default is False
    UNWRAP = False    
if "nra" in dargs.keys():
    N_RA = int(dargs["nra"])
else:
    N_RA = 100
if "ndec" in dargs.keys():
    N_DEC = int(dargs["ndec"])
else:
    N_DEC = 100

RA_LIM = [float(r) for r in dargs["ralim"].split(']')[0].split('[')[1].split(',')]
DEC_LIM = [float(r) for r in dargs["declim"].split(']')[0].split('[')[1].split(',')]

RA_GUESS = float(dargs["raguess"])
DEC_GUESS = float(dargs["raguess"])

# retrieve all astrored files from the datadir
printinf("Found a total of {:d} astroreduced files in {}".format(len(SWAP_FILES), DATA_DIR))

objOis = []
starOis = []

ois1 = []
ois2 = []

# load OBs and put them in two groups based on the two swap pos
for k in range(len(SWAP_FILES)):
    filename = SWAP_FILES[k]
    printinf("Loading file "+filename)    
    oi = gravity.GravityDualfieldAstrored(filename, extension = 10, corrMet = "drs", corrDisp = "drs")
    # replace data by the mean over all DITs if go_fast has been requested in the config file
    if GO_FAST:
        oi.computeMean()        
        oi.visOi.recenterPhase(oi.sObjX, oi.sObjY)
        visData = np.copy(oi.visOi.visData, 'complex')
        visRef = np.copy(oi.visOi.visRef, 'complex')        
        oi.visOi.visData = np.zeros([1, oi.visOi.nchannel, oi.nwav], 'complex')
        oi.visOi.visRef = np.zeros([1, oi.visOi.nchannel, oi.nwav], 'complex')    
        oi.visOi.visData[0, :, :] = visData.mean(axis = 0)
        oi.visOi.visRef[0, :, :] = visRef.mean(axis = 0)
        uCoord = np.copy(oi.visOi.uCoord)
        oi.visOi.uCoord = np.zeros([1, oi.visOi.nchannel, oi.nwav])
        oi.visOi.uCoord[0, :, :] = np.mean(uCoord, axis = 0)
        vCoord = np.copy(oi.visOi.vCoord)
        oi.visOi.vCoord = np.zeros([1, oi.visOi.nchannel, oi.nwav])
        oi.visOi.vCoord[0, :, :] = np.mean(vCoord, axis = 0)
        u = np.copy(oi.visOi.u)
        oi.visOi.u = np.zeros([1, oi.visOi.nchannel])
        oi.visOi.u[0, :] = np.mean(u, axis = 0)
        v = np.copy(oi.visOi.v)
        oi.visOi.v = np.zeros([1, oi.visOi.nchannel])
        oi.visOi.v[0, :] = np.mean(v, axis = 0)        
        oi.visOi.ndit = 1
        oi.ndit = 1
        oi.visOi.recenterPhase(-oi.sObjX, -oi.sObjY)        
    oi.computeMean()
    if (GO_FAST):
        oi.visOi.visErr = np.zeros([1, oi.visOi.nchannel, oi.nwav], 'complex')    
        oi.visOi.visErr[0, :, :] = np.copy(oi.visOi.visErrMean)    
    if oi.swap:
        printinf("Adding OB {} to group 1 (swap=True)".format(filename.split('/')[-1]))
        ois1.append(oi)
    else:
        printinf("Adding OB {} to group 2 (swap=False)".format(filename.split('/')[-1]))
        ois2.append(oi)        
    objOis.append(oi)    
printinf("Found {:d} OBs for Group 1 and {:d} for Group 2".format(len(ois1), len(ois2)))

# check if no group is empty
if len(ois1) == 0:
    printerr("Group 1 does not contain any OB")
    stop()
if len(ois2) == 0:
    printerr("Group 2 does not contain any OB")
    stop()

# calculate the useful w for plotting
w = np.zeros([6, oi.nwav])
for k in range(6):
    w[k, :] = oi.wav*1e6

# we need to coadd the elements of each group
# this is best done in in the SC reference frame, so we need to shift the OBs to the SC frame
printinf("Recentering OBs on SC position")
for oi in ois1+ois2:
    if oi.swap:
        oi.visOi.recenterPhase(-RA_GUESS, -DEC_GUESS)
    else:
        oi.visOi.recenterPhase(RA_GUESS, DEC_GUESS)        

# coadd elements in each group to obtain a reference visibility at each position of the swap
visRefS1 = 0*oi.visOi.visRef.mean(axis = 0)
visRefS2 = 0*oi.visOi.visRef.mean(axis = 0)
printinf("Calculating reference visibility for group 1")
for k in range(len(ois1)):
    oi = ois1[k]
    visRefS1 = visRefS1+oi.visOi.visRef.mean(axis = 0)
visRefS1 = visRefS1/len(ois1)
printinf("Calculating reference visibility for group 2")
for k in range(len(ois2)):
    oi = ois2[k]
    visRefS2 = visRefS2+oi.visOi.visRef.mean(axis = 0)
visRefS2 = visRefS2/len(ois2)

# the phase ref is the average of the phase of the two groups
printinf("Calculating phase ref")
phaseRef = 0.5*(np.angle(visRefS1)+np.angle(visRefS2))

# unwrap
if UNWRAP:
    printinf("Unwrapping the visibilities")
    testCase = np.angle(ois1[0].visOi.visRef.mean(axis = 0))-phaseRef
    unwrapped = 0.5*np.unwrap(2*testCase)
    correction = unwrapped - testCase
    phaseRef = phaseRef - correction


# subtract the phase reference
printinf("Phase referencing OBs")
for oi in ois1+ois2:
    oi.visOi.addPhase(-phaseRef)

# recenter each OB on its initial position
printinf("Recentering OBs on initial positions")
for oi in ois1+ois2:
    if oi.swap:
        oi.visOi.recenterPhase(RA_GUESS, DEC_GUESS)
    else:
        oi.visOi.recenterPhase(-RA_GUESS, -DEC_GUESS)        

# chi2Map
printinf("RA grid: [{:.2f}, {:.2f}] with {:d} points".format(RA_LIM[0], RA_LIM[1], N_RA))
printinf("DEC grid: [{:.2f}, {:.2f}] with {:d} points".format(DEC_LIM[0], DEC_LIM[1], N_DEC))
raValues = np.linspace(RA_LIM[0], RA_LIM[1], N_RA)
decValues = np.linspace(DEC_LIM[0], DEC_LIM[1], N_DEC)

for k in range(len(ois2)):
    oi = ois2[k]
    printinf("Calculating chi2 Map for OB {}".format(oi.filename.split('/')[-1]))
    oi.visOi.computeChi2Map(raValues, decValues, nopds = 100, visInst = np.abs(oi.visOi.visRef.mean(axis = 0)))

for k in range(len(ois1)):
    oi = ois1[k]
    printinf("Calculating chi2 Map for OB {}".format(oi.filename.split('/')[-1]))            
    oi.visOi.computeChi2Map(-raValues[::-1], -decValues[::-1], nopds = 100, visInst = np.abs(oi.visOi.visRef.mean(axis = 0)))    

raBest = np.array([oi.visOi.fit['xBest'] for oi in ois2]+[-oi.visOi.fit['xBest'] for oi in ois1])
decBest = np.array([oi.visOi.fit['yBest'] for oi in ois2]+[-oi.visOi.fit['yBest'] for oi in ois1])


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
    

"""
oi = objOis[-3]
plt.figure()
plt.imshow(oi.visOi.fit["chi2Map"].T, origin = "lower", extent = [np.min(raValues), np.max(raValues), np.min(decValues), np.max(decValues)])

fig = plt.figure()
gravityPlot.reImPlot(w, oi.visOi.visRef[:, :, :].mean(axis = 0), fig = fig, subtitles = oi.basenames)
gravityPlot.reImPlot(w, oi.visOi.visRefFit[:, :, :].mean(axis = 0), fig = fig)

plt.figure()
plt.plot(raBest, decBest, "+")
"""

for key in cfg['swap_ois'].keys():
    cfg["swap_ois"][key]["astrometric_solution"] = [float(raBest.mean()), float(decBest.mean())] # YAML cannot convert numpy types
            
f = open(CONFIG_FILE, "w")
f.write(ruamel.yaml.dump(cfg, Dumper=ruamel.yaml.RoundTripDumper))
f.close()

stop()

        
fig = plt.figure()
gravityPlot.reImPlot(w, ois1[0].visOi.visRef[:, :, :].mean(axis = 0), fig = fig, subtitles = oi.basenames)
gravityPlot.reImPlot(w, ois2[0].visOi.visRef[:, :, :].mean(axis = 0), fig = fig)


"""
for c in range(oi.visOi.nchannel):
    for k in range(1, oi.nwav):
        if phaseRef[c, k]-phaseRef[c, k-1] > np.pi/2:
            phaseRef[c, k:]=phaseRef[c, k:] - np.pi
        elif phaseRef[c, k]-phaseRef[c, k-1] < -np.pi/2:
            phaseRef[c, k:]=phaseRef[c, k:] + np.pi
"""
    
#ois1.visOi.recenterPhase(-ois1.sObjX, -ois1.sObjY)
#ois2.visOi.recenterPhase(-ois2.sObjX, -ois2.sObjY)

stop

"""
fig = plt.figure()
gravityPlot.reImPlot(w, ois1.visOi.visRef[0, :, :]*np.exp(1j*np.angle(ois2.visOi.visRef[0, :, :])), fig = fig, subtitles = oi.basenames)
"""


fig = plt.figure()
gravityPlot.reImPlot(w, ois1.visOi.visRef[0, :, :], fig = fig, subtitles = oi.basenames)

ois1.visOi.addPhase(-phaseRef)

gravityPlot.reImPlot(w, ois1.visOi.visRef[:, :, :].mean(axis = 0), fig = fig)

u = ois1.visOi.uCoord.mean(axis = 0)
v = ois1.visOi.vCoord.mean(axis = 0)
ra =  ois1.sObjX/3600.0/360.0/1000.0*2*np.pi
dec = ois1.sObjY/3600.0/360.0/1000.0*2*np.pi
cosine = np.exp(1j*2*np.pi*(ra*u+dec*v))

#gravityPlot.reImPlot(w, 5000*cosine, fig = fig)
