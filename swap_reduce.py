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

# standard imports
import numpy as np
import sys, os
import glob
import itertools

# cleanGravity imports
import cleanGravity as gravity
from cleanGravity import complexstats as cs

# local stuff
from utils import *

# ruamel to read config yml file
try:
    import ruamel.yaml
    RUAMEL = True
except: # if ruamel not available, switch back to pyyaml, which does not handle comments properly
    import yaml
    RUAMEL = False

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
if RUAMEL:
    cfg = ruamel.yaml.load(open(CONFIG_FILE, "r"), Loader=ruamel.yaml.RoundTripLoader)
else:
    cfg = yaml.safe_load(open(CONFIG_FILE, "r"))

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
if "nopd" in dargs.keys():
    N_OPD = int(dargs["nopd"])
else:
    N_OPD = 200

# figdir to know where to put figures
if "figdir" in dargs.keys():
    FIGDIR = dargs["figdir"]
else:
    FIGDIR = None

# if figdir given, we need to import matplotlib and gravityPlot
if FIGDIR:
    import matplotlib.pyplot as plt
    from cleanGravity import gravityPlot as gPlot
    
RA_LIM = [float(r) for r in dargs["ralim"].split(']')[0].split('[')[1].split(',')]
DEC_LIM = [float(r) for r in dargs["declim"].split(']')[0].split('[')[1].split(',')]

RA_GUESS = float(dargs["raguess"])
DEC_GUESS = float(dargs["decguess"])

# retrieve all astrored files from the datadir
printinf("Found a total of {:d} astroreduced files in {}".format(len(SWAP_FILES), DATA_DIR))

objOis = []
ois1 = []
ois2 = []

# load OBs and put them in two groups based on the two swap pos
for k in range(0, len(SWAP_FILES)):
    filename = SWAP_FILES[k]
    printinf("Loading file "+filename)    
    oi = gravity.GravityDualfieldAstrored(filename, extension = cfg["general"]["extension"], corrMet = "drs", corrDisp = "drs")
    # note: in case go_fast is set, files should not be averaged over DITs at this step, but later
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

printinf("gofast flag is set. Averaging over DITs")
if (GO_FAST):
    for oi in objOis:
        printinf("Averaging file {}".format(oi.filename))
        oi.visOi.recenterPhase(oi.sObjX, oi.sObjY) # mean should be calculated in the frame of the SC
        oi.computeMean()
        oi.visOi.recenterPhase(-oi.sObjX, -oi.sObjY) # back to original (FT) frame        
    
# calculate the useful w for plotting
w = np.zeros([6, oi.nwav])
for k in range(6):
    w[k, :] = oi.wav*1e6

# we need to coadd the elements of each group
# this is best done in in the SC reference frame to avoid issues with rapid oscillations of the phase
# so we need to shift the OBs to the SC frame. We could use the fiber position (oi.visOi.sObjX and Y)
# but sometimes the target is not correctly centered. So the user has to supply an estimate of the
# astrometry of the target
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

# prepare chi2Maps
printinf("RA grid: [{:.2f}, {:.2f}] with {:d} points".format(RA_LIM[0], RA_LIM[1], N_RA))
printinf("DEC grid: [{:.2f}, {:.2f}] with {:d} points".format(DEC_LIM[0], DEC_LIM[1], N_DEC))
raValues = np.linspace(RA_LIM[0], RA_LIM[1], N_RA)
decValues = np.linspace(DEC_LIM[0], DEC_LIM[1], N_DEC)
raValues_swap = np.linspace(-RA_LIM[1], -RA_LIM[0], N_RA)
decValues_swap = np.linspace(-DEC_LIM[-1], -DEC_LIM[0], N_DEC)
chi2Maps = np.zeros([len(objOis), N_RA, N_DEC])
bestFits = []
# to store best fits values on each file
raBests = np.zeros(len(objOis))
decBests = np.zeros(len(objOis))

# calculate chi2Maps using the fitAstrometry method, which itself call fitVisRefOpd
for k in range(len(objOis)):
    oi = objOis[k]
    if oi.swap:
        fitAstro = oi.visOi.fitVisRefAstrometry(raValues_swap, decValues_swap, nopd=N_OPD)
    else:
        fitAstro = oi.visOi.fitVisRefAstrometry(raValues, decValues, nopd=N_OPD)        
    chi2Maps[k, :, :] = fitAstro["map"]
    bestFits.append(fitAstro["fit"])
    raBests[k] = fitAstro["best"][0]
    decBests[k] = fitAstro["best"][1]

# calculate the best RA and DEC taking into account swap mode:
raBest_swap = np.mean(np.array([raBests[k] for k in range(len(objOis)) if objOis[k].swap]))
decBest_swap = np.mean(np.array([decBests[k] for k in range(len(objOis)) if objOis[k].swap]))
raBest = np.mean(np.array([raBests[k] for k in range(len(objOis)) if not(objOis[k].swap)]))
decBest = np.mean(np.array([decBests[k] for k in range(len(objOis)) if not(objOis[k].swap)]))

printinf("Astrometric solution found using NO SWAP file: RA={:f} mas, DEC={:f} mas".format(raBest, decBest))
printinf("Astrometric solution found using SWAP files: RA={:f} mas, DEC={:f} mas".format(raBest_swap, decBest_swap))

# get the astrometric values and calculate the mean, and add it to the YML file (same value for all swap Ois)
for key in cfg['swap_ois'].keys():
    cfg["swap_ois"][key]["astrometric_solution"] = [float(raBest), float(decBest)] # YAML cannot convert numpy types

# rewrite the YML
f = open(CONFIG_FILE, "w")
if RUAMEL:
    f.write(ruamel.yaml.dump(cfg, Dumper=ruamel.yaml.RoundTripDumper))
else:
    f.write(yaml.safe_dump(cfg, default_flow_style = False)) 
f.close()
    
# now its time to plot the chi2Maps in FIGDIR if given
if not(FIGDIR is None):
    for k in range(len(objOis)):
        oi = objOis[k]
        fig = plt.figure(figsize=(10, 8))
        gPlot.reImPlot(w, np.ma.masked_array(oi.visOi.visRef, oi.visOi.flag).mean(axis = 0), subtitles = oi.basenames, fig = fig)
        gPlot.reImPlot(w, np.ma.masked_array(bestFits[k], oi.visOi.flag).mean(axis = 0), fig = fig)
        plt.legend([oi.filename.split("/")[-1], "Astrometry fit"])
        plt.savefig(FIGDIR+"/astrometry_fit_"+str(k)+".pdf")

    fig = plt.figure(figsize = (10, 10))
    n = int(np.ceil(np.sqrt(len(objOis))))
    for k in range(len(objOis)):
        ax = fig.add_subplot(n, n, k+1)
        oi = objOis[k]
        name = oi.filename.split('/')[-1]
        if oi.swap:
            ax.imshow(chi2Maps[k].T, origin = "lower", extent = [np.min(raValues_swap), np.max(raValues_swap), np.min(decValues_swap), np.max(decValues_swap)])
        else:
            ax.imshow(chi2Maps[k].T, origin = "lower", extent = [np.min(raValues), np.max(raValues), np.min(decValues), np.max(decValues)])            
        ax.set_xlabel("$\Delta{}\mathrm{RA}$ (mas)")
        ax.set_ylabel("$\Delta{}\mathrm{DEC}$ (mas)")
        ax.set_title(name)
    plt.tight_layout()
    plt.savefig(FIGDIR+"/astrometry_chi2Maps.pdf")
