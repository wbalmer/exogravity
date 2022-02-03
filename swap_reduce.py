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
    

REQUIRED_ARGS = ["config_file"]
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
    GO_FAST = dargs["gofast"].lower()=="true" # bypass value from config file
else: # default is False
    GO_FAST = False
    
if not("ralim" in dargs.keys()) or not("declim" in dargs.keys()):
    printwar("ralim or declim not provided in args. Default: fiber position +/- 30 mas (UTs) or +/- 120 mas (ATs).")
    dargs["ralim"] = None
    dargs["declim"] = None

if not("nra" in dargs.keys()):
    printwar("nra not provided in args. Default: nra=100")
    dargs["nra"] = 100
if not("ndec" in dargs.keys()):
    printwar("ndec not provided in args. Default: ndec=100")
    dargs["ndec"] = 100
    
# figdir to know where to put figures
FIGDIR = cfg["general"]["figdir"]
if "figdir" in dargs.keys(): # overwrite
    FIGDIR = dargs["figdir"]

# LOAD GRAVITY PLOT is savefig requested
if not(FIGDIR is None):
    from cleanGravity import gravityPlot as gPlot
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib.backends.backend_pdf import PdfPages    
    if not(os.path.isdir(FIGDIR)):
        os.makedirs(FIGDIR)
        printinf("Directory {} was not found and has been created".format(FIGDIR))
    
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

# select the field of the default chi2Maps depending if observations are from UTs or ATs
if (objOis[0].stationNames[0] == "U1"):
    FIELD = 30 # UT field
else:
    FIELD = 120 # UT field
if ((dargs["ralim"] is None) or (dargs["declim"] is None)):
    ra = np.mean(np.array([oi.sObjX for oi in ois2])) # ois2 are the unswapped position
    dec = np.mean(np.array([oi.sObjY for oi in ois2]))
    ralim = [float(ra)-FIELD/2, float(ra)+FIELD/2] # get rid of numpy type so that yaml conversion works    
    declim = [float(dec)-FIELD/2, float(dec)+FIELD/2]
else:
    ralim = [float(r) for r in dargs["ralim"].split(']')[0].split('[')[1].split(',')]
    declim = [float(r) for r in dargs["declim"].split(']')[0].split('[')[1].split(',')]    
printinf("RA grid set to [{:.2f}, {:.2f}] with {:d} points".format(ralim[0], ralim[1], dargs["nra"]))
printinf("DEC grid set to [{:.2f}, {:.2f}] with {:d} points".format(declim[0], declim[1], dargs["ndec"]))

# prepare chi2Maps
N_RA = dargs["nra"]
N_DEC = dargs["ndec"]
raValues = np.linspace(ralim[0], ralim[1], N_RA)
decValues = np.linspace(declim[0], declim[1], N_DEC)
chi2Map = np.zeros([N_RA, N_DEC])
# to keep track of the best fit
chi2Best = np.inf
bestFit = np.zeros([oi.visOi.nchannel, oi.visOi.nwav], "complex")
raBest = 0
decBest = 0

# start the fit
for i in range(N_RA):
    ra = raValues[i]    
    printinf("Calculating chi2 map for ra value {} ({}/{})".format(ra, i+1, N_RA))
    for j in range(N_DEC):
        dec = decValues[j]
        # reference visibility for group 1        
        visRefS1 = 0*oi.visOi.visRef.mean(axis = 0)
        # reference visibility for group 2                
        visRefS2 = 0*oi.visOi.visRef.mean(axis = 0)        
        for oi in ois1+ois2:
            this_u = (ra*oi.visOi.uCoord + dec*oi.visOi.vCoord)/1e7
            phase = 2*np.pi*this_u*1e7/3600.0/360.0/1000.0*2*np.pi
            phi = np.exp(1j*phase)
            if oi.swap:
                visRefS1 = visRefS1+(np.conj(phi)*oi.visOi.visRef).mean(axis = 0)
            else:
                visRefS2 = visRefS2+(phi*oi.visOi.visRef).mean(axis = 0)                
        visRefS1 = visRefS1/len(ois1)
        visRefS2 = visRefS2/len(ois2)
        visRefSwap = np.sqrt(visRefS1*np.conj(visRefS2))
        chi2Map[i, j] = np.sum(np.imag(visRefSwap)**2)
        if chi2Map[i, j] < chi2Best:
            chi2Best = chi2Map[i, j]
            raBest = ra
            decBest = dec
            bestFit = np.real(visRefSwap)+0j
            visRefSwapBest = visRefSwap
            
printinf("Best astrometry solution: RA={}, DEC={}".format(raBest, decBest))
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
    
# we'll need the phase reference for the plots
for oi in ois1+ois2:
    if oi.swap:
        oi.visOi.recenterPhase(-raBest, -decBest)        
    else:
        oi.visOi.recenterPhase(raBest, decBest)

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
printinf("Recentering OBs on initial positions")
for oi in ois1+ois2:
    if oi.swap:
        oi.visOi.recenterPhase(raBest, decBest)
    else:
        oi.visOi.recenterPhase(-raBest, -decBest)        

# the phase ref is the average of the phase of the two groups
printinf("Calculating phase ref")
phaseRef = np.angle(0.5*(visRefS1+visRefS2))

# remove phaseRef from OBs
for oi in objOis:
    oi.visOi.addPhase(-phaseRef)

# now its time to plot the chi2Maps in FIGDIR if given
if not(FIGDIR is None):
    with PdfPages(FIGDIR+"/swap_fit_results.pdf") as pdf:
        
        for k in range(len(objOis)):
            oi = objOis[k]
            fig = plt.figure(figsize=(10, 8))
            gPlot.reImPlot(w, np.ma.masked_array(oi.visOi.visRef, oi.visOi.flag).mean(axis = 0), subtitles = oi.basenames, fig = fig)
            # move the best fit back to star frame to overplot it
            this_u = (raBest*oi.visOi.uCoord + decBest*oi.visOi.vCoord)/1e7
            phase = 2*np.pi*this_u*1e7/3600.0/360.0/1000.0*2*np.pi
            phi = np.exp(1j*phase)
            if oi.swap:
                gPlot.reImPlot(w, np.ma.masked_array(phi*bestFit, oi.visOi.flag).mean(axis = 0), fig = fig)                
            else:
                gPlot.reImPlot(w, np.ma.masked_array(np.conj(phi)*bestFit, oi.visOi.flag).mean(axis = 0), fig = fig)
            plt.legend([oi.filename.split("/")[-1], "Swap fit (from combined files)"])
            pdf.savefig()
            
        fig = plt.figure(figsize = (10, 10))
        ax = fig.add_subplot(1, 1, 1)
        oi = objOis[k]
        name = oi.filename.split('/')[-1]
        ax.imshow(chi2Map.T, origin = "lower", extent = [np.min(raValues), np.max(raValues), np.min(decValues), np.max(decValues)])
        ax.set_xlabel("$\Delta{}\mathrm{RA}$ (mas)")
        ax.set_ylabel("$\Delta{}\mathrm{DEC}$ (mas)")
        ax.set_title(name)
        plt.tight_layout()
        pdf.savefig()
