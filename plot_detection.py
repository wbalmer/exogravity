#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot the chi2Map in the fiber field of view.

This script is part of the exoGravity data reduction package.
To use this script, you need to call it with a configuration file, see example below.

Args:
  config_file (str): the path the to YAML configuration file.

Example:
  python plot_detection config_file=full/path/to/yourconfig.yml

Authors:
  M. Nowak, and the exoGravity team.

Version:
  xx.xx
"""

# BASIC IMPORTS
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import scipy.sparse, scipy.sparse.linalg
from scipy.linalg import lapack
import astropy.io.fits as fits
from utils import * # utils from this exoGravity package
from common import *
# ruamel to read config yml file
try:
    import ruamel.yaml
    RUAMEL = True
except: # if ruamel not available, switch back to pyyaml, which does not handle comments properly
    import yaml
    RUAMEL = False
import sys, os

# load aguments into a dictionnary
dargs = args_to_dict(sys.argv)

if "help" in dargs.keys():
    print(__doc__)
    stop()

# arg should be the path to the config yal file    
REQUIRED_ARGS = ["config_file"]
for req in REQUIRED_ARGS:
    if not(req in dargs.keys()):
        printerr("Argument '"+req+"' is not optional for this script. Required args are: "+', '.join(REQUIRED_ARGS))

CONFIG_FILE = dargs["config_file"]
if not(os.path.isfile(CONFIG_FILE)):
    raise Exception("Error: argument {} is not a file".format(CONFIG_FILE))
if not("save_residuals") in dargs.keys():
    dargs['save_residuals'] = False

if "pmin" in list(dargs.keys()):
    PMIN = dargs["pmin"]
else:
    PMIN = None
if "pmax" in list(dargs.keys()):
    PMAX = dargs["pmax"]
else:
    PMAX = None    
    
# READ THE CONFIGURATION FILE
if RUAMEL:
    cfg = ruamel.yaml.load(open(CONFIG_FILE, "r"), Loader=ruamel.yaml.RoundTripLoader)
else:
    cfg = yaml.safe_load(open(CONFIG_FILE, "r"))
   
FIGDIR = cfg["general"]["figdir"]
DATA_DIR = cfg["general"]["datadir"]
CONTRAST_FILE = cfg["general"]["contrast_file"]
GO_FAST = cfg["general"]["gofast"]
FIGDIR = cfg["general"]["figdir"]

PLANET_FILES = [DATA_DIR+cfg["planet_ois"][preduce[list(preduce.keys())[0]]["planet_oi"]]["filename"] for preduce in cfg["general"]["reduce_planets"]]
PLANET_REJECT_DITS = [preduce[list(preduce.keys())[0]]["reject_dits"] for preduce in cfg["general"]["reduce_planets"]]
PLANET_REJECT_BASELINES = [preduce[list(preduce.keys())[0]]["reject_baselines"] for preduce in cfg["general"]["reduce_planets"]]

STAR_ORDER = cfg["general"]["star_order"]
EXTENSION = cfg["general"]["extension"]
REDUCTION = cfg["general"]["reduction"]

# FILTER DATA
PHASEREF_ARCLENGTH_THRESHOLD = cfg["general"]["phaseref_arclength_threshold"]
FT_FLUX_THRESHOLD = cfg["general"]["ft_flux_threshold"]

# OVERWRITE SOME CONFIGURATION VALUES WITH ARGUMENTS FROM COMMAND LINE
if "gofast" in dargs.keys():
    GO_FAST = dargs["gofast"].lower()=="true" # bypass value from config file
    
# lists to contain the planet, star, and potentially swap OIs
objOis = [] # contains the OIs on the planet itself

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
        printerr("Unknown reduction type '{}'.".format(REDUCTION))        
    objOis.append(oi)

# filter data
if REDUCTION == "astrored":
    # flag points based on FT value and phaseRef arclength
    ftThresholdPlanet = cfg["general"]["ftOnPlanetMeanFlux"]*FT_FLUX_THRESHOLD
    for k in range(len(PLANET_FILES)):
        oi = objOis[k]
        filter_ftflux(oi, ftThresholdPlanet)
        if PHASEREF_ARCLENGTH_THRESHOLD > 0:
            filter_phaseref_arclength(oi, PHASEREF_ARCLENGTH_THRESHOLD) 
        # explicitly set ignored dits to NAN
        if not(PLANET_REJECT_DITS[k] is None):
            if len(PLANET_REJECT_DITS[k]) > 0:
                a = PLANET_REJECT_DITS[k]
                b = range(oi.visOi.nchannel)
                (a, b, c) = np.meshgrid(a, b, range(oi.nwav))
                oi.visOi.flagPoints((a, b, c))
                printinf("Ignoring some dits in file {}".format(oi.filename))           
        # explicitly set ignored baselines to NAN
        if not(PLANET_REJECT_BASELINES[k] is None):
            if len(PLANET_REJECT_BASELINES[k]) > 0:            
                a = range(oi.visOi.ndit)
                b = PLANET_REJECT_BASELINES[k]
                (a, b, c) = np.meshgrid(a, b, range(oi.nwav))
                oi.visOi.flagPoints((a, b, c))
                printinf("Ignoring some baselines in file {}".format(oi.filename))

# replace data by the mean over all DITs if go_fast has been requested in the config file
if (GO_FAST):
    printinf("gofast flag is set. Averaging over DITs")            
    for oi in objOis: # mean should not be calculated on swap before phase correction
        printinf("Averaging file {}".format(oi.filename))
        oi.visOi.recenterPhase(oi.sObjX, oi.sObjY)
        oi.computeMean()
        oi.visOi.recenterPhase(-oi.sObjX, -oi.sObjY)

if REDUCTION == "astrored":
    wFt = np.zeros([oi.visOi.nchannel, oi.visOi.nwavFt])
    for c in range(oi.visOi.nchannel):
        wFt[c, :] = range(oi.visOi.nwavFt)

# retrieve the reference visibility for each OB
printinf("Retrieving visibility references from fits files")
visRefs = [oi.visOi.visRef.mean(axis=0)*0 for oi in objOis]
for k in range(len(objOis)):
    oi = objOis[k]
    try:
        visRefs[k] = fits.getdata(oi.filename, "EXOGRAV_VISREF").field("EXOGRAV_VISREF")
    except:
        printerr("Cannot find visibility reference (EXOGRAV_VISREF) in file {}.".format(oi.filename))

# subtract the reference phase to each OB
printinf("Subtracting phase reference to each planet OI.")
for k in range(len(objOis)):
    oi = objOis[k]
    oi.visOi.addPhase(-np.angle(visRefs[k]))

# create projector matrices
for k in range(len(objOis)):
    oi = objOis[k]
    printinf("Create projector matrices (p_matrices) ({}/{})".format(k+1, len(objOis)))
    # calculate the projector
    wav = oi.wav*1e6
    vectors = np.zeros([STAR_ORDER+1, oi.nwav], 'complex64')
    thisVisRef = visRefs[k]
    thisAmpRef = np.abs(thisVisRef)
    oi.visOi.p_matrices = np.zeros([oi.visOi.ndit, oi.visOi.nchannel, oi.nwav, oi.nwav], 'complex64')
    P = np.zeros([oi.visOi.nchannel, oi.nwav, oi.nwav], 'complex64')
    for dit in range(oi.visOi.ndit): # the loop of DIT number is necessary because of bad points mgmt (bad_indices depends on dit)
        for c in range(oi.visOi.nchannel):
            for j in range(STAR_ORDER+1):
                vectors[j, :] = np.abs(thisAmpRef[c, :])*(wav-np.mean(wav))**j # pourquoi ampRef et pas visRef ?
            bad_indices = np.where(oi.visOi.flag[dit, c, :])
            vectors[:, bad_indices] = 0
            for l in range(oi.nwav):
                x = np.zeros(oi.nwav)
                x[l] = 1
                coeffs = np.linalg.lstsq(vectors.T, x, rcond=-1)[0]
                P[c, :, l] = x - np.dot(vectors.T, coeffs)
        oi.visOi.p_matrices[dit, :, :, :] = P
        
printinf("Starting calculation of H matrices")
for k in range(len(objOis)):
    printinf("Calculating H ({:d}/{:d})".format(k+1, len(objOis)))
    oi = objOis[k]
    oi.visOi.h_matrices = np.zeros([oi.visOi.ndit, oi.visOi.nchannel, oi.visOi.nwav, oi.visOi.nwav], 'complex64')
    oi.visOi.m = np.zeros([oi.visOi.ndit, oi.visOi.nchannel])
    for dit in range(oi.visOi.ndit):
        for c in range(oi.visOi.nchannel):
            (G, d, H) = np.linalg.svd(oi.visOi.p_matrices[dit, c, :, :]) # scipy faster but different result?
            D = np.diag(d)
            oi.visOi.h_matrices[dit, c, :, :] = H
            m = int(np.sum(d))
            oi.visOi.m[dit, c] = m
        
# project visibilities
for k in range(len(objOis)):
    oi = objOis[k]
    printinf("Projecting visibilities ({}/{})".format(k+1, len(objOis)))    
    oi.visOi.visProj = [[[] for c in range(oi.visOi.nchannel)] for dit in range(oi.visOi.ndit)]
    for dit in range(oi.visOi.ndit):
        for c in range(oi.visOi.nchannel):
            m = int(oi.visOi.m[dit, c])
            oi.visOi.visProj[dit][c] = np.dot(oi.visOi.h_matrices[dit, c, 0:m, :], oi.visOi.visRef[dit, c, :])

# invert covariance matrices
for k in range(len(objOis)):
    oi = objOis[k]
    printinf("Inverting covariance matrices ({}/{})".format(k+1, len(objOis)))        
    oi.visOi.W2inv = [[[] for c in range(oi.visOi.nchannel)] for dit in range(oi.visOi.ndit)]
    for dit in range(oi.visOi.ndit): 
        for c in range(oi.visOi.nchannel):
            m = int(oi.visOi.m[dit, c])                   
            # propagate projection of visibilities on errors
            H_sp = scipy.sparse.csc_matrix(oi.visOi.h_matrices[dit, c, 0:m, :])
            W_sp = scipy.sparse.csr_matrix(oi.visOi.visRefCov[dit, c].todense())
            Z_sp = scipy.sparse.csr_matrix(oi.visOi.visRefPcov[dit, c].todense())
            W = np.dot((H_sp.dot(W_sp)).todense(), cs.adj(H_sp).todense())
            Z = np.dot((H_sp.dot(Z_sp)).todense(), H_sp.T.todense())
            W2 = cs.extended_covariance(W, Z)#.real
            oi.visOi.W2inv[dit][c] = np.linalg.inv(W2)

# prepare chi2Maps
chi2Refs = np.zeros(len(objOis))

for k in range(len(objOis)):
    oi = objOis[k]
    printinf("Calculating reference chi2 for file {}".format(oi.filename))
    for dit in range(oi.visOi.ndit):
        for c in range(oi.visOi.nchannel):
            PV2 = cs.conj_extended(oi.visOi.visProj[dit][c])                        
            Q = np.dot(np.dot(cs.adj(PV2), oi.visOi.W2inv[dit][c]), PV2)
            chi2Refs[k] = chi2Refs[k]+Q

hdul = fits.open(FIGDIR+"/chi2Maps.fits")
hdu = hdul[0]
chi2Maps = hdu.data
n = np.shape(chi2Maps)[1]
raValues = hdu.header["CRVAL1"] + hdu.header["CDELT1"]*np.array(range(n))
decValues = hdu.header["CRVAL2"] + hdu.header["CDELT2"]*np.array(range(n))

radius = 30
# mean fiber position
sObjX, sObjY = 0, 0
for k in range(len(cfg["general"]["reduce_planets"])):
    preduce = list(cfg["general"]["reduce_planets"][k].keys())[0]
    sObjX, sObjY = sObjX+cfg["planet_ois"][preduce]["sObjX"], sObjY+cfg["planet_ois"][preduce]["sObjY"]
sObjX, sObjY = sObjX/len(cfg["general"]["reduce_planets"]), sObjY/len(cfg["general"]["reduce_planets"])

# degrees of freedom
ndof = np.sum(np.array([oi.visOi.nchannel*(2*oi.nwav-STAR_ORDER*2) for oi in objOis])) - 2 - len(objOis) # minus 2 for astrometry, and 1 contrast per file

fig = plt.figure()
ax = fig.add_subplot(111)
fov = mpl.patches.Circle((sObjX, sObjY), radius, facecolor='none', edgecolor="silver", linewidth=2, linestyle="--")
ax.add_patch(fov)
im = plt.imshow(-chi2Maps.sum(axis=0)/ndof, origin = "lower", extent=[np.min(raValues), np.max(raValues), np.min(decValues), np.max(decValues)], clip_path=fov, clip_on=True, vmin = PMIN, vmax = PMAX)
cbar = plt.colorbar()
cbar.set_label("Periodogram power")
# hide top and right axis lines
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
# move the x and y axis to center of field-of-view
ax.spines["left"].set_position(["data", sObjX])
ax.spines["bottom"].set_position(["data", sObjY])
# set color of axes, labels, and ticks to white
color = "silver"
ax.spines["left"].set_color(color)
ax.spines["bottom"].set_color(color)
ax.xaxis.label.set_color(color)
ax.yaxis.label.set_color(color)
ax.tick_params(colors=color, which='both')
# select ticks
xticks = [np.round(sObjX)+k*10 for k in range(-2, 3) if k!=0]
yticks = [np.round(sObjY)+k*10 for k in range(-2, 3) if k!=0]
plt.xticks(np.round(xticks))    
plt.yticks(np.round(yticks))
# add title for axes
ax.set_xlabel("$\Delta{}\mathrm{RA}$ (mas)")
ax.xaxis.set_label_coords(0.70, 0.56)
ax.set_ylabel("$\Delta{}\mathrm{DEC}$ (mas)")
ax.yaxis.set_label_coords(0.56, 0.3)
# extent of figure
plt.xlim(sObjX-radius*1.0, sObjX+radius*1.0)
plt.ylim(sObjY-radius*1.0, sObjY+radius*1.0)    
# invert RA axis
ax.invert_xaxis()

plt.savefig(FIGDIR+"/detection.pdf")
plt.close()
