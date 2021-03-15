#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Extract astrometry from an exoGravity observation

This script is part of the exoGravity data reduction package.
The astrometry_reduce script is used to extract the astrometry from an exoplanet observation made with GRAVITY, in dual-field mode.
To use this script, you need to call it with a configuration file, see example below.

Args:
  config_file (str): the path the to YAML configuration file.
  gofast (bool, optional): if set, average over DITs to accelerate calculations (Usage: --gofast, or gofast=True, or goFast=False). This will bypass
                           the value contained in the YAML file.
  noinv (bool, optional): if set, avoid inversion of the covariance matrix and replace it with a gaussian elimination approach. 
                          Bypass the value in the YAML file. (Usage: --noinv, or noinv=True, or noinv=False)
  save_residuals (bool, optional): if set, save the residuals in npy files. figdir must also be provided for this option to be used

Example:
  python astrometry_reduce config_file=full/path/to/yourconfig.yml

Authors:
  M. Nowak, and the exoGravity team.

Version:
  xx.xx
"""

# BASIC IMPORTS
import numpy as np
import scipy.sparse, scipy.sparse.linalg
from scipy.linalg import lapack
import astropy.io.fits as fits
# cleanGravity IMPORTS
import cleanGravity as gravity
from cleanGravity import complexstats as cs
from cleanGravity.utils import loadFitsSpectrum, saveFitsSpectrum
from utils import * # utils from this exooGravity package
from common import *
# other random stuffs
import glob
import itertools
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
    
# READ THE CONFIGURATION FILE
if RUAMEL:
    cfg = ruamel.yaml.load(open(CONFIG_FILE, "r"), Loader=ruamel.yaml.RoundTripLoader)
else:
    cfg = yaml.safe_load(open(CONFIG_FILE, "r"))
DATA_DIR = cfg["general"]["datadir"]
PHASEREF_MODE = cfg["general"]["phaseref_mode"]
CONTRAST_FILE = cfg["general"]["contrast_file"]
NO_INV = cfg["general"]["noinv"]
GO_FAST = cfg["general"]["gofast"]
FIGDIR = cfg["general"]["figdir"]
PLANET_FILES = [DATA_DIR+cfg["planet_ois"][k]["filename"] for k in cfg["general"]["reduce"]] # list of files corresponding to planet exposures
if not("swap_ois" in cfg.keys()):
    SWAP_FILES = []
elif cfg["swap_ois"] is None:
    SWAP_FILES = []
else:
    SWAP_FILES = [DATA_DIR+cfg["swap_ois"][k]["filename"] for k in cfg["swap_ois"].keys()]
STAR_ORDER = cfg["general"]["star_order"]
N_OPD = cfg["general"]["n_opd"]
N_RA = cfg["general"]["n_ra"]
N_DEC = cfg["general"]["n_dec"]
RA_LIM = cfg["general"]["ralim"]
DEC_LIM = cfg["general"]["declim"]
EXTENSION = cfg["general"]["extension"]
REDUCTION = cfg["general"]["reduction"]

# FILTER DATA
PHASEREF_ARCLENGTH_THRESHOLD = cfg["general"]["phaseref_arclength_threshold"]
FT_FLUX_THRESHOLD = cfg["general"]["ft_flux_threshold"]
IGNORE_BASELINES = cfg["general"]["ignore_baselines"]

# OVERWRITE SOME OF THE CONFIGURATION VALUES WITH ARGUMENTS FROM COMMAND LINE
if "gofast" in dargs.keys():
    GO_FAST = dargs["gofast"] # bypass value from config file    
if "noinv" in dargs.keys():
    NO_INV = dargs["noinv"] # bypass value from config file
if "figdir" in dargs.keys():
    FIGDIR = dargs["figdir"] # bypass value from config file

# LOAD GRAVITY PLOT is savefig requested
if not(FIGDIR is None):
    from cleanGravity import gravityPlot as gPlot
    import matplotlib.pyplot as plt
    import matplotlib
    if not(os.path.isdir(FIGDIR)):
        os.makedirs(FIGDIR)
        printinf("Directory {} was not found and has been created".format(FIGDIR))

# THROW AN ERROR IF save_residuals requested, but no FIGDIR provided
if dargs['save_residuals'] and (FIGDIR is None):
    printerr("save_residuals option requested, but no figdir provided.")
        
# extract list of useful star ois from the list indicated in the star_indices fields of the config file:
star_indices = []
for k in cfg["general"]["reduce"]:
    star_indices = star_indices+cfg["planet_ois"][k]["star_indices"]
star_indices = list(set(star_indices)) # remove duplicates
STAR_FILES = [DATA_DIR+cfg["star_ois"][k]["filename"] for k in star_indices]


# read the contrast file if given, otherwise simply set it to 1
if CONTRAST_FILE is None:
    contrast_data = 1
elif CONTRAST_FILE.split('.')[-1].lower() in ["fit", "fits"]:
    contrast_wav, dummy, dummy2, contrast_data, dummy3 = loadFitsSpectrum(CONTRAST_FILE)
else:
    printerr("The given contrast file ({}) should have a fits extension".format(fits))

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
        printerr("Unknonwn reduction type '{}'.".format(REDUCTION))        
    objOis.append(oi)

# filter data
if REDUCTION == "astrored":
    # flag points based on FT value and phaseRef arclength
    ftThresholdPlanet = cfg["general"]["ftOnPlanetMeanFlux"]*FT_FLUX_THRESHOLD
    for oi in objOis:
        filter_ftflux(oi, ftThresholdPlanet)
        if PHASEREF_ARCLENGTH_THRESHOLD > 0:
            filter_phaseref_arclength(oi, PHASEREF_ARCLENGTH_THRESHOLD)
        # explicitly set ignored baselines to NAN
        if len(IGNORE_BASELINES)>0:
            a = range(oi.visOi.ndit)
            b = IGNORE_BASELINES
            (a, b, c) = np.meshgrid(a, b, range(oi.nwav))
            oi.visOi.flagPoints((a, b, c))
            printinf("Ignoring some baselines in file {}".format(oi.filename))

# replace data by the mean over all DITs if go_fast has been requested in the config file
printinf("gofast flag is set. Averaging over DITs")
for oi in objOis: # mean should not be calculated on swap before phase correction
    if (GO_FAST):
        printinf("Averaging file {}".format(oi.filename))
        oi.visOi.recenterPhase(oi.sObjX, oi.sObjY)
        oi.computeMean()
        oi.visOi.recenterPhase(-oi.sObjX, -oi.sObjY)

# calculate the very useful w for plotting
oi = objOis[0]
w = np.zeros([oi.visOi.nchannel, oi.nwav])
for c in range(oi.visOi.nchannel):
    w[c, :] = oi.wav*1e6
    
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
            # invert the covariance matrix
            #ZZ, _ = lapack.dpotrf(W2)
            #T, info = lapack.dpotri(ZZ)
            #oi.visOi.W2inv[dit][c] = np.triu(T) + np.triu(T, k=1).T

# prepare chi2Maps
printinf("RA grid: [{:.2f}, {:.2f}] with {:d} points".format(RA_LIM[0], RA_LIM[1], N_RA))
printinf("DEC grid: [{:.2f}, {:.2f}] with {:d} points".format(DEC_LIM[0], DEC_LIM[1], N_DEC))
raValues = np.linspace(RA_LIM[0], RA_LIM[1], N_RA)
decValues = np.linspace(DEC_LIM[0], DEC_LIM[1], N_DEC)
chi2Maps = np.zeros([len(objOis), N_RA, N_DEC])
# to store best fits values on each file
bestFits = []
kappas = np.zeros(len(objOis))
raBests = np.zeros(len(objOis))
decBests = np.zeros(len(objOis))

for k in range(len(objOis)):
    oi = objOis[k]
    chi2Best = np.inf
    # prepare array
    phi_values = np.zeros([oi.visOi.ndit, oi.visOi.nchannel, oi.nwav], 'complex')
    # calculate As and Bs
    phiBest = np.zeros([oi.visOi.ndit, oi.visOi.nchannel, oi.nwav], "complex")
    printinf("Calculating chi2Map for file {}".format(oi.filename))
    for i in range(N_RA):
        for j in range(N_DEC):
            ra = raValues[i]
            dec = decValues[j]
            A, B = 0., 0.
            kappa2 = 0
            this_u = (ra*oi.visOi.uCoord + dec*oi.visOi.vCoord)/1e7
            phase = 2*np.pi*this_u*1e7/3600.0/360.0/1000.0*2*np.pi
            phi = np.exp(-1j*phase)*np.abs(visRefs[k])
            phiProj = [[[] for c in range(oi.visOi.nchannel)] for dit in range(oi.visOi.ndit)]
            for dit in range(oi.visOi.ndit):
                for c in range(oi.visOi.nchannel):
                    m = int(oi.visOi.m[dit, c])
                    phiProj[dit][c] = np.dot(oi.visOi.h_matrices[dit, c, 0:m, :], phi[dit, c, :])
            for dit in range(oi.visOi.ndit):
                for c in range(oi.visOi.nchannel):
                    phiProj2 = cs.conj_extended(phiProj[dit][c])
                    PV2 = cs.conj_extended(oi.visOi.visProj[dit][c])            
                    Q = np.dot(cs.adj(phiProj2), oi.visOi.W2inv[dit][c])
                    A = A+np.real(np.dot(Q, PV2))
                    B = B+np.real(np.dot(Q, phiProj2))
            kappa = A/B
            chi2Maps[k, i, j] = -A**2/B
            if chi2Maps[k, i, j] < chi2Best:
                phiBest = kappa*phi
                chi2Best = chi2Maps[k, i, j]
                raBests[k] = ra
                decBests[k] = dec
                kappas[k] = kappa
    bestFits.append(phiBest)

# calculate best fit with only a star model by projetting visibilities                
bestFitStars = []
for k in range(len(objOis)):
    oi = objOis[k]
    bestFitStar = np.zeros([oi.visOi.ndit, oi.visOi.nchannel, oi.nwav], "complex")
    for dit in range(oi.visOi.ndit):
        for c in range(oi.visOi.nchannel):
            bestFitStar[dit, c, :] = oi.visOi.visRef[dit, c, :] - np.dot(oi.visOi.p_matrices[dit, c, :, :], oi.visOi.visRef[dit, c, :])
    bestFitStars.append(bestFitStar)

# combine chi2Maps to get best astrometric solution
chi2Map = np.sum(chi2Maps, axis = 0)
ind = np.where(chi2Map == np.min(chi2Map))
raBest = raValues[ind[0][0]]
decBest = decValues[ind[1][0]]

# for each map, look for local minimum close to the best solution
raBests_local = np.zeros(len(objOis))
decBests_local = np.zeros(len(objOis))
for k in range(len(objOis)):
    printinf("Looking for local chi2 minimun for file {}".format(objOis[k].filename))
    i, j = np.where(raValues == raBest)[0][0], np.where(decValues == decBest)[0][0]    
    i, j = findLocalMinimum(chi2Maps[k, :, :], i, j, jump_size=5)
    raBests_local[k] = raValues[i]
    decBests_local[k] = decValues[j]
    if (raBests_local[k] != raBests[k]) or (decBests_local[k] != decBests[k]):
        printwar("Local minimum for file {} is different from global minimum".format(objOis[k].filename))


cov = np.cov(raBests, decBests)
printinf("RA: {:.2f}+-{:.3f} mas".format(np.mean(raBests), cov[0, 0]**0.5))
printinf("DEC: {:.2f}+-{:.3f} mas".format(np.mean(decBests), cov[1, 1]**0.5))
printinf("COV: {:.2f}".format(cov[0, 1]/np.sqrt(cov[0, 0]*cov[1, 1])))

cov_local = np.cov(raBests_local, decBests_local)
printinf("RA (from combined map): {:.2f}+-{:.3f} mas".format(raBest, cov_local[0, 0]**0.5))
printinf("DEC (from combined map): {:.2f}+-{:.3f} mas".format(decBest, cov_local[1, 1]**0.5))

printinf("Contrast obtained (mean, min, max): {:.2e}, {:.2e}, {:.2e}".format(np.mean(kappas), np.min(kappas), np.max(kappas)))

for k in range(len(PLANET_FILES)):
    ind = cfg["general"]["reduce"][k]
    cfg["planet_ois"][ind]["astrometric_solution"] = [float(raBests[k]), float(decBests[k])] # YAML cannot convert numpy types
            
f = open(CONFIG_FILE, "w")
if RUAMEL:
    f.write(ruamel.yaml.dump(cfg, Dumper=ruamel.yaml.RoundTripDumper))
else:
    f.write(yaml.safe_dump(cfg, default_flow_style = False)) 
f.close()


if not(FIGDIR is None):
    plt.figure()
    plt.imshow(chi2Map.T, origin = "lower", extent = [np.min(raValues), np.max(raValues), np.min(decValues), np.max(decValues)])
    plt.xlabel("$\Delta{}\mathrm{RA}$ (mas)")
    plt.ylabel("$\Delta{}\mathrm{DEC}$ (mas)")
    plt.savefig(FIGDIR+"/astrometry_combined.pdf")                

    # UV plot
    gPlot.uvMap(objOis[0].visOi.uCoord[0, :, :], objOis[0].visOi.vCoord[0, :, :], targetCoord=(raBest, decBest), legend = objOis[0].basenames, symmetric = True)
    plt.savefig(FIGDIR+"/uvmap.pdf")
    plt.close()
    
    for k in range(len(objOis)):
        oi = objOis[k]
        fig = plt.figure(figsize=(10, 8))
        gPlot.reImPlot(w, np.ma.masked_array(oi.visOi.visRef-bestFitStars[k], oi.visOi.flag).mean(axis = 0), subtitles = oi.basenames, fig = fig)
        gPlot.reImPlot(w, np.ma.masked_array(bestFits[k], oi.visOi.flag).mean(axis = 0), fig = fig)
        plt.legend([oi.filename.split("/")[-1], "Astrometry fit"])
        plt.savefig(FIGDIR+"/astrometry_fit_"+str(k)+".pdf")
        fig = plt.figure(figsize=(10, 8))        
        gPlot.reImPlot(w, np.ma.masked_array(oi.visOi.visRef, oi.visOi.flag).mean(axis = 0), subtitles = oi.basenames, fig = fig)
        gPlot.reImPlot(w, np.ma.masked_array(bestFitStars[k], oi.visOi.flag).mean(axis = 0), fig = fig)
        plt.legend([oi.filename.split("/")[-1], "Star fit"])
        plt.savefig(FIGDIR+"/star_fit_"+str(k)+".pdf")                

    fig = plt.figure(figsize = (10, 10))
    n = int(np.ceil(np.sqrt(len(objOis))))
    for k in range(len(objOis)):
        ax = fig.add_subplot(n, n, k+1)
        oi = objOis[k]
        name = oi.filename.split('/')[-1]
        ax.imshow(chi2Maps[k, :, :].T, origin = "lower", extent = [np.min(raValues), np.max(raValues), np.min(decValues), np.max(decValues)])
        ax.plot(raBests[k], decBests[k], "+C1")
        ax.plot(raBests_local[k], decBests_local[k], "+C2")                 
        ax.plot(raBest, decBest, "+C3")         
        ax.set_xlabel("$\Delta{}\mathrm{RA}$ (mas)")
        ax.set_ylabel("$\Delta{}\mathrm{DEC}$ (mas)")
        ax.set_title(name)
    plt.tight_layout()
    plt.savefig(FIGDIR+"/astrometry_chi2Maps.pdf")

    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(111)
    ax.plot(raBests, decBests, "oC0")
    ax.plot(raBests_local, decBests_local, "oC1")    
    val, vec = np.linalg.eig(cov)
    e1=matplotlib.patches.Ellipse((raBests.mean(),decBests.mean()), 2*val[0]**0.5, 2*val[1]**0.5, angle=np.arctan2(vec[0,1],-vec[1,1])/np.pi*180, fill=False, color='C0', linewidth=2, linestyle='--')
    e2=matplotlib.patches.Ellipse((np.mean(raBests), np.mean(decBests)), 3*2*val[0]**0.5, 3*2*val[1]**0.5, angle=np.arctan2(vec[0, 1], -vec[1, 1])/np.pi*180.0, fill=False, color='C0', linewidth=2)
    ax.add_patch(e1)
    ax.add_patch(e2)    
    ax.plot([np.mean(raBests)], [np.mean(decBests)], '+C0')
    ax.text(raBests.mean()+val[0]**0.5, decBests.mean()+val[1]**0.5, "RA={:.2f}+-{:.3f}\nDEC={:.2f}+-{:.3f}\nCOV={:.2f}".format(np.mean(raBests), cov[0, 0]**0.5, np.mean(decBests), cov[1, 1]**0.5, cov[0, 1]/np.sqrt(cov[0, 0]*cov[1, 1])))
    
    val, vec = np.linalg.eig(cov_local)
    e1=matplotlib.patches.Ellipse((raBests_local.mean(), decBests_local.mean()), 2*val[0]**0.5, 2*val[1]**0.5, angle=np.arctan2(vec[0,1],-vec[1,1])/np.pi*180, fill=False, color='C1', linewidth=2, linestyle='--')
    e2=matplotlib.patches.Ellipse((np.mean(raBests_local), np.mean(decBests_local)), 3*2*val[0]**0.5, 3*2*val[1]**0.5, angle=np.arctan2(vec[0, 1], -vec[1, 1])/np.pi*180.0, fill=False, color='C1', linewidth=2)
    ax.add_patch(e1)
    ax.add_patch(e2)    
    ax.plot([np.mean(raBests_local)], [np.mean(decBests_local)], '+C1')
    ax.text(raBests_local.mean()-val[0]**0.5, decBests_local.mean()-val[1]**0.5, "RA={:.2f}+-{:.3f}\nDEC={:.2f}+-{:.3f}\nCOV={:.2f}".format(np.mean(raBests_local), cov_local[0, 0]**0.5, np.mean(decBests_local), cov_local[1, 1]**0.5, cov_local[0, 1]/np.sqrt(cov_local[0, 0]*cov_local[1, 1])))    
    ax.set_xlabel("$\Delta{}\mathrm{RA}$ (mas)")
    ax.set_ylabel("$\Delta{}\mathrm{DEC}$ (mas)")
    plt.axis("equal")
    plt.savefig(FIGDIR+"/solution.pdf")
    
if dargs['save_residuals']:
    np.save(FIGDIR+"/"+"wav.npy", w[0, :])    
    for k in range(len(objOis)):
        oi = objOis[k]
        name = oi.filename.split('/')[-1].split('.fits')[0]
        visRef = oi.visOi.visRef
        visFitStar = bestFitStars[k]
        visFit = bestFits[k]
        flags = oi.visOi.flag
        visFt = oi.visOi.visDataFt        
        np.save(FIGDIR+"/"+name+"_ref.npy", visRef)
        np.save(FIGDIR+"/"+name+"_fit.npy", visFit)
        np.save(FIGDIR+"/"+name+"_starfit.npy", visFitStar)
        np.save(FIGDIR+"/"+name+"_flags.npy", flags)
        np.save(FIGDIR+"/"+name+"_ft.npy", visFt)                                

stop()

# A large array to store the coefficients of the fits
fig = plt.figure()
for k in range(len(objOis)):
    opdFit = opdFits[k]
    values = opdFit["grid"]
    params = opdFit["params"]
    if k==0:
        gPlot.baselinePlot(values, params[0, :, :, -1], subtitles = oi.basenames, fig = fig)
    else:
        gPlot.baselinePlot(values, params[0, :, :, -1], fig = fig)
plt.tight_layout()

stop()


for k in range(len(objOis)):
    fig = plt.figure()    
    oi = objOis[k]
    for dit in range(oi.ndit):
        if dit == 0:
            gPlot.reImPlot(w, oi.visOi.visRef[0, :, :], subtitles=oi.basenames, fig=fig)
        else:
            gPlot.reImPlot(w, oi.visOi.visRef[dit, :, :], fig=fig)

wdits = np.zeros([oi.visOi.nchannel, oi.ndit])
for c in range(oi.visOi.nchannel):
    wdits[c, :] = range(oi.visOi.ndit)

for k in range(len(objOis)):
    coeff = bestParamsStars[k][:, :, 0]+1j*bestParamsStars[k][:, :, 1]
    gPlot.baselinePlot(wdits, np.angle(coeff).T, subtitles = oi.basenames)
            

# ACTUAL CODE ENDS HERE!!!


"""
# estimate the distribution of likelihood
all_opts = []
all_smallMaps = []
all_fits = []
for k in range(len(objOis)-1):
    # convert chi2 to likelihoo
    likelihood = np.exp(-0.5*(chi2Map[k, :, :]-np.min(chi2Map[k, :, :])))

    # get indice of max and trim the map
    n = 7
    ra_ind = np.where(raValues == raBest[k])[0][0]
    dec_ind = np.where(decValues == decBest[k])[0][0]
    ra = raValues[ra_ind-n:ra_ind+n]
    dec = decValues[dec_ind-n:dec_ind+n]

    X = np.zeros([2*n, 2])
    X[:, 0] = ra
    X[:, 1] = dec

    smallMap = likelihood[ra_ind-n:ra_ind+n, dec_ind-n:dec_ind+n]

    def gaussian(x):
        W = np.zeros([2, 2])
        W[0, 0] = np.sqrt(1+x[0]**2+x[1]**2)+x[0]
        W[1, 1] = np.sqrt(1+x[0]**2+x[1]**2)-x[0]
        W[0, 1] = x[1]
        W[1, 0] = x[1]
        W = np.exp(x[2])*W
        model = np.zeros([2*n, 2*n])
        for k in range(2*n):
            for i in range(2*n):
                thisX = np.array([[X[k, 0]-x[3], X[i, 1]-x[4]]])
                model[k, i] = np.exp(-0.5*np.dot(np.dot(thisX, np.linalg.inv(W)), thisX.T))[0, 0]
        return model

    def sqrsum(x):
        return np.sum((smallMap - gaussian(x))**2)

    opt = scipy.optimize.minimize(sqrsum, [0.25, -1.8, -9.5, raBest[k], decBest[k]])
    xopt = opt['x']
    
    all_opts.append(opt)
    all_smallMaps.append(smallMap)
    all_fits.append(gaussian(xopt))    

np.savetxt("astrometry.txt", np.array([opt['x'] for opt in all_opts]))

for k in range(len(objOis)):
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    p = ax.imshow(all_smallMaps[k].T, origin = 'lower')
    plt.colorbar(p)
    ax = fig.add_subplot(1, 2, 2)    
    p = ax.imshow(all_fits[k].T, origin = 'lower')
    plt.colorbar(p)
    plt.savefig("./likelihood_fits/fit_"+str(k)+".png")
    plt.close()

# save results
astrometry = np.zeros([len(objOis), 2])
astrometry[:, 0] = raBest
astrometry[:, 1] = decBest
np.save('chi2Map', chi2Map)
np.save('astrometry', astrometry)

# plot maps
for k in range(len(objOis)):
    plt.figure()
    plt.imshow(chi2Map[k, :, :].T/(objOis[k].visOi.ndit*objOis[k].visOi.nchannel*objOis[k].nwav),  origin = "lower", extent = [ra_lim[0], ra_lim[1], dec_lim[0], dec_lim[1]])
    plt.colorbar()
"""

# recalculate best fits
pBar = patiencebar.Patiencebar(valmax = len(objOis), title = "Recalculating best Fits...")



fig = plt.figure(figsize = (6, 6))
plt.plot(raBest, decBest, "o")
plt.xlabel("$\Delta{}\mathrm{RA}$ (mas)")
plt.ylabel("$\Delta{}\mathrm{DEC}$ (mas)")
plt.axis('equal')
plt.tight_layout()
if SAVEFIG:
    plt.savefig("astrometryReport/astrometry.pdf")
    plt.close(fig)

fig = plt.figure(figsize = (12, 6))
ax = fig.add_subplot(211)
ax.plot([oi.time for oi in objOis], raBest, "o")
ax.set_xlabel("Time (s)")
ax.set_ylabel("$\Delta{}\mathrm{RA}$ (mas)")
ax = fig.add_subplot(212)
ax.plot([oi.time for oi in objOis], decBest, "o")
ax.set_xlabel("Time (s)")
ax.set_ylabel("$\Delta{}\mathrm{DEC}$ (mas)")
plt.tight_layout()
if SAVEFIG:
    plt.savefig("astrometryReport/astrometry_time.pdf")
    plt.close(fig)
                     

for k in range(len(objOis)):
    fig = plt.figure(figsize = (12, 6))
    ylim = np.percentile(np.abs(objOis[k].visOi.visRef), 99)
    gravityPlot.reImPlot(w, objOis[k].visOi.visRef.mean(axis = 0), subtitles = oi.basenames, fig = fig, ylim = [-ylim, ylim])
    gravityPlot.reImPlot(w, objOis[k].visOi.bestFitStar.mean(axis = 0), fig = fig)
    if SAVEFIG:
        plt.savefig("astrometryReport/starfit_"+str(k)+".pdf")
        plt.close(fig)

    fig = plt.figure(figsize = (12, 6))
    ylim = np.percentile(np.abs(objOis[k].visOi.visRef-objOis[k].visOi.bestFitStar), 99)
    gravityPlot.reImPlot(w, objOis[k].visOi.visRef.mean(axis = 0) - objOis[k].visOi.bestFitStar.mean(axis = 0), subtitles = oi.basenames, fig = fig, ylim = [-ylim, ylim])
    gravityPlot.reImPlot(w, objOis[k].visOi.bestFit.mean(axis = 0) - objOis[k].visOi.bestFitStar.mean(axis = 0), fig = fig)
    if SAVEFIG:
        plt.savefig("astrometryReport/planetfit_"+str(k)+".pdf")
        plt.close(fig)

    fig = plt.figure(figsize = (6, 6))
    gravityPlot.uvMap(objOis[k].visOi.uCoordMean, objOis[k].visOi.vCoordMean, targetCoord=[objOis[k].sObjX, objOis[k].sObjY], fig = fig, legend = objOis[k].basenames)
    plt.tight_layout()
    if SAVEFIG:
        plt.savefig("astrometryReport/uvmap_"+str(k)+".pdf")
        plt.close(fig)

    fig = plt.figure(figsize = (6, 6))
    plt.imshow(chi2Maps[k][0, :, :].T, origin = "lower", extent = [np.min(raValues), np.max(raValues), np.min(decValues), np.max(decValues)])
    plt.text(np.min(raValues), np.min(decValues)+0.1*(np.max(decValues)-np.min(decValues)), "RA: {:.2f}".format(raBest[k]), color = "w", weight = "bold")
    plt.text(np.min(raValues), np.min(decValues)+0.05*(np.max(decValues)-np.min(decValues)), "DEC: {:.2f}".format(decBest[k]), color = "w", weight = "bold")    
    plt.xlabel("$\Delta{}\mathrm{RA}$ (mas)")
    plt.ylabel("$\Delta{}\mathrm{DEC}$ (mas)")
    plt.tight_layout()
    if SAVEFIG:
        plt.savefig("astrometryReport/chi2Map_"+str(k)+".pdf")
        plt.close(fig)


stop()

f = open("astrometryReport/report.tex", "w")
f.write("\\documentclass[a4paper, landscape]{article}\n")
f.write("\\usepackage{graphicx}\n")
f.write("\\usepackage[left=1cm, right=1cm, top = 1cm, bottom = 1cm]{geometry}\n")
f.write("\\begin{document}\n")
f.write("\\includegraphics[width=0.3\linewidth]{astrometry.pdf}\n")
f.write("\\includegraphics[width=0.60\linewidth]{astrometry_time.pdf}\n")
f.write("\\clearpage\n")
for k in range(len(objOis)):
    f.write("\\includegraphics[width=0.3\linewidth]{uvmap_"+str(k)+".pdf}\n")
    f.write("\\includegraphics[width=0.60\linewidth]{starfit_"+str(k)+".pdf}\n")
    f.write("\n")
    f.write("\\includegraphics[width=0.3\linewidth]{chi2Map_"+str(k)+".pdf}\n")
    f.write("\\includegraphics[width=0.60\linewidth]{planetfit_"+str(k)+".pdf}\n")
    f.write("\n")
f.write("\\end{document}\n")
f.close()


plt.figure()
plt.plot(raBest, decBest, 'o')
for k in range(len(objOis)):
    plt.text(raBest[k], decBest[k], str(k))

stop

"""
w = np.zeros([oi.visOi.nchannel, oi.nwav])
for c in range(oi.visOi.nchannel):
    w[c, :] = oi.wav*1e6
    
m=1
fig = plt.figure()
gravityPlot.reImPlot(w, objOis[0].visOi.visRef[m, :, :], fig = fig, subtitles = objOis[0].basenames)
gravityPlot.reImPlot(w, bestFitStar[m, :, :], fig = fig)

fig = plt.figure()
gravityPlot.reImPlot(w, objOis[0].visOi.visRef[m, :, :]-bestFitStar[m, :, :], fig = fig, subtitles = objOis[0].basenames)
gravityPlot.reImPlot(w, (bestFit[m, :, :]-bestFitStar[m, :, :]), fig = fig)

"""
m=0
oi = objOis[m]
maps = np.copy(opdChi2Maps[m])
for dit in range(oi.visOi.ndit):
    for c in range(oi.visOi.nchannel):
        maps[dit, c, :] = (maps[dit, c, :] - np.min(maps[dit, c, :]))/(np.max(maps[dit, c, :]) - np.min(maps[dit, c, :]))
        
fig = plt.figure()
gravityPlot.waterfall(opdRanges[m, :, :], maps, fig = fig, subtitles = oi.basenames, xlabel = "OPD ($\mu\mathrm{m}$)")

plt.figure()
plt.imshow(chi2Maps[m].sum(axis = 0).T/(oi.visOi.ndit*oi.visOi.nchannel*oi.nwav), origin = "lower", extent = [np.min(raValues), np.max(raValues), np.min(decValues), np.max(decValues)])
plt.colorbar()



oinoref = gravity.GravityDualfieldAstrored(oi.filename, corrMet = "sylvestre", opdDispCorr = True, extension = 10)

fig = plt.figure(figsize = (11.5, 9))
gravityPlot.reImPlot(w, oinoref.visOi.visData[0, :, :], fig = fig, subtitles = oi.basenames, xlabel = "Wavelength ($\mu\mathrm{m}$)")
gravityPlot.reImPlot(w, oinoref.visOi.visRef[0, :, :], fig = fig)

fig = plt.figure(figsize = (11.5, 9))
gravityPlot.reImPlot(w, oinoref.visOi.visRef[0, :, :]+np.float('nan')+1j*np.float('nan'), fig = fig, subtitles = oi.basenames, xlabel = "Wavelength ($\mu\mathrm{m}$)")
gravityPlot.reImPlot(w, oinoref.visOi.visRef[0, :, :], fig = fig)
gravityPlot.reImPlot(w, oi.visOi.visRef[0, :, :], fig = fig)

fig = plt.figure(figsize = (11.5, 9))
gravityPlot.reImPlot(w, oi.visOi.visData.mean(axis = 0)+np.float('nan')+1j*np.float('nan'), fig = fig, subtitles = oi.basenames, xlabel = "Wavelength ($\mu\mathrm{m}$)")
gravityPlot.reImPlot(w, oi.visOi.visData.mean(axis = 0)+np.float('nan')+1j*np.float('nan'), fig = fig)
gravityPlot.reImPlot(w, oi.visOi.visRef.mean(axis = 0), fig = fig)
gravityPlot.reImPlot(w, oi.visOi.bestFitStar.mean(axis = 0), fig = fig)

fig = plt.figure(figsize = (11.5, 9))
gravityPlot.reImPlot(w, oi.visOi.visData.mean(axis = 0)+np.float('nan')+1j*np.float('nan'), fig = fig, subtitles = oi.basenames, xlabel = "Wavelength ($\mu\mathrm{m}$)")
gravityPlot.reImPlot(w, oi.visOi.visData.mean(axis = 0)+np.float('nan')+1j*np.float('nan'), fig = fig)
gravityPlot.reImPlot(w, oi.visOi.visRef.mean(axis = 0)-oi.visOi.bestFitStar.mean(axis = 0), fig = fig)
gravityPlot.reImPlot(w, oi.visOi.bestFit.mean(axis = 0) - oi.visOi.bestFitStar.mean(axis = 0), fig = fig)


w = np.zeros([oi.visOi.nchannel, oi.nwav])
for c in range(oi.visOi.nchannel):
    w[c, :] = oi.wav*1e6

pdf = PdfPages('starfits.pdf')

for k in range(len(objOis)):
    fig = plt.figure()
    oi = objOis[k]
    gravityPlot.reImPlot(w, oi.visOi.visRef.mean(axis = 0), fig = fig, subtitles = oi.basenames)
    oi.visOi.fitStarDitByDit(visRefs[0], order = STAR_ORDER)
    gravityPlot.reImPlot(w, oi.visOi.visStar.mean(axis = 0), fig = fig)
    pdf.savefig()
    plt.close(fig)

for k in range(len(objOis)):
    fig = plt.figure()
    oi = objOis[k]
    gravityPlot.reImPlot(w, oi.visOi.visRef.mean(axis = 0), fig = fig, subtitles = oi.basenames)
    pdf.savefig()
    plt.close(fig)
    
pdf.close()



fig = plt.figure(figsize = (12, 10))
gravityPlot.reImPlot(w, starOis[0].visOi.visRef.mean(axis = 0), fig = fig, subtitles = starOis[0].basenames)
for k in range(1, len(starOis)):
    gravityPlot.reImPlot(w, starOis[k].visOi.visRef.mean(axis = 0), fig = fig)


fig = plt.figure(figsize = (12, 10))
gravityPlot.modPhasePlot(w, starOis[0].visOi.visRef.mean(axis = 0), fig = fig, subtitles = starOis[0].basenames)
for k in range(1, len(starOis)):
    gravityPlot.modPhasePlot(w, starOis[k].visOi.visRef.mean(axis = 0), fig = fig)


###
#wav, flux, fluxCov, contrast, contrastCov = loadFitsSpectrum("/home/mnowak/Documents/exoGravity/spectra/BetaPictorisc_2020-03-08_ADI.fits")
wav, flux, fluxCov, contrast, contrastCov = loadFitsSpectrum("/home/mnowak/Documents/exoGravity/spectra/BetaPictorisb_2018-09-22.fits")
contrast = np.interp(oi.wav*1e6, wav, contrast)
oi = objOis[0]
plt.figure()
for c in range(oi.visOi.nchannel):
    freq = oi.visOi.freq[:, c, :].mean(axis = 0)/1e6
    amp = oi.dit/soi.dit*np.abs(soi.visOi.visRef[:, c, :]).mean()
    plt.semilogy(freq, np.abs(oi.dit/soi.dit*soi.visOi.visRef[:, c, :]).mean(axis = 0)/amp, 'C'+str(c))    
    plt.semilogy(freq, np.abs(oi.dit*oi.visOi.visRef[:, c, :]).mean(axis = 0)/amp, '-.C'+str(c))
#    plt.semilogy(freq, (np.abs(oi.dit*oi.visOi.visRef[:, c, :].mean(axis = 0) - oi.dit*oi.visOi.bestFitStar[:, c, :].mean(axis = 0)))/amp, '--C'+str(c))
    plt.semilogy(freq, np.abs(oi.dit/soi.dit*soi.visOi.visRef[:, c, :]).mean(axis = 0)/amp*contrast, 'C'+str(c))    
    plt.semilogy(freq, np.abs(oi.dit*oi.visOi.visRef[:, c, :]).mean(axis = 0)**0.5/(len(objOis))**0.5/amp, 'k--')

plt.legend(["On-star", "On-planet stellar residual", "On-planet planet flux", "Shot noise on stellar residual"])
    
plt.xlabel("Frequency (Mrad/s)")
plt.ylabel("Contrast")

plt.grid(which = "both")
plt.xlim(15, 65)
plt.ylim(1e-6, 2)
###

    

for k in cfg['planet_ois']:
    print(cfg['planet_ois'][k]['filename'])

chiMin = 0

superChi2Map = np.zeros([len(raValues), len(decValues)])
for k in range(len(objOis)):
    superChi2Map = superChi2Map + chi2Maps[k].sum(axis = 0)
    chiMin = chiMin+opdChi2Maps[k][:, :, 0].sum()
    
#superChi2Map[:, 0] = np.float('nan')
#superChi2Map[:, -1] = np.float('nan')
#superChi2Map[0, :] = np.float('nan')
#superChi2Map[-1, :] = np.float('nan')
#superChi2Map[-2, :] = np.float('nan')

#zmap = (233*12*5-(6*12*5+6+2))*
zmap = (chiMin-superChi2Map)/2
xfib = objOis[0].sObjX
yfib = objOis[0].sObjY

import matplotlib
axcol = 'lightgray'
matplotlib.rc('axes',edgecolor=axcol)
fig = plt.figure()
ax = fig.add_subplot(111)
p = ax.imshow(zmap.T, origin = 'lower', extent = [np.min(raValues), np.max(raValues), np.min(decValues), np.max(decValues)], aspect = 'equal', vmin = 0)
cbar = fig.colorbar(p)
cbar.set_label("Periodogram power")
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')    
ax.yaxis.label.set_color(axcol)
ax.xaxis.label.set_color(axcol)
ax.tick_params(axis='x', colors=axcol)
ax.tick_params(axis='y', colors=axcol)
r = 30
plt.xlim([xfib-r-1, xfib+r+1])
plt.ylim([yfib-r-1, yfib+r+1])

theta = np.linspace(0,2*np.pi,100);      # A vector of 100 angles from 0 to 2*pi
xCircle = xfib+r*1.0*np.cos(theta);              # x coordinates for circle
yCircle = yfib+r*1.0*np.sin(theta);              # y coordinates for circle
s = 1
xSquare = [xfib+r+s, xfib+r+s, xfib-r-s, xfib-r-s, xfib+r+s, xfib+r+s];         # x coordinates for square
ySquare = [yfib, yfib-r-s, yfib-r-s, yfib+r+s, yfib+r+s, yfib];         # y coordinates for square
hp = plt.fill(list(xCircle)+xSquare, list(yCircle)+ySquare, 'w');

ax.set_xlabel('$\Delta\mathrm{RA}$ (mas)')
ax.set_ylabel('$\Delta\mathrm{DEC}$ (mas)')
ax.xaxis.set_label_coords(0.75, 0.55)
ax.yaxis.set_label_coords(0.55, 0.75)

plt.plot(xCircle, yCircle, '--', color='gray')
#plt.plot(xfib+r*np.cos(theta)/2, yfib+r*np.sin(theta)/2, '--', color=axcol, alpha=0.5)

plt.text(xfib+r/2**0.5+1, yfib-r/2**0.5-1, "Fiber Field-of-view", ha='center', va='center', rotation = 45,color = 'gray')
#plt.text(xfib+r/2**0.5/2+1, yfib-r/2**0.5/2-1, "Half-injection", ha='center', va='center', rotation = 45, color = axcol)
#lt.xlabel("U coordinate ($\\times{}10^7 \mathrm{ rad}^{-1}$)")
#lt.ylabel("V coordinate ($\\times{}10^7 \mathrm{ rad}^{-1}$)")

# Clip

phaseRefFit = np.zeros([oi.visOi.ndit, oi.visOi.nchannel, oi.visOi.nwav])
for dit in range(oi.visOi.ndit):
    for c in range(oi.visOi.nchannel):
        phaseRefFit[dit, c, :] = np.polyval(oi.visOi.phaseRefCoeff[dit, c, :], oi.visOi.wav)
        
k=10

pref = oi.visOi.phaseRef[k, 3, :]
coeff = oi.visOi.phaseRefCoeff[k, 3, :]
#wavft = np.array(range(oi.visOi.nwavFt))
#wwft = (wavft.mean()/wavft - 1)/(wavft.max() - wavft.min())*wavft.mean()
w0 = oi.wav.mean()#*0+2.2e-6
ww = (w0/oi.wav - 1)/(np.max(oi.wav) - np.min(oi.wav))*w0

fig = plt.figure()
plt.plot(ww, pref)
plt.plot(ww, np.mod(-np.polyval(coeff[::-1], ww), 2*np.pi))

#plt.plot(wwft, np.angle(oi.visOi.visDataFt[k, 3, :]))
#plt.plot(wwft, np.polyval(coeff[::-1], wwft))


#fig = plt.figure()
#gPlot.baselinePlot(w, oi.visOi.phaseRef[k, :, :], subtitles = oi.basenames, fig = fig)
#gPlot.baselinePlot(w, phaseRefFit[k, :, :], fig = fig)

def calculatePhaseRefArclength(oi):
    # calculate normalized wavelength grid used to fit phaseRef
    w0 = oi.wav.mean()
    wref = (w0/oi.wav - 1)/(np.max(oi.wav) - np.min(oi.wav))*w0
    # prepare array for values
    phaseRefArclength = np.zeros([oi.visOi.ndit, oi.visOi.nchannel])
    for dit in range(oi.visOi.ndit):
        for c in range(oi.visOi.nchannel):
            coeffs = oi.visOi.phaseRefCoeff[dit, c, :]
            print(coeffs)
            # calculate the coeffs of 1+(dy/dx)**2
            arclengthpolycoeff = np.array([1+coeffs[1]**2, 4*coeffs[1]*coeffs[2], 4*coeffs[2]**2+6*coeffs[1]*coeffs[3], 12*coeffs[2]*coeffs[3], 9*coeffs[3]**2])
            # integrate sqrt(1+(dx/dy)**2) do get arclength. Minus sign because wref is decreasing
            phaseRefArclength[dit, c] = -np.trapz(np.sqrt(np.polyval(arclengthpolycoeff[::-1], ww)), wref)
    return phaseRefArclength


nwav = oi.nwav
wav = oi.wav
poly_order = 6
star = visRefs[0][0, :]
vref = objOis[0].visOi.visRef[0, 0, :]
nopd = 1000
opd = np.linspace(0, 4, nopd)

chi2 = np.zeros(nopd)
coeff = np.zeros(nopd)
for k in range(nopd):
    A = np.zeros([nwav, poly_order*2+1], 'complex')
    Y = np.zeros([2*nwav, 1], 'complex')
    Y[:, 0] = cs.conj_extended(vref)
    for p in range(poly_order):
        A[:, 2*p] = np.abs(star)*((wav-np.mean(wav))*1e6)**p
        A[:, 2*p+1] = 1j*np.abs(star)*((wav-np.mean(wav))*1e6)**p
    phase = 2*np.pi*opd[k]/(wav*1e6)
    A[:, -1] = np.exp(-1j*phase)*(0+np.abs(star))
    A2 = cs.conj_extended(A)
    print(np.min(np.linalg.svd(A2)[1])/np.max(np.linalg.svd(A2)[1]))
    X = np.dot(np.linalg.pinv(A2, rcond = 1e-6), Y)
    fit = np.dot(A2, X)
    realfit = np.real(fit[0:nwav, 0])
    imagfit = np.imag(fit[0:nwav, 0])
    chi2[k] = np.sum((np.real(vref)-realfit)**2+(np.imag(vref)-imagfit)**2)
    coeff[k] = X[-1, 0]

plt.figure()
plt.plot(opd, coeff)
plt.plot(values[0, :], params[0, 0, :, -1])
    
fig = plt.figure()
ax = fig.add_subplot(211)
ax.plot(oi.wav, np.real(vref)-realfit)
ax2 = fig.add_subplot(212)
ax2.plot(oi.wav, np.imag(vref)-imagfit)

fig = plt.figure()
opdChi2 = opdFits[0]["map"][0, 0, :]
grid = opdFits[0]["grid"][0, :]
plt.plot(grid, opdChi2)

plt.figure()
plt.plot(opd, chi2)



def findLocalMinimum(chi2Map, xstart, ystart, jump_size = 1):
    """ Find the local minimum closest to starting point on the given chi2Map """
    n, m = np.shape(chi2Map)
    borderedMap = np.zeros([n+2*jump_size, m+2*jump_size])+np.inf
    borderedMap[jump_size:-jump_size, jump_size:-jump_size] = chi2Map
    found = False
    x, y = xstart+jump_size, ystart+jump_size
    while not(found):
        submap = borderedMap[x-jump_size:x+jump_size+1, y-jump_size:y+jump_size+1]
        ind = np.where(submap == np.min(submap))
        print(ind)
        if (x, y) == (x+ind[0][0]-jump_size, y+ind[1][0]-jump_size):
            found = True
        else:
            x = x+ind[0][0]-jump_size
            y = y+ind[1][0]-jump_size
        print(x, y)
        plt.plot(x, y)
    return x-jump_size, y-jump_size

i0, j0 = np.where(raValues == raBest)[0][0], np.where(decValues == decBest)[0][0]
i, j = findLocalMinimum(chi2Maps[0], i0, j0, jump_size = 1)
ind = np.where(chi2Maps[0] == np.min(chi2Maps[0]))
iBest, jBest = ind[0][0], ind[1][0]

plt.figure()
plt.imshow(chi2Maps[0].T, origin = "lower")
plt.plot(i, j, "+C1")
plt.plot(iBest, jBest, "+C2")
