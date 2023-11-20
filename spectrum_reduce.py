#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Extract contrast spectrum from an exoGravity observation

This script is part of the exoGravity data reduction package.
The spectrum_reduce script is used to extract the contrast spectrum from an exoplanet observation made with GRAVITY, in dual-field mode.
To use this script, you need to call it with a configuration file, and an output directory, see example below.

Authors:
  M. Nowak, and the exoGravity team.
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
from exogravity.utils import * # utils from this exooGravity package
from exogravity.common import *
# other random stuffs
from astropy.time import Time
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
# argparse for command line arguments
import argparse
import distutils

# create the parser for command lines arguments
parser = argparse.ArgumentParser(description=
"""
Extract contrast spectrum from an exoGravity observation
""")

# required arguments are the path to the folder containing the data, and the path to the config yml file to write 
parser.add_argument('config_file', type=str, help="the path the to YAML configuration file.")

# some elements from the config file can be overridden with command line arguments
parser.add_argument("--figdir", metavar="DIR", type=str, default=argparse.SUPPRESS,
                    help="name of a directory where to store the output PDF files [overrides value from yml file]")   

parser.add_argument("--go_fast", metavar="EMPTY or TRUE/FALSE", type=lambda x:bool(distutils.util.strtobool(x)), default=argparse.SUPPRESS, nargs="?", const = True,
                    help="if set, average over DITs to accelerate calculations. [overrides value from yml file]")   

parser.add_argument("--save_residuals", metavar="EMPTY or TRUE/FALSE", type=lambda x:bool(distutils.util.strtobool(x)), default=argparse.SUPPRESS, nargs="?", const = True,
                     help="if set, saves fit residuals as npy files for further inspection. mainly a DEBUG option. [overrides value from yml file]")

parser.add_argument('--output', type=str, default="spectrum.fits", help="the name of output fits file where to save the spectrum.")


# IF BEING RUN AS A SCRIPT, LOAD COMMAND LINE ARGUMENTS
if __name__ == "__main__":   
    # load arguments into a dictionnary
    args = parser.parse_args()
    dargs = vars(args) # to treat as a dictionnary

    CONFIG_FILE = dargs["config_file"]
    if not(os.path.isfile(CONFIG_FILE)):
        raise Exception("Error: argument {} is not a file".format(CONFIG_FILE))

    # READ THE CONFIGURATION FILE
    if RUAMEL:
        cfg = ruamel.yaml.load(open(CONFIG_FILE, "r"), Loader=ruamel.yaml.RoundTripLoader)
    else:
        cfg = yaml.safe_load(open(CONFIG_FILE, "r"))

    # override some values with command line arguments
    for key in dargs:
        if key in cfg["general"]:
            cfg["general"][key] = dargs[key]

    SPECTRUM_FILENAME = dargs["output"]
    
# IF THIS FILE IS RUNNING AS A MODULE, WE WILL TAKE CONFIGURATION FILE FROM THE PARENT MODULE
if __name__ != "__main__":
    import exogravity
    cfg = exogravity.cfg
    SPECTRUM_FILENAME = "spectrum.fits"


#######################
# START OF THE SCRIPT #
#######################
DATA_DIR = cfg["general"]["datadir"]
PHASEREF_MODE = cfg["general"]["phaseref_mode"]
CONTRAST_FILE = cfg["general"]["contrast_file"]
NO_INV = cfg["general"]["noinv"]
GO_FAST = cfg["general"]["go_fast"]
REFLAG = cfg['general']['reflag']
FIGDIR = cfg["general"]["figdir"]
PLANET_FILES = [DATA_DIR+cfg["planet_ois"][list(preduce.keys())[0]]["filename"] for preduce in cfg["general"]["reduce_planets"]] # list of files corresponding to planet observations
PLANET_REJECT_DITS = [preduce[list(preduce.keys())[0]]["reject_dits"] for preduce in cfg["general"]["reduce_planets"]]
PLANET_REJECT_BASELINES = [preduce[list(preduce.keys())[0]]["reject_baselines"] for preduce in cfg["general"]["reduce_planets"]]
if not("swap_ois" in cfg.keys()):
    SWAP_FILES = []
elif cfg["swap_ois"] is None:
    SWAP_FILES = []
else:
    SWAP_FILES = [DATA_DIR+cfg["swap_ois"][k]["filename"] for k in cfg["swap_ois"].keys()]
POLY_ORDER = cfg["general"]["poly_order"]
STAR_DIAMETER = cfg["general"]["star_diameter"]
EXTENSION = cfg["general"]["extension"]
REDUCTION = cfg["general"]["reduction"]

# FILTER DATA
PHASEREF_ARCLENGTH_THRESHOLD = cfg["general"]["phaseref_arclength_threshold"]
FT_FLUX_THRESHOLD = cfg["general"]["ft_flux_threshold"]

# FIGDIR is now a compulsory keyword. Throw an error if it is not in the config file
if FIGDIR is None:
    printerr("figdir not given in the config file. figdir is now required")
    stop()
    
# LOAD GRAVITY PLOT is savefig requested
if not(FIGDIR is None):
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages        
    from cleanGravity import gravityPlot as gPlot
    if not(os.path.isdir(FIGDIR)):
        os.makedirs(FIGDIR)
        printinf("Directory {} was not found and has been created".format(FIGDIR))

# THROW AN ERROR IF save_residuals requested, but not FIGDIR provided
if cfg["general"]['save_residuals'] and (FIGDIR is None):
    printerr("save_residuals option requested, but not figdir provided.")
        
# extract the astrometric solutions
try:
    RA_SOLUTIONS = [preduce[list(preduce.keys())[0]]["astrometric_solution"][0] for preduce in cfg["general"]["reduce_planets"]]
    DEC_SOLUTIONS = [preduce[list(preduce.keys())[0]]["astrometric_solution"][1] for preduce in cfg["general"]["reduce_planets"]]
except:
    printinf("Please run the script 'astrometryReduce' first, or provide RA DEC values as arguments:")
    stop()

objOis = []

sFluxOis = []
oFluxOis = []

# LOAD DATA
t = time.time()
# LOAD DATA
for filename in PLANET_FILES:
    printinf("Loading file "+filename)
    if REDUCTION == "astrored":
        oi = gravity.GravityDualfieldAstrored(filename, corrMet = cfg["general"]["corr_met"], extension = EXTENSION, corrDisp = cfg["general"]["corr_disp"], reflag = REFLAG)
        printinf("File is on planet. FT coherent flux: {:.2e}".format(np.mean(np.abs(oi.visOi.visDataFt))))                        
    elif REDUCTION == "dualscivis":
        oi = gravity.GravityDualfieldScivis(filename, extension = EXTENSION)
        printinf("File is on planet")
    else:
        printerr("Unknonwn reduction type '{}'.".format(REDUCTION))        
    objOis.append(oi)

if REFLAG:
    printinf("REFLAG is set, and points have been reflagged according to the REFLAG column of the fits file")

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

printinf("Normalizing on-star visibilities to FT coherent flux.")
for oi in objOis:
    oi.visOi.scaleVisibilities(1.0/np.abs(oi.visOi.visDataFt).mean(axis = -1))
    
# replace data by the mean over all DITs if go_fast has been requested in the config file
if (GO_FAST):
    printinf("gofast flag is set. Averaging over DITs")            
    for oi in objOis: # mean should not be calculated on swap before phase correction
        printinf("Averaging file {}".format(oi.filename))
        oi.visOi.recenterPhase(oi.sObjX, oi.sObjY)
        oi.computeMean()
        oi.visOi.recenterPhase(-oi.sObjX, -oi.sObjY)
        
# calculate the very useful w for plotting
oi = objOis[0]
w = np.zeros([oi.visOi.nchannel, oi.nwav])
for c in range(oi.visOi.nchannel):
    w[c, :] = oi.wav*1e6

# calculate the reference visibility for each OB
printinf("Retrieving visibility references from fits files")
visRefs = [oi.visOi.visRef.mean(axis=0)*0 for oi in objOis]
for k in range(len(objOis)):
    oi = objOis[k]
    try:
        visRefs[k] = fits.getdata(oi.filename, "EXOGRAV_VISREF").field("EXOGRAV_VISREF")
    except:
        printerr("Cannot find visibility reference (EXOGRAV_VISREF) in file {}".format(oi.filename))

# subtract the reference phase to each OB
printinf("Subtracting phase reference to each planet OI.")
for k in range(len(objOis)):
    oi = objOis[k]
    oi.visOi.addPhase(-np.angle(visRefs[k]))

# calculate the phi values
printinf("Calculating the phi values")
for k in range(len(objOis)):
    oi = objOis[k]
    oi.visOi.phi_values = np.zeros([oi.visOi.ndit, oi.visOi.nchannel, oi.nwav], 'complex64')
    this_u = (RA_SOLUTIONS[k]*oi.visOi.uCoord + DEC_SOLUTIONS[k]*oi.visOi.vCoord)/1e7
    phase = 2*np.pi*this_u*1e7/3600.0/360.0/1000.0*2*np.pi
    for dit in range(oi.visOi.ndit):
        for c in range(oi.visOi.nchannel):
            oi.visOi.phi_values[dit, c, :] = np.exp(1j*phase[dit, c, :])

# create projector matrices
for k in range(len(objOis)):
    oi = objOis[k]
    printinf("Create projector matrices (p_matrices) ({}/{})".format(k+1, len(objOis)))
    # calculate the projector
    wav = oi.wav*1e6
    vectors = np.zeros([POLY_ORDER+1, oi.nwav], 'complex64')
    thisVisRef = visRefs[k]
    thisAmpRef = np.abs(thisVisRef)
    oi.visOi.p_matrices = np.zeros([oi.visOi.ndit, oi.visOi.nchannel, oi.nwav, oi.nwav], 'complex64')
    P = np.zeros([oi.visOi.nchannel, oi.nwav, oi.nwav], 'complex64')
    for dit in range(oi.visOi.ndit): # the loop of DIT number is necessary because of bad points mgmt (bad_indices depends on dit)
        for c in range(oi.visOi.nchannel):
            for j in range(POLY_ORDER+1):
                vectors[j, :] = np.abs(thisAmpRef[c, :])*(wav-np.mean(wav))**j # pourquoi ampRef et pas visRef ?
            bad_indices = np.where(oi.visOi.flag[dit, c, :])
            vectors[:, bad_indices] = 0
            for l in range(oi.nwav):
                x = np.zeros(oi.nwav)
                x[l] = 1
                coeffs = np.linalg.lstsq(vectors.T, x, rcond=-1)[0]
                P[c, :, l] = x - np.dot(vectors.T, coeffs)
            oi.visOi.p_matrices[dit, c, :, :] = np.dot(np.diag(oi.visOi.phi_values[dit, c, :]), np.dot(P[c, :, :], np.diag(oi.visOi.phi_values[dit, c, :].conj())))

# change frame
printinf("Switching on-planet visibilities to planet reference frame")
for k in range(len(objOis)):
    oi = objOis[k]
    oi.visOi.recenterPhase(RA_SOLUTIONS[k], DEC_SOLUTIONS[k], radec = True)

# project visibilities
printinf("Projecting visibilities orthogonally to speckle noise")
for k in range(len(objOis)):
    oi = objOis[k]
    for dit in range(oi.visOi.ndit):
        for c in range(oi.visOi.nchannel):
            bad_indices = np.where(oi.visOi.flag[dit, c, :])
            oi.visOi.visRef[dit, c, bad_indices] = 0        
            oi.visOi.visRef[dit, c, :] = np.dot(oi.visOi.p_matrices[dit, c, :, :], oi.visOi.visRef[dit, c, :])
    oi.visOi.visRefMean = oi.visOi.visRef.mean(axis = 0)

# estimate the covariance matrices
#W_obs = []
#Z_obs = []
#for k in range(len(objOis)):
#    oi = objOis[k]
#    visRef = oi.visOi.visRef
#    s = np.shape(visRef)
#    visRef = np.reshape(visRef, [s[0], s[1]*s[2]])
#    W_obs.append(cs.cov(visRef.T))
#    Z_obs.append(cs.pcov(visRef.T))

# calculate all H matrices
printinf("Starting calculation of H matrices")
counter = 1
for k in range(len(objOis)):
    printinf("Calculating H ({:d}/{:d})".format(counter, len(objOis)))
    counter = counter+1
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

# TODO: is it useful?
"""
# if injection data available, load them
if os.path.isfile('injection.npy'):
    inj_coeffs = np.load("injection.npy")
else:
    inj_coeffs = None    
inj_coeffs = None

# calculate chromatic coupling from coefficients and fiber offset
sObjX = np.array([oi.sObjX for oi in objOis])
sObjY = np.array([oi.sObjY for oi in objOis])
offsets = np.sqrt((ra - sObjX)**2+(dec - sObjY)**2)
"""
chromatic_coupling = np.zeros([len(objOis), objOis[0].nwav])
ref_coupling = 1#np.polyval(chromatic_coefficients[0, ::-1], 0)
for k in range(len(objOis)):
   # offset = offsets[k]
   # a0 = np.polyval(chromatic_coefficients[0, ::-1], offset)
   # a1 = np.polyval(chromatic_coefficients[1, ::-1], offset)
   # a2 = np.polyval(chromatic_coefficients[2, ::-1], offset)
    chromatic_coupling[k, :] = 1#np.polyval([a2/ref_coupling, a1/ref_coupling, a0/ref_coupling], objOis[k].wav*1e6 - 2.2)
### END OF TODO
    
# calculate the resolved stellar visibility function
resolved_star_models = []
for k in range(len(objOis)):
    oi = objOis[k]
    model = np.zeros([oi.visOi.ndit, oi.visOi.nchannel, oi.nwav])
    if STAR_DIAMETER == 0:
        model = model+1
    else:
        model = 2*scipy.special.j1(np.pi*STAR_DIAMETER*oi.visOi.freq)/(np.pi*STAR_DIAMETER*oi.visOi.freq)
    resolved_star_models.append(model)
    
# big H and big U
nwav = objOis[0].nwav
nb = objOis[0].visOi.nchannel
nt = len(objOis)
mtot = int(np.sum([np.sum(oi.visOi.m) for oi in objOis]))
H = np.zeros([mtot, nwav], 'complex64')
Y = np.zeros([mtot], 'complex64')
r = 0
for k in range(len(objOis)):
    oi = objOis[k]
    for dit in range(oi.visOi.ndit):
        for c in range(oi.visOi.nchannel):
            m = int(oi.visOi.m[dit, c])
            # TODO: useful?
#            if (inj_coeffs is None):
            G = np.diag(np.abs(visRefs[k][c, :])/resolved_star_models[k][dit, c, :]*chromatic_coupling[k, :])
#            else:
#                G = np.diag((inj_coeffs[k][dit, c, 0]*np.abs(ampRefs[k][c, :])+inj_coeffs[k][dit, c, 1]*(oi.wav - np.min(oi.wav))*1e6*np.abs(ampRefs[k][c, :]))/re#solved_star_models[k][dit, c, :]*chromatic_coupling[k, :])
            H[r:(r+m), :] = np.dot(oi.visOi.h_matrices[dit, c, 0:m, :], G)
            # filter bad datapoints by setting the corresponding columns of H to 0
            indx = np.where(oi.visOi.flag[dit, c, :])
            H[r:(r+m), indx] = 0
            # Y is different from the paper, as we don't really use "U" here. So we don't divide by abs(visRefs[k]), and we add G in the definition of H
            Y[r:r+m] = np.dot(oi.visOi.h_matrices[dit, c, 0:m, :], oi.visOi.visRef[dit, c, :])
            r = r+m

Y2 = cs.conj_extended(Y)
H2 = cs.conj_extended(H)

# block calculate W2inv*H2, cov(U2)*W2inv*H2, and pcov(U2)*W2inv*H2
nb = objOis[0].visOi.nchannel
nwav = objOis[0].nwav
W2invH2 = np.zeros([2*mtot, nwav], 'complex64')
covY2W2invH2 = np.zeros([2*mtot, nwav], 'complex64')
pcovY2W2invH2 = np.zeros([2*mtot, nwav], 'complex64')
r = 0
nwav = objOis[0].nwav
nb = objOis[0].visOi.nchannel
ndittot = np.sum(np.array([oi.visOi.ndit for oi in objOis]))
#Ginv = np.zeros([nb*nwav, nb*nwav], 'complex64')    

printinf("Starting Calculation of W2invH2")
counter = 1
for k in range(len(objOis)):
    oi = objOis[k]
    nchannel = oi.visOi.nchannel
    for dit in range(oi.visOi.ndit):
        printinf("Calculating W2invH2 ({:d}/{:d})".format(counter, ndittot))
        counter = counter+1
        this_dit_mtot = 0
        for c in range(nchannel):        
            m = int(oi.visOi.m[dit, c])
            this_dit_mtot = this_dit_mtot + m
#        W_elem = W_obs[k][0:this_dit_mtot, 0:this_dit_mtot]
#        Z_elem = Z_obs[k][0:this_dit_mtot, 0:this_dit_mtot]
        this_r = 0
        H_elem = np.zeros([this_dit_mtot, oi.visOi.nchannel*oi.nwav], 'complex64')        
        W_elem = np.zeros([this_dit_mtot, this_dit_mtot], 'complex64')
        W2_elem_inv = np.zeros([2*this_dit_mtot, 2*this_dit_mtot], 'complex64')
        Z_elem = np.zeros([this_dit_mtot, this_dit_mtot], 'complex64')
        for c in range(nchannel):
#            Omega_c_sp = scipy.sparse.csc_matrix(np.diag(oi.visOi.phi_values[dit, c, :]))
#            Omega[c*oi.nwav:(c+1)*oi.nwav, c*oi.nwav:(c+1)*oi.nwav] = Omega_c_sp.todense()
#            Ginv_c_sp = scipy.sparse.csc_matrix(np.diag(1.0/np.abs(visRefs[k][c, :])))
#            Ginv[c*oi.nwav:(c+1)*oi.nwav, c*oi.nwav:(c+1)*oi.nwav] = Ginv_c_sp.todense()
            m = int(oi.visOi.m[dit, c])            
            H_elem_c_sp = scipy.sparse.csc_matrix(oi.visOi.h_matrices[dit, c, 0:m, :])
            H_elem[this_r:this_r+m, c*oi.nwav:(c+1)*nwav] = H_elem_c_sp.todense()
#            Wc_sp = scipy.sparse.csr_matrix(np.diag(oi.visOi.visErr[dit, c, :].real**2+oi.visOi.visErr[dit, c, :].imag**2))
#            Zc_sp = scipy.sparse.csr_matrix(np.diag(oi.visOi.visErr[dit, c, :].real**2-oi.visOi.visErr[dit, c, :].imag**2))
            Wc_sp = scipy.sparse.csr_matrix(oi.visOi.visRefCov[dit, c].todense())
            Zc_sp = scipy.sparse.csr_matrix(oi.visOi.visRefPcov[dit, c].todense())
#            Wc_sp = (Ginv_c_sp.dot(Wc_sp)).dot(cs.adj(Ginv_c_sp))
#            Zc_sp = (Ginv_c_sp.dot(Zc_sp)).dot(Ginv_c_sp.T)
            W_elem_c = np.dot((H_elem_c_sp.dot(Wc_sp)).todense(), cs.adj(H_elem_c_sp).todense())
            Z_elem_c = np.dot((H_elem_c_sp.dot(Zc_sp)).todense(), H_elem_c_sp.T.todense())
            W_elem[this_r:this_r+m, this_r:this_r+m] = W_elem_c
            Z_elem[this_r:this_r+m, this_r:this_r+m] = Z_elem_c
            W2_elem_c = cs.extended_covariance(W_elem_c, Z_elem_c)
            W2_elem_c_inv = np.linalg.inv(W2_elem_c)
            W2_elem_inv[this_r:this_r+m, this_r:this_r+m] = W2_elem_c_inv[0:m, 0:m]
            W2_elem_inv[this_r:this_r+m, this_dit_mtot+this_r:this_dit_mtot+this_r+m] = W2_elem_c_inv[0:m, m:2*m]
            W2_elem_inv[this_dit_mtot+this_r:this_dit_mtot+this_r+m, this_r:this_r+m] = W2_elem_c_inv[m:2*m, 0:m]
            W2_elem_inv[this_dit_mtot+this_r:this_dit_mtot+this_r+m, this_dit_mtot+this_r:this_dit_mtot+this_r+m] = W2_elem_c_inv[m:2*m, m:2*m]
            this_r = this_r+m
#        Omega_sp = scipy.sparse.csc_matrix(Omega)
#        H_elem_sp = scipy.sparse.csc_matrix(H_elem)
#        Rcov = np.diag(oi.visOi.visErr[dit, :, :].real.reshape(oi.nwav*oi.visOi.nchannel)**2)
#        Icov = np.diag(oi.visOi.visErr[dit, :, :].imag.reshape(oi.nwav*oi.visOi.nchannel)**2)  
#        W_dit = np.dot(np.dot(Omega, Rcov+Icov), cs.adj(Omega))
#        Z_dit = np.dot(np.dot(Omega, Rcov-Icov), Omega.T)      
#        W_dit = np.dot(np.dot(Omega, W_dit), cs.adj(Omega))
#        Z_dit = np.dot(np.dot(Omega, Z_dit), cs.adj(Omega))
#        W_elem = np.dot(np.dot(H_elem, W_dit), cs.adj(H_elem))
#        Z_elem = np.dot(np.dot(H_elem, Z_dit), H_elem.T)
#        W2_elem = cs.extended_covariance(W_elem, Z_elem)                         
#        W2_elem_sp = scipy.sparse.csc_matrix(W2_elem)
#        W2invA2 = np.zeros([2*oi.nwav, 2*STAR_ORDER+1], 'complex64')
#        ZZ, _ = lapack.dpotrf(W2)
#        T, info = lapack.dpotri(ZZ)
#        W2_elem_inv = np.triu(T) + np.triu(T, k=1).T
#        W2_elem_inv = np.linalg.inv(W2_elem)
        [A, B, C, D] = [W2_elem_inv[0:this_dit_mtot, 0:this_dit_mtot], W2_elem_inv[0:this_dit_mtot, this_dit_mtot:], W2_elem_inv[this_dit_mtot:, 0:this_dit_mtot], W2_elem_inv[this_dit_mtot:, this_dit_mtot:]]
        W2invH2[r:r+this_dit_mtot, :] = np.dot(A, H[r:r+this_dit_mtot])+np.dot(B, H[r:r+this_dit_mtot].conj())
        W2invH2[mtot+r:mtot+r+this_dit_mtot, :] = np.dot(C, H[r:r+this_dit_mtot])+np.dot(D, H[r:r+this_dit_mtot].conj())
        covY2W2invH2[r:r+this_dit_mtot, :] = np.dot(W_elem, W2invH2[r:r+this_dit_mtot, :]) + np.dot(Z_elem, W2invH2[mtot+r:r+mtot+this_dit_mtot, :])
        covY2W2invH2[mtot+r:mtot+r+this_dit_mtot, :] = np.dot(cs.adj(Z_elem), W2invH2[r:r+this_dit_mtot, :]) + np.dot(np.conj(W_elem), W2invH2[mtot+r:r+mtot+this_dit_mtot, :])
        pcovY2W2invH2[r:r+this_dit_mtot, :] = np.dot(Z_elem, W2invH2[r:r+this_dit_mtot, :]) + np.dot(W_elem, W2invH2[mtot+r:r+mtot+this_dit_mtot, :])
        pcovY2W2invH2[mtot+r:mtot+r+this_dit_mtot, :] = np.dot(cs.adj(W_elem), W2invH2[r:r+this_dit_mtot, :]) + np.dot(np.conj(Z_elem), W2invH2[mtot+r:r+mtot+this_dit_mtot, :]) 
        r = r+this_dit_mtot

# calculate spectrum
printinf("Calculating contrast spectrum")
A = np.real(np.dot(cs.adj(Y2), W2invH2))
try:
    B = np.linalg.inv(np.real(np.dot(cs.adj(H2), W2invH2)))
except:
    printwar("Problem with the inversion of model. Is the matrix singular? Trying a pseudo inverse instead.")
    B = np.linalg.pinv(np.real(np.dot(cs.adj(H2), W2invH2)))
C = np.dot(A, B)

# calculate errors
printinf("Calculating covariance matrix")
covY2 = np.dot(cs.adj(W2invH2), covY2W2invH2)
pcovY2 = np.dot(W2invH2.T, pcovY2W2invH2)
Cerr = np.dot(B, np.dot(0.5*np.real(covY2+pcovY2), B.T))

# save raw contrast spectrum and diagonal error terms in a file
printinf("Writting spectrum in FITS file")
# build header 
headers = []
for filename in [cfg["general"]["datadir"]+"/"+cfg["planet_ois"][pkey]["filename"] for pkey in list(cfg["planet_ois"].keys())]:
    headers.append(fits.open(filename)[0].header)    
obstimes = [Time(hdr["DATE-OBS"]).mjd for hdr in headers]

first = headers[np.argmin(obstimes)]
last = headers[np.argmax(obstimes)]
resolution_mode = first["HIERARCH ESO INS SPEC RES"].lstrip().rstrip().lower()
if resolution_mode == "low":
    resolution = 50
elif resolution_mode == "medium":
    resolution = 500
elif resolution_mode == "high":
    resolution = 4000
else:
    resolution = None
header = {"DATE-OBS": Time(np.mean(obstimes), format = "mjd").iso,
          "INSTRU": "GRAVITY",
          "FACILITY": "ESO VLTI",
          "DATE": Time.now().iso,
          "AIRM-BEG": first["HIERARCH ESO ISS AIRM START"],
          "AIRM-END": last["HIERARCH ESO ISS AIRM END"],
          "TAU0-BEG": first["HIERARCH ESO ISS AMBI TAU0 START"],
          "TAU0-END": last["HIERARCH ESO ISS AMBI TAU0 END"],
          "FWHM-BEG": first["HIERARCH ESO ISS AMBI FWHM START"],
          "FWHM-END": last["HIERARCH ESO ISS AMBI FWHM END"],
          "PROGID": first["HIERARCH ESO OBS PROG ID"],
          "SPECRES": resolution,
          "SNR": (np.dot(np.dot(C, np.linalg.pinv(Cerr)), C)/len(wav))**0.5
          }
saveFitsSpectrum(FIGDIR+"/"+SPECTRUM_FILENAME, wav, C*0, Cerr*0, C, Cerr, header = header)

# now we want to recontruct the planet visibilities and the best fit obtained, in the original
# pipeline reference frame. The idea is that we want to compare the residuals to the pipeline
# original errors to check from problems and for cosmic rays
for k in range(len(objOis)):
    oi = objOis[k]
    # we'll store the recontructed planet visibilities and the fit directly in the oi object
    oi.visOi.visPlanet = np.zeros([oi.visOi.ndit, oi.visOi.nchannel, oi.nwav], "complex64")
    oi.visOi.visPlanetFit = np.zeros([oi.visOi.ndit, oi.visOi.nchannel, oi.nwav], "complex64")    
    for dit, c in itertools.product(range(oi.visOi.ndit), range(oi.visOi.nchannel)):
        oi.visOi.visPlanet[dit, c, :] = np.dot(oi.visOi.p_matrices[dit, c, :, :], oi.visOi.visRef[dit, c, :])
        oi.visOi.visPlanet[dit, c, :] = oi.visOi.visPlanet[dit, c, :]*np.exp(1j*np.angle(visRefs[k][c, :]))        
        oi.visOi.visPlanetFit[dit, c, :] = C*np.abs(visRefs[k])[c, :]
        bad_indices = np.where(oi.visOi.flag[dit, c, :])
        oi.visOi.visPlanetFit[dit, c, bad_indices] = 0
        oi.visOi.visPlanetFit[dit, c, :] = np.dot((oi.visOi.p_matrices[dit, c, :, :]), oi.visOi.visPlanetFit[dit, c, :])                
        oi.visOi.visPlanetFit[dit, c, :] = oi.visOi.visPlanetFit[dit, c, :]*np.exp(1j*np.angle(visRefs[k][c, :]))

# now we are going to calculate the residuals, and the distances in units of error bars
for k in range(len(objOis)):
    oi = objOis[k]
    oi.visOi.residuals = np.zeros([oi.visOi.ndit, oi.visOi.nchannel, oi.nwav], "complex64")
    oi.visOi.fitDistance = np.zeros([oi.visOi.ndit, oi.visOi.nchannel, oi.nwav])
    for dit, c in itertools.product(range(oi.visOi.ndit), range(oi.visOi.nchannel)):
        oi.visOi.residuals[dit, c, :] = oi.visOi.visPlanet[dit, c, :]-oi.visOi.visPlanetFit[dit, c, :]
        realErr = np.real(0.5*(np.diag(oi.visOi.visRefCov[dit, c].todense())+np.diag(oi.visOi.visRefPcov[dit, c].todense())))**0.5
        imagErr = np.real(0.5*(np.diag(oi.visOi.visRefCov[dit, c].todense())-np.diag(oi.visOi.visRefPcov[dit, c].todense())))**0.5   
        oi.visOi.fitDistance[dit, c, :] = np.abs(np.real(oi.visOi.residuals[dit, c])/realErr+1j*np.imag(oi.visOi.residuals[dit, c])/imagErr)

# now we want to save a new flag to indicate which points are badly fitted. But ONLY if gofast is set to False (we can't flag point if the individual DITs have been binned)
if (GO_FAST==False):
    printinf("Reflagging bad datapoints based on fit results (5 sigmas)")
    for k in range(len(objOis)):
        oi = objOis[k]
        reflag = np.zeros([oi.visOi.ndit, oi.visOi.nchannel, oi.nwav], 'bool')
        indx = np.where(oi.visOi.fitDistance > 5)
        reflag[indx] = True
        reflag = reflag | oi.visOi.flag
        oi.visOi.reflag = reflag
    # now we need to save this in the fits file. this is a tricky operation when only a subsets of the dits are used in each oi reduced, or worse if dits are splitted
    # because the fits file is shared between different objOis. So we need to concatenate the flags properly. Here we go.
    pkeys = [preduce[list(preduce.keys())[0]]["planet_oi"] for preduce in cfg["general"]["reduce_planets"]]  # list of planet_oi keys (p0, p1, etc.) used
    pkeys = list(set(pkeys)) # make it unique
    for pkey in pkeys:
        filename = DATA_DIR+cfg["planet_ois"][pkey]["filename"] # this is the fits file
        hdul = fits.open(filename, mode = "update")
        if "REFLAG" in [hdu.name for hdu in hdul]:
            hdul.pop([hdu.name for hdu in hdul].index("REFLAG"))        
        ndit = cfg["planet_ois"][pkey]["ndit"] # true number of dits
        reflag = np.zeros([ndit, objOis[0].visOi.nchannel, objOis[0].visOi.nwav])        
        inds = [r for r in range(len(PLANET_FILES)) if PLANET_FILES[r] == filename] # inds of the reduced ois corresponding tho this fits file
        for ind in inds: # we need to retrive the reflag from each of them
            oi = objOis[ind]
            reject_dits = PLANET_REJECT_DITS[ind]
            if (reject_dits is None): 
                reflag[:, :, :] = oi.visOi.reflag[:, :, :]
            else:
                for j in range(ndit):
                    if not(j in reject_dits):
                        reflag[j, :, :] = oi.visOi.reflag[j, :, :]        
        hdul.append(fits.BinTableHDU.from_columns([fits.Column(name="REFLAG", format = str(oi.nwav)+"L", array = reflag.reshape([oi.visOi.nchannel*ndit, oi.visOi.nwav]))], name = "REFLAG"))
        hdul.writeto(oi.filename, overwrite = "True")
        hdul.close()
        printinf("A total of {:d} bad points have be reflagged for file {}".format(len(indx[0]), oi.filename))

if not(FIGDIR is None):
    with PdfPages(FIGDIR+"/spectrum_fit_results.pdf") as pdf:
        fig = plt.figure(figsize=(10, 4))
        plt.plot(wav, C, '.-')
        plt.xlabel("Wavelength ($\mu\mathrm{m}$)")
        plt.ylabel("Contrast")
        pdf.savefig()                   
        for k in range(len(objOis)):
            oi = objOis[k]
            fig = plt.figure(figsize = (10, 8))
            gPlot.reImPlot(w, (oi.visOi.visPlanet*(1-oi.visOi.flag)).mean(axis = 0)*np.exp(-1j*np.angle(visRefs[k])), subtitles = oi.basenames, fig = fig)
            gPlot.reImPlot(w, oi.visOi.visPlanetFit.mean(axis = 0)*np.exp(-1j*np.angle(visRefs[k])), fig = fig)
            pdf.savefig()           
    
if cfg["general"]['save_residuals']:    
    for k in range(len(objOis)):
        oi = objOis[k]
        name = oi.filename.split('/')[-1].split('.fits')[0]
        visRes = (oi.visOi.visPlanet - oi.visOi.visPlanetFit)*np.exp(-1j*np.angle(visRefs[k]))
        np.save(FIGDIR+"/"+name+"_spectrum_residuals.npy", visRes)

        
