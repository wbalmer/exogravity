#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A script to extract the astrometry from an exoplanet observation
@author: mnowak
"""

# IMPORTS
import matplotlib.pyplot as plt
plt.ion()

import numpy as np
import scipy.linalg.lapack as lapack
import scipy.signal
import scipy.sparse, scipy.sparse.linalg
import astropy.io.fits as fits
from astropy import units as u 
import patiencebar
from datetime import datetime
import cleanGravity as gravity
import glob
from cleanGravity import complexstats as cs
from cleanGravity import gravityPlot
from matplotlib.backends.backend_pdf import PdfPages
import yaml
import sys, os
from config import dictToYaml

# arg should be the path to tge config yal file
args = sys.argv
if len(args) != 2:
    raise Exception("This script takes exactly one argument which should be the path_to_config.yal")

CONFIG_FILE = args[1]
if not(os.path.isfile(args[1])):
    raise Exception("Error: argument {} is not a file".format(CONFIG_FILE))


# READ THE CONFIGURATION FILE
cfg = yaml.load(open(CONFIG_FILE, "r"))
DATA_DIR = cfg["general"]["datadir"]
CONTRAST_FILE = cfg["general"]["contrast_file"]
NO_INV = cfg["general"]["no_inv"]
GO_FAST = cfg["general"]["go_fast"]
SAVEFIG = cfg["general"]["save_fig"]
PLANET_FILES = [DATA_DIR+cfg["planet_ois"][k]["filename"] for k in cfg["general"]["reduce"]] # list of files corresponding to planet exposures
STAR_ORDER = cfg["general"]["star_order"]
N_OPD = cfg["general"]["n_opd"]
N_RA = cfg["general"]["n_ra"]
N_DEC = cfg["general"]["n_dec"]
RA_LIM = cfg["general"]["ra_lim"]
DEC_LIM = cfg["general"]["dec_lim"]


# extract list of useful star ois from the list indicated in the star_indices fields of the config file:
star_indices = []
for k in cfg["general"]["reduce"]:
    star_indices = star_indices+cfg["planet_ois"][k]["star_indices"]
star_indices = list(set(star_indices)) # remove duplicates
STAR_FILES = [DATA_DIR+cfg["star_ois"][k]["filename"] for k in star_indices]

# read the contrast file if given, otherwise simply set it to 1
if CONTRAST_FILE is None:
    contrast = 1
else:
    data = np.loadtxt(CONTRAST_FILE).T
    contrast = data[:, 1]


# two lists to contain the planet and star OIs
starOis = []
objOis = []

# LOAD DATA
for filename in PLANET_FILES+STAR_FILES:
    oi = gravity.GravityDualfieldAstrored(filename, corrMet = "sylvestre", extension = 10, opdDispCorr = "sylvestre")
    # replace data by the mean over all DITs if go_fast has been requested in the config file
    if (GO_FAST):
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
    if filename in PLANET_FILES:
        objOis.append(oi)
    else:
        starOis.append(oi)

# calculate the very useful w for plotting
oi = objOis[0]
w = np.zeros([oi.visOi.nchannel, oi.nwav])
for c in range(oi.visOi.nchannel):
    w[c, :] = oi.wav*1e6

# create visRefs fromn star_indices indicated in the config file
visRefs = [oi.visOi.visRefMean*0 for oi in objOis]
ampRefs = [oi.visOi.visRefMean*0 for oi in objOis]
for k in range(len(objOis)):
    planet_ind = cfg["general"]["reduce"][k]
    for ind in cfg["planet_ois"][planet_ind]["star_indices"]:
        # not all star files have ben loaded, so we cannot trust the order and we need to explicitly look for the correct one
        starOis_ind = [soi.filename for soi in starOis].index(DATA_DIR+cfg["star_ois"][ind]["filename"]) 
        soi = starOis[starOis_ind]
        visRefs[k] = visRefs[k]+soi.visOi.visRefMean
        ampRefs[k] = ampRefs[k]+np.abs(soi.visOi.visRefMean)
    visRefs[k] = visRefs[k]/len(cfg["planet_ois"][k]["star_indices"])
    ampRefs[k] = ampRefs[k]/len(cfg["planet_ois"][k]["star_indices"])

# subtract the reference phase to each OB
for k in range(len(objOis)):
    oi = objOis[k]
    oi.visOi.addPhase(-np.angle(visRefs[k]))

# prepare chi2Maps
raValues = np.linspace(RA_LIM[0], RA_LIM[1], N_RA)
decValues = np.linspace(DEC_LIM[0], DEC_LIM[1], N_DEC)
chi2Maps = np.zeros([len(objOis), N_RA, N_DEC])

# calculate limits in OPD
opdLimits = np.zeros([len(objOis), objOis[0].visOi.nchannel, 2])
opdLimits[:, :, 0] = +np.inf
opdLimits[:, :, 1] = -np.inf

for k in range(len(objOis)):
    oi = objOis[k]
    for ra in raValues:
        for dec in decValues:
            opd = oi.visOi.getOpd(ra, dec)
            for c in range(oi.visOi.nchannel):
                if np.min(opd[:, c]) < opdLimits[k, c, 0]:
                    opdLimits[k, c, 0] = np.min(opd[:, c])
                if np.max(opd[:, c]) > opdLimits[k, c, 1]:
                    opdLimits[k, c, 1] = np.max(opd[:, c])

opdRanges = np.zeros([len(objOis), oi.visOi.nchannel, N_OPD])
for k in range(len(objOis)):
    oi = objOis[k]
    for c in range(oi.visOi.nchannel):
        opdRanges[k, c, :] = np.linspace(opdLimits[k, c, 0], opdLimits[k, c, 1], N_OPD)

# calculate maps in OPD
opdChi2Maps = []
bestOpdFits = []
nditsTot = np.sum(np.array([oi.visOi.ndit for oi in objOis]))
pBar = patiencebar.Patiencebar(valmax = nditsTot*oi.visOi.nchannel, title = "Calculating solutions...")
fit = np.zeros([6, 230], 'complex')
for k in range(len(objOis)):
    oi = objOis[k]
    opdChi2Map = np.zeros([oi.visOi.ndit, oi.visOi.nchannel, N_OPD]) 
    bestOpdFit = np.zeros([oi.visOi.ndit, oi.visOi.nchannel, oi.nwav], 'complex')   
    oi = objOis[k]
    visRef = visRefs[k]
#    visRef = totalVisRef
    for dit in range(oi.visOi.ndit):
        for c in range(oi.visOi.nchannel):
            pBar.update()
            # cov and pcov
            W = np.diag((np.real(oi.visOi.visErr[dit, c, :])**2+np.imag(oi.visOi.visErr[dit, c, :])**2)) 
            Z = np.diag((np.real(oi.visOi.visErr[dit, c, :])**2-np.imag(oi.visOi.visErr[dit, c, :])**2))
            W2 = cs.extended_covariance(W, Z).real
            if NO_INV:
                W2sp = scipy.sparse.csr_matrix(W2)
                W2invA2 = np.zeros([2*oi.nwav, 2*STAR_ORDER+1], 'complex')
            else:
                ZZ, _ = lapack.dpotrf(W2)
                T, info = lapack.dpotri(ZZ)
                W2inv = np.triu(T) + np.triu(T, k=1).T
            for j in range(N_OPD):
                opd = opdRanges[k, c, j]
                # calculate A matrix
                A = np.zeros([oi.nwav, STAR_ORDER*2+1], 'complex')
                for p in range(STAR_ORDER):
#                    A[:, 2*p] = np.abs(visRef[c, :])*((oi.wav.mean()/oi.wav-1)*10)**p#((oi.wav-np.mean(oi.wav))*1e6)**p
#                    A[:, 2*p+1] = 1j*np.abs(visRef[c, :])*((oi.wav.mean()/oi.wav-1)*10)**p#((oi.wav-np.mean(oi.wav))*1e6)**p
                    A[:, 2*p] = np.abs(visRef[c, :])*((oi.wav-np.mean(oi.wav))*1e6)**p
                    A[:, 2*p+1] = 1j*np.abs(visRef[c, :])*((oi.wav-np.mean(oi.wav))*1e6)**p                    
                # last column requires phase and contrast
                phase = 2*np.pi*opd/(oi.wav*1e6)
                A[:, -1] = np.exp(-1j*phase)*np.abs(visRef[c, :])*contrast
                # maximum likelihood explicit solution
                A2 = cs.conj_extended(A)
                if NO_INV:
                    for p in range(2*STAR_ORDER+1):
                        W2invA2[:, p] = scipy.sparse.linalg.spsolve(W2sp, A2[:, p])
                else:
                    W2invA2 = np.dot(W2inv, A2)
                left = np.real( np.dot(cs.adj(cs.conj_extended(oi.visOi.visRef[dit, c, :])), W2invA2) )
                right = np.real( np.dot(cs.adj(A2), W2invA2) )
                dzeta = np.transpose(np.dot(left, np.linalg.pinv(right)))
                if dzeta[-1] < 0:
                    dzeta[-1] = 0
                    A[:, -1] = 0
                    A2 = cs.conj_extended(A)
                    if NO_INV:
                        W2invA2[:, -1] = scipy.sparse.linalg.spsolve(W2sp, A2[:, -1])
                    else:
                        W2invA2 = np.dot(W2inv, A2)
                    left = np.real( np.dot(cs.adj(cs.conj_extended(oi.visOi.visRef[dit, c, :])), W2invA2) )
                    right = np.real( np.dot(cs.adj(A2), W2invA2) )
                    dzeta = np.transpose(np.dot(left, np.linalg.pinv(right)))
                # calculate the corresponding chi2
                vec = cs.conj_extended(oi.visOi.visRef[dit, c, :]) - np.dot(A2, dzeta)
                if NO_INV:
                    rightVec = scipy.sparse.linalg.spsolve(W2sp, vec)
                    opdChi2Map[dit, c, j] = np.real(np.dot(cs.adj(vec), rightVec))
                else:
                    opdChi2Map[dit, c, j] = np.real(np.dot(np.dot(cs.adj(vec), W2inv), vec))
                fit[c, :] = np.dot(A2, dzeta)[:230]
    opdChi2Maps.append(opdChi2Map)

# calculate the chi2Maps from the OPD maps
chi2Maps = []
pBar = patiencebar.Patiencebar(valmax = len(objOis), title = "Calculating chi2Map in RA/DEC...")
for k in range(len(objOis)):
    pBar.update()    
    oi = objOis[k]
    chi2Map = np.zeros([oi.visOi.ndit, N_RA, N_DEC])
    for i in range(N_RA):
        for j in range(N_DEC):
            ra = raValues[i]
            dec = decValues[j]
            opd = oi.visOi.getOpd(ra, dec)
            for dit in range(oi.visOi.ndit):
                for c in range(oi.visOi.nchannel):
                    chi2Map[dit, i, j] += np.interp(opd[dit, c], opdRanges[k, c, :], opdChi2Maps[k][dit, c, :])
    chi2Maps.append(chi2Map)

# calculate the combined chi2Map and best parameters for each OB
chi2Map = np.zeros([len(objOis), N_RA, N_DEC])
raBest = np.zeros(len(objOis))
decBest = np.zeros(len(objOis))
for k in range(len(objOis)):
    chi2Map[k, :, :] = np.sum(chi2Maps[k], axis = 0)
    (a, b) = np.where(chi2Map[k, :, :] == np.min(chi2Map[k, :, :]))
    raBest[k] = raValues[a]
    decBest[k] = decValues[b]

print("RA:", np.mean(raBest))
print("DEC:", np.mean(decBest))

for k in range(len(PLANET_FILES)):
    ind = cfg["general"]["reduce"][k]
    cfg["planet_ois"][ind]["astrometric_solution"] = [float(raBest[k]), float(decBest[k])] # YAML cannot convert numpy types
            
f = open(CONFIG_FILE, "w")
f.write(dictToYaml(cfg))
f.close()

stop

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
for k in range(len(objOis)):
    pBar.update()
    oi = objOis[k]
    bestFitStar = np.zeros([oi.visOi.ndit, oi.visOi.nchannel, oi.nwav], "complex")
    bestFit = np.zeros([oi.visOi.ndit, oi.visOi.nchannel, oi.nwav], "complex")    
    (a, b) = np.where(chi2Maps[k].sum(axis = 0) == np.min(chi2Maps[k].sum(axis = 0)))
    thisraBest = raValues[a]
    thisdecBest = decValues[b]
    opdBest = oi.visOi.getOpd(thisraBest, thisdecBest)
    for dit in range(oi.visOi.ndit):
        for c in range(0, oi.visOi.nchannel):
            opd = opdBest[dit, c]
            # calculate A matrix
            A = np.zeros([oi.nwav, STAR_ORDER*2+1], 'complex')
            for p in range(STAR_ORDER):
                A[:, 2*p] = np.abs(visRef[c, :])*((oi.wav-np.mean(oi.wav))*1e6)**p
                A[:, 2*p+1] = 1j*np.abs(visRef[c, :])*((oi.wav-np.mean(oi.wav))*1e6)**p
            A2 = cs.conj_extended(A)
            # cov and pcov
            W = np.diag((np.real(oi.visOi.visErr[dit, c, :])**2+np.imag(oi.visOi.visErr[dit, c, :])**2)) 
            Z = np.diag((np.real(oi.visOi.visErr[dit, c, :])**2-np.imag(oi.visOi.visErr[dit, c, :])**2))
            W2 = cs.extended_covariance(W, Z).real
            if NO_INV:
                W2sp = scipy.sparse.csr_matrix(W2)
                W2invA2 = np.zeros([2*oi.nwav, 2*STAR_ORDER+1], 'complex')
            else:
                ZZ, _ = lapack.dpotrf(W2)
                T, info = lapack.dpotri(ZZ)
                W2inv = np.triu(T) + np.triu(T, k=1).T
            # maximum likelihood explicit solution
            if NO_INV:
                for p in range(2*STAR_ORDER+1):
                    W2invA2[:, p] = scipy.sparse.linalg.spsolve(W2sp, A2[:, p])
            else:
                W2invA2 = np.dot(W2inv, A2)
            left = np.real( np.dot( cs.adj(cs.conj_extended(oi.visOi.visRef[dit, c, :])), W2invA2 ))
            right = np.real( np.dot( cs.adj(A2), W2invA2 ) )
            dzeta = np.transpose(np.dot(left, np.linalg.pinv(right)))
            # calculate the corresponding chi2
            bestFitStar[dit, c, :] = np.dot(A2, dzeta)[0:oi.nwav]
            # last column requires phase and contrast
            phase = 2*np.pi*opd/(oi.wav*1e6)
            A[:, -1] = np.exp(-1j*phase)*np.abs(visRef[c, :])
            A2 = cs.conj_extended(A)       
            if NO_INV:
                W2invA2[:, -1] = scipy.sparse.linalg.spsolve(W2sp, A2[:, -1])
            else:
                W2invA2 = np.dot(W2inv, A2)     
            # maximum likelihood explicit solution
            left = np.real( np.dot( cs.adj(cs.conj_extended(oi.visOi.visRef[dit, c, :])), W2invA2 ) )
            right = np.real( np.dot( cs.adj(A2), W2invA2 ) )
            dzeta = np.transpose(np.dot(left, np.linalg.pinv(right)))
            # calculate the corresponding chi2
            bestFit[dit, c, :] = np.dot(A2, dzeta)[0:oi.nwav]
    oi.visOi.bestFit = bestFit
    oi.visOi.bestFitStar = bestFitStar


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

stop

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
oi = objOis[0]
plt.figure()
for c in range(oi.visOi.nchannel):
    freq = oi.visOi.freq[:, c, :].mean(axis = 0)/1e6
    plt.semilogy(freq, np.abs(oi.visOi.visRef[:, c, :]).mean(axis = 0), 'C'+str(c))
    plt.semilogy(freq, np.abs(oi.visOi.visRef[:, c, :].mean(axis = 0) - oi.visOi.bestFitStar[:, c, :].mean(axis = 0)), '--C'+str(c))
    plt.semilogy(freq, np.abs(oi.visOi.visRef[:, c, :]).mean(axis = 0)**0.5/len(objOis)**0.5, 'k--')

plt.legend(["On-planet stellar residual", "On-planet planet flux", "Shot noise on stellar residual"])
    
plt.xlabel("Frequency (Mrad/s)")
plt.ylabel("Amplitude (ADU)")
###

    
