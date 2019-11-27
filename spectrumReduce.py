#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
A class to load the data from a gravity .astrored file.
@author: mnowak
"""

LOCAL_SYSTEM = 'denebola' # the name of the local system, used to avoid interactive behavior on remote machines 

import os
if os.uname()[1] == LOCAL_SYSTEM:
    import matplotlib.pyplot as plt
    plt.ion()
else:
    import matplotlib as mpl
    mpl.use('Agg')    
    import matplotlib.pyplot as plt

import numpy as np
import scipy.signal
import scipy.special
import astropy.io.fits as fits
from astropy import units as u 
import patiencebar
from datetime import datetime
import cleanGravity as gravity
from cleanGravity import complexstats as cs
from cleanGravity import gravityPlot
import glob
import exosed

star_diameter = 0.03#
star_diameter = star_diameter/360.0/3600.0/1000.0*2*np.pi # convert to rad

#data = np.loadtxt("spectrum.txt").T
#star = data[:, 1]/S

fake = False
goFast = True


# HD206893
datadir = "/data1/gravity/HD206893/2019-07-17/"
#datadir = "/data1/gravity/PDS70/"
datafiles = glob.glob(datadir+'/GRAVI*astrored*.fits')
datafiles.sort()
ra = np.array([130.85151515, 131.05353535, 131.25555556, 130.85151515])
dec = np.array([197.82747475, 197.72646465, 197.32242424, 198.02949495])

#ra = np.array([127.31616162, 127.31616162, 127.01313131, 126.81111111, 127.01313131])
#dec = np.array([199.03959596, 198.73656566, 199.34262626, 199.54464646, 199.54464646])

#ra = np.array([130.85151515, 131.05353535, 131.25555556, 130.85151515])*0+131.00
#dec = np.array([197.82747475, 197.72646465, 197.32242424, 198.02949495])*0+197.72

star_spectrum="./sylvestre/lte076-4.0-0.0a+0.0.BT-NextGen.7"
d = open(star_spectrum, "r")
lines=d.readlines()[10:99900]
if len(lines[0].split()[0]) < 22:
    val=np.double(np.array([(l.split()[0].replace('D','E'), l.split()[1].replace('D','E')) for l in lines[:-1]]))
#else:
#    N=len(lines[0].split('D')[0])
#    val=np.doublefloat(np.array([(l[:N-8].replace('D','E'),l[N-8:N+4].replace('D','E')) for l in lines[:-1]]))
#    print("fixing BT Settl data")
log = False
if log:
    val[:,1]=10**(val[:,1]-8)*1e-7
else:
    val[:, 1] = val[:, 1]*1e-7
val[:,0]*=1e-4
val=val[val[:,0]>1.5]
val=val[val[:,0]<2.8]
starSpectrum = exosed.Spectrum(val[:, 0], val[:, 1])

"""
# beta pic
datafiles = glob.glob('./betaPictoris/2018-09-21_allwavelengths/GRAVI*astrored*.fits')
datafiles.sort()
astrometry = np.loadtxt("./betaPictoris/results_9773/astrometry")
ra = astrometry[:, 0]
dec = astrometry[:, 1]
# star spectrum for normalisation
hdu = fits.open("./betaPictoris/spectra/Mickael/Synthetic_spectrum_BpicA.fits")[0]
data = hdu.data
ind = np.where((data[:, 0] < 2.6) & (data[:, 0] > 1.6))[0]
starSpectrum = exosed.Spectrum(data[ind, 0], data[ind, 1])
"""

order = 3

# PDS70
#ra = [-215.02]*6
#dec = [32.34]*6

starOis = []
objOis = []

sFluxOis = []
oFluxOis = []

for k in range(len(datafiles)):
    filename = datafiles[k]
    oi = gravity.GravityDualfieldAstrored(filename, corrMet = "sylvestre", extension = 10, opdDispCorr = "sylvestre")
    if (goFast):
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
    if oi.sObjX**2+oi.sObjY**2 < 10:
        oi.visOi.computeMean()
        starOis.append(oi)
    else:
        oi.visOi.computeMean()        
        objOis.append(oi)

oi = objOis[0]
oi.visOi.visRef = oi.visOi.visRef

objOis = objOis[0:2]+objOis[3:]
#objOis = objOis[0:-1]

# PDS70
#ra = [-215.02]*6
#dec = [32.34]*6

#ra = [126.9]*len(objOis)
#dec = [200.5]*len(objOis)

chromatic_coefficients = np.load("chromatic_coefficients.npy")

# PDS70
#data = np.loadtxt("SpeX_SED.txt", skiprows = 1)
#star_wav = data[:, 0]
#star_spectrum = data[:, 1]

#star_wav = objOis[0].wav*1e6
#star_spectrum = np.ones(objOis[0].nwav)


# create visRefs
visRefs = [oi.visOi.visRefMean*0 for oi in objOis]
ampRefs = [oi.visOi.visRefMean*0 for oi in objOis]
for k in range(len(objOis)):
    oi = objOis[k]
    oiAfter = oi.getClosestOi(starOis, forceAfter = True)
    oiBefore = oi.getClosestOi(starOis, forceBefore = True)
    visRefs[k] = 0.5*(oiAfter.visOi.visRefMean + oiBefore.visOi.visRefMean)
    ampRefs[k] = 0.5*(np.abs(oiAfter.visOi.visRefMean) + np.abs(oiBefore.visOi.visRefMean))

totalVisRef = oi.visOi.visRefMean*0
for k in range(len(starOis)):
    oi = starOis[k]
    totalVisRef = totalVisRef + oi.visOi.visRefMean
totalVisRef = totalVisRef/len(starOis)

# subtract the reference phase to each OB
for k in range(len(objOis)):
    oi = objOis[k]
#    oi.visOi.addPhase(-np.angle(totalVisRef))
    oi.visOi.addPhase(-np.angle(visRefs[k]))

# calculate the phi values
for k in range(len(objOis)):
    oi = objOis[k]
    oi.visOi.phi_values = np.zeros([oi.visOi.ndit, oi.visOi.nchannel, oi.nwav], 'complex64')
    this_u = (ra[k]*oi.visOi.uCoord + dec[k]*oi.visOi.vCoord)/1e7
    phase = 2*np.pi*this_u*1e7/3600.0/360.0/1000.0*2*np.pi
    for dit in range(oi.visOi.ndit):
        for c in range(oi.visOi.nchannel):
            oi.visOi.phi_values[dit, c, :] = np.exp(1j*phase[dit, c, :])

"""
for k in range(len(starOis)):
    soi = starOis[k]
    soi.visOi.phi_values = np.zeros([soi.visOi.ndit, soi.visOi.nchannel, soi.nwav], 'complex64')
    this_u = (ra[k]*soi.visOi.uCoord + dec[k]*soi.visOi.vCoord)/1e7
    phase = 2*np.pi*this_u*1e7/3600.0/360.0/1000.0*2*np.pi
    for dit in range(soi.visOi.ndit):
        for c in range(soi.visOi.nchannel):
            soi.visOi.phi_values[dit, c, :] = np.exp(1j*phase[dit, c, :])
"""

# create projector matrices
for k in range(len(objOis)):
    oi = objOis[k]
    # calculate the projector
    wav = oi.wav*1e7
    vectors = np.zeros([order+1, oi.nwav], 'complex64')
    thisVisRef = visRefs[k]
    thisAmpRef = np.abs(thisVisRef)
    oi.visOi.visStar = np.zeros([oi.visOi.ndit, oi.visOi.nchannel, oi.nwav], 'complex64')
    oi.visOi.p_matrices = np.zeros([oi.visOi.ndit, oi.visOi.nchannel, oi.nwav, oi.nwav], 'complex64')
    P = np.zeros([oi.visOi.nchannel, oi.nwav, oi.nwav], 'complex64')
    for c in range(oi.visOi.nchannel):
        for j in range(order+1):
            vectors[j, :] = np.abs(thisAmpRef[c, :])*(wav-np.mean(wav))**j # pourquoi ampRef et pas visRef ?
        for l in range(oi.nwav):
            x = np.zeros(oi.nwav)
            x[l] = 1
            coeffs = np.linalg.lstsq(vectors.T, x)[0]
            P[c, :, l] = x - np.dot(vectors.T, coeffs)
        for dit in range(oi.visOi.ndit):
            oi.visOi.p_matrices[dit, c, :, :] = np.dot(np.diag(oi.visOi.phi_values[dit, c, :]), np.dot(P[c, :, :], np.diag(oi.visOi.phi_values[dit, c, :].conj())))

# change frame
for k in range(len(objOis)):
    oi = objOis[k]
    oi.visOi.recenterPhase(ra[k], dec[k], radec = True)

# project visibilities
for k in range(len(objOis)):
    oi = objOis[k]
    for dit in range(oi.visOi.ndit):
        for c in range(oi.visOi.nchannel):
            oi.visOi.visRef[dit, c, :] = np.dot(oi.visOi.p_matrices[dit, c, :, :], oi.visOi.visRef[dit, c, :])
    oi.visOi.visRefMean = oi.visOi.visRef.mean(axis = 0)

# estimate the covariance matrices
W_obs = []
Z_obs = []
for k in range(len(objOis)):
    oi = objOis[k]
    visRef = oi.visOi.visRef
    s = np.shape(visRef)
    visRef = np.reshape(visRef, [s[0], s[1]*s[2]])
    W_obs.append(cs.cov(visRef.T))
    Z_obs.append(cs.pcov(visRef.T))

# fake data
if fake == True:
    for k in range(len(objOis)):
        oi = objOis[k]
        for dit in range(oi.visOi.ndit):
            for c in range(oi.visOi.nchannel):
                oi.visOi.visRef[dit, c, :] = oi.visOi.visRef[dit, c, :] - np.dot(oi.visOi.p_matrices[dit, c, :, :], oi.visOi.visRef[dit, c, :])
                oi.visOi.visRef[dit, c, :] = oi.visOi.visRef[dit, c, :] + S*np.abs(ampRefs[k][c, :]) 
        oi.visOi.visRefMean = oi.visOi.visRef.mean(axis = 0)

# calculate all H matrices
pbar = patiencebar.Patiencebar(title = "Calculating H matrices", valmax = len(objOis))
for k in range(len(objOis)):
    pbar.update()
    oi = objOis[k]
    oi.visOi.h_matrices = np.zeros([oi.visOi.ndit, oi.visOi.nchannel, oi.visOi.nwav, oi.visOi.nwav], 'complex64')
    oi.visOi.m = np.zeros([oi.visOi.ndit, oi.visOi.nchannel])
    for dit in range(oi.visOi.ndit):
        for c in range(oi.visOi.nchannel):
            (G, d, H) = np.linalg.svd(oi.visOi.p_matrices[dit, c, :, :])
            D = np.diag(d)
            oi.visOi.h_matrices[dit, c, :, :] = H
            m = int(np.sum(d))
            oi.visOi.m[dit, c] = m

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
chromatic_coupling = np.zeros([len(objOis), objOis[0].nwav])
ref_coupling = np.polyval(chromatic_coefficients[0, ::-1], 0)
for k in range(len(objOis)):
    offset = offsets[k]
    a0 = np.polyval(chromatic_coefficients[0, ::-1], offset)
    a1 = np.polyval(chromatic_coefficients[1, ::-1], offset)
    a2 = np.polyval(chromatic_coefficients[2, ::-1], offset)
    chromatic_coupling[k, :] = np.polyval([a2/ref_coupling, a1/ref_coupling, a0/ref_coupling], objOis[k].wav*1e6 - 2.2)

# calculate the resolved stellar visibility function
resolved_star_models = []
for k in range(len(objOis)):
    oi = objOis[k]
    model = np.zeros([oi.visOi.ndit, oi.visOi.nchannel, oi.nwav])
    model = 2*scipy.special.j1(np.pi*star_diameter*oi.visOi.freq)/(np.pi*star_diameter*oi.visOi.freq)
    resolved_star_models.append(model)
    
# big H and big U
nwav = objOis[0].nwav
nb = objOis[0].visOi.nchannel
nt = len(objOis)
mtot = int(np.sum([np.sum(oi.visOi.m) for oi in objOis]))
H = np.zeros([mtot, nwav], 'complex64')
U = np.zeros([mtot], 'complex64')
r = 0
for k in range(len(objOis)):
    oi = objOis[k]
    for dit in range(oi.visOi.ndit):
        for c in range(oi.visOi.nchannel):
            m = int(oi.visOi.m[dit, c])
            if (inj_coeffs is None):
                G = np.diag(np.abs(ampRefs[k][c, :])/resolved_star_models[k][dit, c, :]*chromatic_coupling[k, :])
            else:
                G = np.diag((inj_coeffs[k][dit, c, 0]*np.abs(ampRefs[k][c, :])+inj_coeffs[k][dit, c, 1]*(oi.wav - np.min(oi.wav))*1e6*np.abs(ampRefs[k][c, :]))/resolved_star_models[k][dit, c, :]*chromatic_coupling[k, :])
            H[r:(r+m), :] = np.dot(oi.visOi.h_matrices[dit, c, 0:m, :], G)
            U[r:r+m] = np.dot(oi.visOi.h_matrices[dit, c, 0:m, :], oi.visOi.visRef[dit, c, :])            
            r = r+m
            
U2 = cs.conj_extended(U)
H2 = cs.conj_extended(H)

# block calculate W2inv*H2, cov(U2)*W2inv*H2, and pcov(U2)*W2inv*H2
nb = objOis[0].visOi.nchannel
nwav = objOis[0].nwav
W2invH2 = np.zeros([2*mtot, nwav], 'complex64')
covU2W2invH2 = np.zeros([2*mtot, nwav], 'complex64')
pcovU2W2invH2 = np.zeros([2*mtot, nwav], 'complex64')
r = 0
pbar = patiencebar.Patiencebar(title = "Calculating W2invH2", valmax = np.sum(np.array([oi.visOi.ndit for oi in objOis])))
nwav = objOis[0].nwav
nb = objOis[0].visOi.nchannel
Omega = np.zeros([nb*nwav, nb*nwav], 'complex64')    

for k in range(len(objOis)):
    oi = objOis[k]
    for dit in range(oi.visOi.ndit):
        pbar.update()
        this_dit_mtot = 0
        for c in range(oi.visOi.nchannel):        
            m = int(oi.visOi.m[dit, c])
            this_dit_mtot = this_dit_mtot + m
        W_elem = W_obs[k][0:this_dit_mtot, 0:this_dit_mtot]
        Z_elem = Z_obs[k][0:this_dit_mtot, 0:this_dit_mtot]
        this_r = 0
        H_elem = np.zeros([this_dit_mtot, oi.visOi.nchannel*oi.nwav], 'complex64')        
        for c in range(oi.visOi.nchannel):
            Omega[c*oi.nwav:(c+1)*oi.nwav, c*oi.nwav:(c+1)*oi.nwav] = np.diag(oi.visOi.phi_values[dit, c, :])
            m = int(oi.visOi.m[dit, c])            
            H_elem[this_r:this_r+m, c*oi.nwav:(c+1)*nwav] = oi.visOi.h_matrices[dit, c, 0:m, :]
            this_r = this_r+m
        Rcov = np.diag(oi.visOi.visErr[dit, :, :].real.reshape(oi.nwav*oi.visOi.nchannel))**2
        Icov = np.diag(oi.visOi.visErr[dit, :, :].imag.reshape(oi.nwav*oi.visOi.nchannel))**2     
        W_dit = np.dot(np.dot(Omega, Rcov+Icov), cs.adj(Omega))
        Z_dit = np.dot(np.dot(Omega, Rcov-Icov), Omega.T)        
        W_elem = np.dot(np.dot(H_elem, W_dit), cs.adj(H_elem))
        Z_elem = np.dot(np.dot(H_elem, Z_dit), H_elem.T)        
        W2_elem = cs.extended_covariance(W_elem, Z_elem)                         
        W2_elem_inv = np.linalg.inv(W2_elem)
        [A, B, C, D] = [W2_elem_inv[0:this_dit_mtot, 0:this_dit_mtot], W2_elem_inv[0:this_dit_mtot, this_dit_mtot:], W2_elem_inv[this_dit_mtot:, 0:this_dit_mtot], W2_elem_inv[this_dit_mtot:, this_dit_mtot:]]        
        W2invH2[r:r+this_dit_mtot, :] = np.dot(A, H[r:r+this_dit_mtot])+np.dot(B, H[r:r+this_dit_mtot].conj())
        W2invH2[mtot+r:mtot+r+this_dit_mtot, :] = np.dot(C, H[r:r+this_dit_mtot])+np.dot(D, H[r:r+this_dit_mtot].conj())
        covU2W2invH2[r:r+this_dit_mtot, :] = np.dot(W_elem, W2invH2[r:r+this_dit_mtot, :]) + np.dot(Z_elem, W2invH2[mtot+r:r+mtot+this_dit_mtot, :])
        covU2W2invH2[mtot+r:mtot+r+this_dit_mtot, :] = np.dot(cs.adj(Z_elem), W2invH2[r:r+this_dit_mtot, :]) + np.dot(np.conj(W_elem), W2invH2[mtot+r:r+mtot+this_dit_mtot, :])
        pcovU2W2invH2[r:r+this_dit_mtot, :] = np.dot(Z_elem, W2invH2[r:r+this_dit_mtot, :]) + np.dot(W_elem, W2invH2[mtot+r:r+mtot+this_dit_mtot, :])
        pcovU2W2invH2[mtot+r:mtot+r+this_dit_mtot, :] = np.dot(cs.adj(W_elem), W2invH2[r:r+this_dit_mtot, :]) + np.dot(np.conj(Z_elem), W2invH2[mtot+r:r+mtot+this_dit_mtot, :]) 
        r = r+this_dit_mtot


# calculate spectrum
A = np.real(np.dot(cs.adj(U2), W2invH2))
B = np.linalg.inv(np.real(np.dot(cs.adj(H2), W2invH2)))
C = np.dot(A, B)

# calculate errors
covU2 = np.dot(cs.adj(W2invH2), covU2W2invH2)
pcovU2 = np.dot(W2invH2.T, pcovU2W2invH2)
Cerr = np.dot(B, np.dot(0.5*np.real(covU2+pcovU2), B.T))

# spectrum calibration for plot
wav_grav = oi.wav*1e6
(starWav, starFlux) = exosed.utilities.convertSpectrum(starSpectrum, instrument = 'gravity-midres', resolution = 200000)
star = np.interp(wav_grav, starWav, starFlux)

diagErrors = np.array([Cerr[k, k] for k in range(len(C))])**0.5

fig = plt.figure(figsize = (10, 6))

ax1 = fig.add_subplot(211)
ax1.errorbar(wav_grav, C, yerr = diagErrors, fmt = '.')

ax2 = fig.add_subplot(212, sharex = ax1)
ax2.errorbar(wav_grav, C*star, yerr = diagErrors*star, fmt = '.')

ax1.set_xlabel("Wavelength ($\mu\mathrm{m}$)")
ax2.set_xlabel("Wavelength ($\mu\mathrm{m}$)")
ax1.set_ylabel("Contrast")
ax2.set_ylabel("Flux")

fig.tight_layout()

# save the data
data = np.zeros([len(wav), 2])
data[:, 0] = wav
data[:, 1] = np.dot(A, B)
#np.savetxt("betapicb_contrast.txt", data)
#np.savetxt("betapicb_errors.txt", Cerr)

"""
data = np.loadtxt('results_5648/betapicb_contrast.txt')
C = data[:, 1]

all_coeffs = []
ind = np.zeros([6, 400])
for c in range(objOis[0].visOi.nchannel):
    ind[c, :] = range(400)
w = np.zeros([6, objOis[0].nwav])
for c in range(objOis[0].visOi.nchannel):
    w[c, :] = objOis[0].wav
# fit the slope
pbar = patiencebar.Patiencebar(valmax = len(objOis))
for k in range(len(objOis)):
    pbar.update()
    oi = objOis[k]
    coeffs = np.zeros([oi.visOi.ndit, 6, 2])    
    Y = np.zeros([oi.visOi.nchannel, oi.nwav], 'complex')    
    Yfit = np.zeros([oi.visOi.nchannel, oi.nwav], 'complex')
    for dit in range(oi.visOi.ndit):
        for c in range(oi.visOi.nchannel):
            m = int(oi.visOi.m[dit, c])                                    
            X = np.zeros([m, 2], 'complex')
            H = oi.visOi.h_matrices[dit, c, 0:m, :]
            U = np.dot(H, oi.visOi.visRef[dit, c, :])
            U2 = gravity.conj_extended(U)            
            G = np.diag(np.abs(totalVisRef[c, :]))
            Lambda = np.diag((oi.wav - np.mean(oi.wav))*1e6)
            X[:, 0] = np.dot(np.dot(H, G), C)
            X[:, 1] = np.dot(np.dot(H, G), np.dot(Lambda, C))
            X2 = gravity.conj_extended(X)
            Rcov = np.diag(oi.visOi.visErr[dit, c, :].real)**2
            Icov = np.diag(oi.visOi.visErr[dit, c, :].imag)**2
            Omega = np.diag(oi.visOi.phi_values[dit, c, :])
            W = np.dot(np.dot(Omega, Rcov+Icov), gravity.adj(Omega))
            Z = np.dot(np.dot(Omega, Rcov-Icov), Omega.T)        
            W = np.dot(np.dot(H, W), gravity.adj(H))
            Z = np.dot(np.dot(H, Z), H.T)        
            W2 = gravity.extended_covariance(W, Z)                         
            W2 = gravity.extended_covariance(W, Z)
            W2inv = np.linalg.inv(W2)
            K = np.real(np.dot(np.dot(gravity.adj(X2), W2inv), X2))
            A = np.dot(np.real(np.dot(np.dot(gravity.adj(U2), W2inv), X2)), np.linalg.pinv(K))
            coeffs[dit, c, :] = np.real(A)
            Xp = np.zeros([oi.nwav, 2], 'complex')
            Xp[:, 0] = np.dot(G, C)
            Xp[:, 1] = np.dot(G, np.dot(Lambda, C))
            Yfit[c, :] = Yfit[c, :] + np.dot(A, Xp.T)
    all_coeffs.append(coeffs)            
    fig = plt.figure()
    gravity.reImPlot(w, oi.visOi.visRef.sum(axis = 0), fig = fig, subtitles = oi.basenames)
    gravity.reImPlot(w, Yfit, fig = fig)

coeffs = np.concatenate(all_coeffs)

fig = plt.figure()
gravity.baselinePlot(ind, coeffs[:, :, 0].T, fig = fig, subtitles = oi.basenames)

fig = plt.figure()
gravity.baselinePlot(ind, coeffs[:, :, 1].T/coeffs[:, :, 0].T, fig = fig, subtitles = oi.basenames)


slopes = np.zeros([objOis[0].visOi.nchannel, objOis[0].nwav])
for c in range(objOis[0].visOi.nchannel):
    slopes[c, :] = coeffs[:, :, 1].mean(axis = 0)[c]*(objOis[0].wav - np.mean(objOis[0].wav))*1e6+coeffs[:, :, 0].mean(axis = 0)[c]
fig = plt.figure()
gravity.baselinePlot(w, slopes, fig = fig, subtitles = oi.basenames)
"""


### NORMALIZE ###

# ESO K filter
data = np.loadtxt("eso_filter_K.txt", skiprows = 4)
eso_wav = data[:, 0]
eso_filt = data[:, 1]
wav = eso_wav

# star spectrum
starWav = val[:, 0]
starFlux = val[:, 1]

# ESO K mag calib
eso_filt_interp = np.interp(starWav, eso_wav, eso_filt)
flux = np.trapz(eso_filt_interp*starFlux, starWav)/0.33
eso_zp = 4.12e-10
eso_mag = 5.687
eso_flux = eso_zp*10**(-eso_mag/2.5)

plt.figure()
plt.plot(starWav, starFlux)
plt.plot(starWav, starFlux*eso_filt_interp)

norm = eso_flux / flux

"""
stop

# conversion to gaia mag
starPhotons = starFlux  * (starWav)/(10**6*6.626e-34*299792458.0) # planck constant and speed of light. photon flux phot/s/m^2
g_interp = np.interp(starWav, wav, g)
betapic_gaia_flux = np.trapz(g_interp*starPhotons, starWav)*0.7278 # telescope surface
betapic_gaia_mag = gaia_zeropoint - 2.5*np.log(betapic_gaia_flux)/np.log(10)
print(betapic_gaia_mag)
norm = 10**((betapic_gaia_mag - 3.72)/2.5)
"""

"""
# GRAVITY
hdu = fits.open("spectra/Mickael/Synthetic_spectrum_BpicA.fits")[0]
data = hdu.data
ind = np.where((data[:, 0] < 3.0) & (data[:, 0] > 1.5))[0]
starSpectrum = exosed.Spectrum(data[ind, 0], data[ind, 1])
(starWav, starFlux) = exosed.utilities.convertSpectrum(starSpectrum, instrument = 'gravity-midres', resolution = 20000)
starWav = starSpectrum.wav
starFlux = starSpectrum.flux
data = np.loadtxt(GRAVITY_SPECTRUM)
raw_wav = data[:, 0]/10.0
star = np.interp(raw_wav, starWav, starFlux)
raw_flux = data[:, 1]*star#*norm
W = np.loadtxt(GRAVITY_SPECTRUM_ERR)
raw_flux_err = np.array([W[k, k]**0.5 for k in range(len(raw_wav))])*star#*norm

# GPI
data = np.loadtxt("spectra/gpi_spectrum.txt")
wav_gpi = data[:, 0]
flux_gpi = data[:, 1]
flux_err_gpi = data[:, 2]
flux_gpi[-44:-33] = float('nan')

# photometry
zero_point = 3.961e-11*1e4*1e-7*1e4 # W/s/cm2/micron 
bpic_mag = 3.48
mag = np.array([9.2, 9.02, 8.92, 8.8])
mag_err = np.array([0.1, 0.13, 0.13, 0.6])
phot_wav = np.array([2.159, 2.174, 2.174, 2.174])
phot_wav_err = np.array([0.324, 0.269, 0.269, 0.269])/2.0
d_bpic = 19.44
d_vega = 7.68

phot = (10**(-(bpic_mag+mag)/2.5)*zero_point)#*(d_vega/d_bpic)**2
phot_err = ( (10**(-(bpic_mag+mag+mag_err)/2.5) - 10**(-(bpic_mag+mag)/2.5))*zero_point )#*(d_vega/d_bpic)**2

plt.figure(figsize=(10,6));
plt.errorbar(raw_wav, 1e14*raw_flux, yerr = 1e14*raw_flux_err, fmt = 'o')
plt.errorbar(raw_wav, 1e14*raw_flux*norm, yerr = 1e14*raw_flux_err*norm, fmt = 'o')
plt.errorbar(wav_gpi, 1e14*flux_gpi, yerr = 1e14*flux_err_gpi, fmt = 'o')
plt.errorbar(phot_wav[1:], 1e14*phot[1:], xerr = phot_wav_err[1:], yerr = 1e14*phot_err[1:], fmt = 's', capsize=8)
plt.errorbar(phot_wav[0:1], 1e14*phot[0:1], xerr = phot_wav_err[0:1], yerr = 1e14*phot_err[0:1], fmt = 's', capsize=8)
plt.legend(["Gravity calib Mickael", "Gravity ESO K calib", "GPI", "GPI NICI photometry", "VLT NACO photometry"])
plt.xlim(1.8, 2.6)
plt.ylim(0.2, 0.9)
plt.xlabel("Wavelength ($\mu\mathrm{m}$)")
plt.ylabel("Flux ($\\times{}10^{-14}\mathrm{W}\,\mathrm{m}^{-2}\mu\mathrm{m}^{-1}$)")
"""


norm = 1.54e-6

C08 = np.loadtxt("contrast_08.txt")
diagErrors08 = np.loadtxt("contrast_err_08.txt")

C07 = np.loadtxt("contrast_07.txt")
diagErrors07 = np.loadtxt("contrast_err_07.txt")

data = np.loadtxt("hd206893.csv", delimiter = ";")
flux_h = data[:-6, 1]
wav_h = data[:-6, 0]

wav_mag = data[-6:-4, 0]
mag = data[-6:-4, 1]
mag_err = mag - data[-4::2, 1]

plt.figure()
plt.errorbar(wav_grav, C08*star*norm*1e15, yerr = diagErrors08*star*norm*1e15, fmt = '.', capsize=2, color = "gray", alpha = 0.7)
plt.errorbar(wav_grav, C07*star*norm*1e15, yerr = diagErrors07*star*norm*1e15, fmt = '.', capsize=2, color = "orange", alpha = 0.7)
#plt.plot(wav_h, flux_h*1e15, 'o')
plt.errorbar(wav_mag, mag*1e15, yerr = mag_err*1e15, fmt = ".k", markersize = 15, linewidth = 2, capsize = 10)

plt.plot(wav_grav, (C08+C07)/2*1e15*star*norm, 'o--')

plt.xlabel("Wavelength ($\mu$m)")
plt.ylabel("Flux ($\\times{}10^{-15}\mathrm{W}\mathrm{m}^{-2}\mathrm{s}^{-1}\mu{}\mathrm{m}^{-1}$)")

plt.legend(["IRDIS $K1$ and $K_2$ mag", "GRAVITY July 2019", "GRAVITY Aug 2019"])
