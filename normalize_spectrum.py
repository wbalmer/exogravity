#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Normalize a GRAVITY spectrum to a given K-band magnitude

This script is part of the exoGravity data reduction package.
The normalize_spectrum_spectrum is used to normalize a spectrum to a given K band magnitude (ESO filter)
using a given stellar model.

Args:
  file: the path the to fits file containing the contrast spectrum
  file_model: the path to the stellar model to use
  mag_k: the K band magnitude of the star (ESO filer)

Example:
  python normalize_spectrum file=full/path/to/spectrum.fits file_mode=modelspectrum.dat mag_k=3.86

Authors:
  M. Nowak, and the exoGravity team.

Version:
  xx.xx
"""
import numpy as np
from cleanGravity.utils import loadFitsSpectrum, saveFitsSpectrum
from utils import *
import sys
import os

whereami = os.path.realpath(__file__).replace("normalize_spectrum.py", "")

# load aguments into a dictionnary
dargs = args_to_dict(sys.argv)

if "help" in dargs.keys():
    print(__doc__)
    stop()

if not("notitle" in dargs):
    dargs["notitle"] = False
if not("noerr" in dargs):
    dargs["noerr"] = False
if not("cov" in dargs):
    dargs["cov"] = False        
    
# arg should be the path to the spectrum
REQUIRED_ARGS = ["file", "file_model", "mag_k"]
for req in REQUIRED_ARGS:
    if not(req in dargs.keys()):
        printerr("Argument '"+req+"' is not optional for this script. Required args are: "+', '.join(REQUIRED_ARGS))
        stop()

# load spectrum
wav, flux, fluxCov, contrast, contrastCov = loadFitsSpectrum(dargs['file'])

# ESO K filter
data = np.loadtxt(whereami+"/data/eso_filter_K.txt", skiprows = 4)
eso_wav = data[:, 0]
eso_filt = data[:, 1]

# LOAD MODEL
filename_model = dargs['file_model']
lines = open(filename_model, 'r').readlines()[2:]
data = np.zeros([len(lines), 2])
for k in range(len(lines)):
    line = lines[k]
    splitted = line.split()
    w = float(splitted[0].replace('D','E'))
    f = float(splitted[1].replace('D','E'))
    data[k, 0] = w*1e-4
    data[k, 1] = f
ind = np.where(((data[:, 0]<=np.max(wav)) & (data[:, 0]>=np.min(wav))))[0]
data=data[ind, :]
data[:, 1]=10**(data[:, 1]-8)

# guess the resolution from the data
res = wav[0]/(wav[1]-wav[0])
if res > 1000:
    R = 4000 # highres
elif res > 100:
    R = 500 # midres
else:
    R = 30 # lowres

# convert model to proper resolution
Ns=int(1/(np.diff(data[:,0]).mean()/(data[:,0]).mean()*R))/2
G=np.exp(-(np.arange(4*Ns)-2*Ns)**2/Ns**2)
wav_model = data[:, 0]
flux_model = data[:, 1]
ind = np.argsort(wav_model)
wav_model = wav_model[ind]
flux_model = flux_model[ind]
flux_model=np.convolve(flux_model,G,'same')/np.convolve(np.ones(len(data)),G,'same')

# ESO K mag calib
eso_filt_interp = np.interp(wav_model, eso_wav, eso_filt)
fluxTot = np.trapz(eso_filt_interp*flux_model, wav_model)/0.33
eso_zp = 4.12e-10
eso_flux = eso_zp*10**(-float(dargs['mag_k'])/2.5)

norm = eso_flux / fluxTot

# interp stellar model onto GRAVITY wave grid
flux_stellar = np.interp(wav, wav_model, flux_model)

# normalize flux and errors
flux = contrast*flux_stellar*norm
fluxCov = contrastCov*(flux_stellar*norm)**2

# save normalized spectrum
saveFitsSpectrum(dargs['file'], wav, flux, fluxCov, contrast, contrastCov)



