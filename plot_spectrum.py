#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Make a quick plot of an exoGRAVITY spectrum fits file

This script is part of the exoGravity data reduction package.
The plot_spectrum is used to display the content of a spectrum FITS file

Authors:
  M. Nowak, and the exoGravity team.
"""
import numpy as np
import matplotlib.pyplot as plt
from cleanGravity.utils import loadFitsSpectrum
from utils import *
import sys
import os
# argparse for command line arguments
import argparse

# create the parser for command lines arguments
parser = argparse.ArgumentParser(description=
"""
Make a quick plot of an exoGRAVITY spectrum fits file
""")

# required arguments are the path to the folder containing the data, and the path to the config yml file to write 
parser.add_argument('file', type=str, help="fits file containing the spectrum to plot.")

# optional arguments
parser.add_argument('--notitle', metavar="EMPTY or TRUE/FALSE", type=bool, default=False, nargs="?", const = True,
                    help="if set, remove the title from the figure. Default: false")

parser.add_argument('--noerr', metavar="EMPTY or TRUE/FALSE", type=bool, default = True, nargs="?", const = True,
                    help="if set, do no draw and plot random spectra from the covariance matrix. Default: true")

parser.add_argument('--cov', metavar="EMPTY or TRUE/FALSE", type=bool, default = False, nargs="?", const = True,
                    help="if set, also plot the covariance matrix. Default: false")

parser.add_argument('--fig', type=int, 
                    help="figure number for the plot. Default: create a new figure")

parser.add_argument('--color', type=str, default = "orange",
                    help="color for the plot (must be a valid python color string). Default: orange")

whereami = os.path.realpath(__file__).replace("plot_spectrum.py", "")

# load aguments into a dictionnary
args = parser.parse_args()
dargs = vars(args) # to treat as a dictionnary

wav, flux, fluxCov, contrast, contrastCov = loadFitsSpectrum(dargs['file'])

if not(dargs['noerr']):
    fluxErr = np.sqrt(np.diag(fluxCov))
    contrastErr = np.sqrt(np.diag(contrastCov))


# ESO K filter
data = np.loadtxt(whereami+"/data/eso_filter_K.txt", skiprows = 4)
eso_wav = data[:, 0]
eso_filt = data[:, 1]

# ESO K mag calib
eso_filt_interp = np.interp(wav, eso_wav, eso_filt)
#fluxTot = np.trapz(eso_filt_interp*flux*1e15, wav)/0.33
eso_zp = 4.12e-10
# np.trapz in an explicit matrix form, to be able to compute error bars from cov matrix
TZ = (np.roll(wav, -1) - np.roll(wav, 1))/2.0
TZ[0] = (wav[1]-wav[0])/2.0
TZ[-1] = (wav[-1] - wav[-2])/2.0
fluxTot = np.dot(TZ, (flux*eso_filt_interp).T)/0.36 # 0.36 is the width of the filter
fluxCov_filt = np.dot( np.dot(np.diag(eso_filt_interp), fluxCov), np.diag(eso_filt_interp) )
fluxTotErr = np.sqrt( np.dot( np.dot(TZ, fluxCov_filt), TZ.T ) )/0.36
contrastTot = np.dot(TZ, (contrast*eso_filt_interp).T)/0.36 # 0.36 is the width of the filter
mag_k = -2.5*np.log10(fluxTot/eso_zp)
mag_k_min = -2.5*np.log10((fluxTot+fluxTotErr)/eso_zp)
mag_k_max = -2.5*np.log10((fluxTot-fluxTotErr)/eso_zp)
mag_contrast = -2.5*np.log10(contrastTot)
print("K-band magnitude: {:.3f} [{:.3f}, {:.3f}]".format(mag_k, mag_k_min, mag_k_max))
print("K-band contrast: {:.3f}".format(mag_contrast))


# plot
if dargs["fig"] is None:
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)        
else:
    fig = plt.figure(int(dargs["fig"]))
    ax1 = fig.get_axes()[0]
    ax2 = fig.get_axes()[1]
    
# plot cov
if dargs["cov"]:
    plt.figure()
    plt.imshow(contrastCov.T, origin = "lower", vmin = -2e-11, vmax = 2e-11, extent = [np.min(wav), np.max(wav), np.min(wav), np.max(wav)])
    plt.xlabel("Wavelength ($\mu$m)")
    plt.ylabel("Wavelength ($\mu$m)")    
    for k in range(30):
        noise = np.random.multivariate_normal(0*contrast, contrastCov)*1.5
        ax1.plot(wav, (contrast+noise)*1e4, "-C7", alpha = 0.2)


# contrast spectrum
if dargs['noerr']:
    ax1.plot(wav, contrast*1e4, dargs['color'], marker=".")
else:
    ax1.errorbar(wav, contrast*1e4, yerr=contrastErr*1e4, fmt = '.', color = dargs["color"], capsize=2, markeredgecolor = 'k')

if dargs["fig"] is None:        
    ax1.set_ylabel("Contrast ($\\times{}10^{-4}$)")
    ax1.set_xlabel("Wavelength ($\mu\mathrm{m})$")
    if not(dargs["notitle"]):
        ax1.set_title(dargs["file"].split('/')[-1])


# flux spectrum
if dargs['noerr']:
    ax2.plot(wav, flux*1e15, color = dargs['color'], marker=".")
else:
    ax2.errorbar(wav, flux*1e15, yerr=fluxErr*1e15, fmt = '.', color = dargs['color'], capsize=2, markeredgecolor = 'k')
if dargs["fig"] is None:    
    ax2.set_ylabel("Flux ($10^{-15}\,\mathrm{W}/\mathrm{m}^2/\mu\mathrm{m}$)")
    ax2.set_xlabel("Wavelength ($\mu\mathrm{m}$)")

# show figure only is fig not given
if dargs["fig"] is None:
    plt.show()



