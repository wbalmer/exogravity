#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Make a quick plot of an exoGRAVITY spectrum fits file

This script is part of the exoGravity data reduction package.
The plot_spectrum is used to display the content of a spectrum FITS file

Args:
  file: the path the to fits file to plot.
  noerr (bool, optional): if set, do not show the error bars on the plot
  notitle (bool, optional): do not put the name of the file in the title
  cov (bool, optional): if set, the covariance matrices will also be displayed as images

Example:
  python plot_spectrum file=full/path/to/spectrum.fits --cov

Authors:
  M. Nowak, and the exoGravity team.

Version:
  xx.xx
"""
import numpy as np
import matplotlib.pyplot as plt
from cleanGravity.utils import loadFitsSpectrum
from utils import *
import sys

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
REQUIRED_ARGS = ["file"]
for req in REQUIRED_ARGS:
    if not(req in dargs.keys()):
        printerr("Argument '"+req+"' is not optional for this script. Required args are: "+', '.join(REQUIRED_ARGS))
        stop()

wav, flux, fluxCov, contrast, contrastCov = loadFitsSpectrum(dargs['file'])

if not(dargs['noerr']):
    fluxErr = np.sqrt(np.diag(fluxCov))
    contrastErr = np.sqrt(np.diag(contrastCov))

fig = plt.figure(figsize=(12,8))

# contrast spectrum
ax1 = fig.add_subplot(211)
if dargs['noerr']:
    ax1.plot(wav, contrast*1e4, 'C0')
else:
    ax1.errorbar(wav, contrast*1e4, yerr=contrastErr*1e4, fmt = '.', color = 'gray', capsize=2, markeredgecolor = 'k')
ax1.set_ylabel("Contrast ($\\times{}10^{-4}$)")

if not(dargs["notitle"]):
    plt.title(dargs["file"].split('/')[-1])

# flux spectrum
ax2 = fig.add_subplot(212, sharex=ax1)
if dargs['noerr']:
    ax2.plot(wav, flux*1e15, 'C0')
else:
    ax2.errorbar(wav, flux*1e15, yerr=fluxErr*1e15, fmt = '.', color = 'gray', capsize=2, markeredgecolor = 'k')
ax2.set_ylabel("Flux ($10^{-15}\,\mathrm{W}/\mathrm{m}^2/\mu\mathrm{m}$)")


plt.show()
