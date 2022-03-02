#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Make a quick plot of an exoGRAVITY spectrum fits file

This script is part of the exoGravity data reduction package.
The combine_spectra is a simple tool which can be used to combine a set of FITS spectra
using the proper coavariance-weighted sum.

Args:
  files: comma separated list of file names
  output: name of the output spectrum file

Example:
  python combine_spectra files=BetaPictoric_2020-02-09.fits,BetaPictoric_2020-02-11.fits output=BetaPictorisc.fits

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

# load aguments into a dictionnary
dargs = args_to_dict(sys.argv)

if "help" in dargs.keys():
    print(__doc__)
    stop()

# arg should be the path to the spectrum
REQUIRED_ARGS = ["files", "output"]
for req in REQUIRED_ARGS:
    if not(req in dargs.keys()):
        printerr("Argument '"+req+"' is not optional for this script. Required args are: "+', '.join(REQUIRED_ARGS))
        stop()

if os.path.isfile(dargs["output"]):
    printinf("File {} already exists.".format(dargs["output"]))
    r = printinp("Overwrite it? (y/n)")
    if not(r.lower() in ["y", "yes"]):
        printerr("User abort")
        stop()
        
filenames = dargs["files"].split(',')
nfiles = len(filenames)
printinf("Combining a total of {} spectra".format(nfiles))

wav_tot, flux_tot, fluxCov_tot, contrast_tot, contrastCov_tot = loadFitsSpectrum(filenames[0])
fluxCovInv_tot = np.linalg.inv(fluxCov_tot)
contrastCovInv_tot = np.linalg.inv(contrastCov_tot)

flux_tot = np.dot(fluxCovInv_tot, flux_tot)
contrast_tot = np.dot(contrastCovInv_tot, contrast_tot)

for k in range(1, nfiles):
    wav, flux, fluxCov, contrast, contrastCov = loadFitsSpectrum(filenames[k])
    if (len(wav) != len(wav_tot)): 
        printerr("The wavelength sequences are not the same in all spectra. Cannot calculate the combined spectra.")
        stop()
    if np.sum((wav - wav_tot)**2)>0:
        printerr("The wavelength sequences are not the same in all spectra. Cannot calculate the combined spectra.")
        stop()
    fluxCovInv = np.linalg.inv(fluxCov)
    contrastCovInv = np.linalg.inv(contrastCov)
    flux_tot = flux_tot + np.dot(fluxCovInv, flux)
    contrast_tot = contrast_tot + np.dot(contrastCovInv, contrast)
    fluxCovInv_tot = fluxCovInv_tot + fluxCovInv
    contrastCovInv_tot = contrastCovInv_tot + contrastCovInv    

fluxCov_tot = np.linalg.inv(fluxCovInv_tot)    
contrastCov_tot = np.linalg.inv(contrastCovInv_tot)

flux_tot = np.dot(fluxCov_tot, flux_tot)
contrast_tot = np.dot(contrastCov_tot, contrast_tot)

printinf("Saving result in {}".format(dargs["output"]))

saveFitsSpectrum(dargs['output'], wav_tot, flux_tot, fluxCov_tot, contrast_tot, contrastCov_tot)
