#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""make a covariance weighted combination of GRAVITY spectra contained in fits files

This script is part of the exoGravity data reduction package.
The combine_spectra is a simple tool which can be used to combine a set of FITS spectra
using the proper coavariance-weighted sum.

Authors:
  M. Nowak, and the exoGravity team.
"""
import numpy as np
from cleanGravity.utils import loadFitsSpectrum, saveFitsSpectrum
from utils import *
import sys
import os
# argparse for command line arguments
import argparse

# create the parser for command lines arguments
parser = argparse.ArgumentParser(description=
"""
Make a covariance weighted combination of GRAVITY spectra contained in fits files
""")

# required arguments are the path to the folder containing the data, and the path to the config yml file to write 
parser.add_argument('files', metavar = "file", type=str, nargs="+", help="fits containing individual spectra to be combined.")

# required arguments are the path to the folder containing the data, and the path to the config yml file to write 
parser.add_argument('output', type=str, help="the name of output fits file where to save the combined spectrum.")

# load arguments into a dictionnary
args = parser.parse_args()
dargs = vars(args) # to treat as a dictionnary

filenames = dargs["files"]
nfiles = len(filenames)
printinf("Combining a total of {} spectra".format(nfiles))

wav_tot, flux_tot, fluxCov_tot, contrast_tot, contrastCov_tot = loadFitsSpectrum(filenames[0])
fluxCovInv_tot = np.linalg.pinv(fluxCov_tot)
contrastCovInv_tot = np.linalg.pinv(contrastCov_tot)

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
    fluxCovInv = np.linalg.pinv(fluxCov)
    contrastCovInv = np.linalg.pinv(contrastCov)
    flux_tot = flux_tot + np.dot(fluxCovInv, flux)
    contrast_tot = contrast_tot + np.dot(contrastCovInv, contrast)
    fluxCovInv_tot = fluxCovInv_tot + fluxCovInv
    contrastCovInv_tot = contrastCovInv_tot + contrastCovInv    

fluxCov_tot = np.linalg.pinv(fluxCovInv_tot)    
contrastCov_tot = np.linalg.pinv(contrastCovInv_tot)

flux_tot = np.dot(fluxCov_tot, flux_tot)
contrast_tot = np.dot(contrastCov_tot, contrast_tot)

if "smooth" in dargs:
    printinf("Smoothing the spectra...")
    smooth = int(dargs["smooth"])
    flux_tot = np.convolve(flux_tot, np.ones(smooth), mode = "same")/np.convolve(np.ones(len(flux_tot)), np.ones(smooth), mode = "same")
    contrast_tot = np.convolve(contrast_tot, np.ones(smooth), mode = "same")/np.convolve(np.ones(len(contrast_tot)), np.ones(smooth), mode = "same")    

printinf("Saving result in {}".format(dargs["output"]))
    
saveFitsSpectrum(dargs['output'], wav_tot, flux_tot, fluxCov_tot, contrast_tot, contrastCov_tot)
