#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Normalize a GRAVITY spectrum using a model from species

This script is part of the exoGravity data reduction package.
The normalize_spectrum_species is used to normalize a spectrum using a model spectrum from species
using a given stellar model.

Args:
  config_file: the path the yml configuraiton file which should contain all calibration parameters
  file: the path to the fits file containing the spectrum to normalize

Example:
  python normalize_spectrum_species file=full/path/to/spectrum.fits config_file=fill/path/to/config.yml

Authors:
  M. Nowak, and the exoGravity team.

Version:
  xx.xx
"""
import numpy as np
from astropy.io import fits
from cleanGravity.utils import loadFitsSpectrum, saveFitsSpectrum
from utils import *
import sys
import os
# ruamel to read config yml file
try:
    import ruamel.yaml
    RUAMEL = True
except: # if ruamel not available, switch back to pyyaml, which does not handle comments properly
    import yaml
    RUAMEL = False
# import matplotlib before species
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages        
import species
from PyPDF2 import PdfFileReader, PdfFileWriter

whereami = os.path.realpath(__file__).replace("normalize_spectrum_species.py", "")

# load aguments into a dictionnary
dargs = args_to_dict(sys.argv)

if "help" in dargs.keys():
    print(__doc__)
    stop()

# arg should be the path to the spectrum
REQUIRED_ARGS = ["file", "config_file"]
for req in REQUIRED_ARGS:
    if not(req in dargs.keys()):
        printerr("Argument '"+req+"' is not optional for this script. Required args are: "+', '.join(REQUIRED_ARGS))
        stop()

CONFIG_FILE = dargs["config_file"]
if not(os.path.isfile(CONFIG_FILE)):
    raise Exception("Error: argument {} is not a file".format(CONFIG_FILE))        
# READ THE CONFIGURATION FILE
if RUAMEL:
    cfg = ruamel.yaml.load(open(CONFIG_FILE, "r"), Loader=ruamel.yaml.RoundTripLoader)
else:
    cfg = yaml.safe_load(open(CONFIG_FILE, "r"))
    
if not("notitle" in dargs):
    dargs["notitle"] = False
if not("noerr" in dargs):
    dargs["noerr"] = False
if not("cov" in dargs):
    dargs["cov"] = False        

    
FIGDIR = cfg["general"]["figdir"]
    
calib_dict = cfg["spectral_calibration"]
# we need to recast the types to the appropriate type for species
calib_dict["parallax"] = tuple(calib_dict["parallax"])
for key in calib_dict["bounds"].keys():
    calib_dict["bounds"][key] = tuple(calib_dict["bounds"][key])
for key in calib_dict["magnitudes"].keys():
    calib_dict["magnitudes"][key] = tuple(calib_dict["magnitudes"][key])

    
# load spectrum and read resolution from header
printinf("Loading contrast spectrum from {}".format(dargs["file"]))
wav, flux, fluxCov, contrast, contrastCov = loadFitsSpectrum(dargs['file'])
resolution = fits.open(dargs["file"])[0].header["SPECRES"]

# init SPECIES
printinf("Initialising exogravity species instance")
f = open("species_config.ini", "w")
f.write("[species]\ndatabase = {}/species_exograv_database.hdf5\ndata_folder = {}/data/".format(whereami, whereami))
f.close()

species.SpeciesInit()
database = species.Database()
# remove objects and fits to start from fresh
database.delete_data("objects")
database.delete_data("results/fit")
database.delete_data("models")
database.delete_data("filters")
# add object
database.add_object(object_name=calib_dict["star_name"], parallax=calib_dict["parallax"], app_mag=calib_dict["magnitudes"], spectrum=None)

printinf("Preparing species fit")
try:
    fit = species.FitModel(object_name=calib_dict["star_name"], model=calib_dict["model"], bounds = calib_dict["bounds"], inc_phot=True, inc_spec=False)
except ValueError:
    # maybe the model is not in database. Try to download it
    printwar("Species returned a ValueError. Maybe the model {} is not in the database? I'll try to download it.".format(calib_dict["model"]))
    database.add_model(model=calib_dict["model"], teff_range=calib_dict["bounds"]["teff"])
    fit = species.FitModel(object_name=calib_dict["star_name"], model=calib_dict["model"], bounds = calib_dict["bounds"], inc_phot=True, inc_spec=False)

# run multinest fit
printinf("Running species multinest fit")
fit.run_multinest(tag=calib_dict["star_name"], n_live_points=300, output=FIGDIR+'/spectral_calibration/', prior=None)

# extract samples and interpolating on proper grid
printinf("Extracting posterior samples and interpolating on GRAVITY wavelength grid")
samples = database.get_mcmc_spectra(tag=calib_dict["star_name"], random=100, wavel_range=(np.min(wav), np.max(wav)), spec_res=resolution)
for sample in samples:
    sample.flux = np.interp(wav, sample.wavelength, sample.flux)
    sample.wavelength = wav

# calculate stellar flux and covariance matrix
bigMat = np.zeros([len(wav), len(samples)])
for k in range(len(samples)):
    bigMat[:, k] = samples[k].flux
specCov = np.cov(bigMat)
spec_flux = np.mean([box.flux for box in samples], axis=0)
spec_sigma = np.std([box.flux for box in samples], axis=0)

# calibration
printinf("Calibrating spectrum and error bars")
flux = spec_flux*contrast
fluxCov = np.dot(np.dot(np.diag(spec_flux), contrastCov), np.diag(spec_flux))
# add uncertainty from calibration
fluxCov = fluxCov + np.dot(np.dot(np.diag(contrast), specCov), np.diag(contrast))

# save normalized spectrum
# get instrument name from header
printinf("Saving calibrated spectrum in {}".format(dargs["file"]))
hdr = fits.open(dargs["file"])[0].header
instrument = hdr["INSTRU"]
resolution = hdr["SPECRES"]
saveFitsSpectrum(dargs['file'], wav, flux, fluxCov, contrast, contrastCov, instrument=instrument, resolution = resolution)

# posterior mag for checking
printinf("Caluculating posterior magnitudes for verification purposes")
mag_post = []
for filt in calib_dict["magnitudes"].keys():
    phot_mag = database.get_mcmc_photometry(tag=calib_dict["star_name"], burnin = 1000, filter_name=filt, phot_type="magnitude")
    mag_post.append(phot_mag)
    
# plot useful debug info in PDF
fig = species.plot_posterior(tag=calib_dict["star_name"], offset=(-0.3, -0.3), output=FIGDIR+"/species_spectral_calibration_results.pdf")

# create temp pdf to add to species pdf
with PdfPages(FIGDIR+"/temp_spectral_calibration_results.pdf") as pdf:
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.errorbar(wav, contrast, yerr = np.diag(contrastCov)**0.5)
    ax1.set_ylabel("Contrast ($\\times{}10^{-4}$)")
    ax1.set_xlabel("Wavelength ($\mu\mathrm{m})$")    
    ax2 = fig.add_subplot(212)
    ax2.errorbar(wav, contrast*spec_flux, yerr = np.diag(fluxCov)**0.5)
    ax2.set_ylabel("Flux ($10^{-15}\,\mathrm{W}/\mathrm{m}^2/\mu\mathrm{m}$)")
    ax2.set_xlabel("Wavelength ($\mu\mathrm{m}$)")
    plt.tight_layout()
    pdf.savefig()
    plt.close(fig)

    fig = plt.figure()
    nrows = int(np.ceil(len(calib_dict["magnitudes"].keys())/3))
    ncol = 3
    for k in range(len(calib_dict["magnitudes"].keys())):
        filt = list(calib_dict["magnitudes"].keys())[k]
        ax = fig.add_subplot(nrows, ncol, k+1)
        n, bins, patches = ax.hist(mag_post[k], bins = int(np.sqrt(len(mag_post[k]))))
        mag, magerr =  calib_dict["magnitudes"][filt]
        ax.plot([mag, mag], [0, np.max(n)], "k-")
        ax.plot([mag-magerr, mag-magerr], [0, np.max(n)], "k--")
        ax.plot([mag+magerr, mag+magerr], [0, np.max(n)], "k--")                
        ax.set_xlabel(filt)
    plt.tight_layout()        
    pdf.savefig(fig)
    plt.close(fig)
    
# merge the 2 pdfs
output = PdfFileWriter()
pdfOne = PdfFileReader(open(FIGDIR+"/temp_spectral_calibration_results.pdf", "rb"))
pdfTwo = PdfFileReader(open(FIGDIR+"/species_spectral_calibration_results.pdf", "rb"))
for k in range(len(pdfOne.pages)):
    output.addPage(pdfOne.getPage(k))
for k in range(len(pdfTwo.pages)):    
    output.addPage(pdfTwo.getPage(k))
outputStream = open(FIGDIR+"/spectral_calibration_results.pdf", "wb")
output.write(outputStream)
outputStream.close()

# remove temps pdf
os.remove(FIGDIR+"/species_spectral_calibration_results.pdf")
os.remove(FIGDIR+"/temp_spectral_calibration_results.pdf")
