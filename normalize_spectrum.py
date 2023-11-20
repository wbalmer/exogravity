#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Normalize a GRAVITY spectrum to a given K-band magnitude

This script is part of the exoGravity data reduction package.
The normalize_spectrum_spectrum is used to normalize a spectrum to a given K band magnitude (ESO filter)
using a given stellar model.

Example:
  python normalize_spectrum file=full/path/to/spectrum.fits file_mode=modelspectrum.dat mag_k=3.86

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
import argparse
# species is too long to import, so import it later to not wait a long time with bad args
# import species 


whereami = os.path.realpath(__file__).replace("normalize_spectrum.py", "")

# create the parser for command lines arguments
parser = argparse.ArgumentParser(description=
"""
Convert a contrast spectrum to flux using a model spectrum of the star fitted on a given set of magnitudes
""")

# required arguments are the path to the folder containing the data, and the path to the config yml file to write 
parser.add_argument('file', type=str, help="the path the to spectrum fits file")

# required arguments are the path to the folder containing the data, and the path to the config yml file to write 
parser.add_argument('config_file', type=str, help="the path the to YML file containing the parameters for normalisation")

# optional arguments
parser.add_argument('--live_points', type=int, default = 300, help="number of live points to use in species fit.")


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


# species takes a while to load, so only import it now
import species

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
#resolution = 500

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
#database.delete_data("models")
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
fit.run_multinest(tag=calib_dict["star_name"], n_live_points=dargs["live_points"], output=FIGDIR+'/spectral_calibration/', prior=None)

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

fig = species.plot_posterior(tag=calib_dict["star_name"], offset=(-0.3, -0.3))
    
# create temp pdf to add to species pdf
with PdfPages(FIGDIR+"/spectral_calibration_results.pdf") as pdf:
    pdf.savefig(fig)

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(wav, spec_flux)
    ax1.set_ylabel("Stellar Flux ($10^{-15}\,\mathrm{W}/\mathrm{m}^2/\mu\mathrm{m}$)")
    ax1.set_xlabel("Wavelength ($\mu\mathrm{m})$")
    ax2 = fig.add_subplot(212)
    ax2.plot(wav, contrast)
    ax2.set_ylabel("Contrast")
    ax2.set_xlabel("Wavelength ($\mu\mathrm{m})$")        
    plt.tight_layout()
    pdf.savefig()
    plt.close(fig)
    
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

    best = database.get_median_sample(tag=calib_dict["star_name"])
    read_model = species.ReadModel(model=calib_dict["model"], wavel_range=None)
    modelbox = read_model.get_model(model_param=best, spec_res=500., smooth=True)
    objectbox = database.get_object(object_name=calib_dict["star_name"], inc_phot=True, inc_spec=False)
    objectbox = species.update_objectbox(objectbox=objectbox, model_param=best)
    synphot = species.multi_photometry(datatype='model', spectrum=calib_dict["model"], filters=objectbox.filters, parameters=best)
    phot_values = np.array([val[0] for val in objectbox.flux.values()])
    fig = species.plot_spectrum(boxes=[modelbox, objectbox, synphot], filters=objectbox.filters,
                                plot_kwargs=[
                                    {'ls': '-', 'lw': 1., 'color': 'black'},
                                    None,
                                    None],
                                xlim = (0.4, 4.0),
                                offset=(-0.4, -0.05),                                
                                ylim = (phot_values.min()/10.0, phot_values.max()*10.0),
                                scale=('linear', 'log'),
                                legend=[{'loc': 'lower left', 'frameon': False, 'fontsize': 11.},
                                        {'loc': 'upper right', 'frameon': False, 'fontsize': 12.}],
                                figsize=(8., 4.),
                                quantity='flux density',
                                output=None)
    for ax in fig.axes:
        ax.get_gridspec().update(left=0.1, bottom=0.2, right=0.95, top=0.93)
    pdf.savefig(fig)
    plt.close(fig)    
