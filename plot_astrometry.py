#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Make a quick plot of the astrometry contained in a processed exoGRAVITY YAML file

This script is part of the exoGravity data reduction package.
The plot_spectrum is used to display the astrometric solution from a processed YAML file

Authors:
  M. Nowak, and the exoGravity team.
"""
import numpy as np
from cleanGravity.utils import loadFitsSpectrum
from astropy.io import fits
from utils import *
from common import *
import sys
try:
    import ruamel.yaml
    RUAMEL = True
except: # if ruamel not available, switch back to pyyaml, which does not handle comments properly
    import yaml
    RUAMEL = False
import sys, os
import scipy

# argparse for command line arguments
import argparse

# create the parser for command lines arguments
parser = argparse.ArgumentParser(description=
"""
Make a quick plot of an exoGRAVITY spectrum fits file
""")

# required arguments are the path to the folder containing the data, and the path to the config yml file to write 
parser.add_argument('config_file', type=str, help="the path the to YAML configuration file with the normalization parameters.")

# optional arguments
parser.add_argument('--notitle', metavar="EMPTY or TRUE/FALSE", type=bool, default=False, nargs="?", const = True,
                    help="if set, remove the title from the figure. Default: false")

parser.add_argument('--noerr', metavar="EMPTY or TRUE/FALSE", type=bool, default = True, nargs="?", const = True,
                    help="if set, do no draw and plot random spectra from the covariance matrix. Default: true")

parser.add_argument('--noplot', metavar="EMPTY or TRUE/FALSE", type=bool, default = False, nargs="?", const = True,
                    help="if set, just print the astrometry, without a plot. Default: false")

parser.add_argument('--formal_errors', metavar="EMPTY or TRUE/FALSE", type=bool, default = True, nargs="?", const = True,
                    help="if set, also plots the formal errors derived from the chi2 values. Default: true")

parser.add_argument('--labels', metavar="EMPTY or TRUE/FALSE", type=bool, default = False, nargs="?", const = True,
                    help="if set, labels the individual points on the plot. Default: false")

# load arguments into a dictionnary
args = parser.parse_args()
dargs = vars(args) # to treat as a dictionnary
         
CONFIG_FILE = dargs['config_file']

# READ THE CONFIGURATION FILE
if RUAMEL:
    cfg = ruamel.yaml.load(open(CONFIG_FILE, "r"), Loader=ruamel.yaml.RoundTripLoader)
else:
    cfg = yaml.safe_load(open(CONFIG_FILE, "r"))

# extract the astrometric solutions
try:
    ra = [preduce[list(preduce.keys())[0]]["astrometric_solution"][0] for preduce in cfg["general"]["reduce_planets"]]
    dec = [preduce[list(preduce.keys())[0]]["astrometric_solution"][1] for preduce in cfg["general"]["reduce_planets"]]
    printinf("Astrometry extracted from config file {}".format(CONFIG_FILE))
except:
        printinf("Please run the script 'astrometryReduce' first.")    
        printerr("Could not extract the astrometry from the config file {}".format(CONFIG_FILE))    


nfiles = len(cfg['planet_ois'])
        
ra_best = np.mean(ra)
dec_best = np.mean(dec)
cov_mat = np.cov(ra, dec)/nfiles

ra_std = cov_mat[0, 0]**0.5
dec_std = cov_mat[1, 1]**0.5
cov = cov_mat[0, 1]/(ra_std*dec_std)

mjd_mean = np.mean(np.array([cfg["planet_ois"][k]["mjd"] for k in cfg["planet_ois"].keys()]))

printinf("Mean MJD for the observation: MJD={:.2f}".format(mjd_mean))
printinf("Best astrometric solution is: RA={:.2f} mas, DEC={:.2f} mas".format(ra_best, dec_best))
printinf("The dispersion on the solutions ({:d} files) is: stdRA={:.4f} mas, stdDEC={:.4f} mas, cov={:.4f}".format(nfiles, ra_std, dec_std, cov))
printinf("One liner CSV: {:.2f},{:.2f},{:.2f},{:.4f},{:.4f},{:.4f}".format(mjd_mean, ra_best, dec_best, ra_std, dec_std, cov))

if dargs['noplot']:
    sys.exit()

    
# PLOT THE FIGURE
import matplotlib
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)

ax.plot(ra, dec, "oC0")    
ax.plot([ra_best], [dec_best], 'ok')

val, vec = np.linalg.eig(cov_mat)

e1Large = matplotlib.patches.Ellipse((ra_best, dec_best), 2*nfiles**0.5*val[0]**0.5, 2*nfiles**0.5*val[1]**0.5, angle = np.arctan2(vec[0, 1], -vec[1, 1])/np.pi*180.0, fill=False, color = 'gray', alpha=0.3, linewidth=2, linestyle='--')
e2Large = matplotlib.patches.Ellipse((ra_best, dec_best), 3*2*nfiles**0.5*val[0]**0.5, 3*2*nfiles**0.5*val[1]**0.5, angle = np.arctan2(vec[0, 1], -vec[1, 1])/np.pi*180.0, fill=False, color='gray', alpha=0.3, linewidth=2)

e1 = matplotlib.patches.Ellipse((ra_best, dec_best), 2*val[0]**0.5, 2*val[1]**0.5, angle = np.arctan2(vec[0, 1], -vec[1, 1])/np.pi*180.0, fill=False, color = 'k', linewidth=2, linestyle = '--')
e2 = matplotlib.patches.Ellipse((ra_best, dec_best), 3*2*val[0]**0.5, 3*2*val[1]**0.5, angle = np.arctan2(vec[0, 1], -vec[1, 1])/np.pi*180.0, fill=False, color = 'k', linewidth=2)

ax.add_patch(e1Large)
ax.add_patch(e2Large)

ax.add_patch(e1)
ax.add_patch(e2)

# now add individual errors:
if dargs["formal_errors"]:    
    for k in range(len(ra)):
        pred = cfg["general"]["reduce_planets"][k]
        pred = pred[list(pred.keys())[0]]
        # the individual error ellipses derived from chi2
        ra_err, dec_err, rho = pred["formal_errors"]
        cov = np.array([[ra_err**2, rho*ra_err*dec_err], [rho*ra_err*dec_err, dec_err**2]])# reconstruct covariance                
        val, vec = np.linalg.eig(cov) 
        e1=matplotlib.patches.Ellipse((ra[k], dec[k]), 2*val[0]**0.5, 2*val[1]**0.5, angle=np.arctan2(vec[0,1],-vec[1,1])/np.pi*180, fill=False, color='C0', linewidth=2, alpha = 0.5, linestyle='--')
        ax.add_patch(e1)
    
if dargs['labels']:
    for k in range(nfiles):
        plt.text(ra[k], dec[k], cfg["planet_ois"].keys()[k])

#ax.text(ra_best+val[0]**0.5, dec_best+val[1]**0.5, "RA={:.2f}+-{:.3f}\nDEC={:.2f}+-{:.3f}\nCOV={:.2f}".format(ra_best, ra_std**0.5, dec_best, dec_std**0.5, cov[0, 1]/np.sqrt(cov[0, 0]*cov[1, 1])))
ax.set_xlabel("$\Delta{}\mathrm{RA}$ (mas)")
ax.set_ylabel("$\Delta{}\mathrm{DEC}$ (mas)")

if not(dargs["notitle"]):
    plt.title(dargs['config_file'])

ax.legend(["Solutions", "Mean solution", "1$\sigma$ dispersion", "3$\sigma$ dispersion", "1$\sigma$ dispersion on mean", "3$\sigma$ dispersion on mean"])
plt.axis("equal")

plt.show()
