#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Make a quick plot of the astrometry contained in a processed exoGRAVITY YAML file

This script is part of the exoGravity data reduction package.
The plot_spectrum is used to display the astrometric solution from a processed YAML file

Args:
  file: the path the YAML configuration file
  noerr (bool, optional): if set, do not show the error bars on the plot
  notitle (bool, optional): do not put the name of the file in the title
  cov (bool, optional): if set, the general covariance will no be displayed on the plot
  noplot: if set, just output the result in the terminal, and do not plot anything
  labels: if set, add label to the individual points
  notitle: if set, to not add the file name in title

Example:
  python plot_astrometry.py file=full/path/to/config.yaml --noerr 
  python plot_astrometry.py file=full/path/to/config.yaml --noplot 

Authors:
  M. Nowak, and the exoGravity team.

Version:
  xx.xx
"""
import numpy as np
from cleanGravity.utils import loadFitsSpectrum
from utils import *
import sys
try:
    import ruamel.yaml
    RUAMEL = True
except: # if ruamel not available, switch back to pyyaml, which does not handle comments properly
    import yaml
    RUAMEL = False
import sys, os


# load aguments into a dictionnary
dargs = args_to_dict(sys.argv)

if "help" in dargs.keys():
    print(__doc__)

if not("notitle" in dargs):
    dargs["notitle"] = False
if not("noerr" in dargs):
    dargs["noerr"] = False
if not("noplot" in dargs):
    dargs["noplot"] = False    
if not("cov" in dargs):
    dargs["cov"] = False        
if not("labels" in dargs):
    dargs["labels"] = False
if not("notitle" in dargs):
    dargs["notile"] = False            

# arg should be the path to the spectrum
REQUIRED_ARGS = ["file"]
for req in REQUIRED_ARGS:
    if not(req in dargs.keys()):
        printerr("Argument '"+req+"' is not optional for this script. Required args are: "+', '.join(REQUIRED_ARGS))
        stop()

CONFIG_FILE = dargs['file']
        
# READ THE CONFIGURATION FILE
if RUAMEL:
    cfg = ruamel.yaml.load(open(CONFIG_FILE, "r"), Loader=ruamel.yaml.RoundTripLoader)
else:
    cfg = yaml.safe_load(open(CONFIG_FILE, "r"))

# extract the astrometric solutions
try:
    ra = [preduce[list(preduce.keys())[0]]["astrometric_solution"][0] for preduce in cfg["general"]["reduce"]]
    dec = [preduce[list(preduce.keys())[0]]["astrometric_solution"][1] for preduce in cfg["general"]["reduce"]]
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

ax.plot(ra, dec, "o")
ax.plot([ra_best], [dec_best], '+k')

val, vec = np.linalg.eig(cov_mat)
e1 = matplotlib.patches.Ellipse((ra_best, dec_best), 2*val[0]**0.5, 2*val[1]**0.5, angle = np.arctan2(vec[0, 1], -vec[1, 1])/np.pi*180.0, fill=False, color = 'k', linewidth=2, linestyle = '--')
e2 = matplotlib.patches.Ellipse((ra_best, dec_best), 3*2*val[0]**0.5, 3*2*val[1]**0.5, angle = np.arctan2(vec[0, 1], -vec[1, 1])/np.pi*180.0, fill=False, color = 'k', linewidth=2)

e1Large = matplotlib.patches.Ellipse((ra_best, dec_best), nfiles**0.5*2*val[0]**0.5, nfiles**0.5*2*val[1]**0.5, angle = np.arctan2(vec[0, 1], -vec[1, 1])/np.pi*180.0, fill=False, color = 'gray', linewidth=2, linestyle = '--')
e2Large = matplotlib.patches.Ellipse((ra_best, dec_best), nfiles**0.5*3*2*val[0]**0.5, nfiles**0.5*3*2*val[1]**0.5, angle = np.arctan2(vec[0, 1], -vec[1, 1])/np.pi*180.0, fill=False, color = 'gray', linewidth=2)

ax.add_patch(e1)
ax.add_patch(e2)

ax.add_patch(e1Large)
ax.add_patch(e2Large)

if dargs['labels']:
    for k in range(nfiles):
        plt.text(ra[k], dec[k], cfg["planet_ois"].keys()[k])

#ax.text(ra_best+val[0]**0.5, dec_best+val[1]**0.5, "RA={:.2f}+-{:.3f}\nDEC={:.2f}+-{:.3f}\nCOV={:.2f}".format(ra_best, ra_std**0.5, dec_best, dec_std**0.5, cov[0, 1]/np.sqrt(cov[0, 0]*cov[1, 1])))
ax.set_xlabel("$\Delta{}\mathrm{RA}$ (mas)")
ax.set_ylabel("$\Delta{}\mathrm{DEC}$ (mas)")

if not(dargs["notitle"]):
    plt.title(dargs['file'])

ax.legend(["Solutions", "Mean solution", "1$\sigma$ dispersion", "3$\sigma$ dispersion", "1$\sigma$ dispersion on mean", "3$\sigma$ dispersion on mean"])
plt.axis("equal")

plt.show()
