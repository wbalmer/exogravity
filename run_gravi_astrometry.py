#! /usr/bin/env python3
#coding: utf8

# Welcome to run_gravi_astrometry.py
# This is a Python wrapper for the individual scripts of the exogravity package.

# These scripts were initially designed with exoplanet observations in mind, and are supposed to be used
# as self-contained script and run sequentially. This wrapper was designed as way of running the
# scripts to quickly extract an astrometric measurements. All the different scripts normally share a
# common configuration file which is used to pass data from one script to the next.

# In this wrapper, the scripts are called via `import` commands, and the exogravity package is used
# as an object to pass data between them. 

import os, sys
from datetime import datetime
from exogravity.utils import args_to_dict
import numpy as np
import glob
from astropy.io import fits
# import YML to write the yml file at the end
try:
    import ruamel.yaml
    RUAMEL = True
except: # if ruamel not available, switch back to pyyaml, which does not handle comments properly
    import yaml
    RUAMEL = False
# argparse for command line arguments
import argparse
import distutils

# create the parser for command lines arguments
parser = argparse.ArgumentParser(description=
"""
Extract astrometry from an GRAVITY dual-field observation, using a chi2 map approach
""")

# for the astrometry map
parser.add_argument("--ralim", metavar=('MIN', 'MAX'), type=float, nargs=2, default=None,
                    help="specify the RA range (in mas) over which to search for the astrometry. Default: fiber position +/- 30 mas (UTs) or +/- 120 mas (ATs).") 
parser.add_argument("--declim", metavar=('MIN', 'MAX'), type=float, nargs=2, default=None, 
                    help="specify the DEC range (in mas) over which to search for the astrometry. Default: fiber position +/- 30 mas (UTs) or +/- 120 mas (ATs).")  
parser.add_argument("--nra", metavar="N", type=int, default=100,
                    help="number of points over the RA range. Default: 100")  
parser.add_argument("--ndec", metavar="N", type=int, default=100,
                    help="number of points over the DEC range. Default: 100")    

# for the astrometry map on the swap
parser.add_argument("--ralim_swap", metavar=('MIN', 'MAX'), type=float, nargs=2, default=None,
                    help="specify the RA range (in mas) over which to search for the astrometry of the swap. Default: fiber position +/- 30 mas (UTs) or +/- 120 mas (ATs).")     
parser.add_argument("--declim_swap", metavar=('MIN', 'MAX'), type=float, nargs=2, default=None,
                    help="specify the DEC range (in mas) over which to search for the astrometry. Default: fiber position +/- 30 mas (UTs) or +/- 120 mas (ATs).")    
parser.add_argument("--nra_swap", metavar="N", type=int, default=100,
                    help="number of points over the RA range for the swap. Default: 100")    
parser.add_argument("--ndec_swap", metavar="N", type=int, default=100,
                    help="number of points over the DEC range for the swap. Default: 100")    

# whether to zoom
parser.add_argument("--zoom", metavar="ZOOM_FACTOR", type=int, default=5,
                    help="create an additional zoomed in version of the chi2 map around the best guess from the initial map. Default: factor 5")

# some parameters related to how files are loaded
parser.add_argument("--target", metavar="NAME", type=str, default = None,
                    help="restrict the reduction to a particular target (from FITS header). Default: None")
parser.add_argument("--swap_target", metavar="NAME", type=str, default = None,
                    help="target name (from FITS header) for the binary swap calibration observation if off-axis mode. Default: None")
parser.add_argument("--fiber_pos", metavar=("RA", "DEC"), type=int, nargs=2, default = None,
                    help="restrict the reduction to a single fiber position (in mas). Default: all files are taken")

parser.add_argument("--reduction", metavar="LEVEL", type=str, default = "astrored", choices=["astrored", "dualscivis"],
                    help="the data reduction level to use. Default: astrored")
parser.add_argument("--extension", metavar="EXT", type=int, default = 10, choices=[10, 11, 12],
                    help="Which extension to use for the polarimetry. Default: 10 (combined)")
parser.add_argument("--corr_met", metavar="METHOD", type=str, default = "drs", choices = ["drs", "sylvestre"],
                    help="how to calculate the metrology correction for astrored files. can be can be 'sylvestre' or 'drs', or possibly 'none'. Default: drs")
parser.add_argument("--corr_disp", metavar="METHOD", type=str, default = "drs", choices = ["drs", "sylvestre"],
                    help="how to calculate the dispersion correction for astrored files. can be can be 'sylvestre' or 'drs', or possibly 'none'. Default: drs")

# data filtering
parser.add_argument("--phaseref_arclength_threshold", metavar="THRESHOLD", type=float, default = 5,
                    help="remove all dit/baseline with an arclength of the phaseref polyfit larger this threshold. Default: 5")
parser.add_argument("--ft_flux_threshold", metavar="THRESHOLD", type=float, default = 0.2,
                    help="remove all dits/baselines with an ft flux below THRESH*MEAN(FT_FLUX_ON_STAR) (default 0.2)")
parser.add_argument("--reflag", metavar="TRUE/FALSE", type=bool, default=False,
                     help="if set, use the REFLAG table to filter bad datapoints. Requires running the scriot twice! Default: false")  # deprecated?
parser.add_argument("--exclude_targets", metavar="tgt1 tgt2 etc.", type=str, nargs="+", default = [],
                     help="can be used to exclude some target names from the reduction")

# for the OPD calculations on baselines
parser.add_argument("--nopd", type=int, default=100,
                    help="DEPRECATED. Default: 100")    

# how to deal with detrending and calibrations
parser.add_argument("--poly_order", metavar="ORDER", type=int, default=4,
                    help="order of the polynomial used to detrend the data (remove the stellar component in exoplanet observations). Default: 4")   
parser.add_argument("--contrast_file", metavar="FILE", type=str, default=None,
                    help="path to the contrast file (planet/star) to use as a model for the planet visibility amplitude. Default: Constant contrast")
parser.add_argument("--star_diameter", metavar="DIAMETER", type=float, default = 0,
                    help="the diameter of the central star (in mas, to scale the visibility amplitude). Default: 0 mas")
parser.add_argument("--calib_strategy", metavar="STRATEGY", type=str, choices = ["nearest", "self", "all", "none"],
                    help="""
                    how to calculate the reference the coherent flux. 'all' for using all file. 'nearest' for using the two nearest available files.
                    erence. 'none' to skip the calibration. 'self' to calibrate each file by itself.
                    """)

# switches for different behaviors
parser.add_argument("--go_fast", metavar="EMPTY or TRUE/FALSE", type=lambda x:bool(distutils.util.strtobool(x)), nargs="?", const = True, default = False,
                    help="if set, average over DITs to accelerate calculations. Default: false")   
parser.add_argument("--gradient", metavar="EMPTY or TRUE/FALSE", type=lambda x:bool(distutils.util.strtobool(x)), nargs="?", const = True, default = True,
                     help="if set, improves the estimate of the location of chi2 minima by performing a gradient descent from the position on the map. Default: true")   
parser.add_argument("--use_local", metavar="EMPTY or TRUE/FALSE", type=lambda x:bool(distutils.util.strtobool(x)), nargs="?", const = True, default = False,
                     help="if set, uses the local minima will be instead of global ones. Useful when dealing with multiple close minimums. Default: false")   
parser.add_argument("--noinv", metavar="EMPTY or TRUE/FALSE", type=lambda x:bool(distutils.util.strtobool(x)), nargs="?", const = True, default = False,
                     help="if set, avoid inversion of the covariance matrix and replace it with a gaussian elimination approach. DEPRECATED. Default: false")  # deprecated?
parser.add_argument("--save_residuals", metavar="EMPTY or TRUE/FALSE", type=lambda x:bool(distutils.util.strtobool(x)), nargs="?", const = True, default = False,
                     help="if set, saves fit residuals as npy files for further inspection. mainly a DEBUG option. Default: false")
parser.add_argument("--extract_spectrum", metavar="EMPTY or TRUE/FALSE", type=lambda x:bool(distutils.util.strtobool(x)), nargs="?", const = True, default = False,
                     help="if set, also run the spectrum extraction step after the astrometry reduction. Default: false")

# load arguments into a dictionnary
args = parser.parse_args()
dargs = vars(args) # to treat as a dictionnary

#######################
# START OF THE SCRIPT #
#######################

MSG_FORMAT = "[RUN_GRAVI_ASTROMETRY]: {}"

# check which types of observations are present
filenames = glob.glob("./*astroreduced.fits")
# ignore files which are not dual-field
filenames = [filename for filename in filenames if ("ESO INS SOBJ SWAP" in fits.open(filename)[0].header.keys())]
headers = [fits.open(filename)[0].header for filename in filenames]
all_targets = list(set([h["HIERARCH ESO OBS TARG NAME"] for h in headers if h["HIERARCH ESO OBS TARG NAME"] not in dargs["exclude_targets"]]))
swap_targets = list(set([h["HIERARCH ESO OBS TARG NAME"] for h in headers if ((h["ESO INS SOBJ SWAP"] == "YES") and (h["HIERARCH ESO OBS TARG NAME"] not in dargs["exclude_targets"]))]))
print(MSG_FORMAT.format("List of targets found in this directory: {}".format(" | ".join(all_targets))))
print(MSG_FORMAT.format("List of SWAP targets found in this directory: {}".format(" | ".join(swap_targets))))                       
not_swap_targets = list(set([target for target in all_targets if not(target in swap_targets)]))
pola = list(set([h["ESO INS POLA MODE"].lower() for h in headers]))
if "split" in pola:
    if not(dargs["extension"] in [11, 12]):
        print(MSG_FORMAT.format("The data appears to be in 'SPLIT' polarisation. Please specify which extension to reduce (--extension 11 or 12)"))      
        sys.exit()

# we can only deal with one target of each. Just take the first one if not explicitly given by user
if dargs["swap_target"] is None:
    if len(swap_targets) > 0:
        dargs["swap_target"] = swap_targets[0]

if dargs["target"] is None:
    if len(not_swap_targets) > 0:
        dargs["target"] = not_swap_targets[0]

import exogravity
dargs["datadir"] = "./"
# we'll put the output PDFs in the current folder, in a "reduced_astrometry" folder
if not(os.path.isdir("./reduced_astrometry/")):
    os.mkdir("./reduced_astrometry/")   
dargs["figdir"] = "./reduced_astrometry/"
exogravity.dargs = dargs

if dargs["calib_strategy"] is None:
    exogravity.dargs["calib_strategy"] = "nearest"

exogravity.dargs["exclude_targets"] = dargs["exclude_targets"]
    
# CASE 1: ON-AXIS
if (dargs["swap_target"] is None):
    print(MSG_FORMAT.format("Strategy identified: on-axis"))    
    # call create_config to create the proper cfg dictionnary
    print(MSG_FORMAT.format("At {}: entering create_config script".format(datetime.utcnow())))
    from exogravity import create_config
    print(MSG_FORMAT.format("At {}: exiting create_config script".format(datetime.utcnow())))

    # now we can call the create_phase_ref
    print(MSG_FORMAT.format("At {}: entering create_phase_reference script".format(datetime.utcnow())))
    from exogravity import create_phase_reference
    print(MSG_FORMAT.format("At {}: exiting create_phase_reference script".format(datetime.utcnow())))

    # now we can call the astrometry_reduce script
    print(MSG_FORMAT.format("At {}: entering astrometry_reduce script".format(datetime.utcnow())))
    from exogravity import astrometry_reduce
    print(MSG_FORMAT.format("At {}: exiting astrometry_reduce script".format(datetime.utcnow())))

    # print the solution. For this we need to collect all individula solutions and take the mean + error bars
    mjd = [exogravity.cfg["planet_ois"][list(preduce.keys())[0]]["mjd"] for preduce in exogravity.cfg["general"]["reduce_planets"]]        
    ra = [preduce[list(preduce.keys())[0]]["astrometric_solution"][0] for preduce in exogravity.cfg["general"]["reduce_planets"]]
    dec = [preduce[list(preduce.keys())[0]]["astrometric_solution"][1] for preduce in exogravity.cfg["general"]["reduce_planets"]]
    cov_mat = np.cov(ra, dec)/len(ra)
    ra_best = np.mean(ra)
    dec_best = np.mean(dec)
    ra_std = cov_mat[0, 0]**0.5
    dec_std = cov_mat[1, 1]**0.5
    rho = cov_mat[0, 1]/(ra_std*dec_std)
    print(MSG_FORMAT.format( "Mean mjd of observation: {}".format(np.mean(np.array(mjd)))))
    print(MSG_FORMAT.format( "Astrometric solution: RA={} mas, DEC={} mas".format(ra_best, dec_best)))
    print(MSG_FORMAT.format( "Errors: stdRA={} mas, stdDEC={} mas, rho={}".format(ra_std, dec_std, rho)))

    if dargs["extract_spectrum"]:
        print(MSG_FORMAT.format("At {}: entering spectrum_reduce script".format(datetime.utcnow())))
        from exogravity import spectrum_reduce
        print(MSG_FORMAT.format("At {}: exiting spectrum_reduce script".format(datetime.utcnow())))
        print(MSG_FORMAT.format("At {}: spectrum now available in {}/spectrum.fits".format(datetime.utcnow(), dargs["figdir"])))
        
    
# CASE 2: ON-AXIS WITH SWAP
elif not(dargs["swap_target"] is None) and not(dargs["target"] is None):
    print(MSG_FORMAT.format("Strategy identified: off-axis with swap calibration"))
    # call create_config to create the proper cfg dictionnary
    print(MSG_FORMAT.format("At {}: entering create_config script".format(datetime.utcnow())))
    from exogravity import create_config
    print(MSG_FORMAT.format("At {}: exiting create_config script".format(datetime.utcnow())))
    # now we can call the swap_reduce script
    # For this, we'll update a few configuration parameters
    if dargs["calib_strategy"] is None:
        exogravity.cfg["general"]["calib_strategy"] = "self"         
    print(MSG_FORMAT.format("At {}: entering swap_reduce script".format(datetime.utcnow())))
    from exogravity import swap_reduce    
    print(MSG_FORMAT.format("At {}: exiting swap_reduce script".format(datetime.utcnow())))    

    # now we can call the create_phase_ref
    print(MSG_FORMAT.format("At {}: entering create_phase_reference script".format(datetime.utcnow())))
    from exogravity import create_phase_reference
    print(MSG_FORMAT.format("At {}: exiting create_phase_reference script".format(datetime.utcnow())))

    # now we can call the astrometry_reduce script
    # For this, we'll update a few configuration parameters
    if dargs["calib_strategy"] is None:
        exogravity.cfg["general"]["calib_strategy"] = "star"   
    print(MSG_FORMAT.format("At {}: entering astrometry_reduce script".format(datetime.utcnow())))
    from exogravity import astrometry_reduce
    print(MSG_FORMAT.format("At {}: exiting astrometry_reduce script".format(datetime.utcnow())))

    # print the solution. For this we need to collect all individula solutions and take the mean + error bars
    mjd = [exogravity.cfg["planet_ois"][list(preduce.keys())[0]]["mjd"] for preduce in exogravity.cfg["general"]["reduce_planets"]]        
    ra = [preduce[list(preduce.keys())[0]]["astrometric_solution"][0] for preduce in exogravity.cfg["general"]["reduce_planets"]]
    dec = [preduce[list(preduce.keys())[0]]["astrometric_solution"][1] for preduce in exogravity.cfg["general"]["reduce_planets"]]
    cov_mat = np.cov(ra, dec)/len(ra)    
    ra_best = np.mean(ra)
    dec_best = np.mean(dec)
    ra_std = cov_mat[0, 0]**0.5
    dec_std = cov_mat[1, 1]**0.5
    rho = cov_mat[0, 1]/(ra_std*dec_std)
    print(MSG_FORMAT.format( "Mean mjd of observation: {}".format(np.mean(np.array(mjd)))))    
    print(MSG_FORMAT.format( "Astrometric solution: RA={} mas, DEC={} mas".format(ra_best, dec_best)))
    print(MSG_FORMAT.format( "Errors: stdRA={} mas, stdDEC={} mas, rho={}".format(ra_std, dec_std, rho)))

    
    
# CASE 3: PURE SWAP
elif not(dargs["swap_target"] is None) and (dargs["target"] is None):
    print(MSG_FORMAT.format("Strategy identified: off-axis swap"))    
    # call create_config to create the proper cfg dictionnary
    print(MSG_FORMAT.format("At {}: entering create_config script".format(datetime.utcnow())))
    from exogravity import create_config
    print(MSG_FORMAT.format("At {}: exiting create_config script".format(datetime.utcnow())))

    # now we can call the swap_reduce script
    # For this, we'll update a few configuration parameters
    if dargs["calib_strategy"] is None:
        exogravity.cfg["general"]["calib_strategy"] = "self" # start descent from local minimu closests to global minimum
    print(MSG_FORMAT.format("At {}: entering swap_reduce script".format(datetime.utcnow())))
    from exogravity import swap_reduce    
    print(MSG_FORMAT.format("At {}: exiting swap_reduce script".format(datetime.utcnow())))

    # Print the astrometric solution. For swap, the stored solution is identical for all files, so just print the first one
    mjd = [exogravity.cfg["swap_ois"][key]["mjd"] for key in list(exogravity.cfg["swap_ois"].keys())]
    key = list(exogravity.cfg["swap_ois"].keys())[0]
    print(MSG_FORMAT.format( "Mean mjd of observation: {}".format(np.mean(np.array(mjd)))))    
    print(MSG_FORMAT.format( "Astrometric solution: RA={} mas, DEC={} mas".format(*exogravity.cfg["swap_ois"][key]["astrometric_solution"]) ))
    print(MSG_FORMAT.format( "Errors: stdRA={} mas, stdDEC={} mas, rho={}".format(*exogravity.cfg["swap_ois"][key]["errors"]) ))

    
else:
    print(MSG_FORMAT.format("Not supported"))
    sys.exit()


# Write the final CFG file which contains the astrometry
f = open(exogravity.cfg["general"]["figdir"] + "astrometry.yml", "w")
if RUAMEL:
    f.write(ruamel.yaml.dump(exogravity.cfg, default_flow_style = False))
else:
    f.write(yaml.safe_dump(exogravity.cfg, default_flow_style = False)) 
f.close()


