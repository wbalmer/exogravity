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

# import YML to write the yml file at the end
try:
    import ruamel.yaml
    RUAMEL = True
except: # if ruamel not available, switch back to pyyaml, which does not handle comments properly
    import yaml
    RUAMEL = False

MSG_FORMAT = "[RUN_GRAVI_ASTROMETRY]: {}"

# check which types of observations are present
import glob
from astropy.io import fits

filenames = glob.glob("./*astroreduced.fits")
# ignore files which are not dual-field
filenames = [filename for filename in filenames if ("ESO INS SOBJ SWAP" in fits.open(filename)[0].header.keys())]
headers = [fits.open(filename)[0].header for filename in filenames]
all_targets = list(set([h["OBJECT"] for h in headers]))
swap_targets = list(set([h["OBJECT"] for h in headers if (h["ESO INS SOBJ SWAP"] == "YES")]))
not_swap_targets = list(set([target for target in all_targets if not(target in swap_targets)]))

# we can only deal with one target of each
if len(swap_targets) > 0:
    swap_target = swap_targets[0]
else:
    swap_target = None

if len(not_swap_targets) > 0:
    not_swap_target = not_swap_targets[0]
else:
    not_swap_target = None


# CASE 1: ON-AXIS
if (swap_target is None):
    # we assume the the data to reduce are in the current folder and we store this in exogravity
    import exogravity
    dargs = {"datadir": "./"}
    dargs["nra"] = 100 
    dargs["ndec"] = 100     
    exogravity.dargs = dargs

    # call create_config to create the proper cfg dictionnary
    print(MSG_FORMAT.format("At {}: entering create_config script".format(datetime.utcnow())))
    from exogravity import create_config
    print(MSG_FORMAT.format("At {}: exiting create_config script".format(datetime.utcnow())))

    # now we can call the create_phase_ref
    print(MSG_FORMAT.format("At {}: entering create_phase_reference script".format(datetime.utcnow())))
    from exogravity import create_phase_reference
    print(MSG_FORMAT.format("At {}: exiting create_phase_reference script".format(datetime.utcnow())))

    # now we can call the astrometry_reduce script
    # For this, we'll update a few configuration parameters
    exogravity.cfg["general"]["gofast"] = True # average the DITS in the file
    exogravity.cfg["general"]["gradient"] = True # use a gradient descent to get a better estimate of the location of the chi2 minimum
    exogravity.cfg["general"]["use_local"] = True # start descent from local minimu closests to global minimum
    # we'll put the output PDFs in the current folder, in a "reduced_astrometry" folder
    if not(os.path.isdir("./reduced_astrometry/")):
        os.mkdir("./reduced_astrometry/")
    exogravity.cfg["general"]["figdir"] = "./reduced_astrometry/"

    print(MSG_FORMAT.format("At {}: entering astrometry_reduce script".format(datetime.utcnow())))
    from exogravity import astrometry_reduce
    print(MSG_FORMAT.format("At {}: exiting astrometry_reduce script".format(datetime.utcnow())))

    # print the solution. For this we need to collect all individula solutions and take the mean + error bars
    ra = [preduce[list(preduce.keys())[0]]["astrometric_solution"][0] for preduce in exogravity.cfg["general"]["reduce_planets"]]
    dec = [preduce[list(preduce.keys())[0]]["astrometric_solution"][1] for preduce in exogravity.cfg["general"]["reduce_planets"]]
    cov_mat = np.cov(ra, dec)/len(ra)
    ra_best = np.mean(ra)
    dec_best = np.mean(dec)
    ra_std = cov_mat[0, 0]**0.5
    dec_std = cov_mat[1, 1]**0.5
    rho = cov_mat[0, 1]/(ra_std*dec_std)
    print(MSG_FORMAT.format( "Astrometric solution: RA={} mas, DEC={} mas".format(ra_best, dec_best)))
    print(MSG_FORMAT.format( "Errors: stdRA={} mas, stdDEC={} mas, rho={}".format(ra_std, dec_std, rho)))


    
# CASE 2: ON-AXIS WITH SWAP
elif not(swap_target is None) and not(not_swap_target is None):
    # we assume the the data to reduce are in the current folder and we store this in exogravity
    import exogravity
    dargs = args_to_dict(sys.argv)
    dargs["datadir"] = "./"
    dargs["swap_target"] = swap_target
    dargs["nra"] = 100 
    dargs["ndec"] = 100 

    exogravity.dargs = dargs

    # call create_config to create the proper cfg dictionnary
    print(MSG_FORMAT.format("At {}: entering create_config script".format(datetime.utcnow())))
    from exogravity import create_config
    print(MSG_FORMAT.format("At {}: exiting create_config script".format(datetime.utcnow())))

    # now we can call the swap_reduce script
    # For this, we'll update a few configuration parameters
    exogravity.cfg["general"]["gofast"] = True # average the DITS in the file
    exogravity.cfg["general"]["gradient"] = True # use a gradient descent to get a better estimate of the location of the chi2 minimum
    exogravity.cfg["general"]["use_local"] = True # start descent from local minimum closests to global minimum
    exogravity.cfg["general"]["calib_strategy"] = "self" # start descent from local minimu closests to global minimum
    # we'll put the output PDFs in the current folder, in a "reduced_astrometry" folder
    if not(os.path.isdir("./reduced_astrometry/")):
        os.mkdir("./reduced_astrometry/")
    exogravity.cfg["general"]["figdir"] = "./reduced_astrometry/"
    print(MSG_FORMAT.format("At {}: entering swap_reduce script".format(datetime.utcnow())))
    from exogravity import swap_reduce    
    print(MSG_FORMAT.format("At {}: exiting swap_reduce script".format(datetime.utcnow())))    

    # now we can call the create_phase_ref
    print(MSG_FORMAT.format("At {}: entering create_phase_reference script".format(datetime.utcnow())))
    from exogravity import create_phase_reference
    print(MSG_FORMAT.format("At {}: exiting create_phase_reference script".format(datetime.utcnow())))

    # now we can call the astrometry_reduce script
    # For this, we'll update a few configuration parameters
    exogravity.cfg["general"]["gofast"] = True # average the DITS in the file
    exogravity.cfg["general"]["gradient"] = True # use a gradient descent to get a better estimate of the location of the chi2 minimum
    exogravity.cfg["general"]["use_local"] = False # start descent from local minimu closests to global minimum
    exogravity.cfg["general"]["calib_strategy"] = "star" # start descent from local minimu closests to global minimum    
    # we'll put the output PDFs in the current folder, in a "reduced_astrometry" folder
    if not(os.path.isdir("./reduced_astrometry/")):
        os.mkdir("./reduced_astrometry/")
    exogravity.cfg["general"]["figdir"] = "./reduced_astrometry/"

    print(MSG_FORMAT.format("At {}: entering astrometry_reduce script".format(datetime.utcnow())))
    from exogravity import astrometry_reduce
    print(MSG_FORMAT.format("At {}: exiting astrometry_reduce script".format(datetime.utcnow())))

    # print the solution. For this we need to collect all individula solutions and take the mean + error bars
    ra = [preduce[list(preduce.keys())[0]]["astrometric_solution"][0] for preduce in exogravity.cfg["general"]["reduce_planets"]]
    dec = [preduce[list(preduce.keys())[0]]["astrometric_solution"][1] for preduce in exogravity.cfg["general"]["reduce_planets"]]
    cov_mat = np.cov(ra, dec)/len(ra)    
    ra_best = np.mean(ra)
    dec_best = np.mean(dec)
    ra_std = cov_mat[0, 0]**0.5
    dec_std = cov_mat[1, 1]**0.5
    rho = cov_mat[0, 1]/(ra_std*dec_std)
    print(MSG_FORMAT.format( "Astrometric solution: RA={} mas, DEC={} mas".format(ra_best, dec_best)))
    print(MSG_FORMAT.format( "Errors: stdRA={} mas, stdDEC={} mas, rho={}".format(ra_std, dec_std, rho)))

    
    
# CASE 3: PURE SWAP
elif not(swap_target is None) and (not_swap_target is None):
    
    import exogravity
    dargs = args_to_dict(sys.argv)
    dargs["datadir"] = "./"
    dargs["swap_target"] = swap_target

    exogravity.dargs = dargs

    # call create_config to create the proper cfg dictionnary
    print(MSG_FORMAT.format("At {}: entering create_config script".format(datetime.utcnow())))
    from exogravity import create_config
    print(MSG_FORMAT.format("At {}: exiting create_config script".format(datetime.utcnow())))

    # now we can call the swap_reduce script
    # For this, we'll update a few configuration parameters
    exogravity.cfg["general"]["gofast"] = True # average the DITS in the file
    exogravity.cfg["general"]["gradient"] = True # use a gradient descent to get a better estimate of the location of the chi2 minimum
    exogravity.cfg["general"]["use_local"] = False # start descent from local minimu closests to global minimum
    exogravity.cfg["general"]["calib_strategy"] = "self" # start descent from local minimu closests to global minimum
    # we'll put the output PDFs in the current folder, in a "reduced_astrometry" folder
    if not(os.path.isdir("./reduced_astrometry/")):
        os.mkdir("./reduced_astrometry/")
    exogravity.cfg["general"]["figdir"] = "./reduced_astrometry/"
    print(MSG_FORMAT.format("At {}: entering swap_reduce script".format(datetime.utcnow())))
    from exogravity import swap_reduce    
    print(MSG_FORMAT.format("At {}: exiting swap_reduce script".format(datetime.utcnow())))

    # Print the astrometric solution. For swap, the stored solution is identical for all files, so just print the first one
    key = list(exogravity.cfg["swap_ois"].keys())[0]
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


