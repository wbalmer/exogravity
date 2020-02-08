#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Contains common functions used by other scripts
@author: mnowak
"""
from builtins import input
from astropy.io import fits
from datetime import datetime
import time

t0 = time.time()

def stop():
    raise Exception("Stop")

def printinf(msg):
    """Print an information message preceded by [INFO]:"""
    print("[INFO]: t={:.2f}s ".format(time.time()-t0) + msg)
    
def printwar(msg):
    """Print a warning message preceded by [WARNING]:"""    
    print("[WARNING]: t={:.2f}s ".format(time.time()-t0) + msg)
    
def printerr(msg):
    """Print an error message preceded by [ERROR]: and stop the execution"""        
    print("[ERROR]: t={:.2f}s ".format(time.time()-t0) + msg)
    stop()

def printinp(msg):
    """Request an input from the user using a message preceded by [INPUT]:"""
    r = input("[INPUT]: "+msg)
    return r

def args_to_dict(args):
    """Convert arguments to a dict. 
    Key/values are extracted from args given as "key=value".
    Args given as "--key" are converted to key = True in the dict.
    """
    d = {}
    d['script'] = args[0]
    for arg in args[1:]:
        if arg[0:2] == '--':
            d[arg[2:]] = True
        elif len(arg.split('=', 1)) == 2:
            d[arg.split('=', 1)[0]] = arg.split('=', 1)[1]
        else:
            continue
    return d


def loadFitsSpectrum(filename):
    """ Open a fits file written in GRAVITY convention, and return the wav grid, flux, covariance, contrast, and contrast covariance)
    """
    hdu = fits.open(filename)
    spectrum = hdu[1]
    wav = spectrum.data.field("wavelength")
    flux = spectrum.data.field("flux")
    fluxErr = spectrum.data.field("covariance")
    contrast = spectrum.data.field("contrast")        
    contrastErr = spectrum.data.field("covariance_contrast")    
    return wav, flux, fluxErr, contrast, contrastErr


def saveFitsSpectrum(filename, wav, flux, fluxCov, contrast, contrastCov, name = "Unknown", date_obs = "Unknown", mjd = "Unknown"):
    """Save a spectrum and contrast spectrum to GRAVITY convention in a fits file
    """
    hdr = fits.Header()
    hdr['INSTRU'] = 'GRAVITY'
    hdr['FACILITY'] = 'ESO VLTI'
    hdr['DATE'] = str(datetime.utcnow())
    hdr['DATE-OBS'] = date_obs
    hdr['MJD-OBS'] = mjd
    hdr['OBJECT'] = name    
    hdr['COMMENT'] = "FITS file contains multiple extensions"
    primary_hdu = fits.PrimaryHDU(header = hdr)
    col0= fits.Column(name='WAVELENGTH', format='1D', unit="um",  array=wav)
    col1= fits.Column(name='FLUX', format='1D', unit="W/m2/um",  array=flux)
    col2= fits.Column(name='COVARIANCE', format='%iD'%len(wav), unit="[W/m2/um]^2",  array=fluxCov)
    col3= fits.Column(name='CONTRAST', format='1D',unit=" - ",  array=contrast)
    col4= fits.Column(name='COVARIANCE_CONTRAST', format='%iD'%len(wav), unit=" - ^2",  array=contrastCov)
    secondary_hdu = fits.BinTableHDU.from_columns([col0, col1, col2, col3, col4])
    secondary_hdu.name='SPECTRUM'
    hdul = fits.HDUList([primary_hdu, secondary_hdu])
    hdul.writeto(filename, overwrite = True)
    return None
