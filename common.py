#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Common functions used by the different scripts

Authors:
  M. Nowak, and the exoGravity team.

Version:
  xx.xx
"""

# BASIC IMPORTS
import numpy as np
# cleanGravity IMPORTS
import cleanGravity as gravity
from cleanGravity import complexstats as cs
from cleanGravity.utils import loadFitsSpectrum, saveFitsSpectrum
# package related functions
from exogravity.utils import * # utils from this exooGravity package
# other random stuffs
import itertools
    
def filter_ftflux(oi, threshold):
    """ Filter points based on the value of the visDataFt coherent flux"""
    a, b = np.where(np.abs(oi.visOi.visDataFt).mean(axis = -1) < threshold)
    (a, b, c) = np.meshgrid(a, b, range(oi.nwav))
    npoints = len(a.ravel())
    oi.visOi.flagPoints((a, b, c))        
    if npoints > 0:
        printinf("A total of {} points have been flagged in {} (below FT threshold of {:.2e})".format(npoints, oi.filename, threshold))
    return None

def filter_phaseref_arclength(oi, threshold):
    """ Filter points based on the total variations of the phaseRef fitted curve (measured through the arclength)"""
    phaseRefArclength = oi.visOi.calculatePhaseRefArclength()
    a, b = np.where(phaseRefArclength > threshold)
    (a, b, c) = np.meshgrid(a, b, range(oi.nwav))
    npoints = len(a.ravel())
    oi.visOi.flagPoints((a, b, c))        
    if npoints > 0:
        printinf("A total of {} points have been flagged in {} (phaseRefArclength > {:.2e})".format(npoints, oi.filename, threshold))
    return None


def findLocalMinimum(chi2Map, xstart, ystart, jump_size = 1):
    """ Find the local minimum closest to starting point on the given chi2Map """
    n, m = np.shape(chi2Map)
    borderedMap = np.zeros([n+2*jump_size, m+2*jump_size])+np.inf
    borderedMap[jump_size:-jump_size, jump_size:-jump_size] = chi2Map
    found = False
    x, y = xstart+jump_size, ystart+jump_size
    while not(found):
        submap = borderedMap[x-jump_size:x+jump_size+1, y-jump_size:y+jump_size+1]
        ind = np.where(submap == np.nanmin(submap))
        if (x, y) == (x+ind[0][0]-jump_size, y+ind[1][0]-jump_size):
            found = True
        else:
            x = x+ind[0][0]-jump_size
            y = y+ind[1][0]-jump_size
    return x-jump_size, y-jump_size


