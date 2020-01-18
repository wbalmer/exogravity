#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Contains common functions used by other scripts
@author: mnowak
"""
from builtins import input

def stop():
    raise Exception("Stop")

def printinf(msg):
    """Print an information message preceded by [INFO]:"""
    print("[INFO]: "+msg)
    
def printwar(msg):
    """Print a warning message preceded by [WARNING]:"""    
    print("[WARNING]: "+msg)
    
def printerr(msg):
    """Print an error message preceded by [ERROR]: and stop the execution"""        
    print("[ERROR]: "+msg)
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




