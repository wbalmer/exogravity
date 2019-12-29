#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Contains common functions used by other scripts
@author: mnowak
"""

def stop():
    raise Exception("Stop")

def printinf(msg):
    print("[INFO]: "+msg)
def printwar(msg):
    print("[WARNING]: "+msg)
def printerr(msg):
    print("[ERROR]: "+msg)
    stop()

def printinp(msg):
    r = input("[INPUT]: "+msg)
    return r

def args_to_dict(args):
    d = {}
    d['script'] = args[0]
    argstr = ' '.join(args[1:])
    splitted = argstr.replace('=', ' ').split()
    for k in range(len(splitted)//2):
        d[splitted[2*k]] = splitted[2*k+1]
    return d



