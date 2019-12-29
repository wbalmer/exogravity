#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
functions use to manipulate the config giles
@author: mnowak
"""
import yaml

def dictToYaml(d, level = 0):
    """
    convert a dict with arbitrary depth to a proper YML string
    """
    if type(d) is dict:
        res = ""
        for key in list(d.keys()):
            if (type(key) is int):
                res = res+" "*2*level+str(key)+":\n"+dictToYaml(d[key], level = level+1)+"\n"
            else:                
                res = res+" "*2*level+key+":\n"+dictToYaml(d[key], level = level+1)+"\n"
        return res
    else:
        return " "*2*level+yaml.safe_dump(d).split("\n")[0]

