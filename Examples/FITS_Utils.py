#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fits File Utilities
Created on Thu Nov  8 13:31:50 2018

@author: archdaemon
"""

import numpy as np
from astropy.io import fits

# dict of nominal fitsio properties
fitsparams = dict()
fitsparams['dataType'] = 'float64';

# params is a dict of stuff you need, including:
# params['dataType'] - the data type to cast read/write ops to
class FITS_Utils():
    def __init__(self, params=fitsparams):
        self.params = params
        return
    
    def simple_fitsread(filename):
        with fits.open(filename, memmap=True) as hdul:
            data = hdul[0].data;
            del hdul
            return(data)

            
    def simple_fitswrite(data,filename):
        fits.writeto(filename,data,overwrite=True)
        return
        
