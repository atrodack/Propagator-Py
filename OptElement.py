#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 19:32:05 2018

@author: archdaemon
"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PyPropUtils import PyPropUtils as PPU



# params is a dict of parameters for the field
#params = dict()
#params['name']
#params['material']
#params['z_position']
#params['diameter']
#params['gridsizeX']
#params['gridsizeY']
#params['zsag']
#params['phasefac'] 
#params['dtype']
class Element:
    def __init__(self, params):
        
        self.params=params
        
        return


    def setZsag(self, A=None):
        if A is None:
            A = self.params['zsag']
            
        if isinstance(A, str):
            self.zSag_ = PPU.simple_fitsread(A)
            
        elif A.__class__ is np.ndarray:
            self.zSag_ = A
            
        else:
            raise ValueError('A is not the right Value type')
            
        return

class Mirror(Element):
    pass