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
#params['pixD']
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
    
    
    def extractPupil(self, extract=False):
        if self.params['gridsizeX'] is not self.params['pixD']:
            extract=True
        
        if extract:
            if np.iscomplexobj(self.zSag_):
                newGrid = np.zeros(shape=(self.params['pixD'],self.params['pixD']), dtype=np.complex128)
            else:
                newGrid = np.zeros(shape=(self.params['pixD'],self.params['pixD']), dtype=np.float64)
                
            center = (int(np.floor(self.params['gridsizeX'] /2)), int(np.floor(self.params['gridsizeY'] /2)))
            newPxGrid = np.arange(center[0]-int(np.floor(self.params['pixD']/2)), center[0]+int(np.floor(self.params['pixD']/2)), 1, 'int')
            PxGrid = np.arange(0,self.params['pixD'],1,'int')
            for i in range(newPxGrid.size-1):
                for j in range(newPxGrid.size-1):
                    newGrid[PxGrid[i],PxGrid[j]]=self.zSag_[newPxGrid[i],newPxGrid[j]]
                    
            self.params['gridsizeX'] = newGrid.shape[0]
            self.params['gridsizeY'] = newGrid.shape[1]
            self.zSag_ = newGrid
            return
            

class Mirror(Element):
    pass