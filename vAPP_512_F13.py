#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 12:00:31 2018

@author: archdaemon
"""

# import dependencies
#import subprocess
import numpy as np
from timeit import default_timer as timer


# Import the config data
import config_laptop as conf

# Import the PyProp Library
from PyPropUtils import PyPropUtils as PPU


# Field module
import OptWF

# Element Module
import OptElement
# Function Declarations









#def main():
# Print the configuration dictionary
print(conf.F13paramslist,'\n\n\n')

# Check the ATMO_Dir
USB_ = PPU.checkUSBConnect()
if USB_ == 0:
    ATMO_DIR = '/media/archdaemon/USB30FD/Research/Code/MATLAB/Atmo_med_512/WFout'
else:
    ATMO_DIR = '/home/archdaemon/Atmo_med_512'

print('ATMO_DIR is set as ',ATMO_DIR)


# Set the datatype to single prec
dataType = 'float32';



# Initialize
wvl = conf.F13paramslist['lambda0']
k = (2*np.pi)/wvl;
ld = wvl / conf.F13paramslist['beamD']
f = conf.F13paramslist['f_num'] * conf.F13paramslist['beamD']
N = conf.F13paramslist['N']
dx = conf.F13paramslist['beamD'] / conf.F13paramslist['pixD']

# Make an Element Object for the Pupil
mirrorparams = dict()
mirrorparams['name'] = conf.F13paramslist['pupilName']
mirrorparams['material'] = conf.F13paramslist['material_Code']
mirrorparams['z_position'] = conf.F13paramslist['zPos']
mirrorparams['diameter'] = conf.F13paramslist['beamD']
mirrorparams['pixD'] = conf.F13paramslist['pixD']
mirrorparams['gridsizeX'] = conf.F13paramslist['gridsizeX']
mirrorparams['gridsizeY'] = conf.F13paramslist['gridsizeY']
mirrorparams['zsag'] = conf.F13paramslist['pupilFilename']
mirrorparams['dtype'] = dataType
PUPIL = OptElement.Mirror(mirrorparams)
PUPIL.setZsag()


# Load in vAPP Masks
vAPP_l, vAPP_u = PPU.mkMagAOX_vAPP()

vAPPlparams = dict()
vAPPlparams['name'] = 'lower vAPP Mask'
vAPPlparams['material'] = conf.F13paramslist['material_Code']
vAPPlparams['z_position'] = conf.F13paramslist['zPos']
vAPPlparams['diameter'] = conf.F13paramslist['beamD']
vAPPlparams['pixD'] = conf.F13paramslist['pixD']
vAPPlparams['gridsizeX'] = conf.F13paramslist['gridsizeX']
vAPPlparams['gridsizeY'] = conf.F13paramslist['gridsizeY']
vAPPlparams['zsag'] = vAPP_l
vAPPlparams['dtype'] = dataType
vAPP_lower = OptElement.Mirror(vAPPlparams)
vAPP_lower.setZsag()

vAPPuparams = dict()
vAPPuparams['name'] = 'lower vAPP Mask'
vAPPuparams['material'] = conf.F13paramslist['material_Code']
vAPPuparams['z_position'] = conf.F13paramslist['zPos']
vAPPuparams['diameter'] = conf.F13paramslist['beamD']
vAPPuparams['pixD'] = conf.F13paramslist['pixD']
vAPPuparams['gridsizeX'] = conf.F13paramslist['gridsizeX']
vAPPuparams['gridsizeY'] = conf.F13paramslist['gridsizeY']
vAPPuparams['zsag'] = vAPP_u
vAPPuparams['dtype'] = dataType
vAPP_upper = OptElement.Mirror(vAPPuparams)
vAPP_upper.setZsag()


# Extract Pupils to remove zero-padding
PUPIL.extractPupil()
vAPP_lower.extractPupil()
vAPP_upper.extractPupil()


# Make Field with on-axis planewave
fieldparams = dict()
fieldparams['name'] = conf.F13paramslist['name1']
fieldparams['plane'] = 'Pupil_Plane'
fieldparams['wvl'] = wvl
fieldparams['gridsizeX'] = PUPIL.params['gridsizeX']
fieldparams['gridsizeY'] = PUPIL.params['gridsizeY']
fieldparams['pxSize'] = dx
fieldparams['beamD'] = conf.F13paramslist['beamD']
fieldparams['max_chirp_step_deg'] = conf.F13paramslist['max_chirp_step_deg']
fieldparams['max_lens_step_deg'] = 20
fieldparams['interp_style'] = 'cubic'
fieldparams['dtype'] = dataType
F1 = OptWF.Field(fieldparams)
F1.getCoords2D()
F1.getKCoords2D()

fieldparams['name'] = conf.F13paramslist['name2']
F2 = OptWF.Field(fieldparams)
F2.getCoords2D()
F2.getKCoords2D()

fieldparams['name'] = conf.F13paramslist['name3']
fieldparams['plane'] = 'Focal_Plane'
F3 = OptWF.Field(fieldparams)
F3.getCoords2D()
F3.getKCoords2D()

x = F1.x
X = F1.X
Y = F1.Y


#RHO,THETA = PPU.cart2pol(X,Y)
#n,m = PPU.Noll(11)
#Z = PPU.Zernike2D(n,m,RHO,THETA)
##plot.ezimshow(Z,1,'pixels x','pixels y', 'Zernike 11')


ts = timer()
F1.applyPupilPlaneOptic(PUPIL)
F1.applyPupilPlaneOptic(vAPP_lower)
F1.FraunhoferPropWF(pscale=(dx))
#F1.logshow(vmin=-6)    

F2.applyPupilPlaneOptic(PUPIL)
F2.applyPupilPlaneOptic(vAPP_upper)
F2.FraunhoferPropWF(pscale=(dx))
#F2.logshow(vmin=-6)

F3.setGrid(F1.grid_ + F2.grid_)
tmp = F3.grid_
F3.mkPSF()
#F3.logshow(vmin=-6,fignum=1)
F3.show(fignum=1)
te = timer()
print('CPU: %.20fs' % (te - ts))


ts = timer()
F1.setGrid()
F1.changePlane()
F1.applyPupilPlaneOptic(PUPIL)
F1.applyPupilPlaneOptic(vAPP_lower)
F1.FraunhoferPropWF_GPU(pscale=(dx))
#F1.logshow(vmin=-6)    

F2.setGrid()
F2.changePlane()
F2.applyPupilPlaneOptic(PUPIL)
F2.applyPupilPlaneOptic(vAPP_upper)
F2.FraunhoferPropWF_GPU(pscale=(dx))
#F2.logshow(vmin=-6)

F3.setGrid()
F3.setGrid(F1.grid_ + F2.grid_)
tmp2 = F3.grid_
F3.mkPSF()
#F3.logshow(vmin=-6,fignum=2)
F3.show(fignum=2)
te = timer()
print('GPU: %.20fs' % (te - ts))


test = tmp.astype(np.complex64) - tmp2
PPU.ezimshow(np.abs(test),3)

#diam = F1.params['beamD']
#diam1 = 1.e3 + diam
#z = 200.e3
#
#F1.setGrid(PUPIL.zSag_)
#g,xnew = F1.ConvFresnel2D(diam1,z)
#
#F1.setGrid(g)
#F1.x = xnew
##
#g, xnew = F1.ApplyThinLens2D([0,0],z/2.)
#
#F1.setGrid(g)
#F1.x = xnew
#
#g,xnew = F1.ConvFresnel2D(diam1,z/2)
#F1.setGrid(g)
#F1.x = xnew
#
#PSF = np.abs(F3.grid_)**2
#PPU.ezimshow(np.log10(PSF / PSF.max()),3)




#if __name__ == "__main__":
#    main()