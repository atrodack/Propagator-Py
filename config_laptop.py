#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 12:04:25 2018

@author: archdaemon
"""

import numpy as np


F13paramslist = dict()

# AO Config
F13paramslist['AOGain'] = 0.67
F13paramslist['Closed_loop_T'] = 0
F13paramslist['AO_Start_T'] = 42
F13paramslist['lag_delay_T'] = 1
F13paramslist['useWFSAOres'] = False
F13paramslist['useGausAOres'] = False
F13paramslist['kernelsize'] = 5.

# ATMO Config
F13paramslist['ATMO_Dir'] = '/media/archdaemon/USB30FD/Research/Code/MATLAB/Atmo_med_512/WFout'
F13paramslist['ATMO_scale_factor'] = 0.25
F13paramslist['useGausAtmo'] = False
F13paramslist['GausATMOkernelsize'] = 5.
F13paramslist['useATMOAmp'] = False

# Field Config
F13paramslist['name1'] = 'Star_1'
F13paramslist['gridsizeX'] = 512
F13paramslist['gridsizeY'] = 512
F13paramslist['name2'] = 'Planet_1'
F13paramslist['name3'] = 'Star_A'

# Planet Config
F13paramslist['Nplanets'] = 1
F13paramslist['maxAlpha'] = 3.
F13paramslist['fixed_'] = True

# Pupil Config
F13paramslist['pupilName'] = 'System_Pupil'
F13paramslist['material_Code'] = 0
F13paramslist['ampOnly'] = True
F13paramslist['zPos'] = 0.
F13paramslist['pixD'] = 200
F13paramslist['beamD'] = 5.e3
F13paramslist['pupilFilename'] = '/home/archdaemon/Research/GitHub/Propagator-Py/Examples/MagAO-X_pupil_512.fits'

# Science Camera Config
F13paramslist['camera_std'] = False
F13paramslist['camera_mean'] = True

# Simulation Config
F13paramslist['N'] = 512
F13paramslist['f_num'] = 80.
F13paramslist['n0'] = 1.
F13paramslist['lambda0'] = 0.550
F13paramslist['bandwidth'] = 0.2
F13paramslist['max_chirp_step_deg'] = 90

# RTFA Config
F13paramslist['u_star'] = 10**5
F13paramslist['u_dot'] = 10**3
F13paramslist['searchBasis'] = np.array((43,43,44,45,46))
F13paramslist['nexposures'] = np.linspace(25,900,36)
F13paramslist['basisCode'] = 1
F13paramslist['nollList'] = np.array((43,44,45,46))
F13paramslist['randomize'] = False
F13paramslist['scaleFactor'] = 1.
F13paramslist['searchVecX'] = np.linspace(243,269,27)
F13paramslist['searchVecY'] = np.linspace(243,269,27)
F13paramslist['QSGain'] = 0.55
F13paramslist['Static_Start_T'] = 150