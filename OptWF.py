#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 21:20:47 2018


IMPORTANT: gridsizeX is the number of rows and gridsizeY is the number of columns
@author: archdaemon
"""
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import convolve as conv
from scipy.interpolate import interp2d
from PyPropUtils import Propagation_Utils as prop
import OptElement


# params is a dict of parameters for the field
#params = dict()
#params['name'] = conf.F13paramslist['name1']
#params['plane'] = plane
#params['wvl'] = wvl
#params['bandwidth'] = bandwidth
#params['gridsizeX'] = conf.F13paramslist['gridsizeX']
#params['gridsizeY'] = conf.F13paramslist['gridsizeY']
#params['pxSize'] = conf.F13paramslist['pxSize']
#params['plane'] = 'Pupil_Plane'
#params['max_chirp_step_deg'] = conf.F13paramslist['max_chirp_step_deg']
#params['dtype'] = dataType
#
# grid_ can be specified as the field to initialize to; if None, on-axis planewave field created with unit amplitude, 0 phase
# amp_ can be specified as the amplitude of an on-axis planewave: MUST NOT SPECIFIY GRID TO USE amp_
class Field():
    def __init__(self, params, grid_=None, amp_ = None):
        # Load in params
        self.params = params
        
        # Set Dict as properties
        for i in self.params.keys():
            if i == 'name':
                self.setName(self.params.get(i))
            elif i == 'wvl':
                self.setCentralWvl(self.params.get(i))
            elif i == 'bandwidth':
                self.setBand(self.params.get(i))
            elif i == 'gridsizeX':
                tmp = self.params.get(i)
            elif i == 'gridsizeY':
                self.setGridsize(tmp, self.params.get(i))
            elif i == 'pxSize':
                self.setPxsize(self.params.get(i))
            elif i == 'dtype':
                self.setDatatype(self.params.get(i))
        
        # Load in given grid or initialize planewave
        self.setGrid(grid_, amp_)
        self.getCoords2D()
        self.getKCoords2D()
#        self.prevGrid_ = self.grid_
        self.currentPlane_ = self.params['plane']
            
        return
    
    
    
    # Property setting methods
    
    def setName(self,name):
        self.name_ = name
        return
    
    def setCentralWvl(self, wvl=None):
        if wvl is None:
            if self.wvlarray_ is not None:
                ind = int(np.ceil(self.wvlarray_.size / 2)) - 1
                self.wvl0_ = self.wvlarray_[ind]
            else:
                raise ValueError('self.wvlarray must be set to use no input to this method')

        else:
            self.wvl0_ = wvl
        return
    
    def setBand(self, bandwidth):
        self.bandwidth_ = bandwidth
    
    def setWvlarray(self, lambdalist=None):
        if lambdalist is None:
            if self.wvl0_ is not None:
                self.wvlarray_ = self.wvl0_
            else:
                raise ValueError('self.wvl0_ must be set to use no input to this method')
                
        else:
            self.wvlarray_ = lambdalist
            if self.wvl0_ is None:
                self.setCentralWvl()
        return

    
    def setGridsize(self, gridsizeX, gridsizeY):
        self.gridsize_ = (gridsizeX, gridsizeY)
        return
    
    def setPxsize(self, pixel_size):
        self.dx_ = pixel_size
        return
    
    def setDatatype(self, datatype):
        self.defaultDatatype = datatype
        return
    
    def setGrid(self, grid_=None, amp_=None):
        if (grid_ is None and amp_ is None):
            self.grid_ = np.ones(shape=(self.params['gridsizeX'],self.params['gridsizeY']),dtype=self.params['dtype'])
        elif (grid_ is None and amp_ is not None):
            self.grid_ = amp_ * np.ones(shape=(self.params['gridsizeX'],self.params['gridsizeY']),dtype=self.params['dtype'])
        elif grid_.shape == (self.params['gridsizeX'],self.params['gridsizeY']):
            self.grid_=None
            self.grid_ = grid_
        else:
            self.gridsize_ = grid_.shape
            self.grid_ = grid_
        return
    
    def changePlane(self):
        if self.currentPlane_ is 'Pupil_Plane':
            self.currentPlane_ = 'Focal_Plane'
        elif self.currentPlane_ is 'Focal_Plane':
            self.currentPlane_ = 'Pupil_Plane'
        
        return
    
    def move2PrevGrid(self,grid_):
        self.prevGrid_ = grid_
#        self.grid_ = None
        return
    
    
    # Field Utilities
    
    def complexNormalRV(self, sigma, gridsize):
        """
        Returns a complex Normal random variable with zero-mean, sigma std
        """
        gridsize = (gridsize,gridsize)
        rv = (sigma/ np.sqrt(2)) * (np.random.normal(loc = 0.0, scale=sigma,size=gridsize) + 1j*np.random.normal(loc=0.0, scale=sigma,size=gridsize))
        return rv
    
    def WFReIm2AmpPhase(self, WFreal=None, WFimag=None):
        if WFreal is None:
            WFreal=np.real(self.grid_)
            
        if WFimag is None:
            WFimag=np.imag(self.grid_)
            
        
        
        sz = WFreal.shape
        if len(sz) == 2:
            sz = (sz[0], sz[1], 1)
        
        numLambdas = sz[2]
        
        WFamp = np.zeros(shape=sz,dtype=self.defaultDatatype)
        WFphase = np.zeros(shape=sz,dtype=self.defaultDatatype)
        
        if sz[2] > 1:
            for i in range(numLambdas):
                WFamp[:,:,i] = np.sqrt(WFreal[:,:,i] * WFreal[:,:,i] + WFimag[:,:,i] * WFimag[:,:,i])
                WFphase[:,:,i] = np.arctan2(WFimag[:,:,i],WFreal[:,:,i])
        
        elif sz[2] == 1:
            WFamp = np.sqrt(WFreal * WFreal + WFimag * WFimag)
            WFphase = np.arctan2(WFimag,WFreal)
    
        self.ampGrid_ = WFamp
        self.phaseGrid_ = WFphase
        
        del WFamp
        del WFphase
    
    
    def phase(self, wvl=None):
        if wvl is None:
            wvl = self.wvl0_
        
        k = (2*np.pi)/wvl
        return k * self.grid_
    
    def phasor(self,wvl=None):
        if wvl is None:
            wvl = self.wvl0_
        return np.exp(-1j * self.phase(wvl))
    
    def GetDxAndDiam(self, x=None):  # assumes x values are bin centers
        if x is None:
            x = self.x
            
        nx = x.shape[0]
        dx = x[1] - x[0]
        diam = (x[-1] - x[0])*(1 + 1/(nx - 1))
        assert diam > 0
        assert dx > 0
        return([dx, diam])
        
    def ResampleField2D(self, dx_new, kind='cubic', fill_value=None):
        x = self.x
        [dx,diam] = self.GetDxAndDiam(x);
        nx = int(np.round(diam/dx_new))
        dxnew = diam/nx
        xnew = np.linspace(-diam/2 + dxnew/2, diam/2 - dxnew/2, nx)
        #interp2d doesn't like complex number.  So stupid.
        interp_real = interp2d(x, x, np.real(self.grid_), kind=kind, fill_value=fill_value)
        interp_imag = interp2d(x, x, np.imag(self.grid_), kind=kind, fill_value=fill_value)
        g = interp_real(xnew, xnew) + 1j*interp_imag(xnew, xnew)
        return([g, xnew])
        
        
    def getCoords2D(self):
        N = self.gridsize_[0]
        
        self.x = (np.arange(0,N,1,self.defaultDatatype) - (N/2.)) * self.dx_
        self.X, self.Y = np.meshgrid(self.x, self.x, indexing='xy')
        self.R = np.sqrt(self.X**2 + self.Y**2)
        return
    
    def getKCoords2D(self):
        N = self.gridsize_[0]
        dk = (2*np.pi) / (N*self.dx_)
        self.kx = (np.arange(0,N,1) - (N/2.)) * dk
        self.ldx = self.kx / (self.params['wvl'] / self.params['beamD']) / ((2*np.pi) / self.params['wvl'])
        self.KX, self.KY = np.meshgrid(self.kx, self.kx)
        self.KR = np.sqrt(self.KX**2 + self.KY**2)
        return
    
    def show(self, fignum=None):
        if fignum is None:
            fignum = 1
        
        sz = self.grid_.shape
        if len(sz) == 2:
            sz = (sz[0], sz[1], 1)
        
        if hasattr(self,'x') == False:
            self.getCoords2D()
        
        if self.currentPlane_ is 'Pupil_Plane':
            xmax = self.x.max() * 1.e-6
            xmin = self.x.min() * 1.e-6
        elif self.currentPlane_ is 'Focal_Plane':
            xmax = self.ldx.max()
            xmin = self.ldx.min()
        else:
            xmin = 0
            xmax = self.gridsize_[0]-1
        
        if np.iscomplexobj(self.grid_) == False:
            for i in range(sz[2]):
                plt.figure(fignum+i)
                plt.clf()
                plt.imshow(self.grid_, extent=[xmin,xmax,xmin,xmax],origin='lower',cmap='gray')
#                plt.xlabel('Meters')
#                plt.ylabel('Meters')
#                plt.title(title_)
                plt.colorbar()  
                plt.show()
        else:
            self.WFReIm2AmpPhase()
            for i in range(sz[2]):
                
                fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
                img1 = ax1.imshow(self.ampGrid_, extent=[xmin,xmax,xmin,xmax],origin='lower',cmap='gray')
#                plt.xlabel('Meters')
#                plt.ylabel('Meters')
#                plt.title(title_)
                divider = make_axes_locatable(ax1)
                cax1 = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(img1, cax=cax1)
                
                img2 = ax2.imshow(self.phaseGrid_, extent=[xmin,xmax,xmin,xmax],origin='lower',cmap='gray')
#                plt.xlabel('Meters')
#                plt.ylabel('Meters')
#                plt.title(title_)
                divider = make_axes_locatable(ax2)
                cax2 = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(img2, cax=cax2)  

                plt.tight_layout(h_pad=1)
                plt.draw()
                
        return
    
    def logshow(self, fignum=None, vmin=-6, vmax=0):
        if fignum is None:
            fignum = 1
        
        sz = self.grid_.shape
        if len(sz) == 2:
            sz = (sz[0], sz[1], 1)
        
        if hasattr(self,'x') == False:
            self.getCoords2D()
        
        if self.currentPlane_ is 'Pupil_Plane':
            xmax = self.x.max() * 1.e-6
            xmin = self.x.min() * 1.e-6
        elif self.currentPlane_ is 'Focal_Plane':
            xmax = self.ldx.max()
            xmin = self.ldx.min()
        else:
            xmin = 0
            xmax = self.gridsize_[0]-1
        
        if np.iscomplexobj(self.grid_) == False:
            for i in range(sz[2]):
                fig, ax1 = plt.subplots(nrows=1, ncols=1)
                img1 = ax1.imshow(np.log10(self.grid_ / self.grid_.max()), extent=[xmin,xmax,xmin,xmax],origin='lower',cmap='gray',vmin=vmin, vmax=vmax)
#                plt.xlabel('Meters')
#                plt.ylabel('Meters')
#                plt.title(title_)
                divider = make_axes_locatable(ax1)
                cax1 = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(img1, cax=cax1) 
                plt.draw()
        else:
            for i in range(sz[2]):
                
                fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
                img1 = ax1.imshow(np.log10(self.ampGrid_ / self.ampGrid_.max()), extent=[xmin,xmax,xmin,xmax],origin='lower',cmap='gray',vmin=vmin, vmax=vmax)
#                plt.xlabel('Meters')
#                plt.ylabel('Meters')
#                plt.title('Log Scale Amplitude')
                divider = make_axes_locatable(ax1)
                cax1 = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(img1, cax=cax1)
                
                img2 = ax2.imshow((self.phaseGrid_ / self.phaseGrid_.max()), extent=[xmin,xmax,xmin,xmax],origin='lower',cmap='gray')
#                plt.xlabel('Meters')
#                plt.ylabel('Meters')
#                plt.title('Normalized Phase')
                divider = make_axes_locatable(ax2)
                cax2 = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(img2, cax=cax2)  

                plt.tight_layout(h_pad=1)
                plt.draw()
                
        return
    
    # Propagation Methods
    
    def ApplyThinLens2D(self, center, fl, return_derivs=False):
        g = self.grid_
        x = self.x
        if g.shape[0] != x.shape[0]:
            raise Exception("ApplyThinLens2D: input field and grid must have same sampling.")
        if g.ndim != 2:
            raise Exception("ApplyThinLens2D: input field array must be 2D.")
        if g.shape[0] != g.shape[1]:
            raise Exception("ApplyThinLens2D: input field array must be square.")
        if len(center) != 2:
            raise Exception("ApplyThinLens2D: center parameter must have len=2.")

        [dx, diam] = self.GetDxAndDiam(x)
        lam = self.params['wvl']
        max_step = self.params['max_lens_step_deg']*np.pi/180
        dx_tol = max_step*lam*fl/(2*np.pi*(diam/2 + np.sqrt(center[0]**2 + center[1]**2)))
        if dx > dx_tol:  # interpolate onto higher resolution grid
            [g, x] = self.ResampleField2D(dx_tol, kind=self.params['interp_style'])

        [sx, sy] = np.meshgrid(x, x, indexing='xy')
        sx -= center[0]
        sy -= center[1]
        h = g*np.exp(-1j*np.pi*(sx*sx + sy*sy)/(fl*lam))
        if not return_derivs:
            return([h, x])
        dhdc0 = 2j*np.pi*sx*h/(lam*fl)
        dhdc1 = 2j*np.pi*sy*h/(lam*fl)
        return([h, [dhdc0, dhdc1], x])

    
    def applyPupilPlaneOptic(self, A):
#        self.move2PrevGrid_(self.grid_)
        
        if self.currentPlane_ is 'Pupil_Plane':
            if A.__class__ is Field:
                print('Not Yet Supported')
                
            elif A.__class__ is OptElement.Mirror:
                if np.iscomplexobj(A.zSag_) == False:
                    self.grid_ *= A.zSag_
                else:
                    A_amp = np.sqrt(np.real(A.zSag_)*np.real(A.zSag_) + np.imag(A.zSag_)*np.imag(A.zSag_))
                    A_pha = np.arctan2(np.imag(A.zSag_),np.real(A.zSag_))
                    A_field = A_amp * np.exp(1j*A_pha)
                    self.grid_ = np.complex128(self.grid_)
                    self.grid_ *= A_field
                    self.WFReIm2AmpPhase()
                
            else:
                if np.iscomplexobj(A) == False:
                    self.grid_ *= A
                else:
                    A_amp = np.sqrt(np.real(A)*np.real(A) + np.imag(A)*np.imag(A))
                    A_pha = np.arctan2(np.imag(A),np.real(A))
                    A_field = A_amp * np.exp(1j*A_pha)
                    self.grid_ = np.complex128(self.grid_)
                    self.grid_ *= A_field
                    self.WFReIm2AmpPhase()
        elif self.currentPlane_ is 'Focal_Plane':
            raise ValueError('Cannot Apply Aperture in Focal Plane')
            
        return
    
#    def applyPhaseScreen(self,WF):
#        if WF.__class__ is OptWF.Field:
#            
#        elif WF.__class__ is numpy.ndarray:
        
    
    #2D Fresenel prop using convolution in the spatial domain
    # g - matrix of complex-valued field in the inital plane
    # x - vector of 1D coordinates in initial plane, corresponding to bin center locaitons
    #        it is assumed that the x and y directions have the same sampling.
    # diam_out - diameter of region to be calculated in output plane
    #    Note that the field is set to 0 outside of disk defined by r = diam_out/2.  This
    #      is because the chirp sampling criteria could be violated outside of this disk.
    # z - propagation distance
    # index_of_refaction - isotropic index of refraction of medium
    # set_dx = [True | False | dx (microns)]
    #   True  - forces full sampling of chirp according to self.params['max_chirp_step_deg']
    #           Note: this can lead to unacceptably large arrays.
    #   False - zeroes the kernel beyond where self.params['max_chirp_step_deg'] 
    #            exceedes the dx given in the x input array.  Note: the output dx will differ
    #            slightly from dx in the x input array.
    #...dx - sets the resolution to dx (units microns).  Note: the output dx will differ 
    #         slightly from this value
    # return_derivs - True to return deriv. of output field WRT z
    def ConvFresnel2D(self, diam_out, z, index_of_refraction=1,
                      set_dx=True, return_derivs=False):
        g = self.grid_
        x = self.x
        if g.shape[0] != x.shape[0]:
            raise Exception("ConvFresnel2D: input field and grid must have same sampling.")
        if g.ndim != 2:
            raise Exception("ConvFresnel2D: input field array must be 2D.")
        if g.shape[0] != g.shape[1]:
            raise Exception("ConvFresnel2D: input field array must be square.")

        lam = (self.params['wvl']/index_of_refraction)
        dx, diam = self.GetDxAndDiam(x)
        dPhiTol_deg = self.params['max_chirp_step_deg']
        dx_chirp = (dPhiTol_deg/180)*lam*z/(diam + diam_out)  # sampling criterion for chirp (factors of pi cancel)
        if set_dx == False:
            dx_new = dx
        elif set_dx == True:  # use chirp sampling criterion
            dx_new = dx_chirp
        else:  # take dx_new to be value of set_dx
            if str(type(set_dx)) != "<class 'float'>":
                raise Exception("ConvFresnel2D: set_dx must be a bool or a float.")
            if set_dx <= 0:
                raise Exception("ConvFresnel2D: numerical value of set_dx must be > 0.")
            dx_new = set_dx

        if dx != dx_new:  # interpolate g onto a grid with spacing of approx dx_new
            [g, x] = self.ResampleField2D(dx_new, kind=self.params['interp_style'])
            dx = x[1] - x[0]

        # make the kernel grid (s) match x as closely as possible
        ns = int(np.round(diam + diam_out)/dx)  # number of points on extended kernel
        s = np.linspace(-diam/2 - diam_out/2 + dx/2, diam/2 + diam_out/2 - dx/2, ns) # spatial grid of extended kernel
        ind = np.where(np.abs(s) < diam_out/2)[0] # get the part of s within the 1D output grid
        [sx, sy] = np.meshgrid(s, s, indexing='xy')
        i_out = np.where(np.sqrt(sx*sx + sy*sy) > diam_out/2)

        #Calculate Fresnel convoltion kernel, (Goodman 4-16)
        #  Note: the factor p = 1/(lam*z) is applied later
        #  Also note: the factor -1j*np.exp(2j*np.pi*z/lam) causes unwanted oscillations with z
        kern = np.exp(1j*np.pi*(sx*sx + sy*sy)/(lam*z))  # Fresnel kernel
        if dx > dx_chirp:  # Where does |s| exceed the max step for this dx?
            s_max = lam*z*self.params['max_chirp_step_deg']/(360*dx)
            null_ind = np.where(np.sqrt(sx*sx + sy*sy) > s_max)
            kern[null_ind[0], null_ind[1]] = 0
        h = conv(kern, g, mode='same', method='fft')  # h is on the s spatial grid
        h[i_out[0], i_out[1]] = 0.  # zero the field outside the desired region
        h = h[ind[0]:ind[-1] + 1, ind[0]:ind[-1] + 1]
        p = 1/(lam*z)
        if not return_derivs:
            return([p*h, s[ind]])
        #dpdz = (-1j/(lam*z*z) + 2*np.pi/(lam*lam*z))*np.exp(2j*np.pi*z/lam)  # includes unwanted oscillations
        dpdz = -1/(lam*z*z)
        dkerndz = -1j*np.pi*(sx*sx + sy*sy)*kern/(lam*z*z)
        dhdz = conv(dkerndz, g, mode='same', method='fft')
        dhdz[i_out[0], i_out[1]] = 0.
        dhdz = dhdz[ind[0]:ind[-1] + 1, ind[0]:ind[-1] + 1]
        return([p*h, dpdz*h + p*dhdz, s[ind]])

            
    
    def FraunhoferPropWF(self, pscale, direction=1, tiltphase=1.):
        sz = self.grid_.shape
        nx = self.grid_.size
        
        if len(sz) == 2:
            sz = (sz[0], sz[1], 1)
            
        field_ = np.zeros(shape=sz, dtype=self.defaultDatatype)
#        self.move2PrevGrid_(self.grid_)
        if sz[2]>1:
            for i in range(sz[2]):
                if direction == -1:
                    field_[:,:,i] = prop.fft2_back(self.grid_[:,:,i], (pscale*pscale*nx)**-1)
                    field_[:,:,i] *= sz[1]*sz[0]*tiltphase
                else:
                    field_[:,:,i] = prop.fft2_fwd(self.grid_[:,:,i] * tiltphase, (pscale*pscale))
            
        elif sz[2]==1:
            if direction == -1:
                field_ = prop.fft2_back(self.grid_, (pscale*pscale*nx)**-1)
                field_ *= sz[1]*sz[0]*tiltphase
            else:
                field_ = prop.fft2_fwd(self.grid_ * tiltphase, (pscale*pscale))
            
        self.setGrid(field_)
        self.WFReIm2AmpPhase()
        self.grid_ = self.ampGrid_ * np.exp(1j * self.phaseGrid_)
        self.changePlane()
               
        return
    
    def FraunhoferPropWF_GPU(self, pscale, direction=1, tiltphase=1.):
        sz = self.grid_.shape
        nx = self.grid_.size
        
        if len(sz) == 2:
            sz = (sz[0], sz[1], 1)
            
        field_ = np.zeros(shape=sz, dtype=self.defaultDatatype)
#        self.move2PrevGrid_(self.grid_)
        if sz[2]>1:
            for i in range(sz[2]):
                if direction == -1:
                    field_[:,:,i] = prop.cufft2_back(self.grid_[:,:,i], (pscale*pscale*nx)**-1)
                    field_[:,:,i] *= sz[1]*sz[0]*tiltphase
                else:
                    field_[:,:,i] = prop.cufft2_fwd(self.grid_[:,:,i] * tiltphase, (pscale*pscale))
            
        elif sz[2]==1:
            if direction == -1:
                field_ = prop.cufft2_back(self.grid_, (pscale*pscale*nx)**-1)
                field_ *= sz[1]*sz[0]*tiltphase
            else:
                field_ = prop.cufft2_fwd(self.grid_ * tiltphase, (pscale*pscale))
            
        self.setGrid(field_)
        self.WFReIm2AmpPhase()
        self.grid_ = self.ampGrid_ * np.exp(1j * self.phaseGrid_)
        self.changePlane()
               
        return
    
    def mkPSF(self):
        if self.currentPlane_ is 'Focal_Plane':
            self.setGrid(np.abs(self.grid_)**2)
        elif self.currentPlane_ is 'Pupil_Plane':
            raise ValueError('Cannot form a PSF in a Pupil Plane!')
        
        return