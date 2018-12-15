#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fits File Utilities
Created on Thu Nov  8 13:31:50 2018

@author: archdaemon
"""

import numpy as np
import subprocess
from scipy.signal import convolve as conv
from scipy.interpolate import interp1d, interp2d
from scipy.special import gamma
from astropy.io import fits
import matplotlib.pyplot as plt




class FITS_Utils:
    def simple_fitsread(filename):
        with fits.open(filename, memmap=True) as hdul:
            data = hdul[0].data;
            del hdul
            return(data)

            
    def simple_fitswrite(data,filename):
        fits.writeto(filename,data,overwrite=True)
        return
        
class Plot_Utils:
    def ezimshow(A, fignum, xlabel_ = None, ylabel_ = None, title_ = None, origin_='lower',cmap_='gray', ):
        fig = plt.figure(fignum)
        fig.clf()
        plt.imshow(A,cmap=cmap_, origin=origin_,)
        plt.xlabel(xlabel_)
        plt.ylabel(ylabel_)
        plt.title(title_)
        plt.colorbar()  
        
    def printConfig(paramslist):
        
        return
        
        
class Propagation_Utils:   
    def gausKernel(shape=(3,3), sigma=0.5):
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        m,n = [(ss-1.)/2. for ss in shape]
        y,x = np.ogrid[-m:m+1,-n:n+1]
        h = np.exp( - (x*x + y*y) / (2.*sigma*sigma) )
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h
    
    
    def gausConvolve(f,kern):
        return conv(kern,f, mode='same', method='fft')
    
    
    
    def fft2_fwd(x,scale):
        return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(x)))*scale
    
    def fft2_back(x,scale):
        return np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(x)))*scale
    

class General_Utils:
    
    # Check if USB is connented
    # Returns 0 if connected, 1 if not connected
    def checkUSBConnect():
        return subprocess.call("mount | grep /media/archdaemon/", shell=True)
    
    def cart2pol(X,Y):
        rho = np.sqrt(X*X + Y*Y)
        phi = np.arctan2(Y,X)
        return rho, phi
    
    def Noll(j):
        n = np.ceil(-1.5 + np.sqrt(0.25 + 2*j) - 1e-10)
        jrem = j - (n*(n+1)/2+1)
#        m = (jrem + np.mod(2,jrem) * np.abs(np.mod(2,n)-1) + np.abs(np.mod(2,jrem)-1)*np.mod(2,n)) * (-np.sign(np.mod(2,j)-0.5))
        m = (jrem + np.mod(jrem,2)*np.abs(np.mod(n,2)-1) + np.abs(np.mod(jrem,2)-1)*np.mod(n,2))*(-np.sign(np.mod(j,2)-0.5));
        return n,m
            
    def Zernike(n,m,x):
        
#        R = np.zeros(shape=x.shape)
        R = 0.
        lmax = int((n-m)/2) + 1
        if (n-m)/2 == np.round((n-m)/2):
            for l in range(lmax):
                 R+=(x**(n-2*l))*(-1)**l*gamma(n-l+1)/(gamma(l+1)*gamma((n+m)/2-l+1)*gamma((n-m)/2-l+1))
        
        R = np.nan_to_num(R)
        
        R_norm = R * np.sqrt(2*(n+1))
        if m == 0:
            R_norm = R_norm / np.sqrt(2)
        
        R = R_norm
        return R
    
    def Zernike2D(n,m,rho,phi):
        if m>=0:
            Z = General_Utils.Zernike(n,m,rho) * np.cos(m*phi)
        else:
            Z = General_Utils.Zernike(n,-m,rho)*np.sin(-m*phi)
        
        return Z
    
    def mkMagAOX_vAPP(dir='/home/archdaemon/Research/GitHub/Propagator-Py/Examples/vAPP_512/'):
        vAPP_lower_re = FITS_Utils.simple_fitsread(dir+str('vAPP_lower_real.fits'))
        vAPP_lower_im = FITS_Utils.simple_fitsread(dir+str('vAPP_lower_imag.fits'))
        vAPP_upper_re = FITS_Utils.simple_fitsread(dir+str('vAPP_upper_real.fits'))
        vAPP_upper_im = FITS_Utils.simple_fitsread(dir+str('vAPP_upper_imag.fits'))
        
        vAPP_lower = np.complex128(vAPP_lower_re + 1j*vAPP_lower_im)
        vAPP_upper = np.complex128(vAPP_upper_re + 1j*vAPP_upper_im)
        return vAPP_lower, vAPP_upper
        
       
        
    
    
class PyPropUtils(FITS_Utils, Plot_Utils, Propagation_Utils, General_Utils):
    pass