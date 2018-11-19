#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Coronagraph Script
Created on Thu Nov  8 13:17:59 2018
@author: archdaemon
"""

import matplotlib.pyplot as plt
import numpy as np
import FITS_Utils as fits
from scipy.signal import convolve as conv
from scipy.interpolate import interp2d
import numba
from numba import jit

def myfft2D(g,dx):
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(g)))*(dx*dx)

def myifft2D(g,dx):
    return np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(g))) / (dx*dx)

def GetDxAndDiam(x):  # assumes x values are bin centers
     nx = x.shape[0]
     dx = x[1] - x[0]
     diam = (x[-1] - x[0])*(1 + 1/(nx - 1))
     assert diam > 0
     assert dx > 0
     return([dx, diam])

def ResampleField2D(g, x, dx_new, inputs, kind='cubic', fill_value=None):
        [dx,diam] = GetDxAndDiam(x);
        nx = int(np.round(diam/dx_new))
        dxnew = diam/nx
        xnew = np.linspace(-diam/2 + dxnew/2, diam/2 - dxnew/2, nx)
        #interp2d doesn't like complex number.  So stupid.
        interp_real = interp2d(x, x, np.real(g), kind=kind, fill_value=fill_value)
        interp_imag = interp2d(x, x, np.imag(g), kind=kind, fill_value=fill_value)
        g = interp_real(xnew, xnew) + 1j*interp_imag(xnew, xnew)
        return([g, xnew])

def ConvFresnel2D(g, x, diam_out, z, inputs, index_of_refraction=1,
                      set_dx=True):
        lam = inputs[0]/index_of_refraction
        [dx,diam] = GetDxAndDiam(x);
        
        dPhiTol_deg = inputs[3]
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
            [g, x] = ResampleField2D(g, x, dx_new, inputs, kind='cubic')
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
            s_max = lam*z*dPhiTol_deg/(360*dx)
            null_ind = np.where(np.sqrt(sx*sx + sy*sy) > s_max)
            kern[null_ind[0], null_ind[1]] = 0
        h = conv(kern, g, mode='same', method='fft')  # h is on the s spatial grid
        h[i_out[0], i_out[1]] = 0.  # zero the field outside the desired region
        h = h[ind[0]:ind[-1] + 1, ind[0]:ind[-1] + 1]
        p = 1/(lam*z)
        return([p*h, s[ind]])
        
def ApplyThinLens2D(g, x, center, fl, inputs):
        [dx,diam] = GetDxAndDiam(x);
        lam = inputs[0];
        max_step = inputs[4]*np.pi/180
        dx_tol = max_step*lam*fl/(2*np.pi*(diam/2 + np.sqrt(center[0]**2 + center[1]**2)))
        if dx > dx_tol:  # interpolate onto higher resolution grid
            [g, x] = ResampleField2D(g, x, dx_tol, inputs, kind='cubic')

        [sx, sy] = np.meshgrid(x, x, indexing='xy')
        sx -= center[0]
        sy -= center[1]
        h = g*np.exp(-1j*np.pi*(sx*sx + sy*sy)/(fl*lam))
        return([h, x])
        


print(numba.__version__)
dataType = 'float32';


# Load a Pupil Mask
A = fits.FITS_Utils.simple_fitsread('vAPP_512/MagAO-X_pupil_512.fits');

# Pixel Values
N = np.size(A,1);
nPix = np.size(A);
diam = 1e3;
dx = diam / 200;

x = np.linspace(-N/2 + dx/2, N/2 - dx/2, N);


# Plot Aperture
plt.figure()
plt.imshow(A, extent=[x[0],x[-1],x[0],x[-1]]);
plt.colorbar();
plt.set_cmap('gray')
plt.xlabel('microns')
plt.ylabel('microns')


# Initialze flat planewave
F = np.exp(1j * 2 * np.pi * np.zeros([N,N],dataType));

# Apply the Aperture
F_ap = F * A;

# Find FP Field
F_fp = myfft2D(F_ap,dx);

# Find the FP Intensity
PSF = np.square(np.abs(F_fp));
maxval = np.amax(PSF);
PSF_norm = PSF / maxval;

FT_Test = myifft2D(F_fp,dx)


plt.figure()
plt.imshow(np.log10(PSF_norm), extent=[N/2-64,N/2+64,N/2-64,N/2+64]);
plt.set_cmap('gray')
plt.xlabel('pixels')
plt.ylabel('pixels')
plt.colorbar()

#fits.FITS_Utils.simple_fitswrite(np.abs(FT_Test),'amp_F_FP.fits')


wvl = 0.6328;
fl = 100e3;
obd = 200e3;
imd = 200e3;
nx = 150;
diam0 = 1e3;
dx = diam0/nx;
x = np.linspace(-diam0/2 + dx/2,diam0/2 - dx/2, nx)

s = np.linspace(-1+.5/nx, 1 - .5/nx,nx)
[sx,sy] = np.meshgrid(s,s, indexing='xy')
cr = np.where(sx*sx + sy*sy > 1)
f = np.ones((nx,nx))
f[cr[0],cr[1]] = 0



B1 = np.array([wvl,dx,diam,90,20])
F_fres, x2d = ConvFresnel2D(f,x,3e3,obd,B1)

B2 = np.array([wvl,x2d[1]-x2d[0],diam,90,20])
F_lens,x2d = ApplyThinLens2D(F_fres, x2d, [0,0],fl,B2 )

B3 = np.array([wvl,x2d[1]-x2d[0],diam,90,20])
F_FP, x2d = ConvFresnel2D(F_lens,x2d,diam+200,imd,B3)

plt.figure()
plt.imshow(np.abs(F_FP))
fits.FITS_Utils.simple_fitswrite(np.abs(F_FP),'amp_F_FP_4f.fits')
