#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 19:28:34 2018

@author: archdaemon
"""
from timeit import default_timer as timer

import numpy as np
from numba import cuda
from PyPropUtils import PyPropUtils as PPU
from PyPropCUDA import cudaFFT as fft
from PyPropCUDA import cuFFT_Utils as cuU

from accelerate.cuda import fft as FFT

stream1 = cuda.stream()

# Test single precision complex to single precision complex
A = PPU.simple_fitsread('/home/archdaemon/Research/GitHub/Propagator-Py/Examples/MagAO-X_pupil_512.fits').astype(np.complex64)
B = fft.cuFFT_v1(A,otype=np.complex64,stream=stream1)
if B.dtype.type is np.complex64:
    print('single complex to single complex pass')
else:
    print('single complex to single complex fail')
    

# Test single precision complex to double precision complex
A = PPU.simple_fitsread('/home/archdaemon/Research/GitHub/Propagator-Py/Examples/MagAO-X_pupil_512.fits').astype(np.complex64)
B = fft.cuFFT_v1(A,otype=np.complex128,stream=stream1)
if B.dtype.type is np.complex128:
    print('single complex to double complex pass')
else:
    print('single complex to double complex fail')
    

# Test double precision complex to single precision complex
A = PPU.simple_fitsread('/home/archdaemon/Research/GitHub/Propagator-Py/Examples/MagAO-X_pupil_512.fits').astype(np.complex128)
B = fft.cuFFT_v1(A,otype=np.complex64,stream=stream1)
if B.dtype.type is np.complex64:
    print('double complex to single complex pass')
else:
    print('double complex to single complex fail')
    


# Test double precision complex to double precision complex
A = PPU.simple_fitsread('/home/archdaemon/Research/GitHub/Propagator-Py/Examples/MagAO-X_pupil_512.fits').astype(np.complex128)
B = fft.cuFFT_v1(A,otype=np.complex128,stream=stream1)
if B.dtype.type is np.complex128:
    print('double complex to double complex pass')
else:
    print('double complex to double complex fail')
    


# Test single precision to single precision complex
A = PPU.simple_fitsread('/home/archdaemon/Research/GitHub/Propagator-Py/Examples/MagAO-X_pupil_512.fits').astype(np.float32)
B = fft.cuFFT_v1(A,otype=np.complex64,stream=stream1)
if B.dtype.type is np.complex64:
    print('single to single complex pass')
else:
    print('single to single complex fail')
    

# Test single precision to double precision complex
A = PPU.simple_fitsread('/home/archdaemon/Research/GitHub/Propagator-Py/Examples/MagAO-X_pupil_512.fits').astype(np.float32)
B = fft.cuFFT_v1(A,otype=np.complex128,stream=stream1)
if B.dtype.type is np.complex128:
    print('single to double complex pass')
else:
    print('single to double complex fail')
    

# Test double precision to single precision complex
A = PPU.simple_fitsread('/home/archdaemon/Research/GitHub/Propagator-Py/Examples/MagAO-X_pupil_512.fits').astype(np.float64)
B = fft.cuFFT_v1(A,otype=np.complex64,stream=stream1)
if B.dtype.type is np.complex64:
    print('double to single complex pass')
else:
    print('double to single complex fail')
    


# Test double precision to double precision complex
A = PPU.simple_fitsread('/home/archdaemon/Research/GitHub/Propagator-Py/Examples/MagAO-X_pupil_512.fits').astype(np.float64)
B = fft.cuFFT_v1(A,otype=np.complex128,stream=stream1)
if B.dtype.type is np.complex128:
    print('double to double complex pass')
else:
    print('double to double complex fail')


N = 2**12
x = np.arange(-N/2, N/2, 1)
X,Y = np.meshgrid(x,x)
R = np.sqrt(X*X + Y*Y)
A = (R<=25).astype(np.complex64)

ts = timer()
B1 = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(A)))
te = timer()
print('CPU  : %.20fs' % (te - ts))

cuU.triggercuFFT(A.astype(np.complex64))
ts = timer()
B2 = fft.float32tocomplex64_FFT(A)
te = timer()
print('GPU 1: %.20fs' % (te - ts))


ts = timer()
B3 = np.zeros(shape=(N,N),dtype=np.complex64)
B3 = fft.cuFFT_v2(A)
te = timer()
print('GPU 2: %.20fs' % (te - ts))


ts = timer()
B4 = np.zeros(shape=(N,N),dtype=np.complex64)
B4 = fft.cuIFFT_v2(B3)
te = timer()
print('GPU 2: %.20fs' % (te - ts))

