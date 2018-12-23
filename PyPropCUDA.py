#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 14:39:07 2018

@author: archdaemon
"""

from accelerate.cuda import fft
from accelerate.cuda.fft import FFTPlan
from numba import cuda
import numpy as np


class cuFFT_Utils():
    def gpuArray(ary,stream=0):
        return(cuda.to_device(ary,stream))
    
    def gather(ary):
        return ary.copy_to_host()
    
    def triggercuFFT(ary):
        FFTPlan(shape=ary.shape, itype=ary.dtype, otype=ary.dtype)
        return


class cudaFFT():
    
    # complex64 to complex64
    def complex64tocomplex64_Plan(shape, batch=1, stream=0, mode=1):
        return FFTPlan(shape=shape, itype=np.complex64, otype=np.complex64, batch = batch, stream=stream, mode=mode)
            
    def complex64tocomplex64_FFT(ary, batch=1, stream=0, mode=1 ):
        
        # Make the plan
        fftplan = cudaFFT.complex64tocomplex64_Plan(ary.shape, batch, stream, mode)
        
        # Shift the array
        ary = np.fft.fftshift(ary)
        
        # Send to GPU
        devi_ary = cuFFT_Utils.gpuArray(ary,stream=stream)
        
        # do the FFT
        fftplan.forward(devi_ary, out=devi_ary)
        ary = cuFFT_Utils.gather(devi_ary)
        ary = np.fft.fftshift(ary)
        return ary.astype(np.complex64)
    
    def complex64tocomplex64_IFFT(ary, batch=1, stream=0, mode=1 ):

        # Make the plan
        fftplan = cudaFFT.complex64tocomplex64_Plan(ary.shape, batch, stream, mode)
        
        # Shift the array
        ary = np.fft.ifftshift(ary)
        
        # Send to GPU
        devi_ary = cuFFT_Utils.gpuArray(ary,stream=stream)
        
        # do the FFT
        fftplan.inverse(devi_ary, out=devi_ary)
        ary = cuFFT_Utils.gather(devi_ary)
        ary = np.fft.ifftshift(ary)
        return ary.astype(np.complex64)
        
    # complex64 to complex128     
    def complex64tocomplex128_FFT(ary, batch=1, stream=0, mode=1 ):
        
        # cast input to double
        ary = ary.astype(np.complex128)
        
        # Make the plan
        fftplan = cudaFFT.complex128tocomplex128_Plan(ary.shape, batch, stream, mode)
        
        # Shift the array
        ary = np.fft.fftshift(ary)
        
        # Send to GPU
        devi_ary = cuFFT_Utils.gpuArray(ary,stream=stream)
        
        # do the FFT
        fftplan.forward(devi_ary, out=devi_ary)
        ary = cuFFT_Utils.gather(devi_ary)
        ary = np.fft.fftshift(ary)
        return ary.astype(np.complex128)
    
    def complex64tocomplex128_IFFT(ary, batch=1, stream=0, mode=1 ):

         # cast input to double
        ary = ary.astype(np.complex128)
        
        # Make the plan
        fftplan = cudaFFT.complex128tocomplex128_Plan(ary.shape, batch, stream, mode)
        
        # Shift the array
        ary = np.fft.ifftshift(ary)
        
        # Send to GPU
        devi_ary = cuFFT_Utils.gpuArray(ary,stream=stream)
        
        # do the FFT
        fftplan.inverse(devi_ary, out=devi_ary)
        ary = cuFFT_Utils.gather(devi_ary)
        ary = np.fft.ifftshift(ary)
        return ary.astype(np.complex128)
    
    # complex128 to complex64            
    def complex128tocomplex64_FFT(ary, batch=1, stream=0, mode=1 ):
        
        # Cast to single precision
        ary = ary.astype(np.complex64)
        
        # Make the plan
        fftplan = cudaFFT.complex64tocomplex64_Plan(ary.shape, batch, stream, mode)
        
        # Shift the array
        ary = np.fft.fftshift(ary)
        
        # Send to GPU
        devi_ary = cuFFT_Utils.gpuArray(ary,stream=stream)
        
        # do the FFT
        fftplan.forward(devi_ary, out=devi_ary)
        ary = cuFFT_Utils.gather(devi_ary)
        ary = np.fft.fftshift(ary)
        return ary.astype(np.complex64)
    
    def complex128tocomplex64_IFFT(ary, batch=1, stream=0, mode=1 ):

        # Cast to single precision
        ary = ary.astype(np.complex64)
        
        # Make the plan
        fftplan = cudaFFT.complex64tocomplex64_Plan(ary.shape, batch, stream, mode)
        
        # Shift the array
        ary = np.fft.ifftshift(ary)
        
        # Send to GPU
        devi_ary = cuFFT_Utils.gpuArray(ary,stream=stream)
        
        # do the FFT
        fftplan.inverse(devi_ary, out=devi_ary)
        ary = cuFFT_Utils.gather(devi_ary)
        ary = np.fft.ifftshift(ary)
        return ary.astype(np.complex64)
    
    # complex128 to complex128
    def complex128tocomplex128_Plan(shape, batch=1, stream=0, mode=1):
        return FFTPlan(shape=shape, itype=np.complex128, otype=np.complex128, batch = batch, stream=stream, mode=mode)
            
    def complex128tocomplex128_FFT(ary, batch=1, stream=0, mode=1 ):
        
        # Make the plan
        fftplan = cudaFFT.complex128tocomplex128_Plan(ary.shape, batch, stream, mode)
        
        # Shift the array
        ary = np.fft.fftshift(ary)
        
        # Send to GPU
        devi_ary = cuFFT_Utils.gpuArray(ary,stream=stream)
        
        # do the FFT
        fftplan.forward(devi_ary, out=devi_ary)
        ary = cuFFT_Utils.gather(devi_ary)
        ary = np.fft.fftshift(ary)
        return ary.astype(np.complex128)
    
    def complex128tocomplex128_IFFT(ary, batch=1, stream=0, mode=1 ):

        # Make the plan
        fftplan = cudaFFT.complex128tocomplex128_Plan(ary.shape, batch, stream, mode)
        
        # Shift the array
        ary = np.fft.ifftshift(ary)
        
        # Send to GPU
        devi_ary = cuFFT_Utils.gpuArray(ary,stream=stream)
        
        # do the FFT
        fftplan.inverse(devi_ary, out=devi_ary)
        ary = cuFFT_Utils.gather(devi_ary)
        ary = np.fft.ifftshift(ary)
        return ary.astype(np.complex128)
  

    # float32 to complex64
    def float32tocomplex64_FFT(ary, batch=1, stream=0, mode=1 ):
        
        # cast ary to complex single
        ary = ary.astype(np.complex64)
        
        # Make the plan
        fftplan = cudaFFT.complex64tocomplex64_Plan(ary.shape, batch, stream, mode)
        
        # Shift the array
        ary = np.fft.fftshift(ary)
        
        # Send to GPU
        devi_ary = cuFFT_Utils.gpuArray(ary,stream=stream)
        
        # do the FFT
        fftplan.forward(devi_ary, out=devi_ary)
        ary = cuFFT_Utils.gather(devi_ary)
        ary = np.fft.fftshift(ary)
        return ary.astype(np.complex64)
    
    def float32tocomplex64_IFFT(ary, batch=1, stream=0, mode=1 ):

        # cast ary to complex single
        ary = ary.astype(np.complex64)
        
        # Make the plan
        fftplan = cudaFFT.float32tocomplex64_Plan(ary.shape, batch, stream, mode)
        
        # Shift the array
        ary = np.fft.ifftshift(ary)
        
        # Send to GPU
        devi_ary = cuFFT_Utils.gpuArray(ary,stream=stream)
        
        # do the FFT
        fftplan.inverse(devi_ary, out=devi_ary)
        ary = cuFFT_Utils.gather(devi_ary)
        ary = np.fft.ifftshift(ary)
        return ary.astype(np.complex64)
    
    
    
    
    # float32 to complex128            
    def float32tocomplex128_FFT(ary, batch=1, stream=0, mode=1 ):
        
        # cast ary to complex double
        ary = ary.astype(np.complex128)
        
        # Make the plan
        fftplan = cudaFFT.complex128tocomplex128_Plan(ary.shape, batch, stream, mode)
        
        # Shift the array
        ary = np.fft.fftshift(ary)
        
        # Send to GPU
        devi_ary = cuFFT_Utils.gpuArray(ary,stream=stream)
        
        # do the FFT
        fftplan.forward(devi_ary, out=devi_ary)
        ary = cuFFT_Utils.gather(devi_ary)
        ary = np.fft.fftshift(ary)
        return ary.astype(np.complex128)
    
    def float32tocomplex128_IFFT(ary, batch=1, stream=0, mode=1 ):

        # cast ary to complex double
        ary = ary.astype(np.complex128)
        
        # Make the plan
        fftplan = cudaFFT.complex128tocomplex128_Plan(ary.shape, batch, stream, mode)
        
        # Shift the array
        ary = np.fft.ifftshift(ary)
        
        # Send to GPU
        devi_ary = cuFFT_Utils.gpuArray(ary,stream=stream)
        
        # do the FFT
        fftplan.inverse(devi_ary, out=devi_ary)
        ary = cuFFT_Utils.gather(devi_ary)
        ary = np.fft.ifftshift(ary)
        return ary.astype(np.complex128)
    
    
# float64 to complex64
    def float64tocomplex64_FFT(ary, batch=1, stream=0, mode=1 ):
        
        # cast ary to complex sinlge
        ary = ary.astype(np.complex64)
        
        # Make the plan
        fftplan = cudaFFT.complex64tocomplex64_Plan(ary.shape, batch, stream, mode)
        
        # Shift the array
        ary = np.fft.fftshift(ary)
        
        # Send to GPU
        devi_ary = cuFFT_Utils.gpuArray(ary,stream=stream)
        
        # do the FFT
        fftplan.forward(devi_ary, out=devi_ary)
        ary = cuFFT_Utils.gather(devi_ary)
        ary = np.fft.fftshift(ary)
        return ary.astype(np.complex64)
    
    def float64tocomplex64_IFFT(ary, batch=1, stream=0, mode=1 ):

        # cast ary to complex sinlge
        ary = ary.astype(np.complex64)
        
        # Make the plan
        fftplan = cudaFFT.complex64tocomplex64_Plan(ary.shape, batch, stream, mode)
        
        # Shift the array
        ary = np.fft.ifftshift(ary)
        
        # Send to GPU
        devi_ary = cuFFT_Utils.gpuArray(ary,stream=stream)
        
        # do the FFT
        fftplan.inverse(devi_ary, out=devi_ary)
        ary = cuFFT_Utils.gather(devi_ary)
        ary = np.fft.ifftshift(ary)
        return ary.astype(np.complex64)
    
    
    
    
    # float64 to complex128
    def float64tocomplex128_FFT(ary, batch=1, stream=0, mode=1 ):
        
        # cast ary to complex double
        ary = ary.astype(np.complex128)
        
        # Make the plan
        fftplan = cudaFFT.complex128tocomplex128_Plan(ary.shape, batch, stream, mode)
        
        # Shift the array
        ary = np.fft.fftshift(ary)
        
        # Send to GPU
        devi_ary = cuFFT_Utils.gpuArray(ary,stream=stream)
        
        # do the FFT
        fftplan.forward(devi_ary, out=devi_ary)
        ary = cuFFT_Utils.gather(devi_ary)
        ary = np.fft.fftshift(ary)
        return ary.astype(np.complex128)
    
    def float64tocomplex128_IFFT(ary, batch=1, stream=0, mode=1 ):

        # cast ary to complex double
        ary = ary.astype(np.complex128)
        
        # Make the plan
        fftplan = cudaFFT.complex128tocomplex128_Plan(ary.shape, batch, stream, mode)
        
        # Shift the array
        ary = np.fft.ifftshift(ary)
        
        # Send to GPU
        devi_ary = cuFFT_Utils.gpuArray(ary,stream=stream)
        
        # do the FFT
        fftplan.inverse(devi_ary, out=devi_ary)
        ary = cuFFT_Utils.gather(devi_ary)
        ary = np.fft.ifftshift(ary)
        return ary.astype(np.complex128)
    
    
    
    def cuFFT_v1(ary, otype=np.complex64, batch=1, stream=0, mode=1):
        if type(ary) is np.ndarray:
            itype = ary.dtype.type
        else:
            raise Exception('cuFFT must have a numpy array as its input!')
        
        if itype is np.float32:
            if otype is np.complex64:
                oary = cudaFFT.float32tocomplex64_FFT(ary, batch=1, stream=0)
            elif otype is np.complex128:
                oary = cudaFFT.float32tocomplex128_FFT(ary, batch=1, stream=0)
        elif itype is np.float64:
            if otype is np.complex64:
                oary = cudaFFT.float64tocomplex64_FFT(ary, batch=1, stream=0)
            elif otype is np.complex128:
                oary = cudaFFT.float64tocomplex128_FFT(ary, batch=1, stream=0)
        elif itype is np.complex64:
            if otype is np.complex64:
                oary = cudaFFT.complex64tocomplex64_FFT(ary, batch=1, stream=0)
            elif otype is np.complex128:
                oary = cudaFFT.complex64tocomplex128_FFT(ary, batch=1, stream=0)
        elif itype is np.complex128:
            if otype is np.complex64:
                oary = cudaFFT.complex128tocomplex64_FFT(ary, batch=1, stream=0)
            elif otype is np.complex128:
                oary = cudaFFT.complex128tocomplex128_FFT(ary, batch=1, stream=0)
        else:
            raise Exception('cuFFT only support single and double precsion float and complex numpy arrays!')
            
        return oary
    
    # Skip using plans and just use the fft in accelerate.cuda.fft
    def cuFFT_v2(ary, out=None, stream=0):
        itype = ary.dtype.type
        if out is not None:
            otype = out.dtype.type
            if otype is np.complex64:
                fft.fft(np.fft.fftshift(ary.astype(np.complex64)),out=out,stream=stream)
                out = np.fft.fftshift(out)
            else:
                raise Exception('Output must be type numpy.complex64' )
            return out
        else:
            if itype is not np.complex64:
                ary = ary.astype(np.complex64)
            tmp = np.fft.fftshift(ary)
            fft.fft_inplace(tmp, stream=stream)
            ary = np.fft.fftshift(tmp)
            return ary

    # Skip using plans and just use the fft in accelerate.cuda.fft
    def cuIFFT_v2(ary, out=None, stream=0):
        itype = ary.dtype.type
        if out is not None:
            otype = out.dtype.type
            if otype is np.complex64:
                fft.ifft(np.fft.ifftshift(ary.astype(np.complex64)),out=out,stream=stream)
                out = np.fft.ifftshift(out)
            else:
                raise Exception('Output must be type numpy.complex64' )
            return out
        else:
            if itype is not np.complex64:
                ary = ary.astype(np.complex64)
            tmp = np.fft.ifftshift(ary)
            fft.ifft_inplace(tmp, stream=stream)
            ary = np.fft.ifftshift(tmp)
            return ary
                        
class PyPropCUDA(cudaFFT, cuFFT_Utils):
    pass