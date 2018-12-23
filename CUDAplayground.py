#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 10:58:49 2018

@author: archdaemon
"""

import numpy as np
from numba import cuda, float32



TPB = 16

@cuda.jit
def matmul(A, B, C):
    """
    Perform matrix multiplication C = A * B
    """
    
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row,k] * B[k,col]
        C[row,col] = tmp
        
  
@cuda.jit
def fast_matmul(A, B, C):
    """
    Perform matrix multiplication C = A * B
    Each thread computes one element of the result matrix C
    """
    
    # Defind an array in shared memory
    sA = cuda.shared.array(shape=(TPB,TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB,TPB), dtype=float32)
    
    x, y = cuda.grid(2)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    
    if x >= C.shape[0] and y>= C.shape[1]:
        return
    
    # Each Thread computes one element in the result matrix
    # The dot product is chunked into dot products of TPB-long vectors
    tmp = 0.
    for i in range(int(A.shape[1] / TPB)):
        # Preload data into shared memory
        sA[tx,ty] = A[x, ty+i*TPB]
        sB[tx,ty] = B[tx+i*TPB,ty]
        
        # Wait until all threads finish preloading
        cuda.syncthreads()
        
        # Compute partial product on shared memory
        for j in range(TPB):
            tmp += sA[tx,j] * sB[j,ty]
            
        # Wait until all threads finish
        cuda.syncthreads()
    C[x,y] = tmp
    
    
      

# Host Code
print(cuda.gpus)    

N = TPB*2
M = TPB*3
P = TPB

# Initialize the data arrays
A = np.full((N,M),3,np.float32)
B = np.full((M,P),4,np.float32)

# Copy the arrays to the GPU
A_global_mem = cuda.to_device(A)
B_global_mem = cuda.to_device(B)

# Allocate memory on GPU for result
C_global_mem = cuda.device_array((N,P))

# Configure the GPU blocks
threadsperblock = (TPB,TPB)
blockspergrid_x = int(np.ceil(A.shape[0] / threadsperblock[1]))
blockspergrid_y = int(np.ceil(B.shape[1] / threadsperblock[0]))
blockspergrid = (blockspergrid_x, blockspergrid_y)


# Start Kernel
fast_matmul[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)

# Copy Result to CPU
C = C_global_mem.copy_to_host()

print(C)