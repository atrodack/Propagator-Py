#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 19:10:51 2018

@author: archdaemon
"""

import numpy as np
import numba
from numba import jit
import timeit

print(numba.__version__)

@jit(nopython=True)
def go_fast(a):
    trace = 0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i,i])
    return a + trace

x = np.arange(100).reshape(10,10)
tmp = go_fast(x)

tmp2 = go_fast(2*x)


# %timeit go_fast(x)