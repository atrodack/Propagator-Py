#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:51:19 2018

@author: archdaemon
"""

import numpy as np
import scipy.optimize as so


# define the function F(x)
def Quadratic(x,A):
    if x.shape[0] != 3:
        x.shape = [3,1]
    return np.dot(x.transpose(), np.dot(A,x))/2

def Quadratic_der(x,A):
    return np.dot(A,x)

def Quadratic_Hess(x,A):
    return A

# Make a positive definite matrix A
A = np.array([(2.,-1.,0.),(-1.,2.,-1.),(0.,-1.,2)])


# Store some sizes
M = A.shape[0]
N = A.shape[1]

# Check the condition on A
A_cond = np.linalg.cond(A);
print("The condition of A is ", A_cond, "\n")

# Start with a stupid guess to see if it can descend to the correct answer
x0 = 12*np.ones([3,1],dtype=float) * np.random.normal(loc=0., scale = 1., size=[3,1])

# Set up BFGS
opts = {'disp': False, 'maxiter': 50, 'gtol': 1.e-7, 'return_all': True}
fargs = (A)
res_BFGS_noJ = so.minimize(Quadratic, x0, args=fargs, method='BFGS', jac=None, options=opts)
res_BFGS_J = so.minimize(Quadratic, x0, args=fargs, method='BFGS', jac=Quadratic_der, options=opts)

#%timeit res_BFGS_noJ = so.minimize(Quadratic, x0, args=fargs, method='BFGS', jac=None, options=opts)
#%timeit res_BFGS_J = so.minimize(Quadratic, x0, args=fargs, method='BFGS', jac=Quadratic_der, options=opts)


# Set up Newton-CG
opts = {'disp': False, 'maxiter': 50, 'xtol': 1.e-6, 'return_all': True}
res_NewtonCG_J = so.minimize(Quadratic,x0,args=fargs,method='Newton-CG', jac = Quadratic_der, hess=None)
res_NewtonCG = so.minimize(Quadratic,x0,args=fargs,method='Newton-CG', jac = Quadratic_der, hess=Quadratic_Hess)

#%timeit res_NewtonCG_J = so.minimize(Quadratic,x0,args=fargs,method='Newton-CG', jac = Quadratic_der, hess=None)
#%timeit res_NewtonCG = so.minimize(Quadratic,x0,args=fargs,method='Newton-CG', jac = Quadratic_der, hess=Quadratic_Hess)


print('BFGS with no Jacobian Provided found: ', res_BFGS_noJ.x, '\n')

print('BFGS with Jacobian Provided found: ', res_BFGS_J.x, '\n')

print('Newton-CG with Jacobian Provided no Hessian found: ', res_NewtonCG_J.x, '\n')

print('Newton-CG with Jacobian and Hessian Provided found: ', res_NewtonCG.x, '\n')