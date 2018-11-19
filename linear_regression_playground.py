#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 15:00:12 2018

@author: archdaemon
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so
import scipy.sparse as sparsemat


def FindMinCost(Cost, CostHessian, c0, y, wt=None, RegParam=None, method='Newton-CG', MaxIt=10, LsqStart=False, AmpCon=False, ):
    assert method in ['Newton-CG', 'BFGS'], "Invalid method"
    fargs = (y, wt, RegParam, True, AmpCon)
    if method == 'Newton-CG':
        opt = {'disp': True, 'maxiter': MaxIt, 'xtol': 1.e-6, 'return_all': True}
    elif method == 'BFGS':
        opt = {'disp': True, 'maxiter': MaxIt, 'gtol': 1.e-7, 'return_all': True}
        
    res = so.minimize(Cost, c0, args=fargs, method=method, options=opt, jac=False, hess=CostHessian)
    sol = res.x
    return(sol,res)
    

def Cost2d(x0, ):
    x = x0[0:N]
    A = x0[(N):(N*M+N)].reshape([M,N])
    x.shape = [x.shape[0],1]
    y_.shape = [y_.shape[0],1]
    return ((y_ - A@x).transpose() @ C_n_inv @ (y_-A@x)) + (np.ndarray.flatten(A) - np.ndarray.flatten(Z)).transpose() @ C_A_inv @ (np.ndarray.flatten(A) - np.ndarray.flatten(Z))

def Cost_der(x0, ):
    x = x0[0:N]
    A = x0[(N):(N*M+N)].reshape([M,N])
    x.shape = [x.shape[0],1]
    y_.shape = [y_.shape[0],1]
    der = np.vstack(((A.transpose() @ C_n_inv @ A @ x) - A.transpose() @ C_n_inv @ y_ , np.reshape( (-1 * C_n_inv @ (y_ - A@x) @ x.transpose()).reshape([N*M]) + 1* C_A_inv @ (np.ndarray.flatten(A) - np.ndarray.flatten(Z)) @ np.ndarray.flatten(A),[M*N,1])));
    der.shape = [der.shape[0]]
    return der



def Makex0(x,A, ):
    A_ = np.ndarray.flatten(A);
    x0 = np.zeros([x.shape[0]+A_.shape[0]])
    for ii in range((x.shape[0] + A_.shape[0])):
        if ii < x.shape[0]:
            x0[ii] = x[ii]
        else:
            x0[ii] = A_[ii-x.shape[0]]
    
    return x0
 
# Set up the linear Regression
M = 5;
N = 3;
nRealizations = 1000;

# Define y and A
y = np.transpose(np.array([4.,19.,39.,1.,-2.]));
A = np.array([ (1.,2.,-1.), (3.,-1.,4.), (-2.,5.,7.), (3.,1.,-2.), (6.,2.,-5.) ]);
A2 = np.array([(2,-1,0), (-1,2,-1), (0, -1, 2)]);
x_true = np.transpose(np.array([2,3,4]))

# x is N x 1

# Define Noise Stats for Measurement
mu_n = 0.;
var_n = 2;
C_n = var_n * np.eye(M)
C_n_inv = np.linalg.pinv(C_n);


# Least Squares, no noise
x_LS = np.linalg.pinv(A) @ y;
print('True x is:\n', x_LS,'\n')

# Now add noise to the measurement of y
n = np.random.normal(loc=mu_n, scale = var_n,size=M)
y_ = y-n;

# Do the ML estimate
x_hat_ML = np.linalg.pinv(np.transpose(A) @ C_n_inv @ A) @ np.transpose(A) @ C_n_inv @ y_;
C_x_hat_ML = np.linalg.pinv(np.transpose(A) @ C_n_inv @ A)
print('Noise in y, ML Estimate:\n', x_hat_ML, '\n')
print('covar(x_hat_ML | x): \n', C_x_hat_ML, '\n')

# Assume Ergodicity, and compute the expectation of the estimator by running it over multiple realizations of the noise

x_ML_expt = np.zeros([nRealizations,N])
for ii in range(nRealizations):
    n = np.random.normal(loc=mu_n, scale = var_n,size=M)
    y_ = y-n;
    
    # Do the ML estimate
    x_ML_expt[ii,:] = np.linalg.pinv(np.transpose(A) @ C_n_inv @ A) @ np.transpose(A) @ C_n_inv @ y_;

print('Expected x_hat_ML is \n', np.mean(x_ML_expt,0),'\n')


# Define noise stats in A
mu_nA = 0.;
var_nA = 1;
C_A = var_nA * np.eye(M*N)
C_A_inv = np.linalg.pinv(C_A)

nA = np.random.normal(loc = mu_nA, scale = var_nA, size=M*N)
n = np.random.normal(loc=mu_n, scale = var_n,size=M)

y_ = y - n;
A_ = np.ndarray.flatten(A);
Z = A_ + nA;
Z = np.reshape(Z,[M,N]);

# Naive Estimate
x_hat_ML_Naive = np.linalg.pinv(np.transpose(Z) @ C_n_inv @ Z) @ np.transpose(Z) @ C_n_inv @ y_;

C_x_hat_ML_Naive = np.linalg.pinv(np.transpose(Z) @ C_n_inv @ Z)


print('x_hat_ML from the Naive Estimate: \n', x_hat_ML_Naive, '\n')


x_ML_expt_Naive = np.zeros([nRealizations,N])
for ii in range(nRealizations):
    n = np.random.normal(loc=mu_n, scale = var_n,size=M)
    y_ = y-n;
    
    nA = np.random.normal(loc = mu_nA, scale = var_nA, size=M*N)
    Z = A_ - nA;
    Z = np.reshape(Z,[M,N]);
    
    # Do the ML estimate
    x_ML_expt_Naive[ii,:] = np.linalg.pinv(np.transpose(Z) @ C_n_inv @ Z) @ np.transpose(Z) @ C_n_inv @ y_;

print('Expected x_hat_ML_Naive is \n', np.mean(x_ML_expt_Naive,0),'\n')



x0 = Makex0(np.ones(3),np.ones([M,N]))
opts = {'disp': False, 'maxiter': 70, 'gtol': 1.e-7, 'return_all': True}
#opts = {'disp': False}
res = so.minimize(Cost2d, x0, method='BFGS', jac=None, options=opts)

print('BFGS Finds:\nx_hat = \n',res.x[0:N],'\nA_hat = \n' ,res.x[(N):(N+N*M)].reshape([M,N]))



x_ML_expt_Joint = np.zeros([nRealizations,N+N*M])
for ii in range(nRealizations):
    n = np.random.normal(loc=mu_n, scale = var_n,size=M)
    y_ = y-n;
    
    nA = np.random.normal(loc = mu_nA, scale = var_nA, size=M*N)
    Z = A_ - nA;
    Z = np.reshape(Z,[M,N]);
    
    # Do the ML estimate
    res = so.minimize(Cost2d, x0, method='BFGS', jac=None, options=opts)
    x_ML_expt_Joint[ii,:] = res.x

print('BFGS Finds:\nx_hat = \n',np.mean(x_ML_expt_Joint,0)[0:N],'\nA_hat = \n' ,np.mean(x_ML_expt_Joint,0)[(N):(N+N*M)].reshape([M,N]))
