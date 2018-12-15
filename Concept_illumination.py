#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 12:38:46 2018

 Consider the problem y = q*b + n
 with q,b scalars and n scalar noise from G(n;0,sigma_n)
 the value of y is known after the measurement has been made

 In a standard regression, we know q exactly (but not b) and then our estimate
 of b is \hat(b) = y/q, which is unbiased and has variance sigma_n^2

 If we don't know q,but have a noisy measurement of it q_m
 governed by G(q_m; q, sigma_q)
 The Joint Likelihood is given by:
 P(y,q_m | q,b) = P(y|q,b) P(q_m|q) = G(y; q*b, sigma_n) G(q_m; q, sigma_q)

 Make a 2D image of the marginal likelihood (integrating P(y, q_m|q,b) over q_m)
 basically the probability of y as a function of the pair (q,b).

 It should be kind of like an ellipsoid with a curved major axis (more so as sigma_q increases).

 Then look at the distribution that governs the naive estimate \hat(b)_naive = y/q_m,
 in which y and q_m are drawn from the distributions:
 y ~ G(y; q_m*b, sigma_n) and q_m ~ G(q_m; q, sigma_q)
 Monte Carlo by drawing q_m first and then drawing y.

 Should see that this distribution is not maximized at \hat{b}_naive = q/b and 
 that the expectation of \hat{b}_naive is not q/b
 
@author: archdaemon
"""

import numpy as np
import matplotlib.pyplot as plt


# Number of Realizations to contain in loops
nRealizations = 1000000

# Define our true scalars
q = 10.0
b = 3.0

# Stats of n
mu_n = 0.0
sigma_n = 1.0

# Compute the standard regression and find bias and variance
q_ = np.zeros([nRealizations,1])
n_ = np.zeros([nRealizations,1])
b_ML_expt = np.zeros([nRealizations,1])
for i in range(nRealizations):
    # Draw n
#    n_Reals[i,:] = np.random.normal(loc=mu_n, scale = sigma_n,size=1)
    n_[i,0] = np.random.normal(loc=mu_n, scale = sigma_n)
    # Take the measurement
    y_ = q*b + n_[i,0]
    
    b_ML_expt[i,:] = y_ / q
    
# Check the noise stats
n_mean = np.mean(n_,0)
n_var = np.var(n_,0)

#plt.figure()
#count, bins, ignored = plt.hist(n_,81,density=True)
#plt.plot(bins, 1/(sigma_n * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu_n)**2 / (2 * sigma_n**2) ),linewidth=2, color='r')
#plt.show()


print('Noise mean is ',n_mean, '   Noise variance is ',n_var,'\n')

# Compute the bias
b_mean = np.mean(b_ML_expt,0) 
b_bias = b - b_mean

# Compute the variance
b_variance = np.var(b_ML_expt,0)
sigma_b = np.std(b_ML_expt,0)

#plt.figure()
#count, bins, ignored = plt.hist(b_ML_expt,81,density=True)
#plt.plot(bins, 1/(sigma_b * np.sqrt(2 * np.pi)) * np.exp( - (bins - b_mean)**2 / (2 * sigma_b**2) ),linewidth=2, color='r')
#plt.show()

print('The bias of b in a standard regression is: ',b_bias,'   The variance of b in a standard regression is: ',b_variance,'This is (sigma_n / q)^2 \n')


# Do the Monte Carlo

# setup q_m
mu_q = q
sigma_q = 3.

q_m = np.zeros([nRealizations])
y_ = np.zeros([nRealizations])
b_ML_expt2 = np.zeros([nRealizations])

for i in range(nRealizations):   
    # Draw q
    q_m[i] = np.random.normal(loc=mu_q, scale = sigma_q)
    
    # Draw y
    y_[i] = np.random.normal(loc=q_m[i]*b, scale = sigma_n)

    # Do the Estimate
    b_ML_expt2[i] = y_[i] / q_m[i]



# Check the noise stats in q
q_mean = np.mean(q_m,0)
q_var = np.var(q_m,0)

#count, bins, ignored = plt.hist(q_m,81,density=True)
#plt.plot(bins, 1/(sigma_q * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu_q)**2 / (2 * sigma_q**2) ),linewidth=2, color='r')
#plt.show()
# 
print('Noise mean in q is ',q_mean, '   Noise variance in q is ',q_var,'\n')

# Compute the stats of y_ [variance should not be sigma_n^2 because q_m has variance]
y_mean = np.mean(y_)
y_variance = np.var(y_)
sigma_y = np.std(y_)


# Compute the bias
b_mean2 = np.mean(b_ML_expt2,0) 
b_bias2 = b - b_mean2
#
## Compute the variance
b_variance2 = np.var(b_ML_expt2,0)
sigma_b2 = np.std(b_ML_expt2,0)


#count, bins, ignored = plt.hist(y_,81,density=True)
#plt.plot(bins, 1/(sigma_y * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu_q*b)**2 / (2 * sigma_y**2) ),linewidth=2, color='r')
#plt.title('Distribution on y_m')
#plt.show()

print('The mean of y is:  ', y_mean,'   The variance of y is: ',y_variance)
print('The bias of b with noisy q is: ',b_bias2,'   The variance of b with noisy q is: ',b_variance2,'\n')

