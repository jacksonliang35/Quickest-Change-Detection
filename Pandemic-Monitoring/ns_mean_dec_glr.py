import numpy as np
from numpy.random import normal
import numba as nb
from numba import njit, prange

### This script is used to compare with GLR Cusum for the new model.

@njit
def norm_logpdf(x, loc, std):
    r = (x-loc)/std
    return -0.5*r*r-0.5*np.log(2*np.pi*std*std)

true_neggam = -0.2
neggam_set = np.linspace(-0.3,-0.1,100)
mu_0 = 0
sig_0 = 2
mu_1 = 2
sig_1 = sig_0

@njit
def f1_rvs(t,nu,mu_1,sig_1,neggam,size):
    # Require: t >= nu
    return normal(0,sig_1,size=size) + mu_1*(np.arange(t,t+size)-nu+1)**neggam

@njit
def f1_logpdf_sum(y,mu_1,sig_1,neggam) -> float:
    ind = np.arange(float(len(y)))
    return np.sum(norm_logpdf(y-mu_1*(ind+1)**neggam, 0, sig_1))

@njit
def gfun_param(y, param) -> float:
    # Aim for one-dimensional params
    llr = np.zeros((*param.shape, len(y)))
    for i in prange(len(param)):
        for k in prange(len(y)):
            llr[i,k] = f1_logpdf_sum(y[k:],mu_1,sig_1,param[i]) - np.sum(norm_logpdf(y[k:],mu_0,sig_0))
    return np.amax(llr)

N = 100

b_1 = np.exp(np.linspace(np.log(3),np.log(5),5))
b_2 = np.exp(np.linspace(np.log(3),np.log(5),5))

mtfa_1 = np.zeros((len(b_1), N))
mtfa_2 = np.zeros((len(b_2), N))
add_1 = np.zeros((len(b_1), N))
add_2 = np.zeros((len(b_2), N))

W_1 = 25
W_2 = 50

def fa(b, W):
    mtfa = np.zeros(len(b))
    for i in prange(len(b)):
        t = W
        y = normal(mu_0,sig_0,size=2000)
        while True:
            if t >= len(y):
                y = np.append(y,normal(mu_0,sig_0,size=1000))
            v = gfun_param(y[t-W+1:t+1], neggam_set)
            if (v > b[i]):
                mtfa[i] = t - W + 1
                break
            t = t + 1
    return mtfa

def delay(b, W):
    add = np.zeros(len(b))
    for i in prange(len(b)):
        t = W
        y = normal(mu_0,sig_0,size=W)
        while True:
            if t >= len(y):
                y = np.append(y,f1_rvs(t,W,mu_1,sig_1,true_neggam,W))
            v = gfun_param(y[t-W+1:t+1], neggam_set)
            if (v > b[i]):
                add[i] = t - W + 1
                break
            t = t + 1
    return add

for j in range(N):
    mtfa_1[:,j] = fa(b_1, W_1)
    mtfa_2[:,j] = fa(b_2, W_2)

    add_1[:,j] = delay(b_1, W_1)
    add_2[:,j] = delay(b_2, W_2)

    if j % 5 == 4:
        print("Num of Iterations:", (j+1))
        print("mtfa_1:", np.mean(mtfa_1[:,:j+1],axis=1))
        print("mtfa_2:", np.mean(mtfa_2[:,:j+1],axis=1))
        print("add_1:", np.mean(add_1[:,:j+1],axis=1))
        print("add_2:", np.mean(add_2[:,:j+1],axis=1))
        # with open('./t1.npy', 'wb') as f:
        #     np.save(f, mtfa_1)
        #     np.save(f, mtfa_2)
        #     np.save(f, add_1)
        #     np.save(f, add_2)


###
