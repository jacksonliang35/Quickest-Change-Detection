import numpy as np
from numpy.random import normal
from numba import njit

### This script is used to simulate GEM. Use Numba to accelerate.

@njit
def norm_logpdf(x, loc, std):
    r = (x-loc)/std
    return -0.5*r*r-0.5*np.log(2*np.pi*std*std)

mu_0 = 0.1
sig_0 = 100
mu_1 = mu_0
sig_1 = sig_0
c = 0.4

@njit
def f1_rvs(t,nu,size):
    # Require: t >= nu
    return normal(0,sig_1,size=size) + mu_1*np.exp(c*(np.arange(t,t+size)-nu))

@njit
def f1_logpdf_sum(y,c) -> float:
    ind = np.arange(float(len(y)))
    return np.sum(norm_logpdf(y-mu_1*np.exp(c*ind), 0, sig_1))

@njit
def gfun(y) -> float:
    llr = np.zeros(len(y))
    for k in range(len(y)):
        llr[k] = f1_logpdf_sum(y[k:],c) - np.sum(norm_logpdf(y[k:],mu_0,sig_0))
    return np.amax(llr)

b_1 = np.exp(np.linspace(np.log(2.5),np.log(4.8),5))
b_2 = np.exp(np.linspace(np.log(2.5),np.log(4.8),5))
W_1 = 15
W_2 = 25

N = 20000

mtfa_1 = np.zeros((len(b_1), N))
mtfa_2 = np.zeros((len(b_2), N))
add_1 = np.zeros((len(b_1), N))
add_2 = np.zeros((len(b_2), N))

def fa(b, W):
    mtfa = np.zeros(len(b))
    for i in range(len(b)):
        t = W
        y = normal(mu_0,sig_0,size=2000)
        while True:
            if t >= len(y):
                y = np.append(y,normal(mu_0,sig_0,size=1000))
            v = gfun(y[t-W+1:t+1])
            if (v > b[i]):
                mtfa[i] = t - W + 1
                break
            t = t + 1
    return mtfa

def delay(b, W):
    add = np.zeros(len(b))
    for i in range(len(b)):
        t = W
        y = normal(mu_0,sig_0,size=W)
        while True:
            if t >= len(y):
                y = np.append(y,f1_rvs(t,W,W))
            v = gfun(y[t-W+1:t+1])
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

    if j % 2000 == 1999:
        print("Num of Iterations:", (j+1))
        print("mtfa_1:", np.mean(mtfa_1[:,:j+1],axis=1))
        print("mtfa_2:", np.mean(mtfa_2[:,:j+1],axis=1))
        print("add_1:", np.mean(add_1[:,:j+1],axis=1))
        print("add_2:", np.mean(add_2[:,:j+1],axis=1))
        with open('./t1.npy', 'wb') as f:
            np.save(f, mtfa_1)
            np.save(f, mtfa_2)
            np.save(f, add_1)
            np.save(f, add_2)


###
