import numpy as np
from numpy.random import normal
import numba as nb
from numba import njit, prange
# from numba_stats import norm
# from scipy import stats
# from expon_cusum import Lv,F1
# import matplotlib.pyplot as plt

### This script is used to compare WL CuSum under different windows for the new model.

@njit
def norm_logpdf(x, loc, std):
    r = (x-loc)/std
    return -0.5*r*r-0.5*np.log(2*np.pi*std*std)

# F1spec = [('neggam', nb.float32), ('mu_1', nb.float32), ('sig_1', nb.float32)]
# @jitclass(F1spec)
# class F1(object):
#     def __init__(self,gamma,mu_1=1,sig_1=1):
#         self.neggam = -gamma
#         self.mu_1 = mu_1
#         self.sig_1 = sig_1
#         # self.dist = norm(0,sig_1)
#     # def pdf(self,y,nu,t):
#     #     # Require: t > nu
#     #     assert(not isinstance(y,np.ndarray))
#     #     return self.dist.pdf(y-self.mu_1*(t-nu+1)**self.neggam)
#     # def logpdf(self,y,nu,t):
#     #     # Require: t > nu
#     #     # assert(not isinstance(y,np.ndarray))
#     #     return self.dist.logpdf(y-self.mu_1*(t-nu+1)**self.neggam)
#     def rvs(self,size,nu,t):
#         # Require: t >= nu
#         return normal(0,self.sig_1,size=size) + self.mu_1*(np.arange(t,t+size)-nu+1)**self.neggam
#     def logpdf_arr(self,y) -> float:
#         # assumes nu=0
#         # assert(isinstance(y,np.ndarray))
#         ind = np.arange(float(len(y)))
#         return np.sum(norm_logpdf(y-self.mu_1*(ind+1)**self.neggam, 0, self.sig_1))

neggam = -0.2
# f0 = stats.norm(0,2)
mu_0 = 0
sig_0 = 2
# f1 = F1(gamma, mu_1=2, sig_1=sig_0)
mu_1 = 2
sig_1 = sig_0

@njit
def f1_rvs(t,nu,mu_1,sig_1,size):
    # Require: t >= nu
    return normal(0,sig_1,size=size) + mu_1*(np.arange(t,t+size)-nu+1)**neggam

@njit
def f1_logpdf_sum(y,mu_1,sig_1) -> float:
    ind = np.arange(float(len(y)))
    return np.sum(norm_logpdf(y-mu_1*(ind+1)**neggam, 0, sig_1))

@njit
def gfun(y) -> float:
    llr = np.zeros(len(y))
    for k in range(len(y)):
        llr[k] = f1_logpdf_sum(y[k:],mu_1,sig_1) - np.sum(norm_logpdf(y[k:],mu_0,sig_0))
    # print(llr)
    # return (np.argmax(llr),np.amax(llr))
    return np.amax(llr)

# @njit(parallel=True)
# def Lv(y) -> float:
#     # max_k, max_llr = gfun(y)
#     max_llr = gfun(y)
#     # print(t,max_k)
#     return max_llr


b_1 = np.exp(np.linspace(np.log(3),np.log(5),5))
b_2 = np.exp(np.linspace(np.log(3),np.log(5),5))
b_3 = np.exp(np.linspace(np.log(3),np.log(5),5))

N = 100

mtfa_1 = np.zeros((len(b_1), N))
mtfa_2 = np.zeros((len(b_2), N))
mtfa_3 = np.zeros((len(b_3), N))
# mtfa_c = np.zeros((len(b_1), N))
add_1 = np.zeros((len(b_1), N))
add_2 = np.zeros((len(b_2), N))
add_3 = np.zeros((len(b_3), N))
# add_c = np.zeros((len(b_3), N))
W_1 = 15
W_2 = 25
W_3 = 40

# @njit
def fa(b, W):
    mtfa = np.zeros(len(b))
    for i in prange(len(b)):
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

def fa_inf(b):
    mtfa = np.zeros(len(b))
    for i in prange(len(b)):
        t = 1
        y = normal(mu_0,sig_0,size=2000)
        while True:
            if t >= len(y):
                y = np.append(y,normal(mu_0,sig_0,size=1000))
            v = gfun(y[1:t+1])
            if (v > b[i]):
                mtfa[i] = t
                break
            t = t + 1
    return mtfa

# @njit
def delay(b, W):
    add = np.zeros(len(b))
    for i in prange(len(b)):
        t = W
        y = normal(mu_0,sig_0,size=W)
        while True:
            if t >= len(y):
                y = np.append(y,f1_rvs(t=t,nu=W,mu_1=mu_1,sig_1=sig_1,size=W))
            v = gfun(y[t-W+1:t+1])
            if (v > b[i]):
                add[i] = t - W + 1
                break
            t = t + 1
    return add

def delay_inf(b):
    add = np.zeros(len(b))
    for i in prange(len(b)):
        t = 1
        y = normal(mu_0,sig_0,size=1)
        while True:
            if t >= len(y):
                y = np.append(y,f1_rvs(t=t,nu=1,mu_1=mu_1,sig_1=sig_1,size=1000))
            v = gfun(y[1:t+1])
            if (v > b[i]):
                add[i] = t
                break
            t = t + 1
    return add

for j in range(N):
    mtfa_1[:,j] = fa(b_1, W_1)
    mtfa_2[:,j] = fa(b_2, W_2)
    mtfa_3[:,j] = fa(b_3, W_3)
    # mtfa_c[:,j] = fa_inf(b_1)

    add_1[:,j] = delay(b_1, W_1)
    add_2[:,j] = delay(b_2, W_2)
    add_3[:,j] = delay(b_3, W_3)
    # add_c[:,j] = delay_inf(b_1)

    if j % 5 == 4:
        print("Num of Iterations:", (j+1))
        print("mtfa_1:", np.mean(mtfa_1[:,:j+1],axis=1))
        print("mtfa_2:", np.mean(mtfa_2[:,:j+1],axis=1))
        print("mtfa_3:", np.mean(mtfa_3[:,:j+1],axis=1))
        # print("mtfa_c:", np.mean(mtfa_c[:,:j+1],axis=1))
        print("add_1:", np.mean(add_1[:,:j+1],axis=1))
        print("add_2:", np.mean(add_2[:,:j+1],axis=1))
        print("add_3:", np.mean(add_3[:,:j+1],axis=1))
        # print("add_c:", np.mean(add_c[:,:j+1],axis=1))
        # with open('./t1.npy', 'wb') as f:
        #     np.save(f, mtfa_1)
        #     np.save(f, mtfa_2)
        #     np.save(f, mtfa_3)
        #     np.save(f, add_1)
        #     np.save(f, add_2)
        #     np.save(f, add_3)


###
