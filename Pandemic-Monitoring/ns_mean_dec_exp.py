import numpy as np
from numpy.random import exponential
from numba import njit

### This script is used to compare WL CuSum under different windows for the new exponential change model.

@njit
def exp_logpdf(x, lam):
    # x has the same shape as lam
    return -lam * x + np.log(lam)

negtheta = -0.35
lam_0 = 1
beta_0 = 1 / lam_0

@njit
def f1_logpdf_sum(y) -> float:
    ind = np.arange(float(len(y)))
    return np.sum(exp_logpdf(y, lam_0*(1-(ind+2)**negtheta)))

@njit
def gfun(y) -> float:
    llr = np.zeros(len(y))
    for k in range(len(y)):
        llr[k] = f1_logpdf_sum(y[k:]) - np.sum(exp_logpdf(y[k:],lam_0))
    # print(llr)
    # return (np.argmax(llr),np.amax(llr))
    return np.amax(llr)

b_1 = np.exp(np.linspace(np.log(3),np.log(5),5))
b_2 = np.exp(np.linspace(np.log(3),np.log(5),5))
b_3 = np.exp(np.linspace(np.log(3),np.log(5),5))
b_4 = np.exp(np.linspace(np.log(3),np.log(5),5))

N = 100

mtfa_1 = np.zeros((len(b_1), N))
mtfa_2 = np.zeros((len(b_2), N))
mtfa_3 = np.zeros((len(b_3), N))
mtfa_4 = np.zeros((len(b_4), N))
add_1 = np.zeros((len(b_1), N))
add_2 = np.zeros((len(b_2), N))
add_3 = np.zeros((len(b_3), N))
add_4 = np.zeros((len(b_4), N))
W_1 = 15
W_2 = 25
W_3 = 50
W_4 = 100

# @njit
def fa(b, W):
    mtfa = np.zeros(len(b))
    for i in range(len(b)):
        t = W
        y = exponential(beta_0,size=2000)
        while True:
            if t >= len(y):
                y = np.append(y,exponential(beta_0,size=1000))
            v = gfun(y[t-W+1:t+1])
            if (v > b[i]):
                mtfa[i] = t - W + 1
                break
            t = t + 1
    return mtfa

# @njit
def delay(b, W):
    add = np.zeros(len(b))
    for i in range(len(b)):
        t = W
        y = exponential(beta_0,size=W)
        while True:
            if t >= len(y):
                y = np.append(y,np.empty(2*W))
            lam_t = lam_0*(1-(t-W+2)**negtheta)
            y[t] = exponential(1/lam_t)
            v = gfun(y[t-W+1:t+1])
            if (v > b[i]):
                add[i] = t - W + 1
                break
            t = t + 1
    return add

for j in range(N):
    mtfa_1[:,j] = fa(b_1, W_1)
    mtfa_2[:,j] = fa(b_2, W_2)
    mtfa_3[:,j] = fa(b_3, W_3)
    mtfa_4[:,j] = fa(b_4, W_4)

    add_1[:,j] = delay(b_1, W_1)
    add_2[:,j] = delay(b_2, W_2)
    add_3[:,j] = delay(b_3, W_3)
    add_4[:,j] = delay(b_4, W_4)

    if j % 5 == 4:
        print("Num of Iterations:", (j+1))
        print("mtfa_1:", np.mean(mtfa_1[:,:j+1],axis=1))
        print("mtfa_2:", np.mean(mtfa_2[:,:j+1],axis=1))
        print("mtfa_3:", np.mean(mtfa_3[:,:j+1],axis=1))
        print("mtfa_4:", np.mean(mtfa_4[:,:j+1],axis=1))
        print("add_1:", np.mean(add_1[:,:j+1],axis=1))
        print("add_2:", np.mean(add_2[:,:j+1],axis=1))
        print("add_3:", np.mean(add_3[:,:j+1],axis=1))
        print("add_4:", np.mean(add_4[:,:j+1],axis=1))
        # with open('./t1.npy', 'wb') as f:
        #     np.save(f, mtfa_1)
        #     np.save(f, mtfa_2)
        #     np.save(f, mtfa_3)
        #     np.save(f, add_1)
        #     np.save(f, add_2)
        #     np.save(f, add_3)


###
