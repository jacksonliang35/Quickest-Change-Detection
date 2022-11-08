import numpy as np
from scipy.optimize import fsolve
from scipy.special import gamma
import scipy.stats as stats
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import time

# Hyper-parameters
alpha1 = 4
alpha2 = 4.5
beta = 16
f0 = stats.beta(a=alpha1, b=beta)
f1 = stats.beta(a=alpha2, b=beta)
eta = 0.21
mu0 = f0.mean()
print("f0 mean:", mu0)
print("f0 var:", f0.var())
print("f1 mean:", f1.mean())
delta = (eta-mu0) / 2
mid = (eta+mu0) / 2

# True LLR
c =  gamma(alpha1) * gamma(alpha2+beta) / (gamma(alpha2) * gamma(alpha1+beta))
T = lambda x: np.log(c*x**(alpha2-alpha1))

# Asym Robust Test Stat
lam = fsolve(lambda a: f0.expect(lambda x: x*np.exp(a*x))/f0.expect(lambda x: np.exp(a*x))-eta, 0.00001)[0]
kappa0 = np.log(f0.expect(lambda x: np.exp(lam*x)))

# Running simulation...
b_1 = np.exp(np.linspace(np.log(1.3),np.log(2.7),5))    # CuSum
b_2 = np.exp(np.linspace(np.log(.7),np.log(1.9),5))     # Known Pre-change
b_3 = np.exp(np.linspace(np.log(.6),np.log(1.5),5))     # MCT

N = 1000
mtfa_1 = np.zeros(len(b_1))
mtfa_2 = np.zeros(len(b_2))
mtfa_3 = np.zeros(len(b_3))
wadd_1 = np.zeros(len(b_1))
wadd_2 = np.zeros(len(b_2))
wadd_3 = np.zeros(len(b_3))

# start = time.time()
for j in range(N):
    for i in range(len(b_1)):
        t = 0
        w_1 = 0
        while True:
            t = t + 1
            y = f0.rvs(size=1)[0]
            w_1 = np.maximum(0, w_1 + T(y))
            if (w_1 > b_1[i]):
                mtfa_1[i] = mtfa_1[i] + t
                break

    for i in range(len(b_2)):
        t = 0
        w_2 = 0
        while True:
            t = t + 1
            y = f0.rvs(size=1)[0]
            w_2 = np.maximum(0, w_2 + lam*y-kappa0)
            if (w_2 > b_2[i]):
                mtfa_2[i] = mtfa_2[i] + t
                break

    for i in range(len(b_3)):
        t = 0
        w_3 = 0
        while True:
            t = t + 1
            y = f0.rvs(size=1)[0]
            w_3 = np.maximum(0, w_3 + y-mid)
            if (w_3 > b_3[i]):
                mtfa_3[i] = mtfa_3[i] + t
                break

    for i in range(len(b_1)):
        t = 0
        w_1 = 0
        while True:
            t = t + 1
            y = f1.rvs(size=1)[0]
            w_1 = np.maximum(0, w_1 + T(y))
            if (w_1 > b_1[i]):
                wadd_1[i] = wadd_1[i] + t
                break

    for i in range(len(b_2)):
        t = 0
        w_2 = 0
        while True:
            t = t + 1
            y = f1.rvs(size=1)[0]
            w_2 = np.maximum(0, w_2 + lam*y-kappa0)
            if (w_2 > b_2[i]):
                wadd_2[i] = wadd_2[i] + t
                break

    for i in range(len(b_3)):
        t = 0
        w_3 = 0
        while True:
            t = t + 1
            y = f1.rvs(size=1)[0]
            w_3 = np.maximum(0, w_3 + y-mid)
            if (w_3 > b_3[i]):
                wadd_3[i] = wadd_3[i] + t
                break

    if j % 10 == 9:
        print("Num of Iterations:", (j+1))
        print("wadd_1:", wadd_1 / (j+1))
        print("wadd_2:", wadd_2 / (j+1))
        print("wadd_3:", wadd_3 / (j+1))
        print("mtfa_1:", mtfa_1 / (j+1))
        print("mtfa_2:", mtfa_2 / (j+1))
        print("mtfa_3:", mtfa_3 / (j+1))
        # with open('./t1.npy', 'wb') as f:
        #     np.save(f, wadd_1 / (j+1))
        #     np.save(f, wadd_2 / (j+1))
        #     np.save(f, wadd_3 / (j+1))
        #     np.save(f, mtfa_1 / (j+1))
        #     np.save(f, mtfa_2 / (j+1))
        #     np.save(f, mtfa_3 / (j+1))
# end = time.time()
# print("Total time used in min:", (end - start)/60)

## Ploting, somewhere else...


###
