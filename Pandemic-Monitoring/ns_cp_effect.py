import numpy as np
from scipy import stats
# from expon_cusum import Lv,F1
# import matplotlib.pyplot as plt

# This script is used to investiage the effect of change-point greater than 1.

## From first commit
class F1(object):
    def __init__(self,c,mu_1=1,sig_1=1):
        self.c = c
        self.mu_1 = mu_1
        self.dist = stats.norm(0,sig_1)
    def pdf(self,y,nu,t):
        # Require: t > nu
        assert(not isinstance(y,np.ndarray))
        return self.dist.pdf(y-self.mu_1*np.exp(self.c*(t-nu)))
    def logpdf(self,y,nu,t):
        # Require: t > nu
        assert(not isinstance(y,np.ndarray))
        return self.dist.logpdf(y-self.mu_1*np.exp(self.c*(t-nu)))
    def rvs(self,size,nu,t):
        # Require: t > nu
        return self.dist.rvs(size=size) + self.mu_1*np.exp(self.c*(np.arange(t,t+size)-nu))
    def logpdf_arr(self,y):
        assert(isinstance(y,np.ndarray))
        ind = np.arange(float(len(y)))
        return np.sum(self.dist.logpdf(y-self.mu_1*np.exp(self.c*ind)))
def gfun(y,fc,f0):
    llr = np.zeros(len(y))
    for k in range(len(y)):
        llr[k] = fc.logpdf_arr(y[k:]) - np.sum(f0.logpdf(y[k:]))
    # print(llr)
    return (np.argmax(llr),np.amax(llr))
def Lv(y, f0, c, t):
    max_k, max_llr = gfun(y,F1(c,mu_1=f0.mean(),sig_1=np.sqrt(f0.var())),f0)
    # print(t,max_k)
    return max_llr

c = 0.4
f0 = stats.norm(0.1,100)
f1 = F1(c,mu_1=f0.mean(),sig_1=np.sqrt(f0.var()))


b_1 = np.exp(np.linspace(np.log(2.5),np.log(4.8),5))
# b_2 = np.exp(np.linspace(np.log(2.5),np.log(4.8),5))
b_3 = np.exp(np.linspace(np.log(2.5),np.log(4.8),5))

N = 10000

possib_nu = np.array([1,5,10,15,20,50,100])

# mtfa_1 = np.zeros(len(b_1))
# mtfa_2 = np.zeros(len(b_2))
# mtfa_3 = np.zeros(len(b_3))
add_1 = np.zeros((len(possib_nu),len(b_1)))
# add_2 = np.zeros((len(possib_nu),len(b_2)))
add_3 = np.zeros((len(possib_nu),len(b_3)))
invd_1 = np.zeros((len(possib_nu),len(b_1)))
invd_3 = np.zeros((len(possib_nu),len(b_3)))
W_1 = 15
# W_2 = 20
W_3 = 25

for j in range(N):
    # for i in range(len(b_1)):
    #     t = W_1
    #     y = f0.rvs(size=W_1)
    #     while True:
    #         try:
    #             y[t]
    #         except IndexError:
    #             y = np.append(y,f0.rvs(size=W_1))
    #         v_1 = Lv(y[max(1,t-W_1+1):t+1], f0, c, t)
    #         if (v_1 > b_1[i]):
    #             mtfa_1[i] = mtfa_1[i] + t - W_1 + 1
    #             break
    #         t = t + 1
    #     # print(mtfa_1[i])
    #
    # for i in range(len(b_2)):
    #     t = W_2
    #     y = f0.rvs(size=W_2)
    #     while True:
    #         try:
    #             y[t]
    #         except IndexError:
    #             y = np.append(y,f0.rvs(size=W_2))
    #         v_2 = Lg(y[t-W_2+1:t+1], f0, fc_grid, t)
    #         if (v_2 > b_2[i]):
    #             mtfa_2[i] = mtfa_2[i] + t - W_2 + 1
    #             break
    #         t = t + 1
    #     # print(mtfa_2[i])
    #
    # for i in range(len(b_3)):
    #     t = W_3
    #     y = f0.rvs(size=W_3)
    #     while True:
    #         try:
    #             y[t]
    #         except IndexError:
    #             y = np.append(y,f0.rvs(size=W_3))
    #         v_3 = Lg(y[t-W_3+1:t+1], f0, fc_grid, t)
    #         if (v_3 > b_3[i]):
    #             mtfa_3[i] = mtfa_3[i] + t - W_3 + 1
    #             break
    #         t = t + 1

    for ni in range(len(possib_nu)):
        nu = possib_nu[ni]
        for i in range(len(b_1)):
            t = W_1
            y = f0.rvs(size=W_1+nu-1)
            while True:
                try:
                    y[t]
                except IndexError:
                    y = np.append(y,f1.rvs(size=W_1,nu=nu+W_1-1,t=t))
                v_1 = Lv(y[max(1,t-W_1+1):t+1], f0, c, t)
                if (v_1 > b_1[i]) and t >= nu:
                    add_1[ni,i] = add_1[ni,i] + t - W_1 - nu + 1
                    break
                elif v_1 > b_1[i]:
                    invd_1[ni,i] += 1
                    break
                t = t + 1

        # for i in range(len(b_2)):
        #     t = W_2
        #     y = f0.rvs(size=W_2+nu-1)
        #     while True:
        #         try:
        #             y[t]
        #         except IndexError:
        #             y = np.append(y,f1.rvs(size=W_2,nu=nu+W_2-1,t=t))
        #         v_2 = Lg(y[t-W_2+1:t+1], f0, fc_grid, t)
        #         if (v_2 > b_2[i]):
        #             add_2[ni,i] = add_2[ni,i] + t - W_2 - nu + 1
        #             break
        #         t = t + 1

        for i in range(len(b_3)):
            t = W_3
            y = f0.rvs(size=W_3+nu-1)
            while True:
                try:
                    y[t]
                except IndexError:
                    y = np.append(y,f1.rvs(size=W_3,nu=nu+W_3-1,t=t))
                v_3 = Lv(y[max(1,t-W_3+1):t+1], f0, c, t)
                if (v_3 > b_3[i]) and t >= nu:
                    add_3[ni,i] = add_3[ni,i] + t - W_3 - nu + 1
                    break
                elif v_3 > b_3[i]:
                    invd_3[ni,i] += 1
                    break
                t = t + 1

    if j % 10 == 9:
        print("Num of Iterations:", (j+1))
        # print("mtfa_1:", mtfa_1 / (j+1))
        # print("mtfa_2:", mtfa_2 / (j+1))
        # print("mtfa_3:", mtfa_3 / (j+1))
        print("add_1:", add_1 / (j+1-invd_1))
        # print("add_2:", add_2 / (j+1))
        print("add_3:", add_3 / (j+1-invd_3))
        # with open('./t1.npy', 'wb') as f:
        #     np.save(f, mtfa_1 / (j+1))
        #     np.save(f, mtfa_2 / (j+1))
        #     np.save(f, mtfa_3 / (j+1))
        #     np.save(f, add_1 / (j+1))
        #     np.save(f, add_2 / (j+1))
        #     np.save(f, add_3 / (j+1))


###
