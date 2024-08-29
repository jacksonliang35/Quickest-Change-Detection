import numpy as np
from scipy import stats
from scipy.special import gamma, polygamma
# from scipy.optimize import minimize
import matplotlib.pyplot as plt
from itertools import product

alpha = 1e-5
Tc = 10
maxt = 100
# c = .043      # For Gaussian change
c = 0.03      # For Beta change
c_min = 0.01
W = 20
print("False alarm rate:", alpha)
print("Change point:", Tc)
print("Window size:", W)
def dft_mcfun(x,c):
    return np.exp(c[0]*x)

# Gaussian exponential mean-change
# class F1(object):
#     def __init__(self,c,mu_1=1,sig_1=1,mcfun=dft_mcfun):
#         self.c = tuple(c)
#         self.mu_1 = mu_1
#         self.dist = stats.norm(0,sig_1)
#         self.mcfun = mcfun      # Mean-change function: need mcfun(0) = 1
#     def pdf(self,y,nu,t):
#         # Require: t > nu
#         assert(not isinstance(y,np.ndarray))
#         return self.dist.pdf(y-self.mu_1*self.mcfun(t-nu,self.c))
#     def logpdf(self,y,nu,t):
#         # Require: t > nu
#         assert(not isinstance(y,np.ndarray))
#         return self.dist.logpdf(y-self.mu_1*self.mcfun(t-nu,self.c))
#     def rvs(self,size,nu,t):
#         # Require: t > nu
#         return self.dist.rvs(size=size) + self.mu_1*self.mcfun(np.arange(t,t+size)-nu, self.c)
#     def logpdf_arr(self,y):
#         assert(isinstance(y,np.ndarray))
#         ind = np.arange(float(len(y)))
#         return np.sum(self.dist.logpdf(y-self.mu_1*self.mcfun(ind,self.c)))

# Beta exponential alpha-change
class F1(object):
    def __init__(self,c,f0,mcfun=dft_mcfun,Wmax=W):
        self.c = tuple(c)
        self.f0 = f0
        self.alpha_1 = self.f0.args[0]
        self.beta_1 = self.f0.args[1]
        self.mcfun = mcfun      # Mean-change function: need mcfun(0) = 1
        self.library = None
        self.setup_lib(Wmax)
    # def pdf(self,y,nu,t):
    #     # Require: t > nu
    #     assert(not isinstance(y,np.ndarray))
    #     return self.dist.pdf(y-self.mu_1*self.mcfun(t-nu,self.c))
    # def logpdf(self,y,nu,t):
    #     # Require: t > nu
    #     assert(not isinstance(y,np.ndarray))
    #     return self.dist.logpdf(y-self.mu_1*self.mcfun(t-nu,self.c))
    def rvs(self,size,nu,t):
        # Require: t > nu
        ind = np.arange(t,t+size,dtype=float)
        a1_arr = self.alpha_1 * self.mcfun(ind,self.c)
        return np.array([stats.beta.rvs(a,self.beta_1) for a in a1_arr])
    def logpdfdiff(self,y):
        assert(isinstance(y,np.ndarray))
        ind = np.arange(len(y),dtype=float)
        a1_arr = self.alpha_1 * self.mcfun(ind,self.c)
        return np.sum((a1_arr-self.alpha_1) * np.log(y) + self.library[:len(y)])
    def setup_lib(self,W):
        ind = np.arange(W,dtype=float)
        a1_arr = self.alpha_1 * self.mcfun(ind,self.c)
        self.library = (a1_arr-self.alpha_1)*polygamma(0,self.alpha_1+self.beta_1)- (np.log(gamma(a1_arr)) - np.log(gamma(self.alpha_1)))
        # print("Library Setup Successfully:", self.library)
        return


# def L(y, f0, f1, tau, t):
#     return f1.logpdf(y,tau,t) - f0.logpdf(y)
# def Ls(y, f0, ct, taus, t):
#     f1e = F1(ct)
#     return f1e.logpdf(y,taus,t) - f0.logpdf(y)
def gfun(y,f1):
    llr = np.zeros(len(y))
    for k in range(len(y)):
        llr[k] = f1.logpdfdiff(y[k:])
    # print(llr)
    return (np.argmax(llr),np.amax(llr))

def Lv(y, fc, t):
    # max_k, max_llr = gfun(y,F1(c,mu_1=f0.mean(),sig_1=np.sqrt(f0.var()),mcfun=mcf),f0)
    max_k, max_llr = gfun(y,fc[0])
    # print(t,max_k)
    return max_llr

def Lg_setup(f0, low=c_min, high=100*c_min, mcf=dft_mcfun, grid=100):
    cl = np.linspace(low,high,grid).T                           # parameter space
    num_param = len(low)
    cf = np.tile(np.array([0],dtype=object),[grid]*num_param)   # f1 function grid
    for i in product(range(grid),repeat=num_param):
        ci = [cl[cii,cij] for cii,cij in enumerate(i)]
        cf[i] = F1(ci,f0,mcfun=mcf)
    return cf

def Lg(y, fc, t):
    # Assumes that Lg_setup has been run.
    grid = fc.shape[0]
    h = np.zeros(fc.shape)      # llr value grid
    for i in product(range(grid),repeat=len(fc.shape)):
        h[i] = gfun(y,fc[i])[1]
    best_h_ind = np.unravel_index(np.argmax(h, axis=None), h.shape)
    cf_best = fc[best_h_ind]
    k_best,llr_best = gfun(y,cf_best)
    # print(t,cl[np.argmax(h)],llr_best,k_best)
    return llr_best,cf_best.c,k_best

if __name__ == "__main__":
    # f0 = stats.norm(1,1)
    alpha_0 = 1
    beta_0 = 100       # OBSOLETE: largest b s.t. the program still works: 171.6
    f0 = stats.beta(alpha_0,beta_0)

    # Create CuSum grid
    f1v = Lg_setup(f0,low=[c],high=[100*c],grid=1)   # high is not used

    # Create GLR grid
    f1g = Lg_setup(f0,low=[c_min],high=[10*c_min],grid=10)

    # Start detection
    changed = False
    y = np.zeros(maxt)
    w = np.zeros(maxt)
    v = np.zeros(maxt)
    # s = np.zeros(maxt)
    g = np.zeros(maxt)
    for t in range(1, maxt-1):
        y[t] = f0.rvs(size=1)[0]
        if changed:
            y[t] = f1v[0].rvs(1,Tc,t)[0]
        ## w[t]: known c ##
        # w[t] = Lv(y[1:t+1], f0, c, t)

        ## v[t]: known c, windowed ##
        v[t] = Lv(y[max(1,t-W+1):t+1], f1v, t)

        ## s[t]: lower bound ##
        # s[t] = s[t-1] + Ls(y[t], f0, c_min, taus, t)
        # if s[t] < 0:
        #     s[t] = 0
        #     taus = t+1
        # s[t] = s[t-1] + stats.norm(np.exp(c_min),1).logpdf(y[t]) - f0.logpdf(y[t])
        # if s[t] < 0:
        #     s[t] = 0
        #     taus = t+1

        ## g[t]: GLR ##
        g[t],_,_ = Lg(y[max(1,t-W+1):t+1], f1g, t)

        # System update below #
        if t == Tc:
            changed = True

    ## Plot process
    plt.subplot(211)
    plt.plot(y[:-1])
    plt.ylabel("Observations")
    plt.title(r"One Realization of the CuSum Statistic at level $\alpha = %s$" % alpha)

    plt.subplot(212)
    # plt.plot(w[:-1], 'black', label="known c: {:.1e}".format(c))
    plt.plot(v[:-1], 'c', label="known param, windowed")
    # plt.plot(s[:-1], 'g', label="lower bounded: > {:.1e}".format(c_min))
    plt.plot(g[:-1], 'b', label="glr, windowed")
    plt.hlines(-np.log(alpha), 0, maxt, 'r')
    plt.vlines(Tc, 0, -np.log(alpha), 'k')
    plt.ylim(0,-np.log(alpha)*1.25)
    plt.ylabel("Test Stat")
    plt.legend(loc="upper right")

    plt.show()



###
