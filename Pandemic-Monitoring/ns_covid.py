import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import fsolve
# from scipy.special import kv
# import scipy.integrate as integrate
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import matplotlib.dates as mdates
from expon_cusum import Lg,Lg_setup
from scipy.special import gamma

# Cook county population, from https://www.census.gov/quickfacts/cookcountyillinois
# popl = 5150233

## Get COVID-19 Data
# Data source: AWS Rearc New York Times COVID-19 Data for Cook county, updated daily, from Jan. 21st
# dataorig = pd.read_csv("~/covid-19-data/us-counties.csv",sep=',',index_col=False) # if on vvv vm
dataorig = pd.read_csv("./covid19-nyt/us-counties.csv",sep=',',index_col=False)

START = 451+14         # 2021-06-15
TRAINING = 20
AVG_MASK = 4
ALPHA = 0.001
GRID = 10
W = TRAINING
ETA_TIMES = 3.3
def MCFUN(x,c):
    # Assuming x > 0
    try:
        x[0] += 1e-12
    except:
        pass
    # c[0] is y-scale; c[1] is mu; c[2] is sigma.
    return 1+10**c[0] * np.exp(-(np.log(x)-c[1])**2 / 2 / c[2]**2) / c[2]
LOW_C = np.array([0.1,1,.1])
HIGH_C = np.array([5,20,5])
# def dft_mcfun(x,c):
#     return np.exp(c[0]*x)
# LOW_C = [0.0001]
# HIGH_C = [1]
def glr_threshold(alpha, w, low_c, high_c):
    sz = 1.0
    eps = 1.0
    d = len(low_c)
    coef = np.pi**(d/2) / gamma(1+d/2)
    for i in range(len(low_c)):
        sz *= high_c[i] - low_c[i]
    b = fsolve(lambda bb: eps*d/2* np.log(bb) - np.log(alpha) + np.log(sz*np.e/coef) - bb, -np.log(alpha)+np.log(2*w*sz*np.e/coef))[0]
    return b

def mct_threshold(alpha,mu0,sig2):
    eta = mu0 * ETA_TIMES
    delta = (eta-mu0)/2
    R = sig2/(sig2+delta*np.maximum(mu0,1-mu0))
    thr = -sig2 * np.log(alpha)/ 2 / delta
    return thr

fig, axs = plt.subplots(3,3, figsize=[14,9], dpi=200)
axs[0,0].yaxis.set_major_formatter(StrMethodFormatter('{x:1.1e}'))
axs[0,1].yaxis.set_major_formatter(StrMethodFormatter('{x:1.1e}'))
axs[0,2].yaxis.set_major_formatter(StrMethodFormatter('{x:1.1e}'))
axs[2,0].yaxis.set_major_formatter(StrMethodFormatter('{x:1.1e}'))
axs[2,1].yaxis.set_major_formatter(StrMethodFormatter('{x:1.1e}'))
axs[2,2].yaxis.set_major_formatter(StrMethodFormatter('{x:1.1e}'))

def get_data(dataset, state, county, population, start=1, training=30, avg_msk=1):
    data = dataorig.loc[dataorig['state'] == state]
    data = data.loc[data['county'] == county]
    ndt = data[['date', 'cases']].groupby('date').sum().reset_index()
    ndt['inc_cases'] = ndt.cases.diff()
    # print("Starting from:", start_date)
    start_date = ndt.iloc[start].date
    z = np.array(ndt.inc_cases)[1:]
    y = np.zeros(len(z))
    for i in range(AVG_MASK-1, len(z)):
        y[i] = np.mean(z[i-AVG_MASK+1:i+1])
    y_sample = y[start-training:start]
    ynorm = y[start:] / population
    ynorm_sample = y_sample / population
    return (y_sample, ynorm_sample, y[start:], ynorm, start_date)

popl = 1749343  # Wayne county population
y_s, y_sn, y, y_n, sd = get_data(dataorig, 'Michigan', "Wayne", popl, start=START, training=TRAINING, avg_msk=AVG_MASK)
mu0 = np.mean(y_sn)
print("Pre-change mean:", mu0)
sig2 = np.mean(y_sn**2) - mu0**2
# print("Sig2:", sig2)
alpha_fit, beta_fit, _, _ = stats.beta.fit(y_sn, floc=0, fscale=1)
print("Alpha:",alpha_fit)
print("Beta:",beta_fit)

glr_thr = glr_threshold(ALPHA, W, LOW_C, HIGH_C)
print("GLR threshold modification:", glr_thr + np.log(ALPHA))

mct_thr = mct_threshold(ALPHA, mu0, sig2)
eta = mu0 * ETA_TIMES

m = np.zeros(len(y))
g = np.zeros(len(y))
t_cross = 0
t_rem = 1
f0 = stats.beta(alpha_fit, beta_fit)
y_gn = np.append(y_sn,y_n)

# Create GLR grid
f1g = Lg_setup(f0,low=LOW_C,high=HIGH_C,mcf=MCFUN,grid=GRID)

for t in range(len(y)):
    # print(t)
    g[t],glr_c,glr_k = Lg(y_gn[t+1:t+W+1], f1g, t)
    # glr[t] = Lv(y[START:t+1], f0, .01, t)
    m[t] = np.maximum(0, m[t-1] + (y_n[t]-(mu0+eta)/2))
    # print("GLR value at ", t, ":", g[t])
    if g[t] > glr_thr and t_rem > 0:
        t_cross = t
        best_glr_c = glr_c
        best_glr_k = glr_k
        t_rem -= 1
    if t_cross > 0 and t > t_cross + 10:
        break
    if t == len(y)-1 and t_rem > 0:
        t_cross = t
print("Best GLR c:", best_glr_c)
print("Best GLR k:", best_glr_k)
print("Crossing Day:", t_cross)

## Plot
end = min((len(y), t_cross+10))
x_axis = np.array([np.datetime64(sd) + i for i in range(end)])
alpha_arr = alpha_fit*MCFUN(np.arange(end,dtype=float),best_glr_c)
if t_cross-W+best_glr_k > 0:
    alpha_arr = alpha_arr[:end-(t_cross-W+best_glr_k)]
    fun_approx = np.append(np.repeat(f0.mean(),t_cross-W+best_glr_k), alpha_arr/(alpha_arr+beta_fit))
else:
    fun_approx = alpha_arr/(alpha_arr+beta_fit)
axs[0,0].plot(x_axis, y_n[:end], '-o', markersize=4)
axs[0,0].set_xlim(x_axis[0], x_axis[-1])
axs[0,0].set_ylabel("Fraction of Incremental Cases")
# axs[0,0].hlines(f0.mean(), x_axis[0], x_axis[-1], 'c', label="pre-change mean")
# axs[0,0].plot(x_axis, fun_approx, 'orange', label="function approx")
axs[0,0].set_title("Wayne County, MI")
# axs[0,0].legend(loc="upper left")

axs[2,0].plot(x_axis, m[:end], '-o', markersize=4, label="MCT")
axs[2,0].set_xlim(x_axis[0], x_axis[-1])
axs[2,0].set_ylabel("MCT Statistic")
axs[2,0].hlines(mct_thr, x_axis[0], x_axis[-1], 'r')

axs[1,0].plot(x_axis, g[:end], '-o', markersize=4, label="GLR")
# axs[1,0].legend(loc="upper right")
axs[1,0].set_xlim(x_axis[0], x_axis[-1])
axs[1,0].set_ylabel("GLR Statistic")
axs[1,0].hlines(glr_thr, x_axis[0], x_axis[-1], 'r')


##############################################################
print("##############################################################")

popl = 18823000  # NYC population
y_s, y_sn, y, y_n, sd = get_data(dataorig, 'New York', "New York City", popl, start=START, training=TRAINING, avg_msk=AVG_MASK)
mu0 = np.mean(y_sn)
# print("Pre-change mean:", mu0)
sig2 = np.mean(y_sn**2) - mu0**2
# print("Sig2:", sig2)
alpha_fit, beta_fit, _, _ = stats.beta.fit(y_sn, floc=0, fscale=1)
print("Alpha:",alpha_fit)
print("Beta:",beta_fit)

glr_thr = glr_threshold(ALPHA, W, LOW_C, HIGH_C)
print("GLR threshold modification:", glr_thr + np.log(ALPHA))

mct_thr = mct_threshold(ALPHA, mu0, sig2)
eta = mu0 * ETA_TIMES

m = np.zeros(len(y))
g = np.zeros(len(y))
t_cross = 0
t_rem = 1
f0 = stats.beta(alpha_fit, beta_fit)
y_gn = np.append(y_sn,y_n)

# Create GLR grid
f1g = Lg_setup(f0,low=LOW_C,high=HIGH_C,mcf=MCFUN,grid=GRID)

for t in range(len(y)):
    # print(t)
    g[t],glr_c,glr_k = Lg(y_gn[t+1:t+W+1], f1g, t)
    # glr[t] = Lv(y[START:t+1], f0, .01, t)
    m[t] = np.maximum(0, m[t-1] + (y_n[t]-(mu0+eta)/2))
    # print("GLR value at ", t, ":", g[t])
    if g[t] > glr_thr and t_rem > 0:
        t_cross = t
        best_glr_c = glr_c
        best_glr_k = glr_k
        t_rem -= 1
    if t_cross > 0 and t > t_cross + 10:
        break
    if t == len(y)-1 and t_rem > 0:
        t_cross = t
print("Best GLR c:", best_glr_c)
print("Best GLR k:", best_glr_k)
print("Crossing Day:", t_cross)

## Plot
end = min((len(y), t_cross+10))
x_axis = np.array([np.datetime64(sd) + i for i in range(end)])
alpha_arr = alpha_fit*MCFUN(np.arange(end,dtype=float),best_glr_c)
if t_cross-W+best_glr_k > 0:
    alpha_arr = alpha_arr[:end-(t_cross-W+best_glr_k)]
    fun_approx = np.append(np.repeat(f0.mean(),t_cross-W+best_glr_k), alpha_arr/(alpha_arr+beta_fit))
else:
    fun_approx = alpha_arr/(alpha_arr+beta_fit)
axs[0,1].plot(x_axis, y_n[:end], '-o', markersize=4)
axs[0,1].set_xlim(x_axis[0], x_axis[-1])
axs[0,1].set_ylabel("Fraction of Incremental Cases")
# axs[0,1].hlines(f0.mean(), x_axis[0], x_axis[-1], 'c', label="pre-change mean")
# axs[0,1].plot(x_axis, fun_approx, 'orange', label="function approx")
axs[0,1].set_title("New York City, NY")
# axs[0,1].legend(loc="upper left")

axs[1,1].plot(x_axis, g[:end], '-o', markersize=4, label="GLR")
# axs[1,1].legend(loc="upper right")
axs[1,1].set_ylabel("GLR Statistic")
axs[1,1].hlines(glr_thr, x_axis[0], x_axis[-1], 'r')

axs[2,1].plot(x_axis, m[:end], '-o', markersize=4, label="MCT")
axs[2,1].set_xlim(x_axis[0], x_axis[-1])
axs[2,1].set_ylabel("MCT Statistic")
axs[2,1].hlines(mct_thr, x_axis[0], x_axis[-1], 'r')


##############################################################
print("##############################################################")

popl = 817473  # Hamilton, OH population
y_s, y_sn, y, y_n, sd = get_data(dataorig, 'Ohio', "Hamilton", popl, start=START, training=TRAINING, avg_msk=AVG_MASK)
mu0 = np.mean(y_sn)
# print("Pre-change mean:", mu0)
sig2 = np.mean(y_sn**2) - mu0**2
# print("Sig2:", sig2)
alpha_fit, beta_fit, _, _ = stats.beta.fit(y_sn, floc=0, fscale=1)
print("Alpha:",alpha_fit)
print("Beta:",beta_fit)

glr_thr = glr_threshold(ALPHA, W, LOW_C, HIGH_C)
print("GLR threshold modification:", glr_thr + np.log(ALPHA))

mct_thr = mct_threshold(ALPHA, mu0, sig2)
eta = mu0 * ETA_TIMES

m = np.zeros(len(y))
g = np.zeros(len(y))
t_cross = 0
t_rem = 1
f0 = stats.beta(alpha_fit, beta_fit)
y_gn = np.append(y_sn,y_n)

# Create GLR grid
f1g = Lg_setup(f0,low=LOW_C,high=HIGH_C,mcf=MCFUN,grid=GRID)

for t in range(len(y)):
    # print(t)
    g[t],glr_c,glr_k = Lg(y_gn[t+1:t+W+1], f1g, t)
    # glr[t] = Lv(y[START:t+1], f0, .01, t)
    m[t] = np.maximum(0, m[t-1] + (y_n[t]-(mu0+eta)/2))
    # print("GLR value at ", t, ":", g[t])
    if g[t] > glr_thr and t_rem > 0:
        t_cross = t
        best_glr_c = glr_c
        best_glr_k = glr_k
        t_rem -= 1
    if t_cross > 0 and t > t_cross + 10:
        break
    if t == len(y)-1 and t_rem > 0:
        t_cross = t
print("Best GLR c:", best_glr_c)
print("Best GLR k:", best_glr_k)
print("Crossing Day:", t_cross)

## Plot
end = min((len(y), t_cross+10))
x_axis = np.array([np.datetime64(sd) + i for i in range(end)])
alpha_arr = alpha_fit*MCFUN(np.arange(end,dtype=float),best_glr_c)
if t_cross-W+best_glr_k > 0:
    alpha_arr = alpha_arr[:end-(t_cross-W+best_glr_k)]
    fun_approx = np.append(np.repeat(f0.mean(),t_cross-W+best_glr_k), alpha_arr/(alpha_arr+beta_fit))
else:
    fun_approx = alpha_arr/(alpha_arr+beta_fit)
axs[0,2].plot(x_axis, y_n[:end], '-o', markersize=4)
axs[0,2].set_xlim(x_axis[0], x_axis[-1])
axs[0,2].set_ylabel("Fraction of Incremental Cases")
# axs[0,2].hlines(f0.mean(), x_axis[0], x_axis[-1], 'c', label="pre-change mean")
# axs[0,2].plot(x_axis, fun_approx, 'orange', label="function approx")
axs[0,2].set_title("Hamilton, OH")
# axs[0,2].legend(loc="upper left")

axs[1,2].plot(x_axis, g[:end], '-o', markersize=4, label="GLR")
# axs[1,2].legend(loc="upper right")
axs[1,2].set_ylabel("GLR Statistic")
axs[1,2].hlines(glr_thr, x_axis[0], x_axis[-1], 'r')

axs[2,2].plot(x_axis, m[:end], '-o', markersize=4, label="MCT")
axs[2,2].set_xlim(x_axis[0], x_axis[-1])
axs[2,2].set_ylabel("MCT Statistic")
axs[2,2].hlines(mct_thr, x_axis[0], x_axis[-1], 'r')


fig.autofmt_xdate()
# for ai in axs:
#     for aij in ai:
#         aij.set_xticklabels(aij.get_xticklabels(), rotation=30, ha='right')
#         aij.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.96, top=0.95, wspace=0.3)
plt.savefig('./fig1.png')
# plt.show()




###
