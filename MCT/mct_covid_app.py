import numpy as np
import pandas as pd
import scipy.stats as stats
# from scipy.optimize import fsolve
# from scipy.special import kv
# import scipy.integrate as integrate
import matplotlib.pyplot as plt


## Get COVID-19 Data
# Data source: AWS Rearc New York Times COVID-19 Data for Cook county, updated daily, from Jan. 21st
dataorig = pd.read_csv("./covid19-nyt/us-counties.csv",sep=',',index_col=False)

print("Wayne County, MI")
data = dataorig.loc[dataorig['state'] == 'Michigan']
data = data.loc[data['county'] == "Wayne"]
ndt = data[['date', 'cases']].groupby('date').sum().reset_index()
ndt['inc_cases'] = ndt.cases.diff()

# Wayne county population
popl = 1749343

## Preparing data
z = np.array(ndt.inc_cases)[1:] / popl
y = np.zeros(len(z))
AVG_MASK = 3
for i in range(AVG_MASK-1, len(z)):
    y[i] = np.mean(z[i-AVG_MASK+1:i+1])
START = 150
y_sample = y[START-20:START]
mu0 = np.mean(y_sample)
print("Pre-change mean:", mu0)
sig2 = np.mean(y_sample**2) - mu0**2
# sig2 = mu0 * (1-mu0)
print("Sig2:", sig2)

## Setting up test
ETA_TIMES = 3.3
print("Setting eta %d times as the pre-change mean." % ETA_TIMES)
eta = mu0 * ETA_TIMES
print("Threshold Eta:", eta)
def g(x, mu0, et):
    return x-(mu0+et)/2
alpha = 0.001
delta = (eta-mu0)/2
# R = sig2/(sig2+delta*np.maximum(mu0,1-mu0)/3)
# print("R:", R)
thr = -sig2 * np.log(alpha)/ 2 / delta
print("Test Threshold:", thr)

## Test starts
m = np.zeros(len(y))
t_cross = 0
T_CROSS = 1
for t in range(START,len(y)):
    # print(t)
    m[t] = np.maximum(0, m[t-1] + g(y[t], mu0, eta))
    if m[t] > thr and T_CROSS > 0:
        t_cross = t
        T_CROSS -= 1
print("Threshold crossing time:", t_cross)

## Plot
FIRST = min((len(y), t_cross+10))
plt.subplot(231)
plt.plot(range(START,FIRST), y[START:FIRST], '-o', markersize=4)
plt.xlabel("Number of Days starting from Jan. 21st")
plt.ylabel("Fraction of Incremental Cases")
plt.hlines(mu0, START, len(y[0:FIRST]), 'c', label="pre-change mean")
plt.hlines(eta, START, len(y[0:FIRST]), 'g', label="mean-threshold")
plt.title("Wayne County, MI")
plt.legend(loc="upper left")

plt.subplot(234)
plt.plot(range(START,FIRST), m[START:FIRST], '-o', markersize=4, label="MCT stat")
plt.hlines(thr, START, len(y[0:FIRST]), 'r', label="MCT threshold")
plt.xlabel("Number of Days starting from Jan. 21st")
plt.ylabel(r"$\tilde{\Lambda}(t)$")
plt.legend(loc="upper left")

##############################################################
print("##############################################################")

print("St. Louis, MO")
data = dataorig.loc[dataorig['state'] == 'Missouri']
data = data.loc[data['county'] == "St. Louis"]
ndt = data[['date', 'cases']].groupby('date').sum().reset_index()
ndt['inc_cases'] = ndt.cases.diff()

## Preparing data
popl = 993191
z = np.array(ndt.inc_cases)[1:] / popl
y = np.zeros(len(z))
for i in range(AVG_MASK-1, len(z)):
    y[i] = np.mean(z[i-AVG_MASK+1:i+1])
y_sample = y[START-30:START]
mu0 = np.mean(y_sample)
print("Pre-change mean:", mu0)

## Setting up test
eta = ETA_TIMES*mu0
print("Threshold Eta:", eta)
sig2 = np.mean(y_sample**2) - mu0**2
print("Estimated Sig2:", sig2)
delta = (eta-mu0)/2
# R = sig2/(sig2+delta*np.maximum(mu0,1-mu0)/3)
# thr = fsolve(lambda b: (2*abs(b)*np.pi*sig2/delta**3)-(alpha**2)*np.exp(4*delta*abs(b)*R**2/sig2), 0.01)[0]
thr = -sig2 * np.log(alpha)/ 2 / delta
print("Stat Threshold:", thr)

## Running test
m = np.zeros(len(y))
t_cross = 0
T_CROSS = False
for t in range(START,len(y)):
    # print(t)
    m[t] = np.maximum(0, m[t-1] + g(y[t], mu0, eta))
    if m[t] > thr and not T_CROSS:
        t_cross = t
        T_CROSS = True
print("Threshold crossing time:", t_cross)

## Plot
FIRST = min((len(y), t_cross+10))
plt.subplot(232)
plt.plot(range(START,FIRST), y[START:FIRST], '-o', markersize=4)
plt.xlabel("Number of Days starting from Jan. 21st")
plt.ylabel("Fraction of Incremental Cases")
plt.hlines(mu0, START, len(y[0:FIRST]), 'c', label="pre-change mean")
plt.hlines(eta, START, len(y[0:FIRST]), 'g', label="mean-threshold")
plt.title("St. Louis County, MO")
plt.legend(loc="upper left")

plt.subplot(235)
plt.plot(range(START,FIRST), m[START:FIRST], '-o', markersize=4, label="MCT stat")
plt.hlines(thr, START, len(y[0:FIRST]), 'r', label="MCT threshold")
plt.xlabel("Number of Days starting from Jan. 21st")
plt.ylabel(r"$\tilde{\Lambda}(t)$")
plt.legend(loc="upper left")

##############################################################
print("##############################################################")

print("Hamilton County, OH")
data = dataorig.loc[dataorig['state'] == 'Ohio']
data = data.loc[data['county'] == 'Hamilton']
ndt = data[['date', 'cases']].groupby('date').sum().reset_index()
ndt['inc_cases'] = ndt.cases.diff()

## Preparing data
popl = 817473
z = np.array(ndt.inc_cases)[1:] / popl
y = np.zeros(len(z))
for i in range(AVG_MASK-1, len(z)):
    y[i] = np.mean(z[i-AVG_MASK+1:i+1])
y_sample = y[START-20:START]
mu0 = np.mean(y_sample)
print("Pre-change mean:", mu0)

## Setting up test
eta = ETA_TIMES*mu0
print("Threshold Eta:", eta)
sig2 = np.mean(y_sample**2) - mu0**2
print("Estimated Sig2:", sig2)
delta = (eta-mu0)/2
# R = sig2/(sig2+delta*np.maximum(mu0,1-mu0)/3)
# thr = fsolve(lambda b: (2*abs(b)*np.pi*sig2/delta**3)-(alpha**2)*np.exp(4*delta*abs(b)*R**2/sig2), 0.01)[0]
thr = -sig2 * np.log(alpha)/ 2 / delta
print("Stat Threshold:", thr)

## Running test
m = np.zeros(len(y))
t_cross = 0
T_CROSS = 6
for t in range(START,len(y)):
    # print(t)
    m[t] = np.maximum(0, m[t-1] + g(y[t], mu0, eta))
    if m[t] > thr and T_CROSS > 0:
        t_cross = t
        T_CROSS -= 1
print("Threshold crossing time:", t_cross)

## Plot
FIRST = min((len(y), t_cross+12))
plt.subplot(233)
plt.plot(range(START,FIRST), y[START:FIRST], '-o', markersize=4)
plt.xlabel("Number of Days starting from Jan. 21st")
plt.ylabel("Fraction of Incremental Cases")
plt.hlines(mu0, START, len(y[0:FIRST]), 'c', label="pre-change mean")
plt.hlines(eta, START, len(y[0:FIRST]), 'g', label="mean-threshold")
plt.title("Hamilton County, OH")
plt.legend(loc="upper left")

plt.subplot(236)
plt.plot(range(START,FIRST), m[START:FIRST], '-o', markersize=4, label="MCT stat")
plt.hlines(thr, START, len(y[0:FIRST]), 'r', label="MCT threshold")
plt.xlabel("Number of Days starting from Jan. 21st")
plt.ylabel(r"$\tilde{\Lambda}(t)$")
plt.legend(loc="upper left")


plt.show()



###
