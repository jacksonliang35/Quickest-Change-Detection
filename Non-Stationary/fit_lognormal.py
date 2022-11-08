import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import matplotlib.dates as mdates
from itertools import product

dataorig = pd.read_csv("./covid19-nyt/us-counties.csv",sep=',',index_col=False)
LOW_C = np.array([0.1,1,.1])
HIGH_C = np.array([3,10,5])
TRAINING = 20
AVG_MASK = 4
GRID = 200

def get_data(dataset, state, county, population, start=31, end=91, training=30, avg_msk=1):
    data = dataorig.loc[dataorig['state'] == state]
    data = data.loc[data['county'] == county]
    ndt = data[['date', 'cases']].groupby('date').sum().reset_index()
    ndt['inc_cases'] = ndt.cases.diff()
    start_date = ndt.iloc[start].date
    print("Starting from:", start_date)
    z = np.array(ndt.inc_cases)[1:]
    y = np.zeros(len(z))
    for i in range(AVG_MASK-1, len(z)):
        y[i] = np.mean(z[i-AVG_MASK+1:i+1])
    y_sample = y[start-training:start]
    ynorm = y[start:end] / population
    ynorm_sample = y_sample / population
    return (y_sample, ynorm_sample, y[start:end], ynorm, start_date)

def MCFUN(x,c):
    # Assuming x > 0
    try:
        x[0] += 1e-12
    except:
        pass
    # c[0] is y-scale; c[1] is mu; c[2] is sigma.
    return 1+10**c[0] * np.exp(-(np.log(x)-c[1])**2 / 2 / c[2]**2) / c[2]

def fit(y, a0, b0, low=LOW_C, high=HIGH_C, mcf=MCFUN, grid=GRID):
    cl = np.linspace(low,high,grid).T                           # parameter space
    num_param = len(low)
    h = np.tile(np.array([0.0]),[grid]*num_param)   # f1 function grid
    for i in product(range(grid),repeat=num_param):
        ci = [cl[cii,cij] for cii,cij in enumerate(i)]
        ar = a0*MCFUN(np.arange(len(y),dtype=float),tuple(ci))
        h[i] = np.sum((y-ar/(ar+b0))**2)
    best_ind = np.unravel_index(np.argmin(h, axis=None), h.shape)
    return tuple([cl[cii,cij] for cii,cij in enumerate(best_ind)])

fig = plt.figure()

# START = 451+14         # 2021-06-15
START = 210
END = START + 120
popl = 1749343  # Wayne county population
y_s, y_sn, y, y_n, sd = get_data(dataorig, 'Michigan', "Wayne", popl, start=START, end=END, training=TRAINING, avg_msk=AVG_MASK)
x_axis = np.array([np.datetime64(sd) + i for i in range(0,END-START)])
alpha_fit, beta_fit, _, _ = stats.beta.fit(y_sn, floc=0, fscale=1)
print("Alpha:",alpha_fit)
print("Beta:",beta_fit)

best_param = fit(y_n, alpha_fit, beta_fit)
print(best_param)
alpha_arr = alpha_fit * MCFUN(np.arange(len(y_n),dtype=float),best_param)
fun_approx = alpha_arr/(alpha_arr+beta_fit)
plt.rcParams.update({'font.size': 15})
plt.plot(x_axis, y_n, '-o', markersize=4)
plt.ylabel("Fraction of Incremental Cases")
plt.plot(x_axis, fun_approx, 'orange', label="Beta Mean")
plt.title("Wayne County, MI")

fig.autofmt_xdate()
# plt.subplots_adjust(left=0.2, bottom=0.2, right=0.86, top=0.85)
plt.show()




###
