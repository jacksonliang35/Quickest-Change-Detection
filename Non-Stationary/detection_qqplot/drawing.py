import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from os import path
import statsmodels.api as sm
import pylab

b = np.exp(np.linspace(np.log(2),np.log(3.3),5))
W = np.array([25,50,np.inf])
Wstring = ['25','50','\infty']

k = 0
Times = np.zeros((3,5,10000,10))    # 3 window sizes, 5 thresholds
for k in range(10):
    cur_f = "tart_t%d.npy" % (k+1)
    with open(cur_f, 'rb') as f:
        _ = np.load(f)
        _ = np.load(f)
        _ = np.load(f)
        Times[:,:,:,k] = np.load(f)

Times = Times[:,:,:5400,:].reshape((3,5,-1))

## Plot 1: MTFAs and thresholds under different windows
# mtfa = np.mean(Times,axis=2)
#
# plt.plot(b, b, markersize=4, label="Theoretical")
# plt.plot(b, -np.log(1/mtfa[0]), 'bo-', markersize=4, label="Window = 25")
# plt.plot(b, -np.log(1/mtfa[1]), 'co-', markersize=4, label="Window = 50")
# plt.plot(b, -np.log(1/mtfa[2]), 'ro-', markersize=4, label="No window")
# plt.legend(loc="upper left", prop={'size': 15})
# plt.xlabel(r"$|\log(\alpha)|$", fontsize=15)
# plt.ylabel(r"$|\log(FAR)|$", fontsize=15)


## Plot 2: Exponentiality of FA times
fig,axes = plt.subplots(2,2)
for x in [0,2]:
    for y in range(3,5):
        # ax = fig.add_subplot(2, 2, 1)
        # stats.probplot(Times[x,y],dist='expon',plot=axes[x//2,y-3])
        sm.graphics.qqplot(Times[x,y], dist=stats.expon, fit=True, line="q", ax=axes[x//2,y-3])
        axes[x//2,y-3].set_title(r"Threshold: %.2f, Window: $%s$" % (b[y],Wstring[x]), fontsize=12)
        # plt.xlabel(r"$\log|\log(\alpha)|$", fontsize=15)
        # plt.ylabel(r"Delay", fontsize=15)

plt.show()


###
