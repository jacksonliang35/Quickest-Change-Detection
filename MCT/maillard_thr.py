import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

alpha1 = 4
alpha2 = 4.5
beta = 16
f0 = stats.beta(a=alpha1, b=beta)
f1 = stats.beta(a=alpha2, b=beta)
eta = 0.21


def scan_b(t0,s,t,d):
    return .5*np.sqrt((1/(s-t0+1)+1/(t-s))*(1+1/(t-t0+1))*2*np.log(2*(t-t0)*np.log(t-t0+2)/d))


# Fix s=100
s = 1000
t = np.array(range(s,100000))
for d in np.geomspace(np.exp(-1),np.exp(-5),num=5):
    plt.plot(t,scan_b(0,s,t,d),label="fa rate: ln(%.1f)" % np.log(d))
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.axhline(y=f1.mean()-f0.mean())
plt.title("Test Threshold assuming correct change-point s=%d (Mallard,2019)" % s)
plt.show()


###
