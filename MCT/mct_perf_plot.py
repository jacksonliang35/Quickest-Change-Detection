import numpy as np
import matplotlib.pyplot as plt
from os import path

with open("perf.npy", "rb") as f:
    wadd_1 = np.load(f)
    wadd_2 = np.load(f)
    wadd_3 = np.load(f)
    mtfa_1 = np.load(f)
    mtfa_2 = np.load(f)
    mtfa_3 = np.load(f)

plt.plot(np.log(mtfa_1), wadd_1, 'go-', markersize=4, label="CuSum Test")
plt.plot(np.log(mtfa_2), wadd_2, 'yo-', markersize=4, label="Asym. Robust Test")
plt.plot(np.log(mtfa_3), wadd_3, 'ro-', markersize=4, label="MCT Test")
plt.legend(loc="upper left", prop={'size': 15})
plt.xlabel(r"$|\log(\alpha)|$", fontsize=15)
plt.ylabel(r"Worst-case Delay", fontsize=15)
plt.title(r"Detecting Change from Beta(4,16) to Beta(4.5,16)")

plt.show()


###
