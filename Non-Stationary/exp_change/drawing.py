import numpy as np
import matplotlib.pyplot as plt
from os import path


wt = np.ones(6)
for k in range(len(wt)):
    cur_f = "window_15_25_40_a%d.npy" % (k+1)
    assert path.exists(cur_f), "No Path Found!"
    with open(cur_f, 'rb') as f:
        if k == 0:
            mtfa_15 = np.load(f) * wt[k] / np.sum(wt)
            mtfa_25 = np.load(f) * wt[k] / np.sum(wt)
            mtfa_40 = np.load(f) * wt[k] / np.sum(wt)
            add_15 = np.load(f) * wt[k] / np.sum(wt)
            add_25 = np.load(f) * wt[k] / np.sum(wt)
            add_40 = np.load(f) * wt[k] / np.sum(wt)
        else:
            mtfa_15 += np.load(f) * wt[k] / np.sum(wt)
            mtfa_25 += np.load(f) * wt[k] / np.sum(wt)
            mtfa_40 += np.load(f) * wt[k] / np.sum(wt)
            add_15 += np.load(f) * wt[k] / np.sum(wt)
            add_25 += np.load(f) * wt[k] / np.sum(wt)
            add_40 += np.load(f) * wt[k] / np.sum(wt)

print("Found %d files to merge..." % (k+1))


wt = np.ones(3)
for k in range(len(wt)):
    cur_f = "window_50_100_%d.npy" % (k+1)
    assert path.exists(cur_f), "No Path Found!"
    with open(cur_f, 'rb') as f:
        if k == 0:
            mtfa_50 = np.load(f) * wt[k] / np.sum(wt)
            mtfa_100 = np.load(f) * wt[k] / np.sum(wt)
            add_50 = np.load(f) * wt[k] / np.sum(wt)
            add_100 = np.load(f) * wt[k] / np.sum(wt)
        else:
            mtfa_50 += np.load(f) * wt[k] / np.sum(wt)
            mtfa_100 += np.load(f) * wt[k] / np.sum(wt)
            add_50 += np.load(f) * wt[k] / np.sum(wt)
            add_100 += np.load(f) * wt[k] / np.sum(wt)

print("Found %d files to merge..." % (k+1))


wt = np.ones(6)
for k in range(len(wt)):
    cur_f = "glr_25_50_%d.npy" % (k+1)
    assert path.exists(cur_f), "No Path Found!"
    with open(cur_f, 'rb') as f:
        if k == 0:
            mtfa_g25 = np.load(f) * wt[k] / np.sum(wt)
            mtfa_g50 = np.load(f) * wt[k] / np.sum(wt)
            add_g25  = np.load(f) * wt[k] / np.sum(wt)
            add_g50  = np.load(f) * wt[k] / np.sum(wt)
        else:
            mtfa_g25 += np.load(f) * wt[k] / np.sum(wt)
            mtfa_g50 += np.load(f) * wt[k] / np.sum(wt)
            add_g25  += np.load(f) * wt[k] / np.sum(wt)
            add_g50  += np.load(f) * wt[k] / np.sum(wt)

print("Found %d files to merge..." % (k+1))


# plt.plot(np.log(mtfa_15), add_15, 'yo-', markersize=4, label="Window = 15")
# plt.plot(np.log(mtfa_25), add_25, 'r^-', markersize=4, label="Window = 25)")
# plt.plot(np.log(mtfa_40), add_40, 'ro-', markersize=4, label="Window = 40")
# plt.plot(np.log(mtfa_50), add_50, 'bo-', markersize=4, label="Window = 50)")
# plt.plot(np.log(mtfa_100), add_100, 'go-', markersize=4, label="Window = 100")

plt.plot(np.log(np.mean(mtfa_25 , axis=1)), np.mean(add_25 , axis=1), 'r^-', markersize=4, label="WL-CuSum (Window: 25)")
plt.plot(np.log(np.mean(mtfa_50 , axis=1)), np.mean(add_50 , axis=1), 'bo-', markersize=4, label="WL-CuSum (Window: 50)")
plt.plot(np.log(np.mean(mtfa_g25, axis=1)), np.mean(add_g25, axis=1), 'r^--', markersize=4, label="WL-GLR-CuSum (Window: 25)")
plt.plot(np.log(np.mean(mtfa_g50, axis=1)), np.mean(add_g50, axis=1), 'bo--', markersize=4, label="WL-GLR-CuSum (Window: 50)")

plt.legend(loc="upper left", prop={'size': 12})
plt.xlabel(r"$|\log(\alpha)|$", fontsize=15)
plt.ylabel(r"Expected Delay", fontsize=15)

plt.show()


###
