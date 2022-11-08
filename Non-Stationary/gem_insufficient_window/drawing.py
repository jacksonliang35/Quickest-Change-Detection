import numpy as np
import matplotlib.pyplot as plt
from os import path


k = 0
wt = np.array([40,40,41,40,39,42,42,39,42,42])
while True:
    k += 1
    cur_f = "t%d.npy" % k
    if not path.exists(cur_f):
        break
    with open(cur_f, 'rb') as f:
        if k == 1:
            mtfa_1 = np.load(f) * wt[k-1] / np.sum(wt)
            mtfa_2 = np.load(f) * wt[k-1] / np.sum(wt)
            mtfa_3 = np.load(f) * wt[k-1] / np.sum(wt)
            wadd_1 = np.load(f) * wt[k-1] / np.sum(wt)
            wadd_2 = np.load(f) * wt[k-1] / np.sum(wt)
            wadd_3 = np.load(f) * wt[k-1] / np.sum(wt)
        else:
            mtfa_1 += np.load(f) * wt[k-1] / np.sum(wt)
            mtfa_2 += np.load(f) * wt[k-1] / np.sum(wt)
            mtfa_3 += np.load(f) * wt[k-1] / np.sum(wt)
            wadd_1 += np.load(f) * wt[k-1] / np.sum(wt)
            wadd_2 += np.load(f) * wt[k-1] / np.sum(wt)
            wadd_3 += np.load(f) * wt[k-1] / np.sum(wt)

print("Found %d files to merge..." % (k-1))

with open('./g.npy', 'wb') as f:
    np.save(f, wadd_1)
    np.save(f, wadd_2)
    np.save(f, wadd_3)
    np.save(f, mtfa_1)
    np.save(f, mtfa_2)
    np.save(f, mtfa_3)

# plt.subplot(121)
# plt.plot(np.log(mtfa_1), wadd_1, 'bo-', markersize=4, label="Window = 25")
# plt.plot(np.log(mtfa_2), wadd_2, 'co-', markersize=4, label="Window = 50")
# plt.plot(np.log(mtfa_3), wadd_3, 'ro-', markersize=4, label="Window = 100")
# plt.legend(loc="upper left", prop={'size': 15})
# plt.xlabel(r"$|\log(\alpha)|$", fontsize=15)
# plt.ylabel(r"Delay", fontsize=15)

# plt.subplot(122)
plt.plot(np.log(np.log(mtfa_1)), wadd_1, 'bo-', markersize=4, label="Window = 15")
plt.plot(np.log(np.log(mtfa_2)), wadd_2, 'co-', markersize=4, label="Window = 20")
plt.plot(np.log(np.log(mtfa_3)), wadd_3, 'ro-', markersize=4, label="Window = 25")
plt.legend(loc="upper left", prop={'size': 15})
plt.xlabel(r"$\log|\log(\alpha)|$", fontsize=15)
plt.ylabel(r"Delay", fontsize=15)

plt.show()


###
