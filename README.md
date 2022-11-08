# Quickest-Change-Detection

### Problem Definition

Given a sequence of observations whose distribution changes at some unknown change point, the goal is to detect the change in distribution as quickly as possible after it occurs, while not making too many false alarms.

See [here](https://arxiv.org/pdf/1210.5552.pdf) for an overview.

### [Non-Parametric Quickest Mean Change Detection](./MCT)

#### Abstract

The problem of quickest detection of a change in the mean of a sequence of independent observations is studied. The pre-change distribution is assumed to be stationary, while the post-change distributions are allowed to be non-stationary. The case where the pre-change distribution is known is studied first, and then the extension where only the mean and variance of the pre-change distribution are known. No knowledge of the post-change distributions is assumed other than that their means are above some pre- specified threshold larger than the pre-change mean. For the case where the pre-change distribution is known, a test is derived that asymptotically minimizes the worst-case detection delay over all possible post-change distributions, as the false alarm rate goes to zero. Towards deriving this asymptotically optimal test, some new results are provided for the general problem of asymptotic minimax robust quickest change detection in non-stationary settings. Then, the limiting form of the optimal test is studied as the gap between the pre- and post-change means goes to zero, called the Mean-Change Test (MCT). It is shown that the MCT can be designed with only knowledge of the mean and variance of the pre-change distribution. The performance of the MCT is also characterized when the mean gap is moderate, under the additional assumption that the distributions of the observations have bounded support. The analysis is validated through numerical results for detecting a change in the mean of a beta distribution. The use of the MCT in monitoring pandemics is also demonstrated.


### [Quickest Change Detection with Non-Stationary Post-Change Observations](./Non-Stationary)

#### Abstract

The problem of quickest detection of a change in the distribution of a sequence of independent observations is considered. The pre-change observations are assumed to be stationary with a known distribution, while the post-change observations are allowed to be non-stationary with some possible parametric uncertainty in their distribution. In particular, it is assumed that the cumulative Kullback-Leibler divergence between the post-change and the pre-change distributions grows in a certain manner with time after the change-point. For the case where the post-change distributions are known, a universal asymptotic lower bound on the delay is derived, as the false alarm rate goes to zero. Furthermore, a window-limited Cumulative Sum (CuSum) procedure is developed, and shown to achieve the lower bound asymptotically. For the case where the post-change distributions have parametric uncertainty, a window-limited (WL) generalized likelihood-ratio (GLR) CuSum procedure is developed and is shown to achieve the universal lower bound asymptotically. Extensions to the case with dependent observations are discussed. The analysis is validated through numerical results on synthetic data. The use of the WL-GLR-CuSum procedure in monitoring pandemics is also demonstrated.
