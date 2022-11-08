# Non-Parametric Quickest Mean Change Detection

### Paper

arXiv Link: https://arxiv.org/pdf/2110.01581.pdf

### Data

Data source: Rearc. AWS Marketplace: Coronavirus (COVID-19) Data in the United States — The New York Times. Link [here](https://aws.amazon.com/marketplace/pp/prodview-jmb464qw2yg74).

### Covid-19 Monitoring Application


$$h_\theta(x) = 1+\frac{10^{\theta_0}}{\theta_2} \exp\left(-\frac{(x-\theta_1)^2}{2 \theta_2^2} \right)$$

See [here](./lognormal_example.py) for an interactive plot for the three parameters.

![Validation of distribution model](./ns_covid_val.png)

The plot shows the four-day moving average of the daily new cases of COVID-19 as a fraction of the population in Wayne County, MI from October 1, 2020 to February 1, 2021 (in blue). The shape of the pre-change distribution $Beta(a_0, b_0)$ is estimated using data from the previous 20 days (from September 11, 2020 to September 30, 2021), where $\hat{a}_0 = 20.6$ and $\hat{b}_0 = 2.94 \times 10^5$. The mean of the Beta distributions with the best-fit h is also shown (in orange), which minimizes the mean-square distance between the daily incremental fraction and mean of the Beta distributions. The best-fit parameters are: $\hat{\theta}_0 = 0.464$, $\hat{\theta}_1 = 3.894$, and $\hat{\theta}_2 = 0.445$.

![COVID-19 monitoring example](./ns_covid.png)

COVID-19 monitoring example. The upper row shows the four-day moving average of the daily new cases of COVID-19
as a fraction of the population in Wayne County, MI (left), New York City, NY (middle) and Hamilton County, OH (right).
A pre-change B(a0, b0) distribution is estimated using data from the previous 20 days (from May 26, 2021 to June 14, 2021).
The plots in the lower row show the evolution of the WL-GLR-CuSum statistic defined in (49). The FAR α is set to 0.001 and
the corresponding thresholds of the WL-CuSum GLR procedure are shown in red. The post-change distribution at time n with
hypothesized change point k is modeled as B(a0hθ(n−k), b0), where hθ is defined in (60), and Θ = (0.1, 5)×(1, 20)×(0.1, 5).
The parameters θ0, θ1 and θ2 are assumed to be unknown. The window size mα = 20.
