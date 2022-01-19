import bayestremes as bxt
import numpy as np
import pandas as pd


# Sea max water level data from the University of Hawai: Caldwell, P. C., M. A. Merrifield, P. R. Thompson (2015),
# Sea level measured by tide gauges from global oceans — the Joint Archive for Sea Level holdings (NCEI Accession
# 0019568), Version 5.5, NOAA National Centers for Environmental Information, Dataset, doi:10.7289/V5V40S7W.

# Import the seal level data saved in csv format
sea_data = pd.read_csv('/Users/danielsalnikov/Documents/time_series_sea.csv')
# Cast as a np array the max level data and create a pointer/ working variable
max_level_data = np.array(sea_data['Max'])
# Visual exploration of these data
bxt.stat_learning.plt.hist(max_level_data, bins=30)

bxt.hypo_tests.max_tail_test(max_level_data, 20, 0.01, 0.01, 10000, 10)
bxt.hypo_tests.max_tail_test(max_level_data, 20, 0.01, 0.01, 10000, 100)
# Deviations from the mean
bxt.hypo_tests.max_tail_test(max_level_data - max_level_data.mean(), 20, 0.01, 0.01, 10000, 10)
# Standardised deviations from the mean
bxt.hypo_tests.max_tail_test((max_level_data - max_level_data.mean()) / max_level_data.std(), 20, 0.01, 0.01, 10000, 10)


hypo_inter_chain = bxt.hypo_tests.hypo_interval_chain(15000, max_level_data, 4, 1/5, 2, 2)
bxt.plt.hist(hypo_inter_chain, bins=30).show()
print(np.quantile(hypo_inter_chain, [0.025, 0.975]))

# There is significant posterior evidence that the data come from a heavy-tail distribution

prior_threshs = np.array([1000, 1200, 1500])
post_thresh_prob = bxt.stat_learning.gibbs_threshold_search_alg(max_level_data, prior_threshs, [1/3, 1/3, 1/3], 10000)
post_thresh_prob[:, 0].mean()
post_thresh_prob[:, 1].mean()
post_thresh_prob[:, 2].mean()
print(0.4733875149027048 * 1200 + 0.45440816308414794 * 1500)
post_param = bxt.stat_learning.quasi_conjugate_sampling(10000, [1, 20, 1000],
                                                        max_level_data, 1249, [0.1, 0.1, 1, 1], 50)
fig, (ax1, ax2, ax3) = bxt.plt.subplots(3, 1)
fig.suptitle('Histograms of Posterior Distribution Simulation Approximation')
ax1.hist(post_param['Alpha'], bins=55)
ax1.set_ylabel(r'$\hat{p}(\alpha | \mathbf{X})$')
ax2.hist(post_param['Theta'], bins=55)
ax2.set_ylabel(r'$\hat{p}(\theta | \mathbf{X})$')
ax3.hist(post_param['Gamma'], bins=55)
ax3.set_ylabel(r'$\hat{p}(\gamma | \mathbf{X})$')
ax3.set_xlabel('Parameter')
bxt.plt.show()

fig0, (ax01, ax02, ax03) = bxt.plt.subplots(3, 1)
ax01.plot(post_param['Alpha'])
ax01.set_ylabel(r'$\alpha_t$')
ax02.plot(post_param['Theta'])
ax02.set_ylabel(r'$\theta_t$')
ax03.plot(post_param['Gamma'])
ax03.set_ylabel(r'$\gamma_t$')
ax03.set_xlabel('Iteration: t')
bxt.plt.show()

print('Square Error Loss Bayes Estimate: ',
      '\nAlpha: ' + str(post_param['Alpha'].mean()),
      '\nTheta ' + str(post_param['Theta'].mean()),
      '\nGamma: ' + str(post_param['Gamma'].mean())
      )
