#   
# Script with examples for FRANZ
#
# Version 0.1 (June 2019)
#
# Copyright (C) June 2019, Jochen Broecker, University of Reading, UK
#
# See franz.py for more information
#
import numpy as np
import franz as franz
import matplotlib.pyplot as plt

# # Begin import exampledataL8.csv. This is data from an AR1 model with AR-parameter = 0.9 and noise std = 1.
# # Column 1: Verification
# # Column 2 to 6: Ensemble
# # Column 7: Strata
# # There are 5 ensemble members, the lead time is 8, and there are 3 strata.
# # There is also exampledataL4.csv with lead time 4 and 2 strata.
joint_data = np.loadtxt('exampledataL8.csv')
# # End import exampledataL8.csv.

lead_time = 8 # Change this dep on data set
nr_contrasts = 2 # Use 2 contrasts.
# Too many contrasts and the test gets problems. Remember that the test has to estimate (nr_contrasts * nr_strata)^2 * (lead_time - 1) quantities!

s_joint_data = joint_data.shape

# Recompute stratification if desired. 
nr_uq_strata = 3 # Change this number for your desired number of strata
mn = np.median(joint_data[:, 0:s_joint_data[1]-1], axis = 1)
ind = np.argsort(mn.reshape([s_joint_data[0], 1]), axis = 0)
allranks = np.argsort(ind, axis = 0)
strat = np.floor((nr_uq_strata * allranks) / s_joint_data[0])
joint_data[:, s_joint_data[1]-1] = strat.reshape((s_joint_data[0],))

# Normally no need to change anything below this line

# Allocate verification, ensemble, and strata
ver = joint_data[:, 0]
ver = ver.reshape([ver.size, 1])
ens = joint_data[:, 1:s_joint_data[1]-1]
strat = joint_data[:, s_joint_data[1]-1]
strat = strat.reshape([strat.size, 1])

uq_strata = np.unique(strat)
nr_uq_strata = uq_strata.size

pval, rnks_vals, covar_est, rnks_counts = franz.rank_test(ver, ens, lead_time, strat, contrasts=nr_contrasts, return_counts = True)
# Check rnk.rank_test? for explanation of input and output arguments.

print('Pval: ', pval)

fig1 = plt.figure(100)
rnks_array = np.array(rnks_counts)
enslabels = np.array(rnks_counts.axes[0])
for l in range(0, nr_uq_strata):
    ax = fig1.add_subplot(nr_uq_strata, 1, l+1)
    ax.bar(enslabels, rnks_array[:, l])
    ax.yaxis.set_label_text("Counts")
ax.xaxis.set_label_text("Ranks")
fig1.suptitle("Rank Histograms")
fig1.savefig("test_hist.pdf")

fig2 = plt.figure(200)
plt.pcolor(covar_est.sum(axis = 2), cmap="Blues")
plt.colorbar()
fig2.suptitle("Covariance Matrix")
fig2.savefig("test_covar.pdf")

fig3 = plt.figure(300)
plt.plot(np.arange(0, s_joint_data[0]), ens, 'bo')
plt.plot(np.arange(0, s_joint_data[0]), ver, 'ro')
fig3.suptitle("Ensembles and Verifications vs Time")
fig3.savefig("test_ens.pdf")

plt.show()

