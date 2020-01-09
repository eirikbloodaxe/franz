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

# Begin import raindataL2.csv. This is synthetic PoP data
# Column 1: Verification
# Column 2: PoP
joint_data = np.loadtxt('raindataL2.csv')
# End import exampledataL8.csv.

lead_time = 2 # Change this dep on data set
s_joint_data = joint_data.shape

# Compute stratification and append to joint_data
strat = joint_data[:, s_joint_data[1]-1]
strat = 1 * (strat > 0.5)
# change this line if other stratification is desired.

# Normally no need to change anything below this line

# Allocate verification, ensemble, and strata
ver = joint_data[:, 0]
ver = ver.astype(int)
ver = ver.reshape([ver.size, 1])
pop = joint_data[:, 1]
pop = pop.reshape([pop.size, 1])
pop = np.concatenate((pop, 1 - pop), axis = 1)
strat = strat.reshape([strat.size, 1])

uq_strata = np.unique(strat)
nr_uq_strata = uq_strata.size

pval, covar_est = franz.category_test(ver, pop, 2, strat)
# Check franz.category_test? for explanation of input and output arguments.

print('Pval: ', pval)

fig1 = plt.figure(200)
plt.pcolor(covar_est.sum(axis = 2), cmap="Blues")
plt.colorbar()
fig1.suptitle("Covariance Matrix")
fig1.savefig("test_covar.pdf")

plt.show()

