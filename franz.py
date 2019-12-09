#   
# (F)orecast (R)eliability (A)nalysis tool(Z) -- FRANZ
#
# Version 0.1 (June 2019)
#
# Copyright (C) June 2019, Jochen Broecker, University of Reading, UK
#
# FRANZ is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# FRANZ is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License <http://www.gnu.org/licenses/> for more details.
#
#  
# TODO List:
# (1) replace []-lists with size = (tuple) in calls of np.zeros etc

import numpy as np
import scipy.stats as stats
from pandas import crosstab

def extend_to_og_basis(pp):
    """ function basis = extend_to_og_basis(pp):
    
    Extends a given d-dimensional vector v to an orthonormal basis of R^d. The vector v is assumed to have sum(v) = nonzero. The algorithm will extend v with an orthonormal basis of
    S = {w an element of R^d, sum(w) = 0} and then apply the Gram Schmidt procedure.

    Input arguments:
    
    pp -- An array with dimensions [N, d] with each row representing a d-dimensional vector. The routine will be performed for all rows of pp separately.

    Output arguments:

    bss = An array with dimensions [N, d, d]. For every n = 0, ..., N-1 we have that bss[n, 0, :] is equal to pp[n, :] albeit normalised, and bss[n, 1, :], ..., bss[n, d-1, :] are orthogonal and normalised.

    Disclaimer: Use at your own risk!

    (c) Jochen Broecker, 2018

    """
    s_pp = pp.shape
    bss = np.zeros((s_pp[0], s_pp[1], s_pp[1]))
    resl = (1/s_pp[1])
    special_eye = (1/s_pp[1]) - np.eye(s_pp[1], s_pp[1] - 1)

    special_eye = special_eye / np.sqrt((s_pp[1] - 1) * resl**2 + (1 - resl)**2)
    for n in range(0, s_pp[0]):
        dummy = np.concatenate((pp[n, :].reshape((s_pp[1], 1)), special_eye), axis = 1)
        q, r = np.linalg.qr(dummy, mode = 'complete')
        bss[n, :, :] = q.reshape((1, s_pp[1], s_pp[1]))

    return(bss)

def category_test(Y, p, lead_time, strat):

    """function [pval, covar_est] = category_test(Y, p, lead_time, strat)
  
    Performs a generalised GOF test for categorial probability forecasts.

    Input arguments:

    Y -- The verification, an array with dimensions [nr_tstamps, 1], first dimension representing time. The verification must assume the values 0, ..., K-1 only.
    p -- The probabilistic forecasts, an array with dimensions [nr_tstamps, K], first dimension representing time and second dimension representing the forecast probabilities of the categories 0, ..., K-1. Sum over the second dimension must give 1 throughout.
    lead_time -- The lead time of the forecast.
    strat -- stratum, a column vector of dimension nr_tstamps-by-1. Entries to this vector should be from a set of S different symbols (the exact symbol is irrelevant). strat[n, 0] gives the stratum of the n'th sample, and samples with the same stratum will be associated with the same evaluation. A warning is given if S > log(nr_tstams) or if the number of samples in the different strata differs by more than 20%.

    Output arguments:

    pval -- The p-value of the test

    covar_est -- Estimator of the (matrix valued) covariance function of the test statistic. This is an array with the three dimensions [nr_moments * S, nr_moments * S, lead_time]. The entry covar_est[c1 + s1 * nr_moments, c2 + s2 * nr_moments, l] is the correlation between L(c1, pit_vals(n)) resp L(c2, pit_vals(n + l)), projected onto stratum s1+1, resp stratum s2+1. We may calculate C = np.sum(covar_est, axis = 2), then the entry C[c1 + s1 * nr_contrasts, c2 + s2 * nr_contrasts] is the correlation between the same variables but summed over time and divided by sqrt(nr_tstamps).

    Disclaimer: Use at your own risk!

    (c) Jochen Broecker, 2018
    """
    # function body

    s_Y = Y.shape
    s_p = p.shape
    nr_tstamps = s_Y[0]
    nr_cats = s_p[1]
    nr_moms = nr_cats - 1
    
    uq_strata, strat_inv, strata_counts = np.unique(strat, return_inverse = True, return_counts = True)
    strat_inv = strat_inv.reshape([nr_tstamps, 1])
    nr_uq_strata = uq_strata.size

    # sort out strata
    if (nr_uq_strata > np.log(nr_tstamps)):
        print('Warning: Number of strata should not be larger than log(nr_tstamps)')
    elif np.sqrt(strata_counts.std()/strata_counts.mean() > 0.4):
        print('Warning: Number of samples in strata varies by more than 40%')

    # Form orthogonal moment variables
    contrasted_Z = np.zeros((nr_tstamps, nr_moms))
    q = np.sqrt(p)
    bss = extend_to_og_basis(q)

    for cat in range(0, nr_moms):
        contrasted_Z[:, cat] = bss[np.arange(0, nr_tstamps), Y.reshape((nr_tstamps,)), np.ones((nr_tstamps,), dtype = 'int8') * (cat + 1)] / q[np.arange(0, nr_tstamps), Y.reshape((nr_tstamps,))]
        # this is supposed to do the following for each n:
        # * Let q = sqrt(p_n) (element wise)
        # * Extend q to an orthonormal basis q, b_1, ..., b_{d-1}. Variable bss[n, :, :] represents that basis. 
        # * Put y = vector with 1 in position Y and zero else
        # * Put z = y/q (element wise)
        # * Let c_k = <b_k, z> - <b_k, q> = <b_k, z> for each k = 1, ..., d-1
        
    
    # Bring in the strata (TODO dispense with loop using np.repeat)
    new_Z = np.zeros(shape = (nr_tstamps, nr_uq_strata * nr_moms))
    for s in range(0, nr_uq_strata):
        new_Z[:, s * nr_moms : (s+1) * nr_moms] = contrasted_Z * np.tile((strat_inv == s), [1, nr_moms])
        new_Z[:, s * nr_moms : (s+1) * nr_moms] = new_Z[:, s * nr_moms : (s+1) * nr_moms] / np.sqrt(strata_counts[s]/nr_tstamps)

    # perform generalised chi square test
    pval, covar_est = gen_chi_squ(new_Z, lead_time)

    return(pval, covar_est)
    
def moment_test(ver, moms, lead_time, strat):

    """function [pval, covar_est] = def moment_test(ver, moms, lead_time, strat)
   
    Performs a generalised GOF test for first and second moment, using the statistic

    Z = (Y - m1) / sqrt(m2 - m1^2)

    where Y is the verification and m1, m2 are the forecast first and second moment, respectively.

    Input arguments:

    ver -- The verification, an array with dimensions [nr_tstamps, 1], first dimension representing time
    moms -- The forecast moments, an array with dimensions [nr_tstamps, 2], first dimension representing time, moms[:, 0] and moms[:, 1] being the forecast first and second moment, respectively
    lead_time -- The lead time of the forecast.
    strat -- stratum, a column vector of dimension nr_tstamps-by-1. Entries to this vector should be from a set of S different symbols (the exact symbol is irrelevant). strat[n, 0] gives the stratum of the n'th sample. A warning is given if S > log(nr_tstams) or if the number of samples in the different strata differs by more than 20%.

    Output arguments:

    pval -- The p-value of the test

    covar_est -- Estimator of the (matrix valued) covariance function of the test statistic. This is an array with the three dimensions [S, S, lead_time]. The entry covar_est[s1, s2, l] is the correlation between Z(n) resp Z(n + l), projected onto stratum s1+1, resp stratum s2+1. We may calculate C = np.sum(covar_est, axis = 2), then the entry C[s1, s2] is the correlation between the same variables but summed over time and divided by sqrt(nr_tstamps).

    Disclaimer: Use at your own risk!

    (c) Jochen Broecker, 2018
    """
    # function body

    s_ver = ver.shape
    nr_tstamps = s_ver[0]
    s_moms = moms.shape
    uq_strata, strat_inv, strata_counts = np.unique(strat, return_inverse = True, return_counts = True)
    strat_inv = strat_inv.reshape([nr_tstamps, 1])
    nr_uq_strata = uq_strata.size

    # sort out strata
    if (nr_uq_strata > np.log(nr_tstamps)):
        print('Warning: Number of strata should not be larger than log(nr_tstamps)')
    elif np.sqrt(strata_counts.std()/strata_counts.mean() > 0.4):
        print('Warning: Number of samples in strata varies by more than 40%')

    # Form Z
    Z = (ver[:, 0] - moms[:, 0]) / np.sqrt(moms[:, 1] - moms[:, 0]**2)

    # Bring in the strata (TODO dispense with loop using np.repeat)
    new_Z = np.zeros((nr_tstamps, nr_uq_strata))
    for s in range(0, nr_uq_strata):
        new_Z[:, s] = Z * (strat_inv == s).reshape((nr_tstamps,))
        new_Z[:, s] = new_Z[:, s] / np.sqrt(strata_counts[s]/nr_tstamps)

    # perform generalised chi square test
    pval, covar_est = gen_chi_squ(new_Z, lead_time)

    retval = (pval, covar_est)
    
    return(retval)

def pit_test(pit_vals, lead_time, strat, ord_moments=1, return_hists = False):

    """function [pval, covar_est] = pit_test(pit_vals, lead_time, strat, ord_moments=1, return_hists = False):

    function [pval, covar_est, hist_vals] = pit_test(pit_vals, lead_time, strat, ord_moments=1, return_hists = False):
   
    Performs a generalised GOF test for probability integral transform.


    Input arguments:

    pit_vals -- The PIT of the verification, an array with dimensions [nr_tstamps, 1], first dimension representing time
    lead_time -- The lead time of the forecast.
    strat -- stratum, a column vector of dimension nr_tstamps-by-1. Entries to this vector should be from a set of S different symbols (the exact symbol is irrelevant). strat[n, 0] gives the stratum of the n'th sample, and samples with the same stratum will be associated with the same histogram (out of S histograms). A warning is given if S > log(nr_tstams) or if the number of samples in the different strata differs by more than 20%.

    Optional arguments
    
    ord_moments=ord -- The test is performed by checking the moments of pit_vals up to order ord which should be larger than 0. If ord = 1 (default), then the tests essentially just checks that mean(pit_vals) = 1/2. Internally, the code uses Legendre polynomials (written as L(ord, .) in the following) instead of moments as the former are orthonormal under the uniform distribution.

    Output arguments:

    pval -- The p-value of the test

    covar_est -- Estimator of the (matrix valued) covariance function of the test statistic. This is an array with the three dimensions [nr_moments * S, nr_moments * S, lead_time]. The entry covar_est[c1 + s1 * nr_moments, c2 + s2 * nr_moments, l] is the correlation between L(c1+1, pit_vals(n)) resp L(c2+1, pit_vals(n + l)), projected onto stratum s1+1, resp stratum s2+1. We may calculate C = np.sum(covar_est, axis = 2), then the entry C[c1 + s1 * nr_contrasts, c2 + s2 * nr_contrasts] is the correlation between the same variables but summed over time and divided by sqrt(nr_tstamps).

    hist_vals -- (if return_counts=True) Histogram bars in a data frame with dimensions [S, nr_bars] where nr_bars is chosen sensibly.

    Disclaimer: Use at your own risk!

    (c) Jochen Broecker, 2018
    """

# function body
    import numpy.polynomial.legendre as lgnd

    s_pit_vals = pit_vals.shape
    nr_tstamps = s_pit_vals[0]
    uq_strata, strat_inv, strata_counts = np.unique(strat, return_inverse = True, return_counts = True)
    strat_inv = strat_inv.reshape([nr_tstamps, 1])
    nr_uq_strata = uq_strata.size

    # sort out strata
    if (nr_uq_strata > np.log(nr_tstamps)):
        print('Warning: Number of strata should not be larger than log(nr_tstamps)')
    elif np.sqrt(strata_counts.std()/strata_counts.mean() > 0.4):
        print('Warning: Number of samples in strata varies by more than 40%')

    # Form orthogonal moment variables
    contrasted_Z = np.zeros((nr_tstamps, ord_moments))
    c = [0, 1]
    for ord in range(0, ord_moments):
        contrasted_Z[:, ord] = np.sqrt(2 * (ord + 1) + 1) * lgnd.legval(2 * pit_vals - 1, np.array(c)).reshape((nr_tstamps,))
        c = [0] + c
    
    # Bring in the strata (TODO dispense with loop using np.repeat)
    new_Z = np.zeros((nr_tstamps, nr_uq_strata * ord_moments))
    for s in range(0, nr_uq_strata):
        new_Z[:, s * ord_moments : (s+1) * ord_moments] = contrasted_Z * np.tile((strat_inv == s), [1, ord_moments])
        new_Z[:, s * ord_moments : (s+1) * ord_moments] = new_Z[:, s * ord_moments : (s+1) * ord_moments] / np.sqrt(strata_counts[s]/nr_tstamps)

    # perform generalised chi square test
    pval, covar_est = gen_chi_squ(new_Z, lead_time)

    retval = (pval, covar_est)
    
    # compute actual histograms for visual display
    if return_hists:
        hist_vals = []
        retval = retval + (hist_vals,)

    return(retval)


def rank_test(ver, ens, lead_time, strat, contrasts=[], return_counts = False):
    """function [pval, rnks_vals, covar_est] = rank_test(ver, ens, lead_time, strat, contrasts=[])    
    
    function [pval, rnks_vals, covar_est, rnks_counts] = rank_test(ver, ens, lead_time, strat, contrasts=[], return_counts = True)
    
    Performs a generalised GOF test for rank histograms from ensemble
    forecasts. 

    Input arguments:

    ver -- The verification, an array with dimensions [nr_tstamps, 1], first dimension representing time
    ens -- The ensemble, an array with dimensions [nr_tstamps, nr_ens] with first dimension representing time and second dimension representing ensemble members
    lead_time -- The lead time of the forecast.
    strat -- stratum, a column vector of dimension nr_tstamps-by-1. Entries to this vector should be from a set of S different symbols (the exact symbol is irrelevant). strat[n, 0] gives the stratum of the n'th sample, and samples with the same stratum will be associated with the same histogram (out of S histograms). A warning is given if S > log(nr_tstams) or if the number of samples in the different strata differs by more than 20%.

    Optional arguments
    
    contrasts=ctr -- If ctr=[] (default), a set of nr_ens orthonormal contrasts is computed. A contrast is a vector x so that sum(x) = 0, and there are nr_ens possible contrasts. The contrasts are generated using contrast_gen(nr_ens + 1, nr_ens). If ctr is a number m (not exceeding nr_ens), then only m contrasts are computed (by invoking contrast_gen(nr_ens + 1, m)). If ctr is a [m1, m2] shaped array, the columns are interpreted as contrasts, and the shape must be m1 = nr_ens + 1, m2 <= nr_ens.  

    Output arguments:

    pval -- The p-value of the test
    rnks_vals -- The ranks of the verification

    covar_est -- Estimator of the (matrix valued) covariance function of the histogram, projected onto the contrasts. This is a 3-dimensional array with dimensions [nr_ctr * S, nr_ctr * S, lead_time]. The entry covar_est[c1 + s1 * nr_contrasts, c2 + s2 * nr_contrasts, l] is the correlation between Rank(n) resp Rank(n + l), projected onto contrast and stratum (c1+1, s1+1), resp contrast and stratum (c2+1, s2+1). We may calculate C = np.sum(covar_est, axis = 2), then the entry C[c1 + s1 * nr_contrasts, c2 + s2 * nr_contrasts] is the correlation between the histograms for strata s1+1 and s2+1, both projected onto contrasts c1+1 and c2+1, respectively, and divided by sqrt(nr_tstamps).

    rnks_counts -- (if return_counts=True) The rank histogram bars in a data frame with dimensions [S, nr_contrasts]

    Disclaimer: Use at your own risk!

    (c) Jochen Broecker, 2018
    """
    # function body

    s_ver = ver.shape
    s_ens = ens.shape
    nr_tstamps = s_ver[0]
    nr_ens = s_ens[1]
    nr_ranks = 1 + nr_ens
    sqrt_ranks = np.sqrt(nr_ranks)
    nr_contrasts = nr_ens
    uq_strata, strat_inv, strata_counts = np.unique(strat, return_inverse = True, return_counts = True)
    strat_inv = strat_inv.reshape([nr_tstamps, 1])
    nr_uq_strata = uq_strata.size
    
    if (np.size(contrasts) == 0):
        contrasts = nr_ens
    if (type(contrasts) == type(0)):
        if (contrasts < nr_ranks):
            the_contrasts = contrast_gen(nr_ranks, contrasts)
            nr_contrasts = contrasts
        else: 
            print('Error: Number of contrasts must not be larger than number of ensemble members!')
    elif (type(contrasts) == type(np.array([]))):
        the_contrasts = contrasts
        s_the_contrasts = the_contrasts.shape
        if ((s_the_contrasts[0] != nr_ranks) or (s_the_contrasts[1] > nr_ens)):
            print('Error: Contrast matrix has wrong dimensions!')
        nr_contrasts = s_the_contrasts[1]
    else:
        print('Type of argument contrasts not understood')

    # sort out strata
    if (nr_uq_strata > np.log(nr_tstamps)):
        print('Warning: Number of strata should not be larger than log(nr_tstamps)')
    elif np.sqrt(strata_counts.std()/strata_counts.mean() > 0.4):
        print('Warning: Number of samples in strata varies by more than 40%')

    # Form scaled indicator variables
    ind = np.argsort(np.concatenate((ver, ens), axis=1), axis=1)
    allranks = np.argsort(ind, axis=1)
    rnks_vals = allranks[:,0] + 1
    Z = (ind == 0) * 1.0;
    Z = sqrt_ranks * Z
    contrasted_Z = Z @ the_contrasts

    # Bring in the strata (TODO dispense with loop using np.repeat)
    new_Z = np.zeros((nr_tstamps, nr_uq_strata * nr_contrasts))
    for s in range(0, nr_uq_strata):
        new_Z[:, s * nr_contrasts : (s+1) * nr_contrasts] = contrasted_Z * np.tile((strat_inv == s), [1, nr_contrasts])
        new_Z[:, s * nr_contrasts : (s+1) * nr_contrasts] =         new_Z[:, s * nr_contrasts : (s+1) * nr_contrasts] / np.sqrt(strata_counts[s]/nr_tstamps)

    # perform generalised chi square test
    pval, covar_est = gen_chi_squ(new_Z, lead_time)

    # compute actual histograms for visual display
    if return_counts:
        rnks_counts = crosstab(rnks_vals.reshape([nr_tstamps,]),  strat_inv.reshape([nr_tstamps,]))
                                                            
        return(pval, rnks_vals, covar_est, rnks_counts)
    else:
        return(pval, rnks_vals, covar_est)

    
def gen_chi_squ(Z, corr_time, standardised = True):
    """function [pval, covar_func] = gen_chi_squ(Z, corr_time, standardised = True)
    Performs a generalised chi square test. 

    Input arguments:

    Z -- An array with dimensions [nr_tstamps, d]
    corr_time -- integer providing correlation time of Z
    standardised -- Flag regarding zero lag correlations; see below

    Purpose:

    Suppose that Z satisfies a Central Limit Theorem, in the sense that 

    X = (1/sqrt(nr_tstamps)) sum(Z, axis = 0) 

    is asymptotically normal with mean zero and covariance matrix M, then 

    t = X * inv(M) * transpose(X)

    is chi-square distributed with d DOF. The covariance matrix M is in theory given by the sum over the (matrix valued) covariance function of Z (which is assumed to converge).  

    The code will estimate the covariance function and also M. It is assumed however that Z is mean free and has no correlations at a lag given by corr_time and beyond. If standardised = True, the code assumes that Z has (lag zero) covariance matrix given by the unit matrix (default). If this flag is set to False, the (lag zero) covariance matrix is estimated, too.

    Output arguments:

    pval -- The p-value of the test
    covar_func -- Estimator of the covariance function. This is an array with dimensions [d, d, corr_time]. There is some redundancy here since covar_func[:, :, 0] is the d-by-d unit matrix.


    Disclaimer: Use at your own risk!

    (c) Jochen Broecker, 2018
    """
    # function body

    s_Z = Z.shape
    nr_tstamps = s_Z[0]
    dof = s_Z[1]
    Z = Z / np.sqrt(nr_tstamps)
    
    # Prepare estimating variance
    covar_func = np.zeros([dof, dof, corr_time])
        
    # We start with correlation lag = 0
    if standardised:
        covar_func[:, :, 0] = np.eye(dof)
    else:
        buffer_length = nr_tstamps - l
        covar_func[:, :, 0] = np.transpose(Z) @ Z

    var_est = covar_func[:, :, 0]

    if (corr_time > 1):
        for l in range(1, corr_time):
            buffer_length = nr_tstamps - l
            dummy = np.transpose(Z[0:buffer_length, :]) @ Z[l:buffer_length+l, :]
            covar_func[:, :, l] = dummy + np.transpose(dummy)
            var_est = var_est + covar_func[:, :, l]

    inv_var_est = np.linalg.inv(var_est)
    d = np.sum(Z, axis=0)
    gofstat = (d @ inv_var_est) @ np.transpose(d)
    pval = 1 - stats.chi2.cdf(gofstat, dof)
                                                            
    return(pval, covar_func)


def contrast_gen(nr_ranks, nr_contrasts):
    """function contrasts = contrast_gen(nr_ranks, nr_contrasts)
    
    A reasonable set of orthonormal contrasts is computed. A contrast
    is a vector x so that sum(x) = 0.
  
    nr_ranks -- no. of ranks (= 1 + no. of ensemble members)
    nr_contrasts -- no. of contrasts returned. Must not be larger than no. of ensemble members

    return -- Array of dimension [nr_ranks, nr_contrasts] with columns representing orthonormal contrasts.

    Disclaimer: Use at your own risk!

    (c) Jochen Broecker, 2018
    """
    # function body

    V = np.zeros((nr_ranks, nr_contrasts + 1))
    V[:, 0] = np.ones((nr_ranks))
    ranks_recentered = np.linspace(1, nr_ranks, nr_ranks) - (nr_ranks + 1) / 2.0
    for ctr in range(1, nr_contrasts + 1):
        w = ranks_recentered ** ctr
        w = (w - np.mean(w)) / np.std(w)
        V[:, ctr] = w
    G = np.linalg.qr(V)
    return(G[0][:,1:])
