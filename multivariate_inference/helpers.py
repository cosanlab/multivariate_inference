"""Helper function definitions."""

import numpy as np
import statsmodels.api as sm
from scipy.interpolate import interp1d
from scipy.spatial.distance import squareform


__all__ = ['upper',
           'isPSD',
           'nearestPSD',
           'easy_multivariate_normal',
           'kde_pvalue',
           'create_heterogeneous_simulation'
           ]


def upper(mat):
    '''Return upper triangle of matrix'''
    idx = np.triu_indices_from(mat, k=1)
    return mat[idx]


def isPSD(mat, tol=1e-8):
    """
    Check if matrix is positive-semi-definite by virtue of all its eigenvalues being >= 0. The cholesky decomposition does not work for edge cases because np.linalg.cholesky fails on matrices with exactly 0 valued eigenvalues, whereas in Matlab this is not true, so that method appropriate. Ref: https://goo.gl/qKWWzJ
    """

    # We dont assume matrix is Hermitian, i.e. real-valued and symmetric
    # Could swap this out with np.linalg.eigvalsh(), which is faster but less general
    e = np.linalg.eigvals(mat)
    return np.all(e > -tol)


def nearestPSD(A, nit=100):
    """
    Higham (2000) algorithm to find the nearest positive semi-definite matrix that minimizes the Frobenius distance/norm. Sstatsmodels using something very similar in corr_nearest(), but with spectral SGD to search for a local minima. Reference: https://goo.gl/Eut7UU

    Args:
        nit (int): number of iterations to run algorithm; more iterations improves accuracy but increases computation time.
    """

    n = A.shape[0]
    W = np.identity(n)

    def _getAplus(A):
        eigval, eigvec = np.linalg.eig(A)
        Q = np.matrix(eigvec)
        xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
        return Q * xdiag * Q.T

    def _getPs(A, W=None):
        W05 = np.matrix(W**.5)
        return W05.I * _getAplus(W05 * A * W05) * W05.I

    def _getPu(A, W=None):
        Aret = np.array(A.copy())
        Aret[W > 0] = np.array(W)[W > 0]
        return np.matrix(Aret)

    # W is the matrix used for the norm (assumed to be Identity matrix here)
    # the algorithm should work for any diagonal W
    deltaS = 0
    Yk = A.copy()
    for k in range(nit):
        Rk = Yk - deltaS
        Xk = _getPs(Rk, W=W)
        deltaS = Xk - Rk
        Yk = _getPu(Xk, W=W)
    # Double check returned matrix is PSD
    if isPSD(Yk):
        return Yk
    else:
        nearestPSD(Yk)


def easy_multivariate_normal(num_obs, num_features, corrs, mu=0.0, sigma=1.0, seed=None, forcePSD=True, return_new_corrs=False, nit=100):
    """
    Function to more easily generate multivariate normal samples provided a correlation matrix or list of correlations (upper triangle of correlation matrix) instead of a covariance matrix. Defaults to returning approximately standard normal (mu = 0; sigma = 1) variates. Unlike numpy, if the desired correlation matrix is not positive-semi-definite, will by default issue a warning and find the nearest PSD correlation matrix and generate data with this matrix. This new matrix can optionally be returned used the return_new_corrs argument.

    Args:
        num_obs (int): number of observations/samples to generate (rows)
        corrs (ndarray/list/float): num_features x num_features 2d array, flattend numpy array of length (num_features * (num_features-1)) / 2, or scalar for same correlation on all off-diagonals
        num_features (int): number of features/variables/dimensions to generate (columns)
        mu (float/list): mean of each feature across observations; default 0.0
        sigma (float/list): sd of each feature across observations; default 1.0
        forcePD (bool): whether to find and use a new correlation matrix if the requested one is not positive semi-definite; default False
        return_new_corrs (bool): return the nearest correlation matrix that is positive semi-definite used to generate data; default False
        nit (int): number of iterations to search for the nearest positive-semi-definite correlation matrix is the requested correlation matrix is not PSD; default 100

    Returns:
        ndarray: correlated data as num_obs x num_features array
    """

    if seed is not None:
        np.random.seed(seed)

    if isinstance(mu, list):
        assert len(
            mu) == num_features, "Number of means must match number of features"
    else:
        mu = [mu] * num_features
    if isinstance(sigma, list):
        assert len(
            sigma) == num_features, "Number of sds must match number of features"
    else:
        sigma = [sigma] * num_features

    if isinstance(corrs, np.ndarray) and corrs.ndim == 2:
        assert corrs.shape[0] == corrs.shape[1] and np.allclose(corrs, corrs.T) and np.allclose(
            np.diagonal(corrs), np.ones_like(np.diagonal(corrs))), "Correlation matrix must be square symmetric"
    elif (isinstance(corrs, np.ndarray) and corrs.ndim == 1) or isinstance(corrs, list):
        assert len(corrs) == (num_features * (num_features - 1)) / \
            2, "(num_features * (num_features - 1) / 2) correlation values are required for a flattened array or list"
        corrs = squareform(corrs)
        np.fill_diagonal(corrs, 1.0)
    elif isinstance(corrs, float):
        corrs = np.array(
            [corrs] * int(((num_features * (num_features - 1)) / 2)))
        corrs = squareform(corrs)
        np.fill_diagonal(corrs, 1.0)
    else:
        raise ValueError(
            "Correlations must be num_features x num_feature, flattend numpy array/list or scalar")

    if not isPSD(corrs):
        if forcePSD:
            # Tell user their correlations are being recomputed if they didnt ask to save them as they might not realize
            if not return_new_corrs:
                print(
                    "Correlation matrix is not positive semi-definite. Solved for new correlation matrix.")
            _corrs = np.array(nearestPSD(corrs, nit))

        else:
            raise ValueError("Correlation matrix is not positive semi-definite. Pymer4 will not generate inaccurate multivariate data. Use the forcePD argument to automatically solve for the closest desired correlation matrix.")
    else:
        _corrs = corrs

    # Rescale correlation matrix by variances, given standard deviations of features
    sd = np.diag(sigma)
    # R * Vars = R * SD * SD
    cov = _corrs.dot(sd.dot(sd))
    X = np.random.multivariate_normal(mu, cov, size=num_obs)

    if return_new_corrs:
        return X, _corrs
    else:
        return X


def kde_pvalue(permutation_distribution, test_statistic, tails=2, kde_grid_size=200):
    """
    Use a KDE to smooth a permutation distribution and use a interpolation to compute p-values a la:
    https://users.aalto.fi/~eglerean/bramila_mantel.m

    Args:
        permutation_distribution (ndarry): array of permuted test statistics
        test_statistic (float): true value of computed test statistic
        tails (int): two-tailed or one-tailed p-value; default two-tailed
        kde_grid_size (int): size of the kde grid to generate; default 200 if len(permutation_distribution) <= 5000 otherwise multiples of 200 correponding to how many extra permutations were performed in multiples of 5000
    """

    if len(permutation_distribution) > 5000 and kde_grid_size == 200:
        kde_grid_size = int(np.round(200 * len(permutation_distribution) / 5000))

    kde = sm.nonparametric.KDEUnivariate(permutation_distribution)
    # Compute robust bandwith of KDE kernel like matlab
    # BandWidth = sig * (4/(3*N))^(1/5);
    # Where sig = MAD / half-normal distribution median
    bw = (np.median(np.abs(permutation_distribution - np.median(permutation_distribution))) / 0.6745) * (4 / (3 * len(permutation_distribution))) ** (1 / 5)
    kde.fit(gridsize=kde_grid_size, fft=False, bw=bw)

    # Get cumulative distribution function of kde fit
    cdf = kde.cdf
    # Learn a linear interpolation function between kde supports and the CDF to look up p-values with
    # This is similar to tdist in scipy to look up p-values
    pdist_func = interp1d(kde.support, cdf, fill_value='extrapolate')

    # Look up p-value
    left_p = pdist_func(test_statistic)
    right_p = pdist_func(-1 * test_statistic)

    # Deal with scipy extrapolating p-values out of range
    if left_p < 0:
        left_p = 0.000
    elif left_p > 1:
        left_p = 1.000
    if right_p < 0:
        right_p = 0.000
    elif right_p > 1:
        right_p = 1.000

    if test_statistic > 0:
        left_p = 1 - left_p
    else:
        right_p = 1 - right_p

    if tails == 2:
        out = left_p + right_p
    elif tails == 1:
        out = left_p

    return out, pdist_func


def create_heterogeneous_simulation(r_within_1, r_within_2, r_between_1, r_between_2, n_variables):
    '''Create a heterogeneous multivariate covariance matrix based on:
        Omelka, M. and Hudecova, S. (2013) A comparison of the Mantel test
        with a generalised distance covariance test. Environmetrics,
        Vol. 24, 449–460. DOI: 10.1002/env.2238.
    '''
    z = np.zeros((n_variables*2, n_variables*2))
    z[:int(n_variables/2), :int(n_variables/2)] = r_within_1
    z[int(n_variables/2):n_variables, int(n_variables/2):n_variables] = r_within_2
    z[n_variables:int(n_variables + (n_variables/2)), n_variables:int(n_variables + (n_variables/2))] = r_within_1
    z[int(n_variables + (n_variables/2)):, int(n_variables + (n_variables/2)):] = r_within_2
    z[n_variables:int(n_variables + (n_variables/2)), :int(n_variables/2)] = r_between_1
    z[int(n_variables + (n_variables/2)):, int(n_variables/2):n_variables] = r_between_2
    z[:int(n_variables/2), n_variables:int(n_variables+(n_variables/2))] = r_between_1
    z[int(n_variables/2):n_variables, int(n_variables + (n_variables/2)):] = r_between_2
    z[np.diag_indices(n_variables*2)] = 1
    return z
