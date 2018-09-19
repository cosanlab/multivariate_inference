"""Dependence measures functions."""

import numpy as np
from scipy.spatial import procrustes
from scipy.spatial.distance import squareform, pdist
from scipy.stats import t as t_dist
import warnings
from joblib import Parallel, delayed
from sklearn.utils import check_random_state
from nltools.stats import _calc_pvalue


__all__ = ['double_center',
           'u_center',
           'distance_correlation',
           'procrustes_similarity'
           ]


def double_center(mat):
    '''Double center a 2d array.

    Args:
        mat (ndarray): 2d numpy array

    Returns:
        mat (ndarray): double-centered version of input

    '''

    if len(mat.shape) != 2:
        raise ValueError('Array should be 2d')

    # keepdims ensures that row/column means are not incorrectly broadcast during    subtraction
    row_mean = mat.mean(axis=0, keepdims=True)
    col_mean = mat.mean(axis=1, keepdims=True)
    grand_mean = mat.mean()
    return mat - row_mean - col_mean + grand_mean


def u_center(mat):
    '''U-center a 2d array. U-centering is a bias-corrected form of double-centering

    Args:
        mat (ndarray): 2d numpy array

    Returns:
        mat (narray): u-centered version of input
    '''

    if len(mat.shape) != 2:
        raise ValueError('Array should be 2d')

    dim = mat.shape[0]
    u_mu = mat.sum() / ((dim - 1) * (dim - 2))
    sum_cols = mat.sum(axis=0, keepdims=True)
    sum_rows = mat.sum(axis=1, keepdims=True)
    u_mu_cols = np.ones((dim, 1)).dot(sum_cols / (dim - 2))
    u_mu_rows = (sum_rows / (dim - 2)).dot(np.ones((1, dim)))
    out = np.copy(mat)
    # Do one operation at a time, to improve broadcasting memory usage.
    out -= u_mu_rows
    out -= u_mu_cols
    out += u_mu
    # The diagonal is zero
    out[np.eye(dim, dtype=bool)] = 0


def distance_correlation(x, y, bias_corrected=True, return_all_stats=False):
    '''Compute the distance correlation betwen 2 arrays.
        Distance correlation involves computing the normalized covariance of two centered euclidean distance matrices. Each distance matrix is the euclidean distance between rows (if x or y are 2d) or scalars (if x or y are 1d). Each matrix is centered using u-centering, a bias-corrected form of double-centering. This permits inference of the normalized covariance between each distance matrix using a one-tailed directional t-test. (Szekely & Rizzo, 2013). While distance correlation is normally bounded between 0 and 1, u-centering can produce negative estimates, which are never significant. Therefore these estimates are windsorized to 0, ala Geerligs, Cam-CAN, Henson, 2016.

    Args:
        x (ndarray): 1d or 2d numpy array of observations by features
        y (ndarry): 1d or 2d numpy array of observations by features
        bias_corrected (bool): if false use double-centering but no inference test is performed, if true use u-centering and perform inference; default True
        return_all_stats (bool): if true return distance covariance and variances of each array as well; default False

    Returns:
        results (dict): dictionary of results (correlation, t, p, and df.) Optionally, covariance, x variance, and y variance
    '''

    if len(x.shape) > 2 or len(y.shape) > 2:
        raise ValueError("Both arrays must be 1d or 2d")

    # 1 compute euclidean distances between pairs of value in each array
    if len(x.shape) == 1:
        _x = x[:, np.newaxis]
    else:
        _x = x
    if len(y.shape) == 1:
        _y = y[:, np.newaxis]
    else:
        _y = y

    x_dist = squareform(pdist(_x))
    y_dist = squareform(pdist(_y))

    # 2 center each matrix
    if bias_corrected:
        # U-centering
        x_dist_cent = u_center(x_dist)
        y_dist_cent = u_center(y_dist)
        # Compute covariances using N*(N-3) in denominator
        adjusted_n = _x.shape[0] * (_x.shape[0]-3)
        xy = np.multiply(x_dist_cent, y_dist_cent).sum() / adjusted_n
        xx = np.multiply(x_dist_cent, x_dist_cent).sum() / adjusted_n
        yy = np.multiply(y_dist_cent, y_dist_cent).sum() / adjusted_n
    else:
        # double-centering
        x_dist_cent = double_center(x_dist)
        y_dist_cent = double_center(y_dist)
        # Compute covariances using N^2 in denominator
        xy = np.multiply(x_dist_cent, y_dist_cent).mean()
        xx = np.multiply(x_dist_cent, x_dist_cent).mean()
        yy = np.multiply(y_dist_cent, y_dist_cent).mean()

    # 3 compute covariances and variances
    var_x = np.sqrt(xx)
    var_y = np.sqrt(yy)

    # 4 Normalize to get correlation
    denom = np.sqrt(xx * yy)
    if denom > 0:
        r2 = xy / denom
    else:
        r2 = 0
    # Windsorize negative values as a result of u-centering
    if r2 > 0:
        cor = np.sqrt(r2)
    else:
        cor = 0

    out = {}
    out['d_correlation_adjusted'] = cor

    if return_all_stats:
        out['d_covariance_squared'] = xy
        out['d_correlation_adjusted'] = r2
        out['x_var'] = var_x
        out['y_var'] = var_y

    if bias_corrected:
        dof = (adjusted_n / 2) - 1
        t = np.sqrt(dof) * (r2 / np.sqrt(1 - r2**2))
        p = 1-t_dist.cdf(t, dof)
        out['t'] = t
        out['p'] = p
        out['df'] = dof

    return out


def procrustes_similarity(mat1, mat2, n_permute=5000, tail=1, n_jobs=-1, random_state=None):
    """ Use procrustes super-position to perform a similarity test between 2 matrices. Matrices need to match in size on their first dimension only, as the smaller matrix on the second dimension will be padded with zeros. After aligning two matrices using the procrustes transformation, use the computed disparity between them (sum of squared error of elements) as a similarity metric. Shuffle the rows of one of the matrices and recompute the disparity to perform inference (Peres-Neto & Jackson, 2001). Note: by default this function reverses disparity to treat it like a *similarity* measure like correlation, rather than a distance measure like correlation distance, i.e. smaller values mean less similar, larger values mean more similar.

    Args:
        mat1 (ndarray): 2d numpy array; must have same number of rows as mat2
        mat2 (ndarray): 1d or 2d numpy array; must have same number of rows as mat1
        n_permute (int): number of permutation iterations to perform
        tail (int): either 1 for one-tailed or 2 for two-tailed test; default 2
        n_jobs (int): The number of CPUs to use to do permutation; default -1 (all)

    Returns:
        similarity (float): similarity between matrices bounded between 0 and 1
        pval (float): permuted p-value

    """

    warnings.warn("This function needs to be edited to scale SSE to a proportion. It is currently WRONG.")

    if mat1.shape[0] != mat2.shape[0]:
        raise ValueError('Both arrays must match on their first dimension')

    random_state = check_random_state(random_state)

    # Make sure both matrices are 2d and the same dimension via padding
    if len(mat1.shape) < 2:
        mat1 = mat1[:, np.newaxis]
    if len(mat2.shape) < 2:
        mat2 = mat2[:, np.newaxis]
    if mat1.shape[1] > mat2.shape[1]:
        mat2 = np.pad(mat2, ((0, 0), (0, mat1.shape[1] - mat2.shape[1])), 'constant')
    elif mat2.shape[1] > mat1.shape[1]:
        mat1 = np.pad(mat1, ((0, 0), (0, mat2.shape[1] - mat1.shape[1])), 'constant')

    _, _, sse = procrustes(mat1, mat2)
    sse = 1 - sse  # flip to similarity measure

    stats = dict()
    stats['similarity'] = sse

    all_p = Parallel(n_jobs=n_jobs)(delayed(procrustes)(random_state.permutation(mat1), mat2) for i in range(n_permute))
    all_p = [1 - x[2] for x in all_p]

    stats['p'] = _calc_pvalue(all_p, sse, tail)

    return stats
