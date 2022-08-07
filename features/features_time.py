from scipy.stats import kurtosis as _kurt
from scipy.stats import skew as _skew
from itertools import groupby
import pandas as pd
import numpy as np
import config as c

Kmax = 3
n = 4
T = 1
Tau = 4
DE = 10
W = None
sfreq = c.FS

def _hjorth_parameters(epochs, axis, **kwargs):
    """Computes the Hjorth parameters.

        Parameters
        ----------
        epochs : numpy array
            one dimensional embedding.

        Returns
        -------
        activity : float
             The final output is a number which corresponds to the variance of the embedding sequence.

        mobility : float
             The final output is a number which corresponds to the mobility of the embedding sequence.

        complexity : float
             The final output is a number which corresponds to the variance complexity of the embedding sequence.

        References
        ----------
        .. https://en.wikipedia.org/wiki/Hjorth_parameters

    """

    def _hjorth_mobility(epochs, axis, **kwargs):
        diff = np.diff(epochs, axis=axis)
        sigma0 = np.std(epochs, axis=axis)
        sigma1 = np.std(diff, axis=axis)
        return np.divide(sigma1, sigma0)

    activity = np.var(epochs, axis=axis)
    diff1 = np.diff(epochs, axis=axis)
    diff2 = np.diff(diff1, axis=axis)
    sigma0 = np.std(epochs, axis=axis)
    sigma1 = np.std(diff1, axis=axis)
    sigma2 = np.std(diff2, axis=axis)
    mobility = np.divide(sigma1, sigma0)
    complexity = np.divide(np.divide(sigma2, sigma1), _hjorth_mobility(epochs, axis))
    return activity, complexity, mobility


def hurst_exponent(epochs, axis, **kwargs):
    """ Computes the Hurst exponent of a time series. If the output H=0.5,the behavior
        of the time-series is similar to random walk. If H<0.5, the time-series
        cover less "distance" than a random walk, vice versa.

        Parameters
        ----------
        epochs : numpy array
            one dimensional embedding.

        Returns
        -------
        H :  float
            Hurst exponent.

        References
        ----------
        .. https://en.wikipedia.org/wiki/Hurst_exponent
    """

    def hurst_1d(X):
        X = np.array(X)
        N = X.size
        T = np.arange(1, N + 1)
        Y = np.cumsum(X)
        Ave_T = Y / T

        # compute the standard deviation "St" and the range of cumulative deviate series "Rt"
        S_T = np.zeros(N)
        R_T = np.zeros(N)
        for i in range(N):
            S_T[i] = np.std(X[:i + 1])
            X_T = Y - T * Ave_T[i]
            R_T[i] = np.ptp(X_T[:i + 1])

        # check for indifferent measurements at time series start
        # they could be introduced by resampling and have to be removed
        # if not removed, it will cause division by std = 0
        for i in range(1, len(S_T)):
            if np.diff(S_T)[i - 1] != 0:
                break
        for j in range(1, len(R_T)):
            if np.diff(R_T)[j - 1] != 0:
                break
        k = max(i, j)
        assert k < 10, "Cannot compute the differenced series!"

        R_S = R_T[k:] / S_T[k:]
        R_S = np.log(R_S)

        # fit the power law E[R(n)/S(n)]=Cn^{H} to the data by plotting log[R(n)/S(n)] as a function of log n,
        # fit a straight line with y = ax + b, where the slope "a" is the estimated Hurst exponent.
        n = np.log(T)[k:]
        A = np.column_stack((n, np.ones(n.size)))
        [a, b] = np.linalg.lstsq(A, R_S, rcond=None)[0]
        H = a
        return H

    return np.apply_along_axis(hurst_1d, axis, epochs)


def higuchi_fractal_dimension(epochs, axis, **kwargs):
    """ Computes the Fractal Dimension using Higuchi's method.

        Parameters
        ----------
        epochs : numpy array
            one dimensional embedding.

        Returns
        -------
        hdf :  float
            Higuchi's fractal dimension.

        References
        ----------
        .. https://en.wikipedia.org/wiki/Higuchi_dimension
    """

    def hfd_1d(X, Kmax):
        L = []
        x = []
        N = len(X)
        for k in range(1, Kmax):
            Lk = []
            for m in range(0, k):
                Lmk = 0
                # create x_m^k reconstructed sequences
                for i in range(1, int(np.floor((N - m) / k))):
                    # for each produced sequence x_m^k, calculate the average box length Lmk
                    Lmk += abs(X[m + i * k] - X[m + i * k - k])
                Lmk = Lmk * (N - 1) / np.floor((N - m) / float(k)) / k

                # add the average box length of the sequences
                Lk.append(Lmk)

            # take the log of the average box length which behaves proportionally to the fractal dimension D times
            # log of the reciprocal of k
            L.append(np.log(np.mean(Lk)))
            x.append([np.log(1. / k), 1])

        # fit a straight line with y = ax + b, where the slope "a" is the estimated Hurst exponent.
        (p, r1, r2, s) = np.linalg.lstsq(x, L, rcond=None)
        return p[0]

    Kmax = kwargs["Kmax"]
    return np.apply_along_axis(hfd_1d, axis, epochs, Kmax)


def petrosian_fractal_dimension(epochs, axis, **kwargs):
    """ Computes the Petrosian Fractal Dimension.

        Parameters
        ----------
        epochs : numpy array
            one dimensional embedding.

        Returns
        -------
        hdf :  float
            Petrosian fractal dimension.

        References
        ----------
        .. https://www.researchgate.net/publication
        /40902603_Comparison_of_Fractal_Dimension_Algorithms_for_the_Computation_of_EEG_Biomarkers_for_Dementia
    """

    def pfd_1d(X, D=None):
        if D is None:
            D = np.diff(X)
            D = D.tolist()

        # number of sign changes in derivative of the signal
        N_delta = 0
        for i in range(1, len(D)):
            if D[i] * D[i - 1] < 0:
                N_delta += 1
        n = len(X)
        return np.log10(n) / (np.log10(n) + np.log10(n / n + 0.4 * N_delta))

    return np.apply_along_axis(pfd_1d, axis, epochs)


def svd_entropy(epochs, axis, **kwargs):
    """Computes entropy of the singular values retrieved from a singular value decomposition from the original series.

        Parameters
        ----------
        epochs : numpy array
            one dimensional embedding.

        Returns
        -------
        svd_entropy :  float
            total entropy of all the singular values.

        References
        ----------
        .. https://en.wikipedia.org/wiki/Entropy_(information_theory)
    """

    def svd_entropy_1d(X, Tau, DE, W):
        if W is None:
            Y = _embed_seq(X, Tau, DE)

            # compute the singular values
            W = np.linalg.svd(Y, compute_uv=False)

            # normalize singular values
            W /= sum(W)
        return -1 * sum(W * np.log(W))

    Tau = kwargs["Tau"]
    DE = kwargs["DE"]
    W = kwargs["W"]
    return np.apply_along_axis(svd_entropy_1d, axis, epochs, Tau, DE, W)


def fisher_information(epochs, axis, **kwargs):
    """Computes fisher information of the singular values retrieved from the original series.

        Parameters
        ----------
        epochs : numpy array
            one dimensional embedding.

        Returns
        -------
        FI_v :  float
            Total Fisher information for all singular values.

        References
        ----------
        .. https://en.wikipedia.org/wiki/Fisher_information
    """

    def fisher_info_1d(a, tau, de):
        mat = _embed_seq(a, tau, de)

        # compute the singular values
        W = np.linalg.svd(mat, compute_uv=False)

        # normalize singular values
        W /= sum(W)

        # compute the Fisher information for all singular values
        FI_v = (W[1:] - W[:-1]) ** 2 / W[:-1]
        FI_v = np.sum(FI_v)
        return FI_v

    tau = kwargs["Tau"]
    de = kwargs["DE"]
    return np.apply_along_axis(fisher_info_1d, axis, epochs, tau, de)


def largest_lyauponov_exponent(epochs, axis, **kwargs):
    """Computes the largest Lyauponov exponent using Rosenstein's method.

            A n-dimensional trajectory is first reconstructed from the observed data by
            use of embedding delay of tau, using pyeeg function, embed_seq(x, tau, n).
            Algorithm then searches for nearest neighbour of each point on the
            reconstructed trajectory; temporal separation of nearest neighbours must be
            greater than mean period of the time series: the mean period can be
            estimated as the reciprocal of the mean frequency in power spectrum
            Each pair of nearest neighbours is assumed to diverge exponentially at a
            rate given by largest Lyapunov exponent. Now having a collection of
            neighbours, a least square fit to the average exponential divergence is
            calculated. The slope of this line gives an accurate estimate of the
            largest Lyapunov exponent.

    Parameters
        ----------
        epochs : numpy array
            one dimensional embedding.

        Returns
        -------
        Lexp :  float
            Largest Lyauponov Exponent.

        References
        ----------
        ... Rosenstein, Michael T., James J. Collins, and Carlo J. De Luca. "A
            practical method for calculating largest Lyapunov exponents from small data
            sets." Physica D: Nonlinear Phenomena 65.1 (1993): 117-134.

    """

    def LLE_1d(x, tau, n, T, fs):
        Em = _embed_seq(x, tau, n)
        M = len(Em)
        A = np.tile(Em, (len(Em), 1, 1))
        B = np.transpose(A, [1, 0, 2])
        square_dists = (A - B) ** 2  # square_dists[i,j,k] = (Em[i][k]-Em[j][k])^2

        # D[i,j] = ||Em[i]-Em[j]||_2
        D = np.sqrt(square_dists[:, :, :].sum(axis=2))

        # exclude elements within T of the diagonal
        band = np.tri(D.shape[0], k=T) - np.tri(D.shape[0], k=-T - 1)
        band[band == 1] = np.inf
        neighbors = (D + band).argmin(axis=0)  # nearest neighbors more than T steps away

        # locate nearest neighbors of each point on the trajectory
        # in_bounds[i,j] = (i+j <= M-1 and i+neighbors[j] <= M-1)
        inc = np.tile(np.arange(M), (M, 1))
        row_inds = (np.tile(np.arange(M), (M, 1)).T + inc)
        col_inds = (np.tile(neighbors, (M, 1)) + inc.T)
        in_bounds = np.logical_and(row_inds <= M - 1, col_inds <= M - 1)

        row_inds[~in_bounds] = 0
        col_inds[~in_bounds] = 0

        # the nearest neighbor, Xˆj , is found by searching for the point that minimizes
        # the distance to the particular reference point, Xj.
        # neighbor_dists[i,j] = ||Em[i+j]-Em[i+neighbors[j]]||_2
        neighbor_dists = np.ma.MaskedArray(D[row_inds, col_inds], ~in_bounds)
        J = (~neighbor_dists.mask).sum(axis=1)
        neighbor_dists[neighbor_dists == 0] = 1

        # handle division by zero cases
        neighbor_dists.data[neighbor_dists.data == 0] = 1

        # impose the additional constraint that nearest neighbors need to have a temporal separation
        # greater than the mean period of the time series.
        d_ij = np.sum(np.log(neighbor_dists.data), axis=1)
        mean_d = d_ij[J > 0] / J[J > 0]

        # following Rosensteins’ method, the largest
        # Lyapunov exponent can be defined using d(t) = Ce^{λ1 *t}
        # where d(t) is the average divergence at time t and C is a constant
        # that normalizes the initial separation
        x = np.arange(len(mean_d))

        # compute the mean of the set of parallel lines (for j = 1, 2, ..., N ),
        # each with slope roughly proportional to λ1.
        # The largest lyauponov exponent is then
        # calculated using a least-squares fit to the average line
        X = np.vstack((x, np.ones(len(mean_d)))).T
        [m, c] = np.linalg.lstsq(X, mean_d, rcond=None)[0]
        Lexp = fs * m
        return Lexp

    tau = kwargs["Tau"]
    n = kwargs["n"]
    T = kwargs["T"]
    fs = kwargs["fs"]
    return np.apply_along_axis(LLE_1d, axis, epochs, tau, n, T, fs)


def _embed_seq(X, Tau, D):
    """Build a set of embedding sequences from given time series X with lag Tau
    and embedding dimension DE. Let X = [x(1), x(2), ... , x(N)], then for each
    i such that 1 < i <  N - (D - 1) * Tau, we build an embedding sequence,
    Y(i) = [x(i), x(i + Tau), ... , x(i + (D - 1) * Tau)]. All embedding
    sequence are placed in a matrix Y.

    Parameters
    ----------
    X : list
        one dimensional time series data

    Tau : int
        the lag or delay when building embedding sequence

    D : int
        the embedding dimension

    Returns
    -------
    Y : list
        2-D list with the embedding sequences
    """

    shape = (X.size - Tau * (D - 1), D)
    strides = (X.itemsize, Tau * X.itemsize)
    return np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)


def flat_spots(epochs):
    try:
        cut_x = pd.cut(epochs, bins=10, include_lowest=True, labels=False) + 1
    except:
        return np.array([0] * len(epochs))
    return np.array([sum(1 for i in g) for k, g in groupby(cut_x)]).max()


def lumpiness(epochs, freq=sfreq):
    nr = len(epochs)
    lo = np.arange(0, nr, freq)
    up = lo + freq
    nsegs = nr / freq
    varx = [np.nanvar(epochs[lo[idx]:up[idx]], ddof=1) for idx in np.arange(int(nsegs))]
    if nr < 2 * freq:
        lumpiness = 0
    else:
        lumpiness = np.nanvar(varx, ddof=1)
    return lumpiness

def skewness(epochs, axis, **kwargs):
    return _skew(epochs, axis=axis, bias=False)

def kurtosis(epochs, axis, **kwargs):
    return _kurt(epochs, axis=axis, bias=False)

def line_length(epochs, axis, **kwargs):
    return np.sum(np.abs(np.diff(epochs)), axis=axis)

def maximum(epochs, axis, **kwargs):
    return np.max(epochs, axis=axis)

def mean(epochs, axis, **kwargs):
    return np.mean(epochs, axis=axis)

def median(epochs, axis, **kwargs):
    return np.median(epochs, axis=axis)

def minimum(epochs, axis, **kwargs):
    return np.min(epochs, axis=axis)

def energy(epochs, axis, **kwargs):
    return np.mean(epochs * epochs, axis=axis)

def non_linear_energy(epochs, axis, **kwargs):
    return np.apply_along_axis(lambda epoch: np.mean((np.square(epoch[1:-1]) - epoch[2:] * epoch[:-2])), axis, epochs)

def line_length(epochs, axis, **kwargs):
    return np.sum(np.abs(np.diff(epochs)), axis=axis)

def zero_crossing(epochs, axis, **kwargs):
    e = 0.01
    norm = epochs - epochs.mean()
    return np.apply_along_axis(lambda epoch: np.sum((epoch[:-5] <= e) & (epoch[5:] > e)), axis, norm)

def zero_crossing_derivative(epochs, axis, **kwargs):
    e = 0.01
    diff = np.diff(epochs)
    norm = diff - diff.mean()
    return np.apply_along_axis(lambda epoch: np.sum(((epoch[:-5] <= e) & (epoch[5:] > e))), axis, norm)


