import numpy as np

def bounded_variation(coefficients, axis):
    diffs = np.diff(coefficients, axis=axis)
    abs_sums = np.sum(np.abs(diffs), axis=axis)
    max_c = np.max(coefficients, axis=axis)
    min_c =np.min(coefficients, axis=axis)
    return np.divide(abs_sums, max_c - min_c)

def maximum(coefficients, axis):
    return np.max(coefficients, axis=axis)

def mean(coefficients, axis):
    return np.mean(coefficients, axis=axis)

def minimum(coefficients, axis):
    return np.min(coefficients, axis=axis)

def power(coefficients, axis):
    return np.sum(coefficients*coefficients, axis=axis)

def power_ratio(coefficients, axis=-2):
    ratios = coefficients / np.sum(coefficients, axis=axis, keepdims=True)
    return ratios

def spectral_entropy(coefficients, axis=None):
    return -1 * coefficients * np.log(coefficients)

def variance(coefficients, axis):
    return np.var(coefficients, axis=axis)