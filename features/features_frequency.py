import numpy as np

def maximum(power_spectrum, axis=-1):
    return np.max(power_spectrum, axis=axis)

def mean(power_spectrum, axis=-1):
    return np.mean(power_spectrum, axis=axis)

def minimum(power_spectrum, axis=-1):
    return np.min(power_spectrum, axis=axis)

def peak_frequency(power_spectrum, axis=-1):
    return power_spectrum.argmax(axis=axis)

def power(power_spectrum, axis=-1):
    return np.sum(power_spectrum**2, axis=axis)

def power_ratio(power_spectrum, axis=-1):
    ratios = power_spectrum / np.sum(power_spectrum, axis=axis, keepdims=True)
    return ratios

def spectral_entropy(power_spectrum):
    return -1 * power_spectrum * np.log(power_spectrum)

def value_range(power_spectrum, axis=-1):
    return np.ptp(power_spectrum, axis=axis)

def variance(power_spectrum, axis=-1):
    return np.var(power_spectrum, axis=axis)

