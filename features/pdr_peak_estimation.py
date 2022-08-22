from scipy.signal import find_peaks, peak_prominences, find_peaks_cwt
from features_background import WelchEstimate
import numpy as np
import config as c

def normalise(x):
    assert type(x) == np.ndarray, "Exception: Returned Type Mismatch"
    assert np.min(x) != 0, "Exception: Cannot normalise series: division by zero"
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def find_peaks_alpha_power(x_o1_closed, x_o2_closed, threshold=1.5):
    """Find peak frequencies in the estimated power spectrum.
      A combination of both "scipy.signals.find_peaks" and "scipy.signals.find_peaks_cwt"
      were used to find an initial dominant peak location before estimating the dominant frequency.
      The initial peak suggestion is based on neighboring peak frequencies. The total method comprises
      three main steps:

      Step 1) Estimate the log power spectra using Welch’s method and
              detect initial peak frequencies.

      Step 2) Compute dominant peak suggestions from initial peak frequencies
              detected in both log power spectra.

      Step 3) Estimate dominant peak frequency from the estimated mean
              log power spectrum in the occipital region.

    Parameters
    ----------
    x_o1_closed : numpy array
         1D array of the channel values from the o1 channel in the eyes closed state
         Input array requires common reference montage.

    x_o2_closed : numpy array
         1D array of the channel values from the o2 channel in the eyes closed state
         Input array requires common reference montage.

    threshold : float
         Parameter threshold to find neighbouring peak frequencies (e.g., pk +- threshold)

    Returns
    -------
    Qpeak : float
        dominant alpha peak frequency

    References
    ----------
    ..https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
    ..https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks_cwt.html
    """

    def _find_peaks(fnorm, Pnorm):
        peaks_welch, _ = find_peaks(Pnorm)
        peaks_cwt = find_peaks_cwt(Pnorm, np.arange(1, 50))

        if len(peaks_welch) and len(peaks_cwt) > 0:
            npp_welch = peak_prominences(Pnorm, peaks_welch)[0].tolist()
            npp_cwt = peak_prominences(Pnorm, peaks_cwt)[0].tolist()
            peaks_welch = [fnorm[peaks_welch[i]] for i in range(len(peaks_welch))]
            peaks_cwt = fnorm[peaks_cwt]
            peak_all = [i for s in [peaks_welch, peaks_cwt] for i in s]
            npp_all = [i for s in [npp_welch, npp_cwt] for i in s]
            return peak_all, npp_all

        elif len(peaks_welch) > 0:
            npp_welch = peak_prominences(Pnorm, peaks_welch)[0].tolist()
            peaks_wel = [fnorm[peaks_welch[i]] for i in range(len(peaks_welch))]
            return peaks_wel, npp_welch
        else:
            print('No dominant peaks found!')
            return [], []

    def _peak_suggestion(peaks, npps, threshold):
        npp = []
        # add neighboring peak prominences for each detected peak
        for pkf in range(len(peaks)):
            npp_add = npps[pkf]
            peak_init = peaks[pkf]
            for neighbor in range(len(peaks)):
                if (peak_init - peaks[neighbor] >= threshold) or (peak_init - peaks[neighbor] >= -threshold):
                    npp_add += npps[neighbor]
            npp.append(npp_add)

        # save all peak and npp pairs in dict, sort in descending order
        to_dict = dict(zip(peaks, npp))
        return sorted(to_dict.items(), key=lambda x: x[1], reverse=True)[:1]

    fmin = 3
    fmax = 16

    # estimate power spectrum using Welch's method
    f, Pxx_o1 = WelchEstimate(x_o1_closed, c.FS, c.NOVERLAP, c.NFFT, c.WINDOW, fmin, fmax)
    f, Pxx_o2 = WelchEstimate(x_o2_closed, c.FS, c.NOVERLAP, c.NFFT, c.WINDOW, fmin, fmax)
    Pxx_o12 = np.array([Pxx_o1, Pxx_o2]).sum(axis=0) * 0.5

    # create frequency bound for power spectrum
    f_lim = f[(f >= fmin) & (f <= fmax)]            # [fmin,fmax] = [3,16] Hz
    start = f.tolist().index(f_lim[0])              # get first freq index
    end = f.tolist().index(f_lim[-1]) + 1           # get last freq index
    fnorm = f[start:end]                            # bound spectrum

    # normalise power spectrum
    Pnorm_o1 = normalise(Pxx_o1)
    Pnorm_o2 = normalise(Pxx_o2)
    Pnorm_o12 = normalise(Pxx_o12)

    # find alpha peak frequencies
    peaks_o1, npps_o1 = _find_peaks(fnorm, Pnorm_o1)
    peaks_o2, npps_o2 = _find_peaks(fnorm, Pnorm_o2)
    Peaks = [i for s in [peaks_o1, peaks_o2] for i in s]
    Npps = [i for s in [npps_o1, npps_o2] for i in s]
    assert len(Peaks) == len(Npps)

    if len(Peaks) and len(Npps) > 0:

        # apply peak function and retrieve suggested dominant peak frequency
        f_loc = _peak_suggestion(Peaks, Npps, threshold)

        # define narrow frequency bound for the most dominant peak detected
        f_upper_bound = f_loc[0][0] + 1.5
        f_lower_bound = f_loc[0][0] - 1.5
        f_lim = f[(f >= f_lower_bound) & (f <= f_upper_bound)]
        start = f.tolist().index(f_lim[0])
        end = f.tolist().index(f_lim[-1]) + 1
        fnorm = f[start:end]

        # find peaks in the suggested narrow frequency bound
        peak, _ = find_peaks(fnorm, Pnorm_o12)
        npp = peak_prominences(Pnorm_o12, peak)[0]
        map_hz = npp.tolist().index(np.max(npp))
        Qpeak = fnorm[map_hz]
        return Qpeak

    else:
        # if no peaks were found, skip peak suggestion and find peaks
        peak, _ = find_peaks(fnorm, Pnorm_o12)
        npp = peak_prominences(Pnorm_o12, peak)[0]
        map_hz = npp.tolist().index(np.max(npp))
        Qpeak = fnorm[peak[map_hz]]

    if Qpeak:
        print('Alpha peak frequency: ', round(Qpeak, 2), ' Hz')
        return float(Qpeak)
    else:
        print('No dominant peak detected!')
        return 0.
