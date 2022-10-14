from scipy.signal import hilbert
import numpy as np

def instantaneous_phases(band_signals, axis):
    """Computes the instantaneous phase of a band signal.

            Parameters
            ----------
            band_signals : numpy array
                one dimensional array of signal values.

            Returns
            -------
            theta : float
                 outputs the instantaneous phase of a signal.

            References
            ----------
            .. https://en.wikipedia.org/wiki/Instantaneous_phase_and_frequency

        """

    analytical_signal = hilbert(band_signals, axis=axis)
    return np.unwrap(np.angle(analytical_signal), axis=axis)


def phase_locking_values(inst_phases):
    """Computes the Phase Locking Values (PLV's) for each channel pairs.

            Parameters
            ----------
            inst_phases : float
                 the instantaneous phase of the band signal.

            Returns
            -------
            theta : numpy array
                 outputs PLV's for each consecutive channel pair.

            References
            ----------
            .. https://arxiv.org/ftp/arxiv/papers/1710/1710.08037.pdf

    """

    (n_windows, n_bands, n_signals, n_samples) = inst_phases.shape
    PLVs = []
    for electrode_id1 in range(n_signals):
        # only compute upper triangle of the synchronicity matrix and fill
        # lower triangle with identical values
        # +1 since diagonal is always 1
        for electrode_id2 in range(electrode_id1+1, n_signals):
            for band_id in range(n_bands):
                plv = phase_locking_value2(
                    theta1=inst_phases[:, band_id, electrode_id1],
                    theta2=inst_phases[:, band_id, electrode_id2]
                )
                PLVs.append(plv)

    # n_window x n_bands * (n_signals*(n_signals-1))/2
    PLVs = np.array(PLVs).T
    return PLVs


def phase_locking_value2(theta1, theta2):
    """Intermediate step to compute the Phase Locking Values (PLV's) between two instantaneous phases.

            Parameters
            ----------
            theta1 : float
                 the instantaneous phase of signal 1.

            theta2 : float
                 the instantaneous phase of signal 2.

            Returns
            -------
            theta : numpy array
                 outputs the normalised PLV's for a channel pair.

    """
    delta = np.subtract(theta1, theta2)
    xs_mean = np.mean(np.cos(delta), axis=-1)
    ys_mean = np.mean(np.sin(delta), axis=-1)
    PLVs = np.linalg.norm([xs_mean, ys_mean], axis=0)
    return PLVs

