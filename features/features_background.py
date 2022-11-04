import numpy as np
from scipy.signal import coherence, welch
from scipy.optimize import least_squares
from scipy.signal import find_peaks, peak_prominences
import config as c

def WelchEstimate(x, fs, NOVERLAP, NFFT, WINDOW, fmin, fmax):
    """Estimates the power spectral density using Welch's method.
       Welch's method computes an estimate of the power spectral density by dividing the data into overlapping segments,
       computing a modified periodogram for each segment and averaging the periodograms.

    Parameters
    ----------
    x : numpy array
         1D array of the channel values.
         Input array requires either common reference or laplacian montage.

    fs : int
         Sampling frequency.
    NOVERLAP : int
         Number of points to overlap between segments. If None, noverlap = nperseg // 2.
    NFFT : int
         Length of the FFT used, if a zero padded FFT is desired.
    WINDOW : int
        Length of each segment.
    fmin : int
        Minimum frequency bound for the power spectrum.
    fmax : int
        Maximum frequency bound for the power spectrum.

    Returns
    -------
    f : numpy array
        Array containing the frequency values of the power spectrum.
    Pxx : numpy array
        Array containing Welch's coefficients bounded to fmin and fmax.

    References
    ----------
    .. https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html
    """

    # Compute Welch's discrete fourier coefficients
    f, Pxx = welch(x,
                   fs=fs,
                   window='hann',
                   noverlap=NOVERLAP,
                   nfft=NFFT,
                   nperseg=WINDOW)

    # Bound the frequency spectrum
    f_lim = f[(f >= fmin) & (f <= fmax)]
    start = f.tolist().index(f_lim[0])  # get first freq index
    end = f.tolist().index(f_lim[-1]) + 1  # get last freq index
    Pxx = Pxx[start:end]  # bound between [fmin,fmax]
    return f, Pxx


def fitCurve(Pxx, f_bound=[2,18]):
    
    """Find the dominant peak frequencies in the estimated power spectrum.
       The function "scipy.signals.find_peaks" was used to guess an initial dominant peak location 
       before estimating the dominant frequency. We proposed an iterative curve-fitting method to localized segments
       of EEG to approximate the two most dominant peak locations, including their amplitudes and widths. 
       Given the estimated power spectrum for a given segment, we fitted the following curve:
      
       Plog(f) ≈ Pcurve(f) = Ppk1(f) + Ppk2(f) + Pbg(f)
       Ppk1(f) = A1 · exp((f − f1)^2 / Δ1^2)
       Ppk2(f) = A2 · exp((f − f2)^2 / Δ2^2)
       Pbg(f) = B − C · log(f)
    Parameters
    ----------
    Pxx : numpy array
         1D array of the power estimates for a given segment.
    f_bound : numpy array or list
         A given frequency bound to search for peaks.
    
    Returns
    -------
    Pcurve : numpy array
             Optimized spectral curve.
             
    Paramaters : list
              Parameters A1 and A2 are the amplitudes, f1 and f2 the center
              frequencies and Δ1 and Δ2 the widths. C is a power-law approximation
              and B a normalization factor.
    
    References
    ----------
    ..https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
    ..https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
    """
    
    assert type(Pxx) == np.ndarray, "Exception: Returned Type Mismatch"
    assert np.min(Pxx) != 0, "Exception: Division by zero"
    
    def _iter1_approx(x, t, y):
        bg = x[0] - x[1]*np.log10(t)
        return y - bg
    def _iter2_approx(x, t, y):
        bg = x[0]- C *np.log10(t)
        pk1 = x[1]*np.exp(-np.power(t-x[2],2)/np.power(x[3],2))
        return y - pk1 - bg
    def _iter3_approx(x,t,y):
        bg = x[0]- C *np.log10(t)
        pk1 = A1*np.exp(-np.power(t-f1,2)/np.power(d1,2))
        pk2 = x[1]*np.exp(-np.power(t-x[2],2)/np.power(x[3],2))
        return y - pk1 - pk2 - bg
    def _spectral_curve(B, C, A1, A2, f1, f2, d1, d2, t):
        bg = B-C*np.log10(t)
        pk1 = A1*np.exp(-np.power(t-f1,2)/np.power(d1,2))
        pk2 = A2*np.exp(-np.power(t-f2,2)/np.power(d2,2))
        return pk1 + pk2 + bg

    # define the log spectrum in the desired frequency band
    y_train = np.log(Pxx[f_bound[0]:f_bound[1]])
    t_train = np.arange(1,len(y_train)+1)

    # approximate the highest peak in the log spectrum
    peak, _ = find_peaks(y_train)
    prominence = peak_prominences(y_train, peak)[0]
    index = prominence.tolist().index(np.max(prominence))
    f1 = peak[index]
    A1 = y_train[peak[index]]

    # Step 1: init params B and C, optimize for {B,C} = argmin |Plog − Pbg|
    B,C = 0, 0
    x1 = np.array([B, C]) 
    res_lsq1 = least_squares(_iter1_approx, x1, args=(t_train, y_train), method='lm')
    # retrieve the optimized params, keep C fixed
    B,_ = res_lsq1.x

    # Step 2: init params B, A1, f1 and d1, optimize for {A1,f1,Δ1,B} = argmin |Plog − Ppk1 − Pbg|
    x2 = np.array([B, A1, f1, 1])
    res_lsq2 = least_squares(_iter2_approx, x2, args=(t_train, y_train), method='lm')
    B, A1, f1, d1 = res_lsq2.x

    # Step 3: init params B, A2, f2 and d2, optimize for {A2,f2,Δ2,B} = argmin |Plog − Ppk1 − Ppk2 − Pbg|
    x3 = np.array([B, 11, 12, 1])
    res_lsq3 = least_squares(_iter3_approx, x3, args=(t_train, y_train), method='lm')
    B, A2, f2, d2 = res_lsq3.x

    # Step 4: derive the spectral curve from the estimated parameters
    Pcurve = _spectral_curve(B, C, A1, A2, f1, f2, d1, d2, t_train)

    # adjust the frequency parameters for f_bound
    f1, f2 = f1+f_bound[0], f2+f_bound[0]

    return np.array(Pcurve), [B, C, A1, A2, f1, f2, d1, d2]


def QAPG(EEG_closed, channels):
    """Computes the anterior to posterior ratio (gradient) of the Alpha power.
        Rhythmic activity from a normal wake brain should be distributed
        with an anterio-posterior gradient over the scalp: higher
        frequency beta activity with low voltages more prominently over
        the frontal regions fading posteriorly, and slower waves (e.g., alpha
        and mu rhythm) with higher voltages over the parietal and occipital lobes.

    Parameters
    ----------
    EEG_closed : numpy array
         Multidimensional array, where (rows = channels, columns = timepoints).
         Input array requires a re-referenced laplacian montage, using eyes closed states only.

    channels : numpy array
         1D array with the ordered channel names.

    Returns
    -------
    Qapg : float
         The final output is a quantified and normalised value for the anterio-posterior gradient.

        If Qapg < 0.4: Alpha power gradient is categorized as normal.
        If 0.4 < Qapg < 0.6: Alpha power gradient is considered moderately differentiated.
        If Qapg > 0.6: Alpha power gradient is considered abnormal.

    References
    ----------
    .. [1] Lodder and Van Putten (2012), Quantification of the adult EEG background pattern.
    .. DOI 10.1016/j.clinph.2012.07.007
    """

    assert len(EEG_closed) and len(channels) == 19, "Exception: Expected 19 channels"
    assert len(EEG_closed) == len(channels), "Exception: Returned Shape Mismatch"
    assert type(EEG_closed) and type(channels) == np.ndarray, "Exception: Returned Type Mismatch"
    assert EEG_closed.shape[1] >= 250 * 5, "Exception: To few datapoints recorded"

    fmin = 8
    fmax = 12

    anterior = ['fp1', 'fp2', 'f7', 'f8', 'f3', 'fz', 'f4']
    posterior = ['t5', 't6', 'p3', 'p4', 'pz', 'o1', 'o2']

    Power_anterior = []
    Power_posterior = []

    # Compute Welch's coefficients for each anterior and posterior channels
    for x in range(len(channels)):
        for i in range(len(anterior)):
            if anterior[i] in channels[x]:
                _, Pxx = WelchEstimate(EEG_closed[x], c.FS, c.NOVERLAP, c.NFFT, c.WINDOW, fmin, fmax)
                Power_anterior.append(Pxx)

        for i in range(len(posterior)):
            if posterior[i] in channels[x]:
                _, Pxx = WelchEstimate(EEG_closed[x], c.FS, c.NOVERLAP, c.NFFT, c.WINDOW, fmin, fmax)
                Power_posterior.append(Pxx)

    # Sum the total power for Pant and Ppos
    Pant = np.array(Power_anterior).sum(axis=0).sum()
    Ppos = np.array(Power_posterior).sum(axis=0).sum()

    # Compute the normalized anterio–posterior ratio
    return float(Pant / (Pant + Ppos))


def QSLOW(EEG_closed):
    """Computes diffuse slow-wave activity of the EEG.
       Diffused slowing results in an increased power over the Delta [0-4]Hz and Theta [4-8]Hz bands
       and decreased power in the Alpha [8-12]Hz and Beta bands [13-25]Hz.

       Using the guideline above, the mean spectrum of the EEG is calculated and the power ratio between
       Plow = {2. . .8} Hz and Pwide = {2. . .25} Hz

    Parameters
    ----------
    EEG_closed : numpy array
         Multidimensional array, where (rows = channels, columns = timepoints).
         Input array requires a common reference montage, using eyes closed states only.

    Returns
    -------
    Qslow : float
         The final output is a quantified and normalised value for diffused slow-wave activity.

        If Qslow > 0.6: EEG is categorized as abnormal.
        If Qslow < 0.6: EEG is categorized as normal.

    References
    ----------
    .. [1] Lodder and Van Putten (2012), Quantification of the adult EEG background pattern.
    .. DOI 10.1016/j.clinph.2012.07.007
    """

    assert len(EEG_closed) == 19, "Exception: Expected 19 channels"
    assert EEG_closed.shape[1] >= 250 * 5, "Exception: To few datapoints recorded"
    assert type(EEG_closed) == np.ndarray, "Exception: Returned Type Mismatch"

    fmin = 2
    fmax_wide = 25
    fmax_low = 8

    Power = []

    # Compute Welch's coefficients for all 19 channels
    for x in range(EEG_closed.shape[0]):
        f, Pxx = WelchEstimate(EEG_closed[x], c.FS, c.NOVERLAP, c.NFFT, c.WINDOW, fmin, fmax_wide)
        Power.append(Pxx)

    # Pwide, Plow = [2,25], [2,8] Hz
    f_wide = f[(f >= fmin) & (f <= fmax_wide)]
    f_low = f[(f >= fmin) & (f <= fmax_low)]

    # Define slice indices to bound the frequency spectrum
    start_wide = f.tolist().index(f_wide[0])
    start_low = f.tolist().index(f_low[0])
    end_wide = f.tolist().index(f_wide[-1]) + 1
    end_low = f.tolist().index(f_low[-1]) + 1

    # Sum the total power for Pwide and Plow
    Pwide = np.array(Power)[:, start_wide:end_wide].sum(axis=0).sum()
    Plow = np.array(Power)[:, start_low:end_low].sum(axis=0).sum()

    # Compute the power ratio between Pwide and Plow
    return float(Plow / Pwide)


def QASYM(EEG, channels):
    """Computes asymmetrical background patterns by comparing rhythmic activity between the two hemispheres in corresponding
       left and right channel pairs. 

    Parameters
    ----------
    EEG : numpy array
         Multidimensional array, where (rows = channels, columns = timepoints).
         Input array requires a re-referenced laplacian montage.

    channels : numpy array
         1D array with the ordered channel names.

    Returns
    -------
    Qasym : list
         The final output is a list of asymmetry values for each left and right channel pair.

         If Qasym > 0.5: Normal asymmetry.
         If Qasym < 0.5: Abnormal asymmetry.

    References
    ----------
    .. [1] Lodder and Van Putten (2012), Quantification of the adult EEG background pattern.
    .. DOI 10.1016/j.clinph.2012.07.007
    """

    assert len(EEG) and len(channels) == 19, "Exception: Expected 19 channels"
    assert len(EEG) == len(channels), "Exception: Returned Shape Mismatch"
    assert type(EEG) and type(channels) == np.ndarray, "Exception: Returned Type Mismatch"
    assert EEG.shape[1] >= 250 * 5, "Exception: To few datapoints recorded"

    fmin = 0.5
    fmax = 15

    left = ['fp1', 'f7', 'f3', 't3', 'c3', 't5', 'p3', 'o1']
    right = ['fp2', 'f8', 'f4', 't4', 'c4', 't6', 'p4', 'o2']

    pairs = [['fp1', 'fp2'], ['f7', 'f8'], ['f3', 'f4'], ['t3', 't4'],
             ['c3', 'c4'], ['t5', 't6'], ['p3', 'p4'], ['o1', 'o2']]

    Power_left = []
    Power_right = []

    # Compute Welch's coefficients for all left-right channels
    for x in range(channels.shape[0]):
        if channels[x].lower() in left:
            _, Pxx = WelchEstimate(EEG[x], c.FS, c.NOVERLAP, c.NFFT, c.WINDOW, fmin, fmax)
            Power_left.append(Pxx)

        if channels[x].lower() in right:
            _, Pxx = WelchEstimate(EEG[x], c.FS, c.NOVERLAP, c.NFFT, c.WINDOW, fmin, fmax)
            Power_right.append(Pxx)

    asymmetries = []

    # Compute asymmetries between left and right pairs
    for pair in range(Power_right.shape[0]):

        L = np.array(Power_left)[pair].sum()
        R = np.array(Power_right)[pair].sum()

        asymmetries.append((R - L) / (R + L))

    # return asymmetry values
    return asymmetries


def QREAC(x_o1_open, x_o1_closed, dominant_peak_frequency):
    """Computes reactivity for the dominant peak frequency between the eyes_closed and eyes_open states.
       Using the estimated PDR peak value, the reactivity is calculated by
       constructing a 0.5 Hz frequency band on the estimated dominant peak frequency
       when the eyes are open and when the eyes are closed. Based on these values
       a normalised value is found which quantifies the reactivity of the PDR.

    Parameters
    ----------
    x_o1_open : numpy array
         Input array requires a common reference montage, using eyes open states only.

    x_o1_closed: numpy array
         Input array requires a common reference montage, using eyes closed states only.

    dominant_peak_frequency: float
        Dominant alpha peak frequency found using find_peaks function  between 2-18Hz

    Returns
    -------
    Qreac : float
         The final output is a quantified and normalised value for alpha power reactivity.

        If Qreac > 0.5: Substantial reactivity.
        If 0.1 < Qreac < 0.5: low reactivity.
        If Qreac < 0.1: absent reactivity.

    References
    ----------
    .. [1] Lodder and Van Putten (2012), Quantification of the adult EEG background pattern.
    .. DOI 10.1016/j.clinph.2012.07.007
    """

    assert type(x_o1_open) and type(x_o1_closed) == np.ndarray, "Exception: Returned Type Mismatch"
    assert x_o1_closed.shape[0] and x_o1_open.shape[0] >= 250 * 5, "Exception: To few datapoints recorded"
    assert (dominant_peak_frequency >= 2) and (dominant_peak_frequency <= 16), "Exception: Peak frequency must be " \
                                                                               "between [2-16]Hz. "
    # Compute narrow frequency bound between the dominant peak frequency
    fmin = dominant_peak_frequency - 0.5
    fmax = dominant_peak_frequency + 0.5 

    # Compute Welch's coefficients for all eyes closed and open segments
    _, Pxx_closed = WelchEstimate(x_o1_closed, c.FS, c.NOVERLAP, c.NFFT, c.WINDOW, fmin, fmax)
    _, Pxx_open = WelchEstimate(x_o1_open, c.FS, c.NOVERLAP, c.NFFT, c.WINDOW, fmin, fmax)

    # Compute the mean power of the eyes open and eyes closed segments
    Pec = Pxx_closed.sum()
    Peo = Pxx_open.sum()

    # Compute the power ratio between eyes open and eyes closed state
    return float(1 - (Peo / Pec))


def rsBSI(EEG, channels):
    """Computes revised Brain Symmetry Index.

    Parameters
    ----------
    EEG : numpy array
         Multidimensional array, where (rows = channels, columns = timepoints).
         Input array requires a common reference montage.

    channels : numpy array
         one-dimensional array including the channel names.

    Returns
    -------
    rBSI : float
         The final output is a normalised value known as the revised Brain Symmetry Index.

    References
    ----------
    .. [1] Van Putten (2007), The revised brain symmetry index.
    .. DOI https://doi.org/10.1016/j.clinph.2007.07.019

    """

    assert len(EEG) and len(channels) == 19, "Exception: Expected 19 channels"
    assert len(EEG) == len(channels), "Exception: Returned Shape Mismatch"
    assert type(EEG) and type(channels) == np.ndarray, "Exception: Returned Type Mismatch"
    assert EEG.shape[1] >= 250 * 5, "Exception: To few datapoints recorded"

    fmin = 0.5
    fmax = 25

    left_outer = ['o1', 't5', 't3', 'f7']
    left_mid = ['o1', 'p3', 'c3', 'f3']
    right_mid = ['o2', 'p4', 'c4', 'f4']
    right_outer = ['o2', 't6', 't4', 'f8']

    P_left_outer, P_left_mid = [], []
    P_right_outer, P_right_mid = [], []

    # Compute Welch's coefficients for all left and right chains
    for x in range(EEG.shape[0]):
        if channels[x].lower() in left_outer:
            _, Pxx = WelchEstimate(EEG[x], c.FS, c.NOVERLAP, c.NFFT, c.WINDOW, fmin, fmax)
            P_left_outer.append(Pxx)

        if channels[x].lower() in left_mid:
            _, Pxx = WelchEstimate(EEG[x], c.FS, c.NOVERLAP, c.NFFT, c.WINDOW, fmin, fmax)
            P_left_mid.append(Pxx)

        if channels[x].lower() in right_mid:
            _, Pxx = WelchEstimate(EEG[x], c.FS, c.NOVERLAP, c.NFFT, c.WINDOW, fmin, fmax)
            P_right_mid.append(Pxx)

        if channels[x].lower() in right_outer:
            _, Pxx = WelchEstimate(EEG[x], c.FS, c.NOVERLAP, c.NFFT, c.WINDOW, fmin, fmax)
            P_right_outer.append(Pxx)

    # Compute the squared values of the estimated Fourier coefficients for left and right channels
    Ln = (np.square(P_left_outer).sum() + np.square(P_left_mid).sum()) * 0.5
    Rn = (np.square(P_right_outer).sum() + np.square(P_right_mid).sum()) * 0.5

    # Compute the ratio between the power for left and right channels
    return float((Rn - Ln) / (Rn + Ln))


def center_of_gravity(EEG_segment):
    """Computes the left-right and anterior-posterior power distributions measured along the scalp using
       van Putten's center-of-gravity feature.

       Parameters
       ----------
       EEG_segment : numpy array
            Multidimensional array, where (rows = channels, columns = timepoints).
            Input array requires a re-referenced laplacian montage.

       Returns
       -------
       X_cog : list Outputs the power distributions in the left-right direction by weighting the power
       with the channel positions.

       Y_cog : list Outputs the power distributions in the anterior-posterior direction by weighting the power with the
       channel positions.

       References
       ----------
       .. [1] Van Putten (2007), The colorful brain: compact visualisation of routine EEG recordings.
       .. 10.1007/978-3-540-73044-6_127
    """

    assert len(EEG_segment) == 19, "Exception: Expected 19 channels"
    assert type(EEG_segment) == np.ndarray, "Exception: Returned Type Mismatch"
    assert EEG_segment.shape[1] >= 250 * 5, "Exception: To few datapoints recorded"

    # compute Welch's discrete power coefficients
    power = []
    for i in range(EEG_segment.shape[0]):
        f, Pxx = welch(EEG_segment[i], fs=c.FS, window='hann', noverlap=c.NOVERLAP, nfft=c.NFFT, nperseg=c.WINDOW)
        power.append(Pxx)

    # output from source: X=[Fp2 F8 T4 T6 O2 F4 C4 P4 Fz Cz Pz F3 C3 P3 Fp1 F7 T3 T5 O1]
    # compute the euclidean distances for each channel location from the center location (i.e., channel Cz)
    X = np.array([0, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1, 0, -2, -2, -2, -1])  # x-direction (LR)
    Y = np.array([0, 1, 0, -1, -2, 1, 0, -1, 1, 0, -1, 1, 0, -1, 0, 1, 0, -1, -2])  # y-direction (AP)
    P = np.array(power).T

    # compute the normalised power distribution in the X and Y directions
    X_cog = np.array((P @ X) / P.sum(axis=1))
    Y_cog = np.array((P @ Y) / P.sum(axis=1))

    return X_cog, Y_cog


def mNNC(EEG_segment):
    """Computes signal coherence's for each neighboring channel (mNNC = maximum nearest neighbor coherence).

    Parameters
    ----------
    EEG_segment : numpy array
         Multidimensional array, where (rows = channels, columns = timepoints).
         Input array requires a re-referenced laplacian montage.

    Returns
    -------
    COH_max : List
         The final outputs are the maximum coherence values from all 19 channels.

    References
    ----------
    .. [1] Van Putten (2007), The colorful brain: compact visualisation of routine EEG recordings.
    .. 10.1007/978-3-540-73044-6_127
    """

    assert len(EEG_segment) == 19, "Exception: Expected 19 channels"
    assert type(EEG_segment) == np.ndarray, "Exception: Returned Type Mismatch"
    assert EEG_segment.shape[1] >= 250 * 5, "Exception: To few datapoints recorded"

    EEG = EEG_segment

    fp2, f8, t4, t6, o2, f4, c4, p4, fz, cz = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    pz, f3, c3, p3, fp1, f7, t3, t5, o1 = 10, 11, 12, 13, 14, 15, 16, 17, 18

    f, FP1a = coherence(EEG[fp1], EEG[fz], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, FP1b = coherence(EEG[fp1], EEG[f3], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, FP1c = coherence(EEG[fp1], EEG[f7], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    cFp1 = np.mean([FP1a, FP1b, FP1c])

    _, FP2a = coherence(EEG[fp2], EEG[fz], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, FP2b = coherence(EEG[fp2], EEG[f4], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, FP2c = coherence(EEG[fp2], EEG[f8], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    cFp2 = np.mean([FP2a, FP2b, FP2c])

    _, F4a = coherence(EEG[f4], EEG[fz], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, F4b = coherence(EEG[f4], EEG[f8], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, F4c = coherence(EEG[f4], EEG[c4], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    cF4 = np.mean([F4a, F4b, F4c])

    _, F3a = coherence(EEG[f3], EEG[fz], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, F3b = coherence(EEG[f3], EEG[f7], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, F3c = coherence(EEG[f3], EEG[c3], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    cF3 = np.mean([F3a, F3b, F3c])

    _, Fza = coherence(EEG[fz], EEG[f3], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, Fzb = coherence(EEG[fz], EEG[f4], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, Fzc = coherence(EEG[fz], EEG[cz], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    cFz = np.mean([Fza, Fzb, Fzc])

    _, Cza = coherence(EEG[cz], EEG[fz], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, Czb = coherence(EEG[cz], EEG[c4], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, Czc = coherence(EEG[cz], EEG[c3], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, Czd = coherence(EEG[cz], EEG[pz], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    cCz = np.mean([Cza, Czb, Czc, Czd])

    _, Pza = coherence(EEG[pz], EEG[p4], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, Pzb = coherence(EEG[pz], EEG[p3], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, Pzc = coherence(EEG[pz], EEG[o2], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, Pzd = coherence(EEG[pz], EEG[o1], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    cPz = np.mean([Pza, Pzb, Pzc, Pzd])

    _, C4a = coherence(EEG[c4], EEG[cz], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, C4b = coherence(EEG[c4], EEG[f4], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, C4c = coherence(EEG[c4], EEG[p4], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, C4d = coherence(EEG[c4], EEG[t4], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    cC4 = np.mean([C4a, C4b, C4c, C4d])

    _, C3a = coherence(EEG[c3], EEG[cz], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, C3b = coherence(EEG[c3], EEG[f3], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, C3c = coherence(EEG[c3], EEG[p3], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, C3d = coherence(EEG[c3], EEG[t3], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    cC3 = np.mean([C3a, C3b, C3c, C3d])

    _, P4a = coherence(EEG[p4], EEG[c4], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, P4b = coherence(EEG[p4], EEG[t6], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, P4c = coherence(EEG[p4], EEG[pz], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, P4d = coherence(EEG[p4], EEG[o2], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    cP4 = np.mean([P4a, P4b, P4c, P4d])

    _, P3a = coherence(EEG[p3], EEG[c3], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, P3b = coherence(EEG[p3], EEG[t5], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, P3c = coherence(EEG[p3], EEG[pz], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, P3d = coherence(EEG[p3], EEG[o1], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    cP3 = np.mean([P3a, P3b, P3c, P3d])

    _, O2a = coherence(EEG[o2], EEG[pz], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, O2b = coherence(EEG[o2], EEG[p4], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, O2c = coherence(EEG[o2], EEG[t6], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    cO2 = np.mean([O2a, O2b, O2c])

    _, O1a = coherence(EEG[o1], EEG[pz], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, O1b = coherence(EEG[o1], EEG[p3], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, O1c = coherence(EEG[o1], EEG[t5], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    cO1 = np.mean([O1a, O1b, O1c])

    _, F8a = coherence(EEG[f8], EEG[t4], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, F8b = coherence(EEG[f8], EEG[c4], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, F8c = coherence(EEG[f8], EEG[f4], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    cF8 = np.mean([F8a, F8b, F8c])

    _, F7a = coherence(EEG[f7], EEG[t3], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, F7b = coherence(EEG[f7], EEG[c3], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, F7c = coherence(EEG[f7], EEG[f3], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    cF7 = np.mean([F7a, F7b, F7c])

    _, T4a = coherence(EEG[t4], EEG[f8], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, T4b = coherence(EEG[t4], EEG[c4], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, T4c = coherence(EEG[t4], EEG[t6], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    cT4 = np.mean([T4a, T4b, T4c])

    _, T3a = coherence(EEG[t3], EEG[f7], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, T3b = coherence(EEG[t3], EEG[c3], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, T3c = coherence(EEG[t3], EEG[t5], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    cT3 = np.mean([T3a, T3b, T3c])

    _, T6a = coherence(EEG[t6], EEG[t4], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, T6b = coherence(EEG[t6], EEG[p4], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, T6c = coherence(EEG[t6], EEG[o2], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    cT6 = np.mean([T6a, T6b, T6c])

    _, T5a = coherence(EEG[t5], EEG[t3], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, T5b = coherence(EEG[t5], EEG[p3], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    _, T5c = coherence(EEG[t5], EEG[o1], c.FS, 'hann', c.WINDOW, c.NOVERLAP, c.NFFT)
    cT5 = np.mean([T5a, T5b, T5c])

    # coherence values from all 19 channels at each frequency
    AllCOH = [cFp2, cF8, cT4, cT6, cO2, cF4, cC4, cP4, cFz, cCz, cPz, cF3, cC3, cP3, cFp1, cF7, cT3, cT5, cO1]

    return AllCOH 
