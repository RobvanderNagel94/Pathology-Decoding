import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks, peak_prominences

def fitCurve(P, f_bound=[2,18]):
    
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
    P : numpy array
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
    
    assert type(P) == np.ndarray, "Exception: Returned Type Mismatch"
    assert np.min(P) != 0, "Exception: Cannot normalise series: division by zero"
    
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
    y_train = np.log(P[f_bound[0]:f_bound[1]])
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
