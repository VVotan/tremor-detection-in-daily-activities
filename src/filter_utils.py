import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, welch


def highpass(data, rate, highpass, order=2):
    '''
    Highpassfilter a signal.
    
    Parameters: 
    ----------------
    data: 1D array of Float
        The signal
    rate: float
        Sampling rate of the signal
    highpass: float
        Cutoff frequency of the highpass filter
    order: int
        Order of the filter. (steepness of the filter)
    
    Returns
    --------------------
    filtered: 1D array of float
        The filtered signal.
    '''
    sos = butter(order, highpass, 'highpass', fs=rate, output='sos') 
    filtered = sosfiltfilt(sos,data)
    return filtered

def lowpass(data, rate, lowpass, order=2):
    '''
    Lowpassfilter a signal.
    
    Parameters: 
    ----------------
    data: 1D array of Float
        The signal
    rate: float
        Sampling rate of the signal
    lowpass: float
        Cutoff frequency of the lowpass filter
    order: int
        Order of the filter. (steepness of the filter)
    
    Returns
    --------------------
    filtered: 1D array of float
        The filtered signal.
    '''
    sos = butter(order, lowpass, 'lowpass', fs=rate, output='sos') 
    filtered = sosfiltfilt(sos,data)
    return filtered


# things that may be useful for later:
# parameters
#nfft = 2**10 # adjustable
#t = np.arange(len(data))/rate #rate kommt von Daten (mit welcher rate sie aufgenommen wurden)
# get power spectra
#freq, power = welch(data, fs=rate, nperseg=nfft)
#decibel_power = 10*np.log10(power/pref)
