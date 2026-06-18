"""Wavelet-based time-frequency analysis for IMU tremor signals."""
import numpy as np
import pywt
from scipy.signal import welch
from src.visualization import VisualizationConfig, Visualizer


def start_wavelet_analysis(data, min_frequency=2, max_frequency=6, sampling_rate=120):
    """
    Perform CWT-based tremor analysis.

    Parameters
    ----------
    data : np.ndarray
        IMU signal 
    min_frequency : float
        lower frequency bound (Hz)
    max_frequency : float
        upper frequency bound (Hz)
    sampling_rate : float
        sampling rate in Hz

    Returns
    -------
    coeffs : np.ndarray
        CWT coefficients (freq x time)
    frequencies : np.ndarray
        corresponding frequencies in Hz
    """

    fs = sampling_rate
    dt = 1 / fs
    
    # compute scales for the desired frequency range
    freqs = np.linspace(min_frequency, max_frequency, 30)
    scales = pywt.frequency2scale('morl', freqs / fs)

    # compute CWT
    coefs, freqs = pywt.cwt(
        data,
        scales,
        'morl',
        sampling_period=dt,
    )

    return coefs, freqs