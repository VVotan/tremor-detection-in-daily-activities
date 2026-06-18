"""Wavelet-based time-frequency analysis for IMU tremor signals."""
import numpy as np
import pywt
from matplotlib import pyplot as plt

from src.visualization import Visualizer


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
    freqs = np.linspace(min_frequency, max_frequency, 128)
    scales = pywt.frequency2scale('morl', freqs / fs)

    # compute CWT
    coefs, freqs = pywt.cwt(
        data,
        scales,
        'morl',
        sampling_period=dt,
    )


    Visualizer.plot_scalogram(
        time=np.arange(len(data)) / sampling_rate,
        frequencies=freqs,
        scalogram=np.abs(coefs),
        log_scale=True,
        #tremor_band=(1, 10),
        frequency_limits=(0, 10),
        title="CWT-Scalogram"
    )

    return coefs, freqs