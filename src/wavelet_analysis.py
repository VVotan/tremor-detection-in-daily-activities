"""Wavelet-based time-frequency analysis for IMU tremor signals."""
import numpy as np
import pywt
from matplotlib import pyplot as plt

from src.visualization import Visualizer
from src.filter_utils import moving_average


def start_wavelet_analysis(data, wavelet, min_frequency=2, max_frequency=6, sampling_rate=120):
    """
    Perform CWT-based tremor analysis.

    Parameters
    ----------
    data : np.ndarray
        IMU signal
    wavelet : str
        Wavelet type
    min_frequency : float
        lower frequency bound (Hz)
    max_frequency : float
        upper frequency bound (Hz)
    sampling_rate : float
        sampling rate in Hz

    Returns
    -------
    power : np.ndarray
        CWT coefficients (freq x time) - absolute value and squared
    frequencies : np.ndarray
        corresponding frequencies in Hz
    f_mean_t : np.ndarray
        mean frequency over time
    """

    fs = sampling_rate
    dt = 1 / fs
    
    # compute scales for the desired frequency range
    freqs = np.linspace(min_frequency, max_frequency, 50)
    scales = pywt.frequency2scale(wavelet, freqs / fs)

    # compute CWT
    coefs, freqs = pywt.cwt(
        data,
        scales,
        wavelet,
        sampling_period=dt,
        precision=12
    )

    power = np.abs(coefs)**2
    # calculate mean frequency over time

    f_mean_t = (
        np.sum(freqs[:, None] * power, axis=0)
        / np.sum(power, axis=0)
    )

    # smooth via moving average ~ maybe delete later test for now -> look in literature if this is common practice
    f_mean_ma = moving_average(f_mean_t, window_size=int(sampling_rate * 3))  

    Visualizer.plot_scalogram(
        time=np.arange(len(data)) / sampling_rate,
        frequencies=freqs,
        scalogram=power,
        log_scale=False,
        #tremor_band=(1, 10),
        #maybe set limits to min and max frequency?
        frequency_limits=(min_frequency, max_frequency),
        title="CWT-Scalogram"
    )

    Visualizer.plot_time_series(
        signal=f_mean_ma,
        constant=np.mean(f_mean_ma),
        constant_label="Overall Mean Frequency [Hz]",
        sampling_rate=sampling_rate,
        title="Instantaneous Mean Frequency",
        y_label="Mean Frequency [Hz]"
    )

    return power, freqs, f_mean_t