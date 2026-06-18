"""FFT-based frequency analysis for IMU tremor signals."""
import numpy as np
from scipy.signal import welch

from src.visualization import VisualizationConfig, Visualizer


def start_fft_analysis(data, sampling_rate, nfft):
    signal = np.asarray(data, dtype=float).squeeze()
    if signal.ndim != 1:
        raise ValueError(f"data must be one-dimensional, got shape {signal.shape}")
    if signal.size < 2:
        raise ValueError("data must contain at least two samples")
    
    nfft = min(int(nfft), signal.size)
    fft_freqs, power = welch(signal, fs=sampling_rate, nperseg=nfft) # welch used because it is more robust

    Visualizer.plot_spectrum(
        fft_freqs,
        power, 
        #10*np.log10(power/np.max(power)),
        title="Power spectrum",
        x_limits=(0, 20),
        #y_limits=(0, 2500), #TODO
        y_label="Power spectral density",
    )

    return fft_freqs, power