"""FFT-based frequency analysis for IMU tremor signals."""
import numpy as np
import scipy

from src.visualization import VisualizationConfig, Visualizer


def start_fft_analysis(data, sampling_rate, nfft):
    fft_freqs, power = None, None





    Visualizer.plot_spectrum(
        fft_freqs,
        power,
        title="FFT-Spektrum",
        x_limits=(0, 20),
        y_limits=(0, 2500),
        y_label="Amplitute",
    )

    return fft_freqs, power