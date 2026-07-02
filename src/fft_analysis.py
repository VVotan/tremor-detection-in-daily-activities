"""FFT-based frequency analysis for IMU tremor signals."""
import numpy as np
import os
from scipy.signal import welch

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

from src.visualization import Visualizer


def start_fft_analysis(
    data,
    sampling_rate,
    nfft,
    output_dir=None,
    animation_duration_sec=8,
):
    signal = np.asarray(data, dtype=float).squeeze()

    if signal.ndim != 1:
        raise ValueError(f"data must be one-dimensional, got shape {signal.shape}")
    if signal.size < 2:
        raise ValueError("data must contain at least two samples")

    nfft = min(int(nfft), signal.size)

    fft_freqs, power = welch(signal, fs=sampling_rate, nperseg=nfft)


    Visualizer.plot_spectrum(
         fft_freqs,
        power, 
        #10*np.log10(power/np.max(power)),
        title="Power spectrum",
        x_limits=(0, 20),
        #y_limits=(0, 2500), #TODO
        y_label="Power spectral density",
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.set_title("FFT Power Spectrum")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Power Spectral Density")

    ax.set_xlim(fft_freqs[0], fft_freqs[-1])
    ax.set_ylim(0, np.max(power) * 1.1)

    ax.grid(True, alpha=0.3)

    line, = ax.plot([], [], lw=2, color="#1f77b4")

    fps = 30
    total_frames = int(animation_duration_sec * fps)

    frame_indices = np.linspace(
        0,
        len(fft_freqs) - 1,
        total_frames
    ).astype(int)

    def update(frame):
        x = fft_freqs[:frame]
        y = power[:frame]
        line.set_data(x, y)
        return line,

    ani = FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        interval=1000 / fps
    )


    if output_dir is None:
        output_dir = "results"

    animation_dir = output_dir 
    animation_dir.mkdir(parents=True, exist_ok=True)

    out_path = animation_dir / "fft_power_spectrum.mp4"

    writer = FFMpegWriter(fps=fps, bitrate=1800)
    ani.save(out_path, writer=writer)

    plt.close(fig)

    print(f"[FFT Animation saved] {out_path}")

    return fft_freqs, power