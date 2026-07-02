"""Wavelet-based time-frequency analysis for IMU tremor signals."""
import numpy as np
import os
import pywt

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

from src.visualization import Visualizer
from src.filter_utils import moving_average


def start_wavelet_analysis(
    data,
    wavelet,
    min_frequency=2,
    max_frequency=6,
    sampling_rate=120,
    output_dir=None,
    animation_duration_sec=8,
):
    """
    Perform CWT-based tremor analysis.
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

    power = np.abs(coefs) ** 2

    # mean frequency over time
    f_mean_t = (
        np.sum(freqs[:, None] * power, axis=0)
        / np.sum(power, axis=0)
    )


    Visualizer.plot_scalogram(
        time=np.arange(len(data)) / sampling_rate,
        frequencies=freqs,
        scalogram=power,
        log_scale=False,
        frequency_limits=(min_frequency, max_frequency),
        title="CWT-Scalogram"
    )

    Visualizer.plot_time_series(
        signal=f_mean_t,
        constant=np.mean(f_mean_t),
        constant_label="Overall Mean Frequency [Hz]",
        sampling_rate=sampling_rate,
        title="Instantaneous Mean Frequency",
        y_label="Mean Frequency [Hz]"
    )

    time = np.arange(len(data)) / sampling_rate

    fps = 30
    total_frames = int(animation_duration_sec * fps)

    animation_dir = output_dir
    if animation_dir is None:
        animation_dir = "results"
    animation_dir.mkdir(parents=True, exist_ok=True)


    fig, ax = plt.subplots(figsize=(8, 4))

    ax.set_title("Instantaneous Mean Frequency (CWT)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")

    ax.set_xlim(time[0], time[-1])
    ax.set_ylim(min_frequency, max_frequency)
    ax.grid(True, alpha=0.3)

    line, = ax.plot([], [], lw=2, color="#1f77b4")
    point, = ax.plot([], [], "o", color="#1f77b4", markersize=3)

    frame_indices = np.linspace(
        0,
        len(time) - 1,
        total_frames
    ).astype(int)

    def update_mean(frame):
        x = time[:frame]
        y = f_mean_t[:frame]

        line.set_data(x, y)

        if frame > 0:
            point.set_data([time[frame - 1]], [f_mean_t[frame - 1]])

        return line, point

    ani = FuncAnimation(
        fig,
        update_mean,
        frames=frame_indices,
        interval=1000 / fps
    )

    out_path_mean = animation_dir / "cwt_mean_frequency.mp4"

    writer = FFMpegWriter(fps=fps, bitrate=1800)
    ani.save(out_path_mean, writer=writer)

    plt.close(fig)

    print(f"[Animation saved] {out_path_mean}")


    fig2, ax2 = plt.subplots(figsize=(8, 4))
    
    ax2.set_title("CWT Scalogram (Progressive Draw)")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Frequency [Hz]")
    
    extent = [time[0], time[-1], min_frequency, max_frequency]
    
    img = ax2.imshow(
        np.zeros_like(power),
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="magma",
        vmin=0,
        vmax=np.max(power)
    )
    
    cbar = plt.colorbar(img, ax=ax2)
    cbar.set_label("Power")
    
    frame_indices2 = np.linspace(
        1,
        power.shape[1],
        total_frames
    ).astype(int)
    
    def update_scalogram(frame):
        partial = np.zeros_like(power)
        partial[:, :frame] = power[:, :frame]   
    
        img.set_data(partial)
    
        return [img]
    
    ani2 = FuncAnimation(
        fig2,
        update_scalogram,
        frames=frame_indices2,
        interval=1000 / fps,
        blit=False
    )
    
    out_path_scalogram = animation_dir / "cwt_scalogram_build.mp4"
    
    ani2.save(out_path_scalogram, writer=writer)
    
    plt.close(fig2)
    
    print(f"[Animation saved] {out_path_scalogram}")

    return power, freqs, f_mean_t, data