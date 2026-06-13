import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import welch
from pathlib import Path
from src.hdf5_utils import load_acceleration_axes, load_orientation_euler

# extract data from hdf5 file
# signal = h5py.File('../collected_datasets/Haltetremor_mit_frequenzaenderung_2_aligned_dataset.h5', 'r')    

base = Path(__file__).parent

path = base.parent / "collected_datasets" / "Haltetremor_mit_frequenzaenderung_2_aligned_dataset.h5"

acceleration = load_acceleration_axes(path, "right_forearm", missing_policy="raise")
orientation = load_orientation_euler(path, "right_forearm", missing_policy="raise")

# The logical bundle exposes the three spatial axes explicitly.
accel_x = acceleration.components["accel_x"]
accel_y = acceleration.components["accel_y"]
accel_z = acceleration.components["accel_z"]
print(accel_x.shape, accel_y.shape, accel_z.shape)
print(acceleration.inspections["accel_y"])
print(f"Acceleration bundle processable: {acceleration.is_processable}")
print(f"Orientation bundle processable: {orientation.is_processable}")

# sampling frequency of 60 Hz doesnt add up with global time -> ~108s instead of 216s
fs = 60 # Hz, maybe 120 Hz?
dt = 1 / fs
scales = np.arange(1, 128)

# compute CWT
coeffs, freqs = pywt.cwt(accel_y, scales, 'morl', sampling_period=dt)

# plot scalogram
power = np.abs(coeffs)
# plt.figure(figsize=(10, 6))
# plt.imshow(
#     power,
#     extent=[0, len(accel_y)/fs, freqs.max(), freqs.min()],
#     aspect='auto'
# )

# plt.xlabel("Zeit [s]")
# plt.ylabel("Frequenz [Hz]")
# plt.colorbar(label="Amplitude")

t = np.arange(len(accel_y)) * dt

# subtract mean from signal to remove DC component
accel_y = accel_y - np.mean(accel_y)

# compute fft
fft_result = np.fft.fft(accel_y)
fft_freqs = np.fft.fftfreq(len(accel_y), d=dt)

positive = fft_freqs >= 0

fft_freqs = fft_freqs[positive]
fft_result = fft_result[positive]
amplitude = np.abs(fft_result)

# plot fft
# plt.figure(figsize=(10, 6))
# plt.plot(fft_freqs, amplitude)
# plt.xlabel("Frequenz (Hz)")
# plt.ylabel("Amplitude")

# compute PSD
freqs_psd, psd = welch(accel_y, fs=fs)

# plot PSD
# plt.figure(figsize=(10, 6))
# plt.semilogy(freqs_psd, psd)
# plt.xlabel("Frequenz (Hz)")
# plt.ylabel("Leistung")
# plt.show()

# delete all plots and put in one 2x2 subplot
plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
plt.plot(t, accel_y)
plt.title("Beschleunigungssignal (accel_y)")
plt.xlabel("Zeit [s]")
plt.ylabel("Beschleunigung [m/s²]")

plt.subplot(2, 2, 2)
plt.imshow(
    power,
    extent=[0, len(accel_y)/fs, freqs.max(), freqs.min()],
    aspect='auto'
)
plt.title("CWT-Skalogramm")
plt.xlabel("Zeit [s]")
plt.ylabel("Frequenz [Hz]")
plt.colorbar(label="Amplitude")

plt.subplot(2, 2, 3)
plt.plot(fft_freqs, amplitude)
plt.title("FFT-Spektrum")
plt.xlabel("Frequenz (Hz)")
plt.ylabel("Amplitude")

plt.subplot(2, 2, 4)
plt.semilogy(freqs_psd, psd)
plt.title("PSD")
plt.xlabel("Frequenz (Hz)")
plt.ylabel("Leistung")
plt.show()
