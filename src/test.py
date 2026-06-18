import h5py
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import welch
from pathlib import Path

# extract data from hdf5 file
# signal = h5py.File('../collected_datasets/Haltetremor_mit_frequenzaenderung_2_aligned_dataset.h5', 'r')    

base = Path(__file__).parent

path = base.parent / "collected_datasets" / "Haltetremor_mit_frequenzaenderung_2_aligned_dataset.h5"

signal = h5py.File(path, "r")

# print(list(signal.keys()))
# print(list(signal['modalities'].keys()))
# print((signal['modalities']['movella__DOT_D422CD008603']['accel_x']))
# movella__DOT_D422CD009F5B = Forearm_right
# movella__DOT_D422CD008CC7 = Hand_right

accel_x = signal['modalities']['movella__DOT_D422CD009F5B']['accel_y'][:]

# test-prints
# print(accel_x)
# print(type(accel_x))
print(accel_x.shape)

# plot data     
# plt.figure(figsize=(10, 6))
# plt.plot(accel_x)

# check if array has NaN values
if np.isnan(accel_x).any():
    # if so print warning and remove
    print("Warning: NaN values found in the array.")
    accel_x = accel_x[~np.isnan(accel_x)]
else:
    print("No NaN values found in the array.")

# sampling frequency of 60 Hz doesnt add up with global time -> ~108s instead of 216s
fs = 120 # Hz, maybe 120 Hz?
dt = 1 / fs
freqs = np.linspace(2, 6, 30)
# freqs = np.geomspace(2, 8, 30)
scales = pywt.frequency2scale('morl', freqs/120)

# compute CWT
coeffs, freqs = pywt.cwt(accel_x, scales, 'morl', sampling_period=dt, precision=12)

print(coeffs)
# plot scalogram
power = np.abs(coeffs)**2 # squared - maybe dont?
print(power)
# plt.figure(figsize=(10, 6))
# plt.imshow(
#     power,
#     extent=[0, len(accel_x)/fs, freqs.max(), freqs.min()],
#     aspect='auto'
# )

# plt.xlabel("Zeit [s]")
# plt.ylabel("Frequenz [Hz]")
# plt.colorbar(label="Amplitude")

t = np.arange(len(accel_x)) * dt

# subtract mean from signal to remove DC component
accel_x = accel_x - np.mean(accel_x)

# compute fft
fft_result = np.fft.fft(accel_x)
fft_freqs = np.fft.fftfreq(len(accel_x), d=dt)

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
freqs_psd, psd = welch(accel_x, fs=fs)

# plot PSD
# plt.figure(figsize=(10, 6))
# plt.semilogy(freqs_psd, psd)
# plt.xlabel("Frequenz (Hz)")
# plt.ylabel("Leistung")
# plt.show()

# delete all plots and put in one 2x2 subplot
plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
plt.plot(t, accel_x)
plt.title("Beschleunigungssignal (accel_x)")
plt.xlabel("Zeit [s]")
plt.ylabel("Beschleunigung [m/s²]")

plt.subplot(2, 2, 2)
plt.imshow(
    power,
    extent=[0, len(accel_x)/fs, freqs.max(), freqs.min()],
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