import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

def process_command(file_path, bw_start=300, bw_end=3400, num_bands=4):
    fs, signal = wav.read(file_path)
    if len(signal.shape) == 2:
        signal = np.mean(signal, axis=1)
    N = len(signal)
    freqs = np.fft.fftfreq(N, 1/fs)
    spectrum = np.fft.fft(signal)
    magnitude = np.abs(spectrum)
    band_width = (bw_end - bw_start) / num_bands
    energies = []
    powers = []
    for i in range(num_bands):
        band_start = bw_start + i * band_width
        band_end = band_start + band_width

        band_mask = (freqs >= band_start) & (freqs < band_end)
        filtered_spectrum = np.zeros_like(spectrum)
        filtered_spectrum[band_mask] = spectrum[band_mask]
        filtered_signal = np.fft.ifft(filtered_spectrum).real
        energy = np.sum(filtered_signal**2) / N
        power = np.sum(filtered_signal**2) / (N**2)

        energies.append(energy)
        powers.append(power)

    return energies, powers

def plot_spectrum(file_path):
    fs, signal = wav.read(file_path)
    if len(signal.shape) == 2:
        signal = np.mean(signal, axis=1)

    N = len(signal)
    freqs = np.fft.fftfreq(N, 1/fs)
    spectrum = np.fft.fft(signal)
    magnitude = np.abs(spectrum)

    plt.plot(freqs[:N//2], magnitude[:N//2])
    plt.title("Espectro de Frecuencias")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud")
    plt.show()

file_path = "command_recordings/80/80_2.wav"
energies, powers = process_command(file_path)

print("Energías por banda:", energies)
print("Potencias por banda:", powers)


