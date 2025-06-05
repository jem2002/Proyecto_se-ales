import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

def process_command(file_path, bw_start=300, bw_end=3400, num_bands=4):
    # Leer archivo de audio
    fs, signal = wav.read(file_path)

    # Si el audio tiene dos canales (estéreo), convertir a mono
    if len(signal.shape) == 2:
        signal = np.mean(signal, axis=1)

    # Aplicar FFT para obtener el espectro
    N = len(signal)
    freqs = np.fft.fftfreq(N, 1/fs)
    spectrum = np.fft.fft(signal)
    magnitude = np.abs(spectrum)

    # Dividir el ancho de banda BW en num_bands
    band_width = (bw_end - bw_start) / num_bands
    energies = []
    powers = []

    # Aplicar filtros pasa banda (en el dominio de frecuencia)
    for i in range(num_bands):
        band_start = bw_start + i * band_width
        band_end = band_start + band_width

        # Crear máscara para la banda actual
        band_mask = (freqs >= band_start) & (freqs < band_end)

        # Filtrar el espectro
        filtered_spectrum = np.zeros_like(spectrum)
        filtered_spectrum[band_mask] = spectrum[band_mask]

        # Transformar de nuevo al dominio del tiempo
        filtered_signal = np.fft.ifft(filtered_spectrum).real

        # Calcular energía y potencia
        energy = np.sum(filtered_signal**2) / N
        power = np.sum(filtered_signal**2) / (N**2)

        energies.append(energy)
        powers.append(power)

    return energies, powers

# Graficar espectro
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

# Prueba con un archivo de audio
file_path = "command_recordings/80/80_2.wav"  # Cambia por la ruta de un archivo
energies, powers = process_command(file_path)

print("Energías por banda:", energies)
print("Potencias por banda:", powers)


