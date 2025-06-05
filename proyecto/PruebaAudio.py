import os
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import struct
import math
import json

# Parámetros globales
fs = 44100  # Frecuencia de muestreo
bw_start = 300  # Inicio del ancho de banda
bw_end = 3400  # Fin del ancho de banda
num_bands = 4

# Cargar vectores de referencia
def load_reference_vectors(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Cálculo de coeficientes del filtro
def calculate_coefficients(f1, f2, fs):
    f0 = math.sqrt(f1 * f2)
    bw = f2 - f1
    Q = f0 / bw
    w0 = 2 * math.pi * f0 / fs
    alpha = math.sin(w0) / (2 * Q)
    b0 = alpha
    b1 = 0
    b2 = -alpha
    a0 = 1 + alpha
    a1 = -2 * math.cos(w0)
    a2 = 1 - alpha
    b = [b0 / a0, b1 / a0, b2 / a0]
    a = [1, a1 / a0, a2 / a0]
    return b, a

# Aplicar el filtro manual
def apply_filter(signal, b, a):
    x1, x2 = 0, 0  # Entradas pasadas
    y1, y2 = 0, 0  # Salidas pasadas
    filtered_signal = []

    for x in signal:
        y = b[0] * x + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
        x2, x1 = x1, x
        y2, y1 = y1, y
        filtered_signal.append(y)

    return np.array(filtered_signal, dtype=np.float32)

# Grabar audio
def record_audio(filename, duration, fs):
    print("Grabando audio...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    wavfile.write(filename, fs, audio)
    print("Grabación completada.")

# Calcular energías en subbandas
def calculate_band_energies(signal, fs, bw_start, bw_end, num_bands):
    N = len(signal)
    freqs = np.fft.fftfreq(N, 1 / fs)
    spectrum = np.fft.fft(signal)

    band_width = (bw_end - bw_start) / num_bands
    energies = []

    for i in range(num_bands):
        band_start = bw_start + i * band_width
        band_end = band_start + band_width

        # Máscara para la banda actual
        band_mask = (freqs >= band_start) & (freqs < band_end)
        filtered_spectrum = np.zeros_like(spectrum)
        filtered_spectrum[band_mask] = spectrum[band_mask]

        # Transformar de nuevo al tiempo y calcular energía
        filtered_signal = np.fft.ifft(filtered_spectrum).real
        energy = np.sum(filtered_signal**2) / N
        energies.append(energy)

    return energies

# Detectar comando
def find_command(audio_path, reference_vectors, fs, bw_start, bw_end, num_bands):
    # Leer el archivo de audio
    fs_audio, audio = wavfile.read(audio_path)

    # Convertir a mono si es necesario
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1).astype(np.int16)

    # Normalizar la señal
    audio = audio / np.max(np.abs(audio))

    # Calcular energías de subbandas
    filtered_energies = []
    for i in range(num_bands):
        f1 = bw_start + i * (bw_end - bw_start) / num_bands
        f2 = f1 + (bw_end - bw_start) / num_bands
        b, a = calculate_coefficients(f1, f2, fs)
        filtered_audio = apply_filter(audio, b, a)
        energy = np.sum(filtered_audio**2) / len(filtered_audio)
        filtered_energies.append(energy)

    # Normalizar energías calculadas
    filtered_energies = np.array(filtered_energies) / np.sum(filtered_energies)

    # Comparar con los vectores de referencia
    min_difference = float('inf')
    detected_command = None

    for command, reference_vector in reference_vectors.items():
        reference_vector = np.array(reference_vector) / np.sum(reference_vector)  # Normalizar referencia
        difference = np.linalg.norm(reference_vector - filtered_energies)
        print(f"Diferencia con '{command}': {difference}")
        if difference < min_difference:
            min_difference = difference
            detected_command = command

    # Ajuste del umbral
    THRESHOLD = 1.0  # Ajusta este valor según las diferencias observadas
    if min_difference > THRESHOLD:
        return "Comando no reconocido"

    return detected_command


# Main
if __name__ == "__main__":
    vector_referencias = load_reference_vectors("C:/Users/nicol/Downloads/proyecto (2)/proyecto/reference_vectors.json")
    audio_path = "recorded_audio.wav"

    # Grabar audio
    record_audio(audio_path, duration=2, fs=fs)

    # Detectar comando
    command = find_command(audio_path, vector_referencias, fs, bw_start, bw_end, num_bands)
    print(f"El comando detectado es: {command}")
