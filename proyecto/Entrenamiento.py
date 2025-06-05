import os
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import butter, lfilter
from vector_referencias import save_reference_vectors

def bandpass_filter(signal, fs, lowcut, highcut, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, signal)

def calculate_band_energies(signal, fs, bw_start, bw_end, num_bands):
    N = len(signal)
    freqs = np.fft.fftfreq(N, 1 / fs)[:N // 2]
    spectrum = np.abs(np.fft.fft(signal))[:N // 2]

    band_width = (bw_end - bw_start) / num_bands
    energies = []

    for i in range(num_bands):
        band_start = bw_start + i * band_width
        band_end = band_start + band_width

        band_mask = (freqs >= band_start) & (freqs < band_end)
        energy = np.sum(spectrum[band_mask]**2)
        energies.append(energy)

    return energies

def generate_reference_vectors(input_folder="filtered_recordings", bw_start=300, bw_end=3400, num_bands=4):

    reference_vectors = {}
    for root, _, files in os.walk(input_folder):
        command_name = os.path.basename(root)
        if command_name not in reference_vectors:
            reference_vectors[command_name] = []

        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                try:
                    fs, signal = wav.read(file_path)
                    if len(signal.shape) == 2:
                        signal = np.mean(signal, axis=1)
                    signal = signal / np.max(np.abs(signal))
                    signal = bandpass_filter(signal, fs, bw_start, bw_end)
                    band_energies = calculate_band_energies(signal, fs, bw_start, bw_end, num_bands)
                    reference_vectors[command_name].append(band_energies)

                except Exception as e:
                    print(f"Error procesando {file_path}: {e}")
    for command in list(reference_vectors.keys()):
        if len(reference_vectors[command]) == 0:
            print(f"Advertencia: No se encontraron archivos en la carpeta {command}. Se eliminará del conjunto.")
            del reference_vectors[command]
        else:
            reference_vectors[command] = np.mean(reference_vectors[command], axis=0).tolist()

    return reference_vectors

def save_reference_vectors_to_json(reference_vectors, output_file="reference_vectors.json"):
    import json
    with open(output_file, 'w') as f:
        json.dump(reference_vectors, f, indent=4)

if __name__ == "__main__":
    input_folder = "command_recordings/filtered_recordings"
    bw_start = 300
    bw_end = 3400
    num_bands = 4
    reference_vectors = generate_reference_vectors(input_folder, bw_start, bw_end, num_bands)
    save_reference_vectors_to_json(reference_vectors)
    print("Vectores de referencia generados y guardados exitosamente.")
