import os
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import butter, lfilter
from vector_referencias import save_reference_vectors

# Función para aplicar un filtro pasa banda
def bandpass_filter(signal, fs, lowcut, highcut, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, signal)

# Función para calcular energía o potencia en subbandas
def calculate_band_energies(signal, fs, bw_start, bw_end, num_bands):
    N = len(signal)
    freqs = np.fft.fftfreq(N, 1 / fs)[:N // 2]  # Frecuencias positivas
    spectrum = np.abs(np.fft.fft(signal))[:N // 2]  # Espectro de magnitud (positivas)

    band_width = (bw_end - bw_start) / num_bands
    energies = []

    for i in range(num_bands):
        band_start = bw_start + i * band_width
        band_end = band_start + band_width

        # Máscara para la banda actual
        band_mask = (freqs >= band_start) & (freqs < band_end)
        energy = np.sum(spectrum[band_mask]**2)  # Energía en la banda
        energies.append(energy)

    return energies

# Generar vectores de referencia
def generate_reference_vectors(input_folder="filtered_recordings", bw_start=300, bw_end=3400, num_bands=4):
    """
    Genera vectores de referencia a partir de los archivos en las carpetas de comandos.
    - input_folder: carpeta base que contiene subcarpetas por comando.
    - bw_start, bw_end: rango de frecuencias para dividir en subbandas.
    - num_bands: número de subbandas para mayor precisión.
    """
    reference_vectors = {}

    # Recorrer subcarpetas y procesar archivos .wav
    for root, _, files in os.walk(input_folder):
        command_name = os.path.basename(root)
        if command_name not in reference_vectors:
            reference_vectors[command_name] = []

        for file in files:
            if file.endswith(".wav"):
                # Leer archivo
                file_path = os.path.join(root, file)
                try:
                    fs, signal = wav.read(file_path)

                    # Convertir a mono si es estéreo
                    if len(signal.shape) == 2:
                        signal = np.mean(signal, axis=1)

                    # Normalizar la señal
                    signal = signal / np.max(np.abs(signal))

                    # Aplicar filtro pasa banda
                    signal = bandpass_filter(signal, fs, bw_start, bw_end)

                    # Calcular energías por banda
                    band_energies = calculate_band_energies(signal, fs, bw_start, bw_end, num_bands)
                    reference_vectors[command_name].append(band_energies)

                except Exception as e:
                    print(f"Error procesando {file_path}: {e}")

    # Calcular el promedio de energías por comando
    for command in list(reference_vectors.keys()):
        if len(reference_vectors[command]) == 0:  # Si no hay archivos en la carpeta
            print(f"Advertencia: No se encontraron archivos en la carpeta {command}. Se eliminará del conjunto.")
            del reference_vectors[command]
        else:
            reference_vectors[command] = np.mean(reference_vectors[command], axis=0).tolist()  # Convertir a lista

    return reference_vectors

# Guardar vectores de referencia en JSON
def save_reference_vectors_to_json(reference_vectors, output_file="reference_vectors.json"):
    import json
    with open(output_file, 'w') as f:
        json.dump(reference_vectors, f, indent=4)

# Entrenamiento
if __name__ == "__main__":
    input_folder = "command_recordings/filtered_recordings"  # Carpeta con los comandos
    bw_start = 300  # Frecuencia mínima
    bw_end = 3400   # Frecuencia máxima
    num_bands = 4   # Número de subbandas

    # Generar vectores de referencia
    reference_vectors = generate_reference_vectors(input_folder, bw_start, bw_end, num_bands)

    # Guardar los vectores en un archivo JSON
    save_reference_vectors_to_json(reference_vectors)
    print("Vectores de referencia generados y guardados exitosamente.")
