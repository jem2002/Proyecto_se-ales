from vector_referencias import load_reference_vectors
import math
import numpy as np
import struct
from scipy.io import wavfile

vector_referencias = load_reference_vectors("C:\Users\joshu\OneDrive\Escritorio\proyecto (2)\proyecto\reference_vectors.json")

fs = 44100
bw_start = 300
bw_end = 3400
num_bands = 4

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

def apply_filter(frames, b, a, chunk_size):
    x1, x2 = 0, 0
    y1, y2 = 0, 0
    filtered_frames = []

    for frame in frames:
        for i in range(0, len(frame), chunk_size * 2):
            chunk = frame[i:i + chunk_size * 2]
            chunk = bytes(chunk) if isinstance(chunk, np.ndarray) else chunk
            if len(chunk) < chunk_size * 2:
                chunk += b'\x00' * (chunk_size * 2 - len(chunk))
            samples = struct.unpack(f'{len(chunk)//2}h', chunk)
            filtered_chunk = b''

            for x in samples:
                y = b[0] * x + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
                x2, x1 = x1, x
                y2, y1 = y1, y
                filtered_chunk += struct.pack('h', int(y))
            filtered_frames.append(filtered_chunk)

    return filtered_frames

def calculate_band_energies(signal, fs, bw_start, bw_end, num_bands):
    N = len(signal)
    freqs = np.fft.fftfreq(N, 1/fs)
    spectrum = np.fft.fft(signal)
    
    band_width = (bw_end - bw_start) / num_bands
    energies = []

    for i in range(num_bands):
        band_start = bw_start + i * band_width
        band_end = band_start + band_width

        band_mask = (freqs >= band_start) & (freqs < band_end)
        filtered_spectrum = np.zeros_like(spectrum)
        filtered_spectrum[band_mask] = spectrum[band_mask]
        filtered_signal = np.fft.ifft(filtered_spectrum).real
        energy = np.sum(filtered_signal**2) / N
        energies.append(energy)

    return energies

def find_command(audio_path, reference_vectors, fs, bw_start, bw_end, num_bands):
    fs_audio, audio = wavfile.read(audio_path)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1).astype(np.int16)
    
    b, a = calculate_coefficients(bw_start, bw_end, fs)
    senal_filtrada = apply_filter([audio], b, a, chunk_size=1024)
    senal_filtrada = b''.join(senal_filtrada)
    senal_filtrada = np.frombuffer(senal_filtrada, dtype=np.int16)
    audio_energies = calculate_band_energies(senal_filtrada, fs_audio, bw_start, bw_end, num_bands)
    min_difference = float('inf')
    detected_command = None
    
    for command, reference_vector in reference_vectors.items():
        difference = np.linalg.norm(np.array(reference_vector) - np.array(audio_energies))
        print(difference)
        
        if difference < min_difference:
            min_difference = difference
            detected_command = command
    
    return detected_command

audio_path = "path_to_test_file.wav"
command = find_command(audio_path, vector_referencias, fs, bw_start, bw_end, num_bands)
print(f"El comando detectado es: {command}")
