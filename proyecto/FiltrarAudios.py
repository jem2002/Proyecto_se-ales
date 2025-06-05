import os
import wave
import math
import struct
import numpy as np

CHUNK = 1024
RATE = 44100

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
    x1, x2 = 0, 0  # Entradas pasadas
    y1, y2 = 0, 0  # Salidas pasadas
    filtered_frames = []

    for frame in frames:
        for i in range(0, len(frame), chunk_size * 2):
            chunk = frame[i:i + chunk_size * 2]
            if len(chunk) < chunk_size * 2:
                chunk += b'\x00' * (chunk_size * 2 - len(chunk))

            samples = struct.unpack(f'{chunk_size}h', chunk)
            filtered_chunk = b''

            for x in samples:
                y = b[0] * x + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
                x2, x1 = x1, x
                y2, y1 = y1, y
                filtered_chunk += struct.pack('h', int(y))
            filtered_frames.append(filtered_chunk)

    return filtered_frames

def process_folder(input_folder):
    output_folder = os.path.join(input_folder, "filtered_recordings")

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".wav"):
                input_path = os.path.join(root, file)

                relative_path = os.path.relpath(root,input_folder)
                output_path_folder = os.path.join(output_folder, relative_path)
                os.makedirs(output_path_folder, exist_ok=True)
                output_path = os.path.join(output_path_folder, file)

                process_audio(input_path, output_path)

def process_audio(input_path, output_path):
    with wave.open(input_path, 'rb') as wf:
        params = wf.getparams()
        n_channels, sampwidth, framerate, n_frames, _, _ = params
        frames = wf.readframes(n_frames)

    b, a = calculate_coefficients(300, 3400, RATE)
    filtered_frames = apply_filter([frames], b, a, CHUNK)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.writeframes(b''.join(filtered_frames))

    print(f"Procesado y guardado: {output_path}")

input_folder = "command_recordings"
process_folder(input_folder)
