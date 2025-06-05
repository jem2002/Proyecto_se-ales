import wave
import pyaudio
import os
import time  # Para implementar la pausa

def record_audio(file_name, duration=2):
    """
    Graba un archivo de audio y lo guarda en formato WAV.
    :param file_name: Nombre del archivo WAV de salida.
    :param duration: Duración de la grabación en segundos.
    """
    chunk = 1024  # Tamaño del buffer
    format = pyaudio.paInt16  # Formato de audio
    channels = 1  # Monoaural
    rate = 44100  # Frecuencia de muestreo

    p = pyaudio.PyAudio()

    stream = p.open(format=format,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)

    print(f"Recording {file_name}...")
    frames = []

    for _ in range(0, int(rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Guardar el archivo
    with wave.open(file_name, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

    print(f"{file_name} saved!")

# Crear carpetas para cada comando
commands = ["dibujo","segmentación"]
output_dir = "command_recordings"
os.makedirs(output_dir, exist_ok=True)

for command in commands:
    command_dir = os.path.join(output_dir, command)
    os.makedirs(command_dir, exist_ok=True)

    # Grabar 10 audios para cada comando
    for i in range(31, 41):
        file_name = os.path.join(command_dir, f"{command}_{i}.wav")
        record_audio(file_name, duration=2)
        
        # Pausa de 2 segundos entre grabaciones
        if i < 10:  # No aplica pausa después de la última grabación
            print("Waiting 2 seconds before the next recording...")
            time.sleep(2)
