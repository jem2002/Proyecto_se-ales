import os
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import struct
import math
import json
import tkinter as tk
from tkinter import messagebox
import subprocess
import sys

# Verificar dependencias
def check_dependencies():
    missing_deps = []
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
        
    try:
        import torchvision
    except ImportError:
        missing_deps.append("torchvision")
    
    if missing_deps:
        msg = "Faltan las siguientes dependencias: " + ", ".join(missing_deps)
        msg += "\n\n¿Desea instalarlas ahora?"
        if messagebox.askyesno("Dependencias faltantes", msg):
            for dep in missing_deps:
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
                    messagebox.showinfo("Instalación", f"{dep} instalado correctamente.")
                except subprocess.CalledProcessError:
                    messagebox.showerror("Error", f"No se pudo instalar {dep}. Por favor, instálelo manualmente.")
            messagebox.showinfo("Reinicio necesario", "Por favor, reinicie la aplicación para aplicar los cambios.")
            sys.exit(0)
        else:
            messagebox.showwarning("Advertencia", "La aplicación puede no funcionar correctamente sin estas dependencias.")

# Parámetros globales
fs = 44100  # Frecuencia de muestreo
bw_start = 300  # Inicio del ancho de banda
bw_end = 3400  # Fin del ancho de banda
num_bands = 4
audio_path = "recorded_audio.wav"

# Cargar vectores de referencia
def load_reference_vectors(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        messagebox.showerror("Error", f"No se encontró el archivo {file_path}. Verifique que existe.")
        return {}
    except json.JSONDecodeError:
        messagebox.showerror("Error", f"El archivo {file_path} no tiene un formato JSON válido.")
        return {}

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct path to reference vectors file
reference_vectors_path = os.path.join(current_dir, "reference_vectors.json")
vector_referencias = load_reference_vectors(reference_vectors_path)

# Grabar audio
def record_audio(filename, duration, fs):
    try:
        print("Grabando audio...")
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        wavfile.write(filename, fs, audio)
        print("Grabación completada.")
        return True
    except Exception as e:
        print(f"Error al grabar audio: {e}")
        messagebox.showerror("Error", f"Error al grabar audio: {e}")
        return False

# Calcular coeficientes del filtro
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

# Detectar comando
# Detectar comando
def find_command(audio_path, reference_vectors, fs, bw_start, bw_end, num_bands):
    try:
        # Verificar si hay vectores de referencia
        if not reference_vectors:
            return "No hay vectores de referencia cargados"
            
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
        differences = {}

        for command, reference_vector in reference_vectors.items():
            reference_vector = np.array(reference_vector) / np.sum(reference_vector)  # Normalizar referencia
            difference = np.linalg.norm(reference_vector - filtered_energies)
            differences[command] = difference
            print(f"Diferencia con '{command}': {difference}")
            if difference < min_difference:
                min_difference = difference
                detected_command = command

        # Umbrales específicos para cada comando
        command_thresholds = {
            "80": 0.8,
            "dibujo": 0.7,
            "segmentación": 0.7
        }
        
        # Verificación adicional para diferenciar entre dibujo y segmentación
        if detected_command in ["dibujo", "segmentación"]:
            # Si la diferencia entre las diferencias es pequeña, aplicar un criterio más estricto
            dibujo_diff = differences.get("dibujo", float('inf'))
            segmentacion_diff = differences.get("segmentación", float('inf'))
            diff_between = abs(dibujo_diff - segmentacion_diff)
            
            # Si las diferencias son muy cercanas (menos de 0.1 de diferencia)
            if diff_between < 0.1:
                audio_first_band = filtered_energies[0]
                dibujo_first_band = reference_vectors["dibujo"][0] / sum(reference_vectors["dibujo"])
                segmentacion_first_band = reference_vectors["segmentación"][0] / sum(reference_vectors["segmentación"])
                
                # Comparar con cuál se parece más en la primera banda
                if abs(audio_first_band - dibujo_first_band) < abs(audio_first_band - segmentacion_first_band):
                    detected_command = "dibujo"
                else:
                    detected_command = "segmentación"
                
                print(f"Comando detectado: {detected_command}")
        
        # Verificar umbral específico del comando detectado
        if detected_command in command_thresholds:
            threshold = command_thresholds[detected_command]
            if min_difference > threshold:
                return "Comando no reconocido (umbral específico)"
        else:
            # Umbral general para comandos sin umbral específico
            GENERAL_THRESHOLD = 0.9
            if min_difference > GENERAL_THRESHOLD:
                return "Comando no reconocido (umbral general)"

        return detected_command
    except Exception as e:
        print(f"Error al procesar el audio: {e}")
        return "Error al procesar el audio"

# Función para procesar el comando de voz
def process_voice_command():
    global label_status
    if not record_audio(audio_path, 2, fs):
        label_status.config(text="Error al grabar audio. Intente nuevamente.")
        return
        
    command = find_command(audio_path, vector_referencias, fs, bw_start, bw_end, num_bands)

    if command == "80":
        label_status.config(text="Comando '80' ejecutado: Programa para comprimir al 80%.")
        try:
            subprocess.run([sys.executable, os.path.join(current_dir, "comprimirImagen.py")])
        except Exception as e:
            label_status.config(text=f"Error al ejecutar comprimirImagen.py: {e}")
    elif command == "dibujo":
        label_status.config(text="Comando 'Dibujo' ejecutado: Programa para señalar dibujos.")
        try:
            subprocess.run([sys.executable, os.path.join(current_dir, "identificadorDibujo", "contar_triangulos.py")])
        except Exception as e:
            label_status.config(text=f"Error al ejecutar contar_triangulos.py: {e}")
    elif command == "segmentación":
        label_status.config(text="Comando 'Segmentación' ejecutando: Programa para segmentar imagenes.")
        try:
            subprocess.run([sys.executable, os.path.join(current_dir, "segmentacion.py")])
        except Exception as e:
            label_status.config(text=f"Error al ejecutar segmentacion.py: {e}")
    else:
        label_status.config(text=f"Comando no reconocido: {command}. Intenta nuevamente.")

# Función principal
def main():
    global label_status
    
    # Verificar dependencias
    check_dependencies()
    
    # Crear la ventana principal
    window = tk.Tk()
    window.title("Reconocimiento de Voz")
    window.geometry("600x400")

    # Mensaje de instrucciones
    instructions = """
Reconocimiento de voz:
1. Di "80" para procesar una imagen al 80%
2. Di "Dibujo" para contar la cantidad de dibujos repetidos
3. Di "Segmentación" para segmentar la imagen
"""
    label_instructions = tk.Label(window, text=instructions, justify="left", font=("Arial", 12))
    label_instructions.pack(pady=20)

    # Botón para activar los comandos
    btn_activate = tk.Button(window, text="Activar Comandos", command=process_voice_command, font=("Arial", 14), bg="lightblue")
    btn_activate.pack(pady=20)

    # Etiqueta para el estado del comando
    label_status = tk.Label(window, text="", font=("Arial", 12), fg="green")
    label_status.pack(pady=10)

    # Ejecutar la ventana principal
    window.mainloop()

if __name__ == "__main__":
    main()
