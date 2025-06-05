import os
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import struct
import math
import json
import customtkinter as ctk
from tkinter import messagebox
import subprocess
import sys

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

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
        label_status.configure(text="Error al grabar audio. Intente nuevamente.")
        return
        
    command = find_command(audio_path, vector_referencias, fs, bw_start, bw_end, num_bands)

    if command == "80":
        label_status.configure(text="Comando '80' ejecutado: Programa para comprimir al 80%.")
        try:
            subprocess.run([sys.executable, os.path.join(current_dir, "comprimirImagen.py")])
        except Exception as e:
            label_status.configure(text=f"Error al ejecutar comprimirImagen.py: {e}")
    elif command == "dibujo":
        label_status.configure(text="Comando 'Dibujo' ejecutado: Programa para señalar dibujos.")
        try:
            subprocess.run([sys.executable, os.path.join(current_dir, "identificadorDibujo", "contar_triangulos.py")])
        except Exception as e:
            label_status.configure(text=f"Error al ejecutar contar_triangulos.py: {e}")
    elif command == "segmentación":
        label_status.configure(text="Comando 'Segmentación' ejecutando: Programa para segmentar imagenes.")
        try:
            subprocess.run([sys.executable, os.path.join(current_dir, "segmentacion.py")])
        except Exception as e:
            label_status.configure(text=f"Error al ejecutar segmentacion.py: {e}")
    else:
        label_status.configure(text=f"Comando no reconocido: {command}. Intenta nuevamente.")

# Función principal
def main():
    global label_status
    
    # Verificar dependencias
    check_dependencies()
    
    # Crear la ventana principal
    window = ctk.CTk()
    window.title("🎤 Reconocimiento de Voz Inteligente")
    window.geometry("700x600")  # Aumentar el tamaño vertical de la ventana
    window.resizable(True, True)  # Permitir redimensionar la ventana
    
    # Configurar el grid
    window.grid_columnconfigure(0, weight=1)
    window.grid_rowconfigure(0, weight=1)
    
    # Frame principal con padding
    main_frame = ctk.CTkFrame(window, corner_radius=20)
    main_frame.grid(row=0, column=0, padx=30, pady=30, sticky="nsew")
    main_frame.grid_columnconfigure(0, weight=1)
    
    # Título principal
    title_label = ctk.CTkLabel(
        main_frame, 
        text="🎤 Reconocimiento de Voz",
        font=ctk.CTkFont(size=28, weight="bold")
    )
    title_label.grid(row=0, column=0, pady=(30, 20))
    
    # Subtítulo
    subtitle_label = ctk.CTkLabel(
        main_frame,
        text="Controla la aplicación con comandos de voz",
        font=ctk.CTkFont(size=16),
        text_color=("gray70", "gray30")
    )
    subtitle_label.grid(row=1, column=0, pady=(0, 30))
    
    # Frame de instrucciones
    instructions_frame = ctk.CTkFrame(main_frame, corner_radius=15)
    instructions_frame.grid(row=2, column=0, padx=20, pady=20, sticky="ew")
    
    instructions_title = ctk.CTkLabel(
        instructions_frame,
        text="📋 Comandos Disponibles",
        font=ctk.CTkFont(size=18, weight="bold")
    )
    instructions_title.pack(pady=(20, 10))
    
    # Lista de comandos con iconos
    commands = [
        ("🔢", "'80'", "Comprimir imagen al 80%"),
        ("🎨", "'Dibujo'", "Detectar y contar figuras"),
        ("✂️", "'Segmentación'", "Segmentar imágenes")
    ]
    
    for icon, command, description in commands:
        command_frame = ctk.CTkFrame(instructions_frame, fg_color="transparent")
        command_frame.pack(fill="x", padx=20, pady=5)
        
        command_text = ctk.CTkLabel(
            command_frame,
            text=f"{icon} Di {command} para {description}",
            font=ctk.CTkFont(size=14),
            anchor="w"
        )
        command_text.pack(side="left", padx=10, pady=5)
    
    # Espaciador
    ctk.CTkLabel(instructions_frame, text="").pack(pady=10)
    
    # Botón principal de activación
    btn_activate = ctk.CTkButton(
        main_frame,
        text="🎙️ GRABAR VOZ",
        command=process_voice_command,
        font=ctk.CTkFont(size=20, weight="bold"),
        height=60,
        width=300,
        corner_radius=10,
        fg_color="#E74C3C",
        hover_color="#C0392B",
        border_width=2,
        border_color="#ECF0F1"
    )
    btn_activate.grid(row=3, column=0, pady=40)
    
    # Frame de estado
    status_frame = ctk.CTkFrame(main_frame, corner_radius=15)
    status_frame.grid(row=4, column=0, padx=20, pady=(0, 20), sticky="ew")
    
    status_title = ctk.CTkLabel(
        status_frame,
        text="📊 Estado del Sistema",
        font=ctk.CTkFont(size=16, weight="bold")
    )
    status_title.pack(pady=(15, 5))
    
    # Etiqueta para el estado del comando
    label_status = ctk.CTkLabel(
        status_frame,
        text="Listo para recibir comandos de voz",
        font=ctk.CTkFont(size=14),
        text_color=("#2CC985", "#2FA572"),
        wraplength=500
    )
    label_status.pack(pady=(5, 20))
    
    # Información adicional en la parte inferior
    info_label = ctk.CTkLabel(
        main_frame,
        text="💡 Tip: Habla claro y espera a que termine la grabación",
        font=ctk.CTkFont(size=12),
        text_color=("gray60", "gray40")
    )
    info_label.grid(row=5, column=0, pady=(0, 20))
    
    # Ejecutar la ventana principal
    window.mainloop()

if __name__ == "__main__":
    main()