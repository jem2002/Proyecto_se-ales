import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import sys
import numpy as np

# Verificar dependencias
def check_dependencies():
    missing_deps = []
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")
    
    if missing_deps:
        msg = "Faltan las siguientes dependencias: " + ", ".join(missing_deps)
        msg += "\n\n¿Desea instalarlas ahora?"
        if messagebox.askyesno("Dependencias faltantes", msg):
            for dep in missing_deps:
                try:
                    import subprocess
                    subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
                    messagebox.showinfo("Instalación", f"{dep} instalado correctamente.")
                except subprocess.CalledProcessError:
                    messagebox.showerror("Error", f"No se pudo instalar {dep}. Por favor, instálelo manualmente.")
            messagebox.showinfo("Reinicio necesario", "Por favor, reinicie la aplicación para aplicar los cambios.")
            sys.exit(0)
        else:
            messagebox.showwarning("Advertencia", "La aplicación puede no funcionar correctamente sin estas dependencias.")
            return False
    return True

def detect_triangles(image_path):
    try:
        # Cargar la imagen
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            messagebox.showerror("Error", f"No se pudo cargar la imagen: {image_path}")
            return
        
        # Convertir la imagen a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detección de bordes en la imagen
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detección de contornos
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Contar triángulos
        triangle_count = 0
        
        # Dibujar los contornos de los triángulos detectados en la imagen original
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.03 * cv2.arcLength(contour, True), True)
            if len(approx) == 3:
                triangle_count += 1
                # Obtener el rectángulo delimitador del contorno
                x, y, w, h = cv2.boundingRect(approx)
                # Dibujar un rectángulo alrededor del triángulo
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Mostrar el resultado
        result_text = f"Se han detectado {triangle_count} triángulos"
        messagebox.showinfo("Resultado", result_text)
        
        # Mostrar la imagen con los triángulos resaltados
        cv2.imshow('Triangulos Detectados', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    except Exception as e:
        messagebox.showerror("Error", f"Error al procesar la imagen: {e}")


def upload_image():
    # Abrir un cuadro de diálogo para seleccionar la imagen
    file_path = filedialog.askopenfilename(initialdir="/", title="Seleccionar Imagen", 
                                          filetypes=(("Archivos de imagen", "*.jpg;*.jpeg;*.png"), ("Todos los archivos", "*.*")))
    
    if not file_path:
        return
    
    try:
        # Mostrar la imagen seleccionada en una ventana
        image = Image.open(file_path)
        image = image.resize((300, 300), Image.Resampling.LANCZOS)  # Usar LANCZOS en lugar de BILINEAR
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo
        
        # Guardar la ruta de la imagen para el botón de detección
        global current_image_path
        current_image_path = file_path
        
        # Habilitar el botón de detección
        detect_button.config(state=tk.NORMAL)
    
    except Exception as e:
        messagebox.showerror("Error", f"Error al cargar la imagen: {e}")


def main():
    # Verificar dependencias
    if not check_dependencies():
        return
    
    global image_label, detect_button, current_image_path
    current_image_path = None
    
    # Configurar ventana principal
    root = tk.Tk()
    root.title("Detector de Triángulos")
    root.geometry("400x450")
    
    # Instrucciones
    instructions = tk.Label(root, text="Seleccione una imagen para detectar triángulos", font=("Arial", 12))
    instructions.pack(pady=10)
    
    # Botón para seleccionar imagen
    select_button = tk.Button(root, text="Seleccionar Imagen", command=upload_image, 
                             font=("Arial", 12), bg="lightblue")
    select_button.pack(pady=10)
    
    # Etiqueta para mostrar la imagen
    image_label = tk.Label(root)
    image_label.pack(pady=10)
    
    # Botón para detectar triángulos
    detect_button = tk.Button(root, text="Detectar Triángulos", command=lambda: detect_triangles(current_image_path), 
                             font=("Arial", 12), bg="lightgreen", state=tk.DISABLED)
    detect_button.pack(pady=10)
    
    root.mainloop()


if __name__ == "__main__":
    main()

