import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import sys
import numpy as np

# Configurar apariencia de CustomTkinter
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

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
        image_label.configure(image=photo)
        image_label.image = photo
        
        # Guardar la ruta de la imagen para el botón de detección
        global current_image_path
        current_image_path = file_path
        
        # Habilitar el botón de detección
        detect_button.configure(state="normal")
    
    except Exception as e:
        messagebox.showerror("Error", f"Error al cargar la imagen: {e}")


def main():
    # Verificar dependencias
    if not check_dependencies():
        return
    
    global image_label, detect_button, current_image_path
    current_image_path = None
    
    # Configurar ventana principal
    root = ctk.CTk()
    root.title("Detector de Triángulos")
    root.geometry("500x600")
    root.resizable(False, False)
    
    # Frame principal con padding
    main_frame = ctk.CTkFrame(root, corner_radius=20)
    main_frame.pack(padx=20, pady=20, fill="both", expand=True)
    
    # Título principal
    title_label = ctk.CTkLabel(
        main_frame, 
        text="Detector de Triángulos", 
        font=ctk.CTkFont(size=24, weight="bold")
    )
    title_label.pack(pady=(20, 5))
    
    # Subtítulo
    subtitle_label = ctk.CTkLabel(
        main_frame,
        text="Identifica y cuenta triángulos en imágenes",
        font=ctk.CTkFont(size=14),
        text_color=("gray70", "gray30")
    )
    subtitle_label.pack(pady=(0, 20))
    
    # Instrucciones
    instructions_frame = ctk.CTkFrame(main_frame, corner_radius=10)
    instructions_frame.pack(padx=20, pady=10, fill="x")
    
    instructions = ctk.CTkLabel(
        instructions_frame, 
        text="Seleccione una imagen para detectar triángulos", 
        font=ctk.CTkFont(size=14)
    )
    instructions.pack(pady=15)
    
    # Botón para seleccionar imagen
    select_button = ctk.CTkButton(
        main_frame, 
        text="Seleccionar Imagen", 
        command=upload_image, 
        font=ctk.CTkFont(size=16, weight="bold"),
        height=40,
        corner_radius=10,
        fg_color="#3498DB",
        hover_color="#2980B9"
    )
    select_button.pack(pady=20)
    
    # Frame para la imagen
    image_frame = ctk.CTkFrame(main_frame, corner_radius=10)
    image_frame.pack(pady=10, fill="both", expand=True)
    
    # Etiqueta para mostrar la imagen
    image_label = ctk.CTkLabel(image_frame, text="")
    image_label.pack(pady=20)
    
    # Botón para detectar triángulos
    detect_button = ctk.CTkButton(
        main_frame, 
        text="Detectar Triángulos", 
        command=lambda: detect_triangles(current_image_path), 
        font=ctk.CTkFont(size=16, weight="bold"),
        height=40,
        corner_radius=10,
        fg_color="#2ECC71",
        hover_color="#27AE60",
        state="disabled"
    )
    detect_button.pack(pady=20)
    
    # Información adicional
    info_label = ctk.CTkLabel(
        main_frame,
        text="💡 Para mejores resultados, use imágenes con bordes bien definidos",
        font=ctk.CTkFont(size=12),
        text_color=("gray60", "gray40")
    )
    info_label.pack(pady=(0, 10))
    
    root.mainloop()


if __name__ == "__main__":
    main()

