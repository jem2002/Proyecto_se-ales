import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import sys
import numpy as np

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

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
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            messagebox.showerror("Error", f"No se pudo cargar la imagen: {image_path}")
            return
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        triangle_count = 0
        
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.03 * cv2.arcLength(contour, True), True)
            if len(approx) == 3:
                triangle_count += 1
                x, y, w, h = cv2.boundingRect(approx)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        result_text = f"Se han detectado {triangle_count} triángulos"
        messagebox.showinfo("Resultado", result_text)
        
        cv2.imshow('Triangulos Detectados', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    except Exception as e:
        messagebox.showerror("Error", f"Error al procesar la imagen: {e}")


def upload_image():
    file_path = filedialog.askopenfilename(initialdir="/", title="Seleccionar Imagen", 
                                            filetypes=(("Archivos de imagen", "*.jpg;*.jpeg;*.png"), ("Todos los archivos", "*.*")))
    
    if not file_path:
        return
    
    try:
        image = Image.open(file_path)
        image = image.resize((300, 300), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        image_label.configure(image=photo)
        image_label.image = photo
        
        global current_image_path
        current_image_path = file_path
        
        detect_button.configure(state="normal")
    
    except Exception as e:
        messagebox.showerror("Error", f"Error al cargar la imagen: {e}")


def main():
    if not check_dependencies():
        return
    
    global image_label, detect_button, current_image_path
    current_image_path = None
    
    root = ctk.CTk()
    root.title("Detector de Triángulos")
    root.geometry("500x800")
    root.resizable(False, False)
    
    main_frame = ctk.CTkFrame(root, corner_radius=20)
    main_frame.pack(padx=20, pady=20, fill="both", expand=True)
    
    title_label = ctk.CTkLabel(
        main_frame, 
        text="Detector de Triángulos", 
        font=ctk.CTkFont(size=24, weight="bold")
    )
    title_label.pack(pady=(20, 5))
    
    subtitle_label = ctk.CTkLabel(
        main_frame,
        text="Identifica y cuenta triángulos en imágenes",
        font=ctk.CTkFont(size=14),
        text_color=("gray70", "gray30")
    )
    subtitle_label.pack(pady=(0, 20))
    
    instructions_frame = ctk.CTkFrame(main_frame, corner_radius=10)
    instructions_frame.pack(padx=20, pady=10, fill="x")
    
    instructions = ctk.CTkLabel(
        instructions_frame, 
        text="Seleccione una imagen para detectar triángulos", 
        font=ctk.CTkFont(size=14)
    )
    instructions.pack(pady=15)
    
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
    
    image_frame = ctk.CTkFrame(main_frame, corner_radius=10)
    image_frame.pack(pady=10, fill="both", expand=True)
    
    image_label = ctk.CTkLabel(image_frame, text="")
    image_label.pack(pady=20)
    
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