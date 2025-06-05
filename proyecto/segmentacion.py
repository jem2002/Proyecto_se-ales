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

# Función para cargar el modelo
def load_model():
    try:
        import torch
        from torchvision.models.segmentation import deeplabv3_resnet101
        model = deeplabv3_resnet101(pretrained=True)
        model.eval()
        return model
    except Exception as e:
        messagebox.showerror("Error", f"Error al cargar el modelo: {e}")
        return None

# Función para preprocesar la imagen
def preprocess_image(image):
    try:
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return preprocess(image).unsqueeze(0)  # Agregar dimensión de batch
    except Exception as e:
        messagebox.showerror("Error", f"Error al preprocesar la imagen: {e}")
        return None

# Paleta de colores para visualización
def decode_segmentation(mask):
    # Mapear cada clase a un color
    colors = [
        (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
        (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
        (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
        (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
        (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
        (0, 64, 128)
    ]

    # Asignar colores a las clases en la máscara
    color_mask = Image.new("RGB", (mask.shape[1], mask.shape[0]))
    pixels = color_mask.load()

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            class_idx = mask[y, x]
            if class_idx < len(colors):
                pixels[x, y] = colors[class_idx]
            else:
                pixels[x, y] = (0, 0, 0)  # Color negro para clases desconocidas

    return color_mask


# Función para segmentar la imagen
def segment_image(image, model):
    try:
        import torch
        # Verificar que el modelo está cargado
        if model is None:
            messagebox.showerror("Error", "El modelo no está cargado correctamente.")
            return None
            
        input_image = preprocess_image(image)
        if input_image is None:
            return None
            
        with torch.no_grad():
            output = model(input_image)["out"][0]  # Salida del modelo
        mask = output.argmax(0).byte().cpu().numpy()  # Clase con mayor puntuación
        return decode_segmentation(mask)
    except Exception as e:
        messagebox.showerror("Error", f"Error al segmentar la imagen: {e}")
        return None


# Función para cargar y procesar la imagen seleccionada
def select_image(model):
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return

    try:
        # Cargar la imagen
        original_image = Image.open(file_path).convert("RGB")
        segmented_image = segment_image(original_image, model)
        
        if segmented_image is not None:
            # Mostrar las imágenes en la interfaz
            display_images(original_image, segmented_image)
    except Exception as e:
        messagebox.showerror("Error", f"Error al procesar la imagen: {e}")


# Función para mostrar imágenes en la ventana
def display_images(original, segmented):
    # Redimensionar imágenes para visualización
    original_resized = original.resize((300, 300))
    segmented_resized = segmented.resize((300, 300))

    # Convertir a formato compatible con Tkinter
    original_tk = ImageTk.PhotoImage(original_resized)
    segmented_tk = ImageTk.PhotoImage(segmented_resized)

    # Actualizar las etiquetas con las imágenes
    original_label.configure(image=original_tk)
    original_label.image = original_tk
    segmented_label.configure(image=segmented_tk)
    segmented_label.image = segmented_tk


def main():
    # Verificar dependencias
    if not check_dependencies():
        return
        
    # Cargar el modelo
    model = load_model()
    if model is None:
        return
    
    # Configuración de la ventana principal
    global original_label, segmented_label
    
    root = ctk.CTk()
    root.title("Segmentación Semántica")
    root.geometry("750x650")
    root.resizable(False, False)
    
    # Frame principal con padding
    main_frame = ctk.CTkFrame(root, corner_radius=20)
    main_frame.pack(padx=20, pady=20, fill="both", expand=True)
    
    # Título principal
    title_label = ctk.CTkLabel(
        main_frame, 
        text="Segmentación Semántica", 
        font=ctk.CTkFont(size=24, weight="bold")
    )
    title_label.pack(pady=(20, 5))
    
    # Subtítulo
    subtitle_label = ctk.CTkLabel(
        main_frame,
        text="Identifica y separa objetos en la imagen",
        font=ctk.CTkFont(size=14),
        text_color=("gray70", "gray30")
    )
    subtitle_label.pack(pady=(0, 20))

    # Instrucciones
    instructions_frame = ctk.CTkFrame(main_frame, corner_radius=10)
    instructions_frame.pack(padx=20, pady=10, fill="x")
    
    instructions = ctk.CTkLabel(
        instructions_frame, 
        text="Seleccione una imagen para realizar segmentación semántica", 
        font=ctk.CTkFont(size=14)
    )
    instructions.pack(pady=15)

    # Botón para seleccionar la imagen
    select_button = ctk.CTkButton(
        main_frame, 
        text="Seleccionar Imagen", 
        command=lambda: select_image(model), 
        font=ctk.CTkFont(size=16, weight="bold"),
        height=40,
        corner_radius=10,
        fg_color="#3498DB",
        hover_color="#2980B9"
    )
    select_button.pack(pady=20)

    # Contenedores para mostrar las imágenes
    frame_images = ctk.CTkFrame(main_frame, corner_radius=15)
    frame_images.pack(pady=20, fill="both", expand=True)

    # Imagen original
    original_frame = ctk.CTkFrame(frame_images, corner_radius=10)
    original_frame.pack(side="left", padx=20, pady=20, fill="both", expand=True)
    ctk.CTkLabel(original_frame, text="Imagen Original", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
    original_label = ctk.CTkLabel(original_frame, text="")
    original_label.pack(pady=10)

    # Imagen segmentada
    segmented_frame = ctk.CTkFrame(frame_images, corner_radius=10)
    segmented_frame.pack(side="right", padx=20, pady=20, fill="both", expand=True)
    ctk.CTkLabel(segmented_frame, text="Imagen Segmentada", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
    segmented_label = ctk.CTkLabel(segmented_frame, text="")
    segmented_label.pack(pady=10)
    
    # Información adicional
    info_label = ctk.CTkLabel(
        main_frame,
        text="💡 La segmentación puede tardar unos segundos dependiendo del tamaño de la imagen",
        font=ctk.CTkFont(size=12),
        text_color=("gray60", "gray40")
    )
    info_label.pack(pady=(0, 10))

    root.mainloop()


if __name__ == "__main__":
    main()
