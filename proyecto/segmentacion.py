import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import sys
import numpy as np

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
    original_label.config(image=original_tk)
    original_label.image = original_tk
    segmented_label.config(image=segmented_tk)
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
    
    root = tk.Tk()
    root.title("Segmentación Semántica")
    root.geometry("650x400")

    # Instrucciones
    instructions = tk.Label(root, text="Seleccione una imagen para realizar segmentación semántica", font=("Arial", 12))
    instructions.pack(pady=10)

    # Botón para seleccionar la imagen
    select_button = tk.Button(root, text="Seleccionar Imagen", command=lambda: select_image(model), font=("Arial", 12), bg="lightblue")
    select_button.pack(pady=10)

    # Contenedores para mostrar las imágenes
    frame_images = tk.Frame(root)
    frame_images.pack(pady=10)

    # Imagen original
    original_frame = tk.Frame(frame_images)
    original_frame.pack(side=tk.LEFT, padx=10)
    tk.Label(original_frame, text="Imagen Original").pack()
    original_label = tk.Label(original_frame)
    original_label.pack()

    # Imagen segmentada
    segmented_frame = tk.Frame(frame_images)
    segmented_frame.pack(side=tk.RIGHT, padx=10)
    tk.Label(segmented_frame, text="Imagen Segmentada").pack()
    segmented_label = tk.Label(segmented_frame)
    segmented_label.pack()

    root.mainloop()


if __name__ == "__main__":
    main()
