from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import sys

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

# Parámetros de la compresión
block_size_image = 16  # Tamaño del bloque para DCT
compression_rate = 0.2  # Mantener solo el 20% de las frecuencias más significativas


def compress_image_to_80(image_obj):
    """
    Comprimir una imagen utilizando DCT manteniendo el 80% de compresión.
    :param image_obj: Objeto PIL.Image.Image de entrada.
    :return: Imagen comprimida en formato PIL.Image.Image.
    """
    if not isinstance(image_obj, Image.Image):
        raise TypeError("El parámetro debe ser un objeto de tipo PIL.Image.Image")
    
    try:
        import cv2
        # Convertir imagen de PIL a formato numpy (RGB)
        image = np.array(image_obj)
        
        def dct_compress_blocks(img, compression_rate):
            h, w = img.shape

            # Ajustar dimensiones al múltiplo más cercano del tamaño de bloque
            new_h = h + (block_size_image - h % block_size_image) if h % block_size_image != 0 else h
            new_w = w + (block_size_image - w % block_size_image) if w % block_size_image != 0 else w
            padded_img = np.zeros((new_h, new_w), dtype=np.float32)
            padded_img[:h, :w] = img

            compressed_img = np.zeros((new_h, new_w), dtype=np.float32)

            for i in range(0, new_h, block_size_image):
                for j in range(0, new_w, block_size_image):
                    block = padded_img[i:i+block_size_image, j:j+block_size_image]
                    dct_block = cv2.dct(block)  # Aplicar DCT
                    mask = np.zeros((block_size_image, block_size_image), dtype=np.float32)
                    keep_size = int(block_size_image * compression_rate)
                    mask[:keep_size, :keep_size] = 1
                    dct_block *= mask
                    compressed_img[i:i+block_size_image, j:j+block_size_image] = dct_block

            return compressed_img, (h, w)

        def dct_decompress_blocks(compressed_img, original_shape):
            new_h, new_w = compressed_img.shape
            decompressed_img = np.zeros((new_h, new_w), dtype=np.float32)

            for i in range(0, new_h, block_size_image):
                for j in range(0, new_w, block_size_image):
                    dct_block = compressed_img[i:i+block_size_image, j:j+block_size_image]
                    block = cv2.idct(dct_block)  # Aplicar IDCT
                    decompressed_img[i:i+block_size_image, j:j+block_size_image] = block

            h, w = original_shape
            return decompressed_img[:h, :w]

        def dct_compress_rgb(img, compression_rate):
            h, w, c = img.shape
            compressed_img = []
            shapes = []

            for channel in range(c):  # Procesar cada canal
                compressed_channel, shape = dct_compress_blocks(img[:, :, channel], compression_rate)
                compressed_img.append(compressed_channel)
                shapes.append(shape)

            return np.stack(compressed_img, axis=-1), shapes

        def dct_decompress_rgb(compressed_img, shapes):
            c = compressed_img.shape[-1]
            decompressed_img = []

            for channel in range(c):
                decompressed_channel = dct_decompress_blocks(compressed_img[:, :, channel], shapes[channel])
                decompressed_img.append(decompressed_channel)

            return np.stack(decompressed_img, axis=-1)

        # Convertir a flotante para la DCT
        img_array = image.astype(np.float32)

        # Comprimir y descomprimir la imagen
        compressed_image, shapes = dct_compress_rgb(img_array, compression_rate)
        decompressed_image = np.clip(dct_decompress_rgb(compressed_image, shapes), 0, 255).astype(np.uint8)

        # Convertir de nuevo a objeto PIL.Image
        return Image.fromarray(decompressed_image)
    except Exception as e:
        messagebox.showerror("Error", f"Error al comprimir la imagen: {e}")
        return image_obj  # Devolver la imagen original en caso de error


# ===== Tkinter GUI =====
def select_and_compress_image():
    # Seleccionar imagen
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if not file_path:
        return
    
    try:
        # Cargar la imagen original
        original_image = Image.open(file_path)
        
        # Comprimir la imagen
        compressed_image = compress_image_to_80(original_image)
        
        # Mostrar ambas imágenes
        display_images(original_image, compressed_image)
        
        # Guardar la imagen comprimida
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg")])
        if save_path:
            compressed_image.save(save_path, "JPEG")
            messagebox.showinfo("Éxito", "Imagen comprimida guardada correctamente.")
    except Exception as e:
        messagebox.showerror("Error", f"Error al procesar la imagen: {e}")


def display_images(original, compressed):
    # Crear una nueva ventana para mostrar las imágenes
    display_window = tk.Toplevel()
    display_window.title("Comparación de Imágenes")
    
    # Redimensionar imágenes para visualización
    max_size = (400, 400)
    original_resized = original.copy()
    original_resized.thumbnail(max_size)
    compressed_resized = compressed.copy()
    compressed_resized.thumbnail(max_size)
    
    # Convertir a formato Tkinter
    original_tk = ImageTk.PhotoImage(original_resized)
    compressed_tk = ImageTk.PhotoImage(compressed_resized)
    
    # Mostrar imágenes
    frame = tk.Frame(display_window)
    frame.pack(padx=10, pady=10)
    
    # Imagen original
    original_frame = tk.Frame(frame)
    original_frame.pack(side=tk.LEFT, padx=10)
    tk.Label(original_frame, text="Imagen Original").pack()
    tk.Label(original_frame, image=original_tk).pack()
    
    # Imagen comprimida
    compressed_frame = tk.Frame(frame)
    compressed_frame.pack(side=tk.RIGHT, padx=10)
    tk.Label(compressed_frame, text="Imagen Comprimida (80%)").pack()
    tk.Label(compressed_frame, image=compressed_tk).pack()
    
    # Mantener referencia a las imágenes
    display_window.original_tk = original_tk
    display_window.compressed_tk = compressed_tk


def main():
    # Verificar dependencias
    if not check_dependencies():
        return
        
    # Configurar ventana principal
    root = tk.Tk()
    root.title("Compresor de Imágenes - 80%")
    root.geometry("400x200")
    
    # Instrucciones
    tk.Label(root, text="Compresor de Imágenes usando DCT", font=("Arial", 14)).pack(pady=10)
    tk.Label(root, text="Seleccione una imagen para comprimir al 80%").pack(pady=5)
    
    # Botón para seleccionar imagen
    tk.Button(root, text="Seleccionar Imagen", command=select_and_compress_image, 
              font=("Arial", 12), bg="lightblue").pack(pady=20)
    
    root.mainloop()


if __name__ == "__main__":
    main()
