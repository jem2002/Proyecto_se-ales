from PIL import Image, ImageTk
import customtkinter as ctk
from tkinter import filedialog, messagebox
import numpy as np
import sys

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

block_size_image = 16
compression_rate = 0.2


def compress_image_to_80(image_obj):
    if not isinstance(image_obj, Image.Image):
        raise TypeError("El parámetro debe ser un objeto de tipo PIL.Image.Image")
    
    try:
        import cv2
        image = np.array(image_obj)
        
        def dct_compress_blocks(img, compression_rate):
            h, w = img.shape

            new_h = h + (block_size_image - h % block_size_image) if h % block_size_image != 0 else h
            new_w = w + (block_size_image - w % block_size_image) if w % block_size_image != 0 else w
            padded_img = np.zeros((new_h, new_w), dtype=np.float32)
            padded_img[:h, :w] = img

            compressed_img = np.zeros((new_h, new_w), dtype=np.float32)

            for i in range(0, new_h, block_size_image):
                for j in range(0, new_w, block_size_image):
                    block = padded_img[i:i+block_size_image, j:j+block_size_image]
                    dct_block = cv2.dct(block)
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
                    block = cv2.idct(dct_block)
                    decompressed_img[i:i+block_size_image, j:j+block_size_image] = block

            h, w = original_shape
            return decompressed_img[:h, :w]

        def dct_compress_rgb(img, compression_rate):
            h, w, c = img.shape
            compressed_img = []
            shapes = []

            for channel in range(c):
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

        img_array = image.astype(np.float32)

        compressed_image, shapes = dct_compress_rgb(img_array, compression_rate)
        decompressed_image = np.clip(dct_decompress_rgb(compressed_image, shapes), 0, 255).astype(np.uint8)

        return Image.fromarray(decompressed_image)
    except Exception as e:
        messagebox.showerror("Error", f"Error al comprimir la imagen: {e}")
        return image_obj


def select_and_compress_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if not file_path:
        return
    
    try:
        original_image = Image.open(file_path)
        
        compressed_image = compress_image_to_80(original_image)
        
        display_images(original_image, compressed_image)
        
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg")])
        if save_path:
            compressed_image.save(save_path, "JPEG")
            messagebox.showinfo("Éxito", "Imagen comprimida guardada correctamente.")
    except Exception as e:
        messagebox.showerror("Error", f"Error al procesar la imagen: {e}")


def display_images(original, compressed):
    display_window = ctk.CTkToplevel()
    display_window.title("Comparación de Imágenes")
    display_window.geometry("900x500")
    
    max_size = (400, 400)
    original_resized = original.copy()
    original_resized.thumbnail(max_size)
    compressed_resized = compressed.copy()
    compressed_resized.thumbnail(max_size)
    
    original_tk = ImageTk.PhotoImage(original_resized)
    compressed_tk = ImageTk.PhotoImage(compressed_resized)
    
    frame = ctk.CTkFrame(display_window, corner_radius=15)
    frame.pack(padx=20, pady=20, fill="both", expand=True)
    
    original_frame = ctk.CTkFrame(frame, corner_radius=10)
    original_frame.pack(side="left", padx=20, pady=20, fill="both", expand=True)
    ctk.CTkLabel(original_frame, text="Imagen Original", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
    original_label = ctk.CTkLabel(original_frame, image=original_tk, text="")
    original_label.pack(pady=10)
    
    compressed_frame = ctk.CTkFrame(frame, corner_radius=10)
    compressed_frame.pack(side="right", padx=20, pady=20, fill="both", expand=True)
    ctk.CTkLabel(compressed_frame, text="Imagen Comprimida (80%)", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
    compressed_label = ctk.CTkLabel(compressed_frame, image=compressed_tk, text="")
    compressed_label.pack(pady=10)
    
    display_window.original_tk = original_tk
    display_window.compressed_tk = compressed_tk


def main():
    if not check_dependencies():
        return
        
    root = ctk.CTk()
    root.title("Compresor de Imágenes - 80%")
    root.geometry("500x350")
    root.resizable(False, False)
    
    main_frame = ctk.CTkFrame(root, corner_radius=20)
    main_frame.pack(padx=20, pady=20, fill="both", expand=True)
    
    ctk.CTkLabel(
        main_frame, 
        text="Compresor de Imágenes", 
        font=ctk.CTkFont(size=24, weight="bold")
    ).pack(pady=(20, 5))
    
    ctk.CTkLabel(
        main_frame,
        text="Usando Transformada Discreta del Coseno (DCT)",
        font=ctk.CTkFont(size=14),
        text_color=("gray70", "gray30")
    ).pack(pady=(0, 20))
    
    instructions_frame = ctk.CTkFrame(main_frame, corner_radius=10)
    instructions_frame.pack(padx=20, pady=10, fill="x")
    
    ctk.CTkLabel(
        instructions_frame,
        text="Seleccione una imagen para comprimir al 80%",
        font=ctk.CTkFont(size=14)
    ).pack(pady=15)
    
    ctk.CTkButton(
        main_frame, 
        text="Seleccionar Imagen", 
        command=select_and_compress_image,
        font=ctk.CTkFont(size=16, weight="bold"),
        height=50,
        corner_radius=10,
        fg_color="#3498DB",
        hover_color="#2980B9"
    ).pack(pady=30)
    
    ctk.CTkLabel(
        main_frame,
        text="💡 La imagen comprimida se guardará en la ubicación que elija",
        font=ctk.CTkFont(size=12),
        text_color=("gray60", "gray40")
    ).pack(pady=(10, 0))
    
    root.mainloop()


if __name__ == "__main__":
    main()
