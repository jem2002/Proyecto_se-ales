import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def extract_features(image):
    # Convertir la imagen a escala de grises usando promedio
    gray = np.mean(image, axis=2).astype(np.uint8)

    # Cálculo de gradientes (Sobel)
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Gx = cv2.filter2D(gray, -1, kernel_x)
    Gy = cv2.filter2D(gray, -1, kernel_y)

    # Cálculo correcto de la magnitud de gradiente
    Gt = np.sqrt(Gx**2 + Gy**2)

    # Normalización de Gt
    GtN = (Gt / np.max(Gt)) * 255
    GtN = GtN.astype(np.uint8)

    # Normalización de Gx y Gy con offset
    VminGx = np.min(Gx)
    VminGy = np.min(Gy)
    GradOffx = Gx - VminGx
    GradOffy = Gy - VminGy
    VmaxGx = np.max(GradOffx)
    VmaxGy = np.max(GradOffy)
    GxN = (GradOffx / VmaxGx) * 255
    GyN = (GradOffy / VmaxGy) * 255
    GxN = GxN.astype(np.uint8)
    GyN = GyN.astype(np.uint8)

    # Binarización de la imagen gradiente
    _, B = cv2.threshold(GtN, 50, 255, cv2.THRESH_BINARY)

    # Calcular momentos de Hu
    moments = cv2.HuMoments(cv2.moments(B)).flatten()

    return moments

# Rutas específicas de las imágenes de entrenamiento

circle_files = [
    r"C:\Users\nicol\Downloads\proyecto (2)\proyecto\identificadorDibujo\circulos\circle_PNG2.png",
    r"C:\Users\nicol\Downloads\proyecto (2)\proyecto\identificadorDibujo\circulos\circle_PNG27.png",
    r"C:\Users\nicol\Downloads\proyecto (2)\proyecto\identificadorDibujo\circulos\OIP.png",
    r"C:\Users\nicol\Downloads\proyecto (2)\proyecto\identificadorDibujo\circulos\R (2).png"
]

square_files = [
    r"C:\Users\nicol\Downloads\proyecto (2)\proyecto\identificadorDibujo\cuadrados\61cddf9cfe50f4baaa8f472c253d1cb4-2d-cuadrado-by-vexels.png",
    r"C:\Users\nicol\Downloads\proyecto (2)\proyecto\identificadorDibujo\cuadrados\R (1).png",
    r"C:\Users\nicol\Downloads\proyecto (2)\proyecto\identificadorDibujo\cuadrados\R.png",
    r"C:\Users\nicol\Downloads\proyecto (2)\proyecto\identificadorDibujo\cuadrados\square_PNG24.png"
]

triangle_files = [
    r"C:\Users\nicol\Downloads\proyecto (2)\proyecto\identificadorDibujo\triangulos\triangle_PNG16.png",
    r"C:\Users\nicol\Downloads\proyecto (2)\proyecto\identificadorDibujo\triangulos\triangle_PNG51.png",
    r"C:\Users\nicol\Downloads\proyecto (2)\proyecto\identificadorDibujo\triangulos\triangle_PNG72.png",
    r"C:\Users\nicol\Downloads\proyecto (2)\proyecto\identificadorDibujo\triangulos\triangle_PNG82.png",
    r"C:\Users\nicol\Downloads\proyecto (2)\proyecto\identificadorDibujo\triangulos\triangle_PNG84.png"
]

def load_images(file_list):
    images = []
    for file in file_list:
        img = cv2.imread(file)
        if img is not None:
            images.append(img)
        else:
            print(f"Warning: No se pudo cargar la imagen: {file}")
    return images

# Cargar imágenes
circle_images = load_images(circle_files)
square_images = load_images(square_files)
triangle_images = load_images(triangle_files)

# Diccionario para mapear etiquetas numéricas a nombres
shape_mapping = {0: "Círculo", 1: "Cuadrado", 2: "Triángulo"}

# Extraer características y etiquetas
X_train = []
y_train = []

image_sets = [circle_images, square_images, triangle_images]

for label, images in enumerate(image_sets):
    for img in images:
        features = extract_features(img)
        X_train.append(features)
        y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Entrenar el clasificador SVM
clf = make_pipeline(StandardScaler(), SVC(kernel='linear'))
clf.fit(X_train, y_train)

# Función para predecir la forma
def predict_shape(image):
    features = extract_features(image)
    prediction = clf.predict([features])
    return prediction[0]

# Probar con una imagen
test_image_path = r"C:\Users\nicol\Downloads\proyecto (2)\proyecto\identificadorDibujo\cuadrados\61cddf9cfe50f4baaa8f472c253d1cb4-2d-cuadrado-by-vexels.png"
test_image = cv2.imread(test_image_path)

if test_image is None:
    raise FileNotFoundError(f"No se pudo cargar la imagen: {test_image_path}")

shape = predict_shape(test_image)
print("Forma detectada:", shape_mapping.get(shape, "Desconocida"))

cv2.imshow("Imagen a detectar", test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
