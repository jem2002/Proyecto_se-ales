import json

# Guardar vectores de referencia en un archivo JSON
def save_reference_vectors(reference_vectors, file_path="reference_vectors.json"):
    with open(file_path, 'w') as f:
        json.dump(reference_vectors, f)
    print(f"Vectores de referencia guardados en {file_path}")
    
    
# Cargar vectores de referencia desde un archivo JSON
def load_reference_vectors(file_path="reference_vectors.json"):
    with open(file_path, 'r') as f:
        reference_vectors = json.load(f)
    print(f"Vectores de referencia cargados desde {file_path}")
    return reference_vectors
