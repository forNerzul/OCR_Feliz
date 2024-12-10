import os
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import cv2
import numpy as np

# Configuración de Tesseract (asegúrate de que esté en tu PATH)
pytesseract.pytesseract.tesseract_cmd = r"tesseract"  # Cambia si estás en Windows

# Ruta del archivo PDF
PDF_PATH = "proyecto.pdf"  # Cambia el nombre por tu archivo PDF
OUTPUT_DIR = "output"  # Carpeta para guardar imágenes y resultados

# Crear la carpeta de salida si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)


import re


def correct_orientation(image_path):
    """
    Corrige la orientación de una imagen y verifica que el texto extraído sea legible.
    Reintenta con diferentes rotaciones si no se obtiene texto legible.
    """
    print(f"Corrigiendo orientación de la imagen: {image_path}")

    # Leer la imagen original con OpenCV
    original_image = cv2.imread(image_path)
    angles_to_try = [0, 90, 180, 270]

    for angle in angles_to_try:
        # Rotar la imagen según el ángulo actual
        rotated_image = rotate_image(original_image, angle)

        # Extraer texto de la imagen rotada (en español)
        print(f"Intentando rotación de {angle} grados...")
        extracted_text = pytesseract.image_to_string(rotated_image, lang="spa")

        # Verificar si el texto extraído es legible
        if is_text_legible_in_spanish(extracted_text):
            print(f"Texto legible encontrado tras rotar {angle} grados.")
            # Guardar la imagen corregida
            corrected_path = image_path.replace(".png", f"_{angle}_corrected.png")
            cv2.imwrite(corrected_path, rotated_image)
            return corrected_path

    print(f"No se encontró un texto legible para la imagen {image_path}.")
    # Guardar la última rotación como resultado fallido
    failed_path = image_path.replace(".png", "_failed.png")
    cv2.imwrite(failed_path, rotated_image)
    return failed_path


def rotate_image(image, angle):
    """Rota una imagen por el ángulo dado sin recortar el contenido."""
    (h, w) = image.shape[:2]
    diagonal = int((w**2 + h**2) ** 0.5)
    new_width = diagonal
    new_height = diagonal

    # Crear lienzo expandido
    expanded_image = cv2.copyMakeBorder(
        image,
        (diagonal - h) // 2,
        (diagonal - h) // 2,
        (diagonal - w) // 2,
        (diagonal - w) // 2,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )

    center = (new_width // 2, new_height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(
        expanded_image, rotation_matrix, (new_width, new_height)
    )
    return rotated_image


def is_text_legible_in_spanish(text):
    """
    Determina si el texto extraído es legible en español.
    Evalúa el texto basándose en palabras comunes del idioma.
    """
    # Lista básica de palabras comunes en español
    common_words = set(
        [
            "el",
            "la",
            "de",
            "que",
            "en",
            "y",
            "a",
            "los",
            "del",
            "se",
            "las",
            "por",
            "un",
            "para",
            "con",
            "no",
            "una",
            "su",
            "al",
            "es",
        ]
    )

    # Extraer palabras del texto
    words = re.findall(r"\b\w+\b", text.lower())

    # Contar palabras válidas
    valid_words = sum(1 for word in words if word in common_words)

    # Considerar el texto legible si al menos el 20% de las palabras son válidas
    if len(words) == 0:
        return False
    return valid_words / len(words) >= 0.2


def rotate_image(image, angle):
    """Rota una imagen por el ángulo dado sin recortar el contenido."""
    (h, w) = image.shape[:2]
    diagonal = int((w**2 + h**2) ** 0.5)
    new_width = diagonal
    new_height = diagonal

    # Crear lienzo expandido
    expanded_image = cv2.copyMakeBorder(
        image,
        (diagonal - h) // 2,
        (diagonal - h) // 2,
        (diagonal - w) // 2,
        (diagonal - w) // 2,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )

    center = (new_width // 2, new_height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(
        expanded_image, rotation_matrix, (new_width, new_height)
    )
    return rotated_image


def is_text_legible(text):
    """
    Determina si el texto extraído es legible.
    Usa un umbral de caracteres válidos frente a caracteres totales.
    """
    valid_chars = sum(c.isalnum() or c.isspace() for c in text)
    total_chars = len(text)

    # Evitar divisiones por cero
    if total_chars == 0:
        return False

    # Considerar legible si más del 50% de los caracteres son válidos
    return valid_chars / total_chars > 0.5


def pdf_to_images(pdf_path, output_dir):
    """Convierte cada página de un PDF en una imagen y las guarda."""
    print("Convirtiendo PDF a imágenes...")
    images = convert_from_path(pdf_path)
    image_paths = []

    for i, image in enumerate(images):
        image_path = os.path.join(output_dir, f"page_{i + 1}.png")
        image.save(image_path, "PNG")
        image_paths.append(image_path)
        print(f"Página {i + 1} convertida a imagen: {image_path}")

    return image_paths


def extract_text_from_images(image_paths):
    """Extrae texto de una lista de imágenes usando Tesseract OCR."""
    all_text = ""
    for i, image_path in enumerate(image_paths):
        # Corregir la orientación de la imagen
        corrected_image_path = correct_orientation(image_path)

        # Extraer texto de la imagen corregida
        print(f"Procesando texto de la imagen: {corrected_image_path}...")
        text = pytesseract.image_to_string(Image.open(corrected_image_path))
        all_text += f"\n--- Texto de la página {i + 1} ---\n{text}"

    return all_text


if __name__ == "__main__":
    # Extraer imágenes del PDF
    image_paths = pdf_to_images(PDF_PATH, OUTPUT_DIR)

    # Extraer texto de las imágenes
    print("Extrayendo texto de las imágenes...")
    extracted_text = extract_text_from_images(image_paths)

    # Guardar el texto extraído en un archivo
    text_output_path = os.path.join(OUTPUT_DIR, "extracted_text.txt")
    with open(text_output_path, "w", encoding="utf-8") as f:
        f.write(extracted_text)

    print(f"Texto extraído guardado en: {text_output_path}")
