import cv2
import numpy as np
import os

def tiene_canal_alpha(ruta_imagen):
    """
    Verifica si una imagen tiene un canal alfa usando OpenCV.

    Args:
        ruta_imagen (str): La ruta al archivo de la imagen.

    Returns:
        bool: True si la imagen tiene canal alfa, False en caso contrario.
              También devuelve False si la imagen no se puede cargar.
    """
    # 1. Leer la imagen con el flag IMREAD_UNCHANGED
    # Este flag es CRUCIAL. Asegura que OpenCV cargue la imagen "tal cual",
    # incluyendo el canal alfa si existe. Por defecto, lo descartaría.
    img = cv2.imread(ruta_imagen, cv2.IMREAD_UNCHANGED)

    # Verificar si la imagen se cargó correctamente
    if img is None:
        print(f"Error: No se pudo cargar la imagen en la ruta: {ruta_imagen}")
        return False

    # 2. Comprobar la "forma" (shape) de la imagen
    # img.shape es una tupla: (alto, ancho, número_de_canales)
    # Si la imagen es a color (BGR), tendrá 3 canales.
    # Si es a color con alfa (BGRA), tendrá 4 canales.
    # Si es en escala de grises, la tupla solo tiene 2 elementos (alto, ancho).
    if len(img.shape) == 3 and img.shape[2] == 4:
        return True
    else:
        return False

if __name__ == "__main__":
    ruta1 = "pocs/img_alpha.png"
    ruta2 = "pocs/cr7.png"

    if tiene_canal_alpha(ruta1):
        print(f"Resultado: SÍ, la imagen '{ruta1}' tiene canal alfa.\n")
    else:
        print(f"Resultado: NO, la imagen '{ruta1}' no tiene canal alfa.\n")
        
    if tiene_canal_alpha(ruta2):
        print(f"Resultado: SÍ, la imagen '{ruta2}' tiene canal alfa.\n")
    else:
        print(f"Resultado: NO, la imagen '{ruta2}' no tiene canal alfa.\n")
