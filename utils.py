import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
from cv2 import VideoCapture

from constants import *

logger = logging.getLogger(__name__)


def verify_system() -> None:
    """Verifica que todo este en orden, directorios, archivos, etc."""

    try:
        path = verify_directory(DS_SOCCERNET)
        path = verify_directory(YOLO_PATH_RESULTS)
        path = verify_directory(LOG_DIR)
        path = verify_directory(DS_SOCCERNET_FINAL)

        logger.info("Verificación completada, iniciando sistema...")
    except Exception as e:
        logger.error(e)
        sys.exit(1)


def verify_directory(dir_path: str) -> Path:
    path = Path(dir_path)

    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)

        msg = f"Directorio no existía y fue creado '{path.resolve()}'"
        print(msg)
        logger.info(msg)

    return path


def verify_file(file_path: str) -> Path:
    path = Path(file_path)

    if not path.is_file():
        raise Exception(f"El archivo no existe '{path.resolve()}'")

    return path


def load_video(video: str) -> VideoCapture:
    try:
        path = verify_file(video)
        cap = VideoCapture(filename=video)

        if not cap.isOpened():
            raise Exception(f"No se puede abrir el video '{path.resolve()}'")

        return cap
    except Exception as e:
        logger.error(e)
        sys.exit(1)


def write_list_in_txt(
        lista: list,
        file: str,
        modo: str = "a"
) -> None:
    """Escribe cada elemento de la lista en un archivo .txt"""
    # print(f"type(lista): {type(lista)}")

    try:
        # Abrimos el archivo en modo escritura ('w'). El modo 'w' crea el archivo si no existe, o lo sobrescribe si ya existe.
        # Usar 'with' asegura que el archivo se cierre correctamente al final.
        with open(file, mode=modo, encoding='utf-8') as archivo:
            for elemento in lista:
                # Convertimos el elemento a cadena (por si no lo es) y le añadimos un salto de línea '\n'
                # print(elemento)
                elemento = str(elemento).replace('\\', '/')
                archivo.write(elemento + '\n')

        logger.info(f"La lista de {len(lista)} elementos, se ha escrito correctamente en '{file}'")
    except IOError as e:
        logger.error(e)
    except Exception as e:
        logger.error(e)


def load_annotations(annotation_file: str) -> list:
    """Carga las anotaciones de un archivo JSON de SoccerNet."""
    # print(f"annotation_file: {annotation_file}")

    try:
        with open(annotation_file, 'r') as f:
            data = json.load(f)
            # Busca la clave que contiene la lista de eventos, sino existe devuelve una lista vacía
            events = data.get("annotations", [])

            return events
    except FileNotFoundError:
        logger.error(f"Archivo de anotaciones no ubicado en {annotation_file}")
        return []
    except json.JSONDecodeError:
        logger.error(f"No se pudo leer el archivo JSON de anotaciones.")
        return []


def write_dictionary_in_txt(
        diccionario: Dict[Any, Any],
        archivo_salida: str,
        formato: str = "items",  # Opciones: "keys", "values", "items"
        modo: str = "w",  # Opciones: "w" (sobrescribir), "a" (añadir)
        separador: str = ": ",  # Separador para formato "items"
) -> None:
    """
    Escribe un diccionario en un archivo de texto, con control sobre el formato y el modo de escritura.

    Args:
        diccionario: Diccionario a escribir.
        archivo_salida: Ruta del archivo de salida.
        formato: Qué parte del diccionario escribir ("keys", "values" o "items").
        modo: Modo de apertura del archivo ("w" para sobrescribir, "a" para añadir).
        separador: Separador entre clave y valor (solo aplica si formato="items").

    Ejemplo de uso:
        escribir_diccionario_en_txt(
            {1: "gol", 2: "tarjeta"}, 
            "eventos.txt", 
            formato="items", 
            separador=" -> "
        )
    """
    try:
        with open(archivo_salida, mode=modo, encoding="utf-8") as f:
            if formato == "items":
                for key, value in diccionario.items():
                    f.write(f"{key}{separador}{value}\n")
            elif formato == "keys":
                for key in diccionario.keys():
                    f.write(f"{key}\n")
            elif formato == "values":
                for value in diccionario.values():
                    f.write(f"{value}\n")
            else:
                raise ValueError(f"Formato no válido: '{formato}'. Use 'keys', 'values' o 'items'.")

        logger.info(f"Diccionario escrito en '{archivo_salida}'")
    except IOError as e:
        logger.error(str(e))
    except Exception as e:
        logger.error(str(e))


def get_time_employed(
        t_start: time,
        message: str,
) -> None:
    t_end = time.time() - t_start
    min = np.round(t_end / 60.0, decimals=2)
    logger.info(f"Tiempo total: {min} minutos. Tarea: {message}")


def write_soccernet_games_in_txt(
        lista: list,
        nombre_archivo: str,
) -> None:
    try:
        # Abrimos el archivo en modo escritura ('w'). El modo 'w' crea el archivo si no existe, o lo sobrescribe si ya existe.
        # Usar 'with' asegura que el archivo se cierre correctamente al final.
        with open(nombre_archivo, mode='w', encoding='utf-8') as archivo:
            for elemento in lista:
                # Convertimos el elemento a cadena (por si no lo es) y le añadimos un salto de línea '\n'
                # print(elemento)
                elemento = str(elemento).replace('\\', '/')
                archivo.write(elemento + '\n')

        logger.info(f"Lsta de {len(lista)} partidos, se ha escrito correctamente en '{nombre_archivo}'")
    except IOError as e:
        logger.error(str(e))
    except Exception as e:
        logger.error(str(e))
