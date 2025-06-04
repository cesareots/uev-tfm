import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
from cv2 import VideoCapture

from src.utils.constants import *

logger = logging.getLogger(__name__)


def verify_system() -> None:
    """Verifica que todo este en orden, directorios, archivos, etc."""

    try:
        list_path = [
            DS_SOCCERNET_RAW,
            RESULTS,
            MODELS_DIR,
            M_BASIC,
            SOCCERNET_RESULTS,
            YOLO_PATH_RESULTS,
            LOG_DIR,
            DS_SOCCERNET_ACTIONS,
            DS_SOCCERNET_TENSORS,
        ]

        for path in list_path:
            verify_directory(path)

        logger.info("Verificación completada, iniciando sistema 🚀🚀🚀")
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


def non_negative_int(value):
    try:
        val = int(value)

        if val < 0:
            raise argparse.ArgumentTypeError(f"{value} es un entero negativo")
        return val
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} no es un entero válido")


def non_negative_float(value):
    try:
        val = float(value)

        if val < 0.0:
            raise argparse.ArgumentTypeError(f"{value} es un decimal negativo")
        return val
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} no es un decimal válido")


def extraer_claves_ordenadas(diccionario: dict) -> list:
    """
    Dado un diccionario cuyas claves son strings y los valores son índices enteros,
    devuelve una lista de claves ordenadas por sus valores.

    Args:
        diccionario (dict): Diccionario con claves de tipo str y valores enteros.

    Returns:
        list: Lista de claves ordenadas según sus valores.
    """
    return [clave for clave, _ in sorted(diccionario.items(), key=lambda item: item[1])]


def obtener_numeros(rango):
    """Devuelve una lista con todos los números enteros entre los dos valores"""
    inicio, fin = rango

    return list(range(inicio, fin + 1))


def escribir_csv(
        nombre_archivo,
        cabecera,
        registro,
) -> None:
    """
    Escribe un registro en un archivo CSV.
    Si el archivo no existe, lo crea con la cabecera.
    Si el archivo existe, agrega el registro en una nueva línea.

    Args:
        nombre_archivo (str): La ruta y nombre del archivo CSV.
        cabecera (list): Lista de cadenas para la cabecera del CSV.
        registro (list): Lista de valores para el registro a escribir.
    """
    archivo_existe = os.path.exists(nombre_archivo)

    # modo append ('a') para añadir.
    # newline='' para evitar líneas en blanco extra en CSV.
    with open(nombre_archivo, mode='a', newline='', encoding='utf-8') as archivo_csv:
        escritor_csv = csv.writer(archivo_csv)

        # Si el archivo no existe, escribe la cabecera
        if not archivo_existe:
            escritor_csv.writerow(cabecera)

        # Escribe el registro actual
        escritor_csv.writerow(registro)


def juego_ya_registrado(
        nombre_archivo_games_csv,
        game_index,
):
    """
    Verifica si un índice de juego ya está presente en el games.csv para evitar duplicados.
    """
    if not os.path.exists(nombre_archivo_games_csv):
        return False

    with open(nombre_archivo_games_csv, mode='r', newline='', encoding='utf-8') as archivo_csv:
        lector_csv = csv.reader(archivo_csv)
        next(lector_csv, None)  # Saltar la cabecera
        for fila in lector_csv:
            try:
                # Asumiendo que el índice es la primera columna
                if len(fila) > 0 and int(fila[0]) == game_index:
                    return True
            except ValueError:
                # Si el valor no es un entero, lo ignoramos y seguimos
                continue

    return False


def leer_registros_txt(ruta_archivo: str) -> list:
    """
    Lee un archivo de texto (.txt) y almacena cada línea como un elemento en una lista.

    Args:
        ruta_archivo (str): La ruta completa al archivo .txt.

    Returns:
        list: Una lista de strings, donde cada string es una línea del archivo.
              Retorna una lista vacía si el archivo no se encuentra o está vacío.
    """
    registros = []

    try:
        with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
            for linea in archivo:
                # .strip() elimina los espacios en blanco al principio/final y, muy importante, el carácter de nueva línea ('\n')
                registros.append(linea.strip())
        logger.info(f"Archivo '{ruta_archivo}' leído exitosamente. Se encontraron {len(registros)} registros.")
    except FileNotFoundError:
        logger.error(f"El archivo '{ruta_archivo}' no fue encontrado.")
    except Exception as e:
        logger.error(f"Al leer el archivo '{ruta_archivo}': {str(e)}")

    return registros
