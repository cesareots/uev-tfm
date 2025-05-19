import glob
import json
import logging

from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.utils import getListGames
from moviepy import VideoFileClip

import utils as ut
from constants import *

logger = logging.getLogger(__name__)
downloader = SoccerNetDownloader(LocalDirectory=DS_SOCCERNET)
downloader.password = SOCCERNET_PASSWORD

PARTIDOS_CANTIDAD = 10
FILE_VIDEO_224p = ["1_224p.mkv", "2_224p.mkv"]
FILE_VIDEO_720p = ["1_720p.mkv", "2_720p.mkv"]
file_labels = ["Labels.json", "Labels-v2.json", "Labels-cameras.json", "Labels-v3.json"]

ACCIONES_RECORTAR = [  # TODO que acciones recortare?
    "Goal",
    "Yellow card",
    "Shots on target",
]

FILES = [
    # acciones detectadas
    file_labels[0],
    file_labels[1],
    file_labels[2],

    # video 224 pixeles de alto, primer y segundo tiempo de cada partido
    FILE_VIDEO_224p[0],
    FILE_VIDEO_224p[1],

    # video 720 pixeles de alto
    FILE_VIDEO_720p[0],
    FILE_VIDEO_720p[1],

    # video - Low Quality
    # "1.mkv",
    # "2.mkv",

    # video - High Quality, falta algun tipo de permiso (HTTP Error 401: Unauthorized)
    # "1_HQ.mkv",
    # "2_HQ.mkv",
    # "video.ini",

    # version 3
    # "Frames-v3.zip",
    # file_labels[3],
]

SPLIT = [
    "train",
    "valid",
    "test",
    # "challenge",  # dataset de retos para nuevos releases
]

PARTIDOS_INDEX = [
    0, 1,  # england_epl/2014-2015/
    4, 5,  # england_epl/2015-2016/
    29, 30,  # england_epl/2016-2017/
    58, 59,  # europe_uefa-champions-league/2014-2015/
    80, 81,  # europe_uefa-champions-league/2015-2016/
    109, 110,  # europe_uefa-champions-league/2016-2017/
    120,  # france_ligue-1/2014-2015/
    121, 122,  # france_ligue-1/2015-2016/
    123, 124,  # france_ligue-1/2016-2017/
    147, 148,  # germany_bundesliga/2014-2015/
    152, 153,  # germany_bundesliga/2015-2016/
    162, 163,  # germany_bundesliga/2016-2017/
    178, 179,  # italy_serie-a/2014-2015/
    185, 186,  # italy_serie-a/2015-2016/
    190, 191,  # italy_serie-a/2016-2017/
    234, 235,  # spain_laliga/2014-2015/
    240, 241,  # spain_laliga/2015-2016/
    258, 259,  # spain_laliga/2016-2017/
]


def config_log() -> None:
    ut.verify_directory(LOG_DIR)

    logging.basicConfig(
        filename=LOG_SOCCERNET,
        filemode="a",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        # https://docs.python.org/3/library/logging.html
        level=logging.INFO,  # NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL
    )


def cut_video_segments_by_label(
        directory: str,
        video_filename: str,
        json_filename: str,
        label_to_find: str,
        # fps: int = 25,  # los videos (224p y 720p) de soccernet estan a 25 fps: https://www.soccer-net.org/data
        mas_menos: float = 2.0,
) -> list:
    """
    Busca anotaciones con un label específico en un archivo JSON,
    calcula los frames correspondientes en un video y extrae segmentos
    de video alrededor de esos puntos (+- 2 segundos).

    Args:
        directory (str): Ruta al directorio que contiene el video y el JSON.
        video_filename (str): Nombre del archivo de video (formato mkv).
        json_filename (str): Nombre del archivo JSON de anotaciones.
        label_to_find (str): El valor del 'label' a buscar en las anotaciones.
        mas_menos (float): +- N segundos para recortar el video a partir de 'label_to_find'

    Returns:
        list: Una lista de las rutas completas de los archivos de video cortados guardados.
              Retorna una lista vacía si no se encontraron anotaciones con el label
              o si ocurrió un error.
    """
    video_path = os.path.join(directory, video_filename)
    json_path = os.path.join(directory, json_filename)
    saved_clip_paths = []

    # Validaciones iniciales
    if not os.path.isdir(directory):
        # logger.error(f"Directorio no encontrado: '{directory}'")
        return []

    if not os.path.exists(video_path):
        logger.error(f"Video no encontrado: '{video_path}'")
        return []

    if not os.path.exists(json_path):
        logger.error(f"JSON no encontrado: '{json_path}'")
        return []

    # cargar y parsear JSON
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        logger.error(f"JSON no válido: '{json_path}'")
        return []
    except Exception as e:
        logger.error(str(e))
        return []

    if not isinstance(data, dict) or \
            "annotations" not in data or \
            not isinstance(data["annotations"], list):
        logger.error(f"El archivo JSON debe contener una clave 'annotations' que sea una lista.")
        return []

    annotations = data["annotations"]
    matching_annotations = []
    gt = video_filename.split("_")[0]

    # buscar anotaciones que coincidan con 'label_to_find'
    for annotation in annotations:
        if isinstance(annotation, dict) and \
                annotation.get("label") == label_to_find and \
                annotation.get("gameTime").split(" - ")[0] == gt:
            matching_annotations.append(annotation)

    if not matching_annotations:
        # logger.info(f"No se encontraron anotaciones con el label '{label_to_find}'. No se realizarán cortes.")
        return []

    # print(f"{len(matching_annotations)} anotaciones encontradas con el label '{label_to_find}'.")

    # cargar el video para obtener su duración
    try:
        video_clip = VideoFileClip(video_path)
        video_duration = video_clip.duration  # Duración en segundos
        # print(f"Video cargado. Duración total: {video_duration:.2f} segundos.")
    except Exception as e:
        logger.error(f"Al cargar el video con moviepy: {e}")
        return []

    # Crear el directorio de salida
    output_dir = os.path.join(DS_SOCCERNET_FINAL, label_to_find)
    os.makedirs(output_dir, exist_ok=True)

    # Contar archivos .mkv existentes en el directorio de salida
    existing_files = glob.glob(os.path.join(output_dir, "*.mkv"))
    segment_index = len(existing_files) + 1
    # segment_index = 1

    # procesar cada anotación coincidente y cortar el video
    for annotation in matching_annotations:
        game_time_str = annotation.get("gameTime")
        # position_str = annotation.get("position", "N/A")

        if not game_time_str:
            logger.warning(f"Anotación encontrada sin 'gameTime'. Saltando anotación: {annotation}")
            continue

        # Extraer MM:SS de "Parte - MM:SS"
        time_part = game_time_str.split(' - ')[-1]

        try:
            minutes, seconds = map(int, time_part.split(':'))
            total_seconds = minutes * 60 + seconds
        except ValueError:
            logger.warning(
                f"Formato de 'gameTime' inesperado '{game_time_str}'. Esperado 'MM:SS'. Saltando anotación: {annotation}")
            continue

        # Aseguramos que el tiempo de inicio no sea negativo y el tiempo de fin no exceda la duración del video
        # mas_menos = 2.0
        cut_start_time = max(0.0, total_seconds - mas_menos)
        cut_end_time = min(video_duration, total_seconds + mas_menos)

        # Aseguramos que el tiempo de fin sea mayor que el tiempo de inicio
        if cut_end_time <= cut_start_time:
            logger.warning(
                f"La ventana de corte calculada para '{game_time_str}' ({cut_start_time:.2f}s a {cut_end_time:.2f}s) es inválida o demasiado corta. Saltando.")
            continue

        # print(f"Procesando '{label_to_find}' en gameTime '{game_time_str}' ({total_seconds:.2f}s).")
        logger.info(f"Ventana de corte calculada: {cut_start_time:.2f}s a {cut_end_time:.2f}s")

        try:
            # Realizar el corte del video
            subclip = video_clip.subclipped(cut_start_time, cut_end_time)

            # Definir el nombre y ruta del archivo de salida
            base_filename = os.path.splitext(video_filename)[0]  # Nombre del archivo original sin extensión
            output_filename = f"{label_to_find}_{segment_index:04d}_{base_filename}.mkv"  # Nombre con label e índice
            output_path = os.path.join(output_dir, output_filename)

            # Especificamos el codec para MKV, libx264 es común
            subclip.write_videofile(output_path, codec='libx264', audio_codec='aac')

            logger.info(f"Segmento guardado exitosamente en: '{output_path}'")
            saved_clip_paths.append(output_path)
            segment_index += 1  # Incrementar el índice para el siguiente segmento
        except Exception as e:
            logger.error(f"Al procesar o guardar el segmento: {str(e)}")

    # liberar recursos del video
    video_clip.close()
    logger.info(f"Procesamiento completado. Se guardaron {len(saved_clip_paths)} segmentos.")

    return saved_clip_paths


def download_files(
        list_games: list,
        cantidad: int = 4,
) -> None:
    ut.write_soccernet_games_in_txt(list_games, f"SoccerNet_list_games.txt")

    for i in range(cantidad):
        # logger.info(f"Identificando partido: {i + 1}")
        # 'downloadGame' solo crea el directorio del 'game' actual
        downloader.downloadGame(game=list_games[i], files=FILES)


def download_files_by_index(
        list_games: list,
        partidos_seleccionados: list = [0],
) -> None:
    ut.write_soccernet_games_in_txt(list_games, f"SoccerNet_list_games.txt")
    result_list = [list_games[index] for index in partidos_seleccionados]
    ut.write_soccernet_games_in_txt(result_list, f"SoccerNet_list_games_downloaded.txt")

    for game in result_list:
        # 'downloadGame' solo crea el directorio del 'game' actual
        downloader.downloadGame(game=game, files=FILES)


def cut_video_loop(
        games: list,
        videos: list,
        actions: list,
) -> None:
    # recorro todos los directorios
    for game in games:
        for video in videos:
            for action in actions:
                saved_clips = cut_video_segments_by_label(
                    directory=os.path.join(DS_SOCCERNET, str(game).replace('\\', '/')),
                    video_filename=video,
                    json_filename=file_labels[1],
                    label_to_find=action,
                )


def main() -> None:
    list_games = getListGames(split=SPLIT)

    # download_files(list_games, PARTIDOS_CANTIDAD)  # descarga los primeros N partidos
    download_files_by_index(list_games, PARTIDOS_INDEX)

    # cut_video_loop(list_games, FILE_VIDEO_224p, ACCIONES_RECORTAR)
    # cut_video_loop(list_games, FILE_VIDEO_720p, ACCIONES_RECORTAR)

    """saved_clips = cut_video_segments_by_label(
        directory=os.path.join(DS_SOCCERNET, f"{"england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley".replace('\\', '/')}"),
        #directory="D:\\cesareots\\uev-master-ia\\0-tfm\\proy_tfm\\dataset\\soccernet\\england_epl\\2014-2015\\2015-02-21 - 18-00 Chelsea 1 - 1 Burnley",
        video_filename="1_720p.mkv",
        json_filename=file_labels[1],
        label_to_find=acciones_recortar[0],
    )"""


if __name__ == "__main__":
    config_log()
    ut.verify_system()
    main()
