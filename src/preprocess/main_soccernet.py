import time
import argparse
import glob
import json
import logging
import sys

from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.utils import getListGames
from moviepy import VideoFileClip

from src.utils import utils as ut
from src.utils.constants import *

logger = logging.getLogger(__name__)
downloader = SoccerNetDownloader(LocalDirectory=DS_SOCCERNET_RAW)
downloader.password = SOCCERNET_PASSWORD

FILE_VIDEO_224p = ["1_224p.mkv", "2_224p.mkv"]
FILE_VIDEO_720p = ["1_720p.mkv", "2_720p.mkv"]
file_labels = ["Labels.json", "Labels-v2.json", "Labels-cameras.json", "Labels-v3.json"]

ACCIONES_RECORTAR = ut.extraer_claves_ordenadas(SOCCERNET_LABELS)

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


def config_log() -> None:
    ut.verify_directory(LOG_DIR)

    logging.basicConfig(
        filename=LOG_SOCCERNET,
        filemode="a",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        # https://docs.python.org/3/library/logging.html
        level=logging.INFO,  # NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL
        encoding="utf-8",
    )


def cut_video_segments_by_label(
        directory: str,
        video_filename: str,
        json_filename: str,
        # label_to_find: str,
        # fps: int = 25,  # los videos (224p y 720p) de soccernet estan a 25 fps: https://www.soccer-net.org/data
        mas_menos: float = 2.0,
) -> list:
    """
    Busca anotaciones en un archivo JSON para todas las 'ACCIONES_RECORTAR', calcula los frames correspondientes en un video y extrae segmentos de video alrededor de esos puntos (+- 2 segundos).
    Si un 'gameTime' se repite para un mismo partido-tiempo (sin importar la etiqueta), solo se considera la primera aparición.
    Las acciones a recortar se priorizan de acuerdo al orden establecido en 'ACCIONES_RECORTAR'.

    Args:
        directory (str): Ruta al directorio que contiene el video y el JSON.
        video_filename (str): Nombre del archivo de video (formato mkv).
        json_filename (str): Nombre del archivo JSON de anotaciones.
        mas_menos (float): +- N segundos para recortar el video por cada accion encontrada.

    Returns:
        list: Una lista de las rutas completas de los archivos de video cortados guardados.
              Retorna una lista vacía si no se encontraron anotaciones con los labels o si ocurrió un error.
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

    logger.info(f"Analizando video '{video_path}'")

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
    gt_prefix = video_filename.split("_")[0]  # '1' o '2' para el primer o segundo tiempo

    # Set para llevar un registro de los gameTime (parte - MM:SS) ya procesados para cualquier label
    processed_gametimes_for_video = set()
    # Para almacenar todas las anotaciones únicas encontradas, sin importar el label
    all_matching_annotations = []

    for label_to_find in ACCIONES_RECORTAR:
        # buscar anotaciones que coincidan con 'label_to_find' y manejar repeticiones
        for annotation in annotations:
            if isinstance(annotation, dict) and \
                    annotation.get("label") == label_to_find:

                game_time_full_str = annotation.get("gameTime")

                if not game_time_full_str:
                    logger.warning(f"Anotación encontrada sin 'gameTime'. Saltando anotación: {annotation}")
                    continue

                game_time_parts = game_time_full_str.split(" - ")

                # Asegurarse de que el gameTime corresponde al tiempo del video actual (1 o 2)
                if len(game_time_parts) < 2 or game_time_parts[0] != gt_prefix:
                    continue

                time_part_mm_ss = game_time_parts[-1]  # MM:SS
                unique_identifier = f"{gt_prefix}-{time_part_mm_ss}"  # Ej: '1-07:15'

                if unique_identifier in processed_gametimes_for_video:
                    logger.warning(
                        f"Anotación duplicada (gameTime) para '{game_time_full_str}' con label '{label_to_find}' (video: {video_filename}). Esta ocurrencia será ignorada."
                    )
                    continue

                # Si es una anotación única para este gameTime en este video, la agregamos
                all_matching_annotations.append(annotation)
                processed_gametimes_for_video.add(unique_identifier)

    # Ordenar las anotaciones por gameTime para asegurar un orden consistente en los clips. Puede ser útil para la organización de los clips generados
    all_matching_annotations.sort(key=lambda x: (int(x["gameTime"].split(" - ")[0]),  # Parte del juego
                                                 int(x["gameTime"].split(" - ")[1].split(":")[0]),  # Minutos
                                                 int(x["gameTime"].split(" - ")[1].split(":")[1])))  # Segundos

    if not all_matching_annotations:
        logger.info(f"No se encontraron anotaciones únicas para recortar en el video '{video_filename}'.")
        return []

    logger.info(f"{len(all_matching_annotations)} anotaciones únicas encontradas para el video '{video_filename}'.")

    try:
        video_clip = VideoFileClip(video_path)
        video_duration = video_clip.duration
    except Exception as e:
        logger.error(f"Al cargar el video con moviepy: {e}")
        return []

    for annotation in all_matching_annotations:
        label_for_clip = annotation.get("label")
        game_time_str = annotation.get("gameTime")

        # Crear el directorio de salida para la acción actual
        output_dir = os.path.join(DS_SOCCERNET_ACTIONS, label_for_clip)
        os.makedirs(output_dir, exist_ok=True)

        # Contar archivos .mkv existentes en el directorio de salida (para el índice)
        existing_files = glob.glob(os.path.join(output_dir, "*.mkv"))
        segment_index = len(existing_files) + 1

        # Extraer MM:SS de "Parte - MM:SS"
        time_part = game_time_str.split(' - ')[-1]

        try:
            minutes, seconds = map(int, time_part.split(':'))
            total_seconds = minutes * 60 + seconds
        except ValueError:
            logger.warning(
                f"Formato de 'gameTime' inesperado '{game_time_str}'. Esperado 'MM:SS'. Saltando anotación: {annotation}")
            continue

        # Asegurar que el tiempo de inicio no sea negativo y el tiempo de fin no exceda la duración del video
        cut_start_time = max(0.0, total_seconds - mas_menos)
        cut_end_time = min(video_duration, total_seconds + mas_menos)

        # Aseguramos que el tiempo de fin sea mayor que el tiempo de inicio
        if cut_end_time <= cut_start_time:
            logger.warning(
                f"La ventana de corte calculada para '{game_time_str}' (label '{label_for_clip}') "
                f"({cut_start_time:.2f}s a {cut_end_time:.2f}s) es inválida o demasiado corta. Saltando.")
            continue

        logger.info(
            f"Procesando '{label_for_clip}' en gameTime '{game_time_str}'. Ventana: {cut_start_time:.2f}s a {cut_end_time:.2f}s")

        try:
            subclip = video_clip.subclipped(cut_start_time, cut_end_time)

            # Definir el nombre y ruta del archivo de salida
            base_filename = os.path.splitext(video_filename)[0]  # Nombre del archivo original sin extensión
            base_filename = base_filename.split("_")[1]
            #base_filename = f"{base_filename}_{directory.replace("/", "_").replace("\\", "_")}"
            output_filename = f"{label_for_clip}_{segment_index:04d}_{base_filename}_{game_time_str.replace(' - ', '_').replace(':', '')}.mkv"
            output_path = os.path.join(output_dir, output_filename)

            # Especificamos el codec para MKV, libx264 es común
            subclip.write_videofile(output_path, codec='libx264', audio_codec='aac')

            logger.info(f"Segmento guardado exitosamente en: '{output_path}'")
            saved_clip_paths.append(output_path)
            # segment_index se incrementa para cada acción individual, ya que los directorios son por acción
        except Exception as e:
            logger.error(f"Al procesar o guardar el segmento: {str(e)}")

    # liberar recursos del video
    video_clip.close()
    logger.info(
        f"Procesamiento completado para video '{video_filename}'. Se guardaron {len(saved_clip_paths)} segmentos únicos.")

    return saved_clip_paths


def download_files(
        list_games: list,
        cantidad: int = 1,
) -> None:
    ut.write_soccernet_games_in_txt(list_games, os.path.join(SOCCERNET_RESULTS, "SoccerNet_list_games.txt"))

    for i in range(cantidad):
        # logger.info(f"Identificando partido: {i + 1}")
        # 'downloadGame' solo crea el directorio del 'game' actual
        downloader.downloadGame(game=list_games[i], files=FILES)


def download_files_by_index(
        list_games: list,
        partidos_seleccionados: list = [0],
) -> None:
    ut.write_soccernet_games_in_txt(list_games, os.path.join(SOCCERNET_RESULTS, "SoccerNet_list_games.txt"))
    result_list = [list_games[index] for index in partidos_seleccionados]
    # TODO crear un .csv a partir de 'result_list', headers [index, game], y guardarlo en local
    ut.write_soccernet_games_in_txt(result_list, os.path.join(SOCCERNET_RESULTS, "SoccerNet_list_games_downloaded_by_index.txt"))

    for game in result_list:
        # 'downloadGame' solo crea el directorio del 'game' actual
        downloader.downloadGame(game=game, files=FILES)


def cut_video_loop(
        games: list,
        videos: list,
        # actions: list,
) -> None:
    # recorro todos los directorios
    for game in games:
        #time.sleep(4.0)
        for video in videos:
            # for action in actions:
            saved_clips = cut_video_segments_by_label(
                directory=os.path.join(DS_SOCCERNET_RAW, str(game).replace('\\', '/')),
                video_filename=video,
                json_filename=file_labels[1],
                # label_to_find=action,
                mas_menos=1.5,
            )


def main(args) -> None:
    list_games = getListGames(split=SPLIT)

    if args.omitir_descarga == 0:
        t_start = time.time()
        
        if args.partidos_cantidad > 0:
            download_files(list_games, args.partidos_cantidad)
        else:
            download_files_by_index(list_games, args.partidos_indice)
        
        ut.get_time_employed(t_start, "Descarga de archivos desde SoccerNet.")

    if args.omitir_recorte == 0:
        t_start = time.time()
        result_list = [list_games[index] for index in args.partidos_indice]
        
        if args.calidad_video == "224p":
            cut_video_loop(result_list, FILE_VIDEO_224p)
        elif args.calidad_video == "720p":
            cut_video_loop(result_list, FILE_VIDEO_720p)
        
        ut.get_time_employed(t_start, "Recorte de acciones etiquetadas en los partidos (crudos) de SoccerNet.")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Descargar archivos de SoccerNet y recortar acciones etiquetadas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--omitir_descarga", default=0, type=int, choices=[0, 1],
                        help="Omite el proceso de descarga en SoccerNet. 0: apagado, 1: prendido.")
    parser.add_argument("--partidos_cantidad", default=0, type=ut.non_negative_int,
                        help="Descarga los primeros N partidos, valor entero entre 0 y 500.")
    parser.add_argument("--partidos_indice", default=[0], type=ut.non_negative_int, nargs="+",
                        help="Índice de cada partido que desea descargar, valores enteros entre 0 y 500.")
    parser.add_argument("--omitir_recorte", default=0, type=int, choices=[0, 1],
                        help="Omite el proceso de recortar las acciones en los videos de SoccerNet. 0: apagado, 1: prendido.")
    parser.add_argument("--calidad_video", default="224p", type=str, choices=["224p", "720p"],
                        help="Tamaño del video que se utilizará para recortar las acciones.")

    return parser.parse_args()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        args = parse_arguments()
    else:
        # Valores por defecto si no se proporcionan argumentos desde la línea de comandos
        class Args:
            omitir_descarga = 1
            partidos_cantidad = 0
            partidos_indice = PARTIDOS_INDICE_LOTE_1
            #partidos_indice = PARTIDOS_INDICE_LOTE_2
            #partidos_indice = PARTIDOS_INDICE_LOTE_3
            omitir_recorte = 0
            #calidad_video = "224p"
            calidad_video = "720p"


        args = Args()

    config_log()
    ut.verify_system()
    main(args)
