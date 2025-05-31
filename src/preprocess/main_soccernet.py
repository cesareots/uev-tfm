import argparse
import glob
import json
import logging
import sys
import time

from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.utils import getListGames
from moviepy import VideoFileClip

from src.utils import utils as ut
from src.utils.constants import *

logger = logging.getLogger(__name__)
downloader = SoccerNetDownloader(LocalDirectory=DS_SOCCERNET_RAW)
downloader.password = SOCCERNET_PASSWORD
ACCIONES_RECORTAR = ut.extraer_claves_ordenadas(SOCCERNET_LABELS)

PARTIDOS_INDICE_LOTE_old = [
    [0, 9],
    [10, 19],
    [20, 29],
    [30, 39],
    [40, 49],
    [50, 59],
    [60, 69],
    [70, 78],
    [79, 79],  # da error por el codec del audio
    [80, 89],
    [90, 99],

    [100, 109],
    [110, 119],
    [120, 129],
    [130, 139],
    [140, 149],
    [150, 159],
    [160, 169],
    [170, 179],
    [180, 189],
    [190, 199],

    [200, 209],
    [210, 219],
    [220, 229],
    [230, 239],
    [240, 249],
    [250, 259],
    [260, 269],
    [270, 279],
    [280, 289],
    [290, 299],

    [300, 309],
    [310, 319],
    [320, 329],
    [330, 339],
    [340, 349],
    [350, 359],
    [360, 369],
    [370, 379],
    [380, 389],
    [390, 399],

    [400, 409],
    [410, 419],
    [420, 429],
    [430, 439],
    [440, 449],
    [450, 459],
    [460, 469],
    [470, 479],
    [480, 489],
    [490, 499],
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
            # base_filename = f"{base_filename}_{directory.replace("/", "_").replace("\\", "_")}"
            output_filename = f"{label_for_clip}_{segment_index:04d}_{base_filename}_{game_time_str.replace(' - ', '_').replace(':', '')}.mkv"
            output_path = os.path.join(output_dir, output_filename)

            # Especificamos el codec para MKV, libx264 es común
            subclip.write_videofile(
                output_path,
                codec="libx264",
                # algunos videos de soccernet tienen 'errores' en la codificacion del audio (ejemplo game_id=79), y sinceramente no estamos analizando audio, solo necesitamos los frames
                audio=False,
                # audio_codec="aac",
                # audio_codec="libvorbis",
                # write_logfile=True,
            )

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


def procesar_csv(
        list_games,
        partidos_indice,
        json_filename,
):
    # Rutas completas a los archivos CSV de salida
    name_g = "games.csv"
    name_d = "detalle.csv"
    games_csv_path = os.path.join(SOCCERNET_RESULTS, name_g)
    detalle_csv_path = os.path.join(SOCCERNET_RESULTS, name_d)

    # Cabeceras para los CSV
    cabecera_games = ["index", "UrlLocal", "UrlYoutube", "gameAwayTeam", "gameDate", "gameHomeTeam", "gameScore"]
    cabecera_detalle = ["game_index", "gameTime", "label", "position", "team", "visibility"]

    for indice_juego in partidos_indice:
        if not (0 <= indice_juego < len(list_games)):
            logger.error(f"El índice {indice_juego} está fuera del rango de [0 - {len(list_games) - 1}]. Saltando.")
            continue

        game_dir_name = list_games[indice_juego]
        json_file_path = os.path.join(DS_SOCCERNET_RAW, game_dir_name, json_filename)
        # logger.info(f"Procesando juego con índice: {indice_juego}, directorio: {game_dir_name}")

        try:
            # Procesar 'name_g' (solo una vez por juego)
            if not ut.juego_ya_registrado(games_csv_path, indice_juego):
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    game_data = json.load(f)

                # Preparar el registro para 'name_g'
                registro_game = [
                    indice_juego,
                    game_data.get("UrlLocal", ""),
                    game_data.get("UrlYoutube", ""),
                    game_data.get("gameAwayTeam", ""),
                    game_data.get("gameDate", ""),
                    game_data.get("gameHomeTeam", ""),
                    game_data.get("gameScore", "")
                ]
                ut.escribir_csv(games_csv_path, cabecera_games, registro_game)
                logger.info(f"Registro del juego '{game_dir_name}' añadido a '{name_g}'.")
            else:
                logger.info(f"Juego '{game_dir_name}' ya existe en '{name_g}'. Saltando adición.")

            # Procesar 'name_d' (por cada anotación)
            with open(json_file_path, 'r', encoding='utf-8') as f:
                game_data = json.load(f)

            annotations = game_data.get("annotations", [])
            if not annotations:
                logger.warning(f"No se encontraron anotaciones para el juego '{game_dir_name}' en {json_file_path}.")

            for annotation in annotations:
                registro_detalle = [
                    indice_juego,
                    annotation.get("gameTime", ""),
                    annotation.get("label", ""),
                    annotation.get("position", ""),
                    annotation.get("team", ""),
                    annotation.get("visibility", "")
                ]
                ut.escribir_csv(detalle_csv_path, cabecera_detalle, registro_detalle)
            logger.info(f"Anotaciones del juego '{game_dir_name}' añadidas a '{name_d}'.")
        except FileNotFoundError:
            logger.error(f"Archivo JSON no encontrado: {json_file_path}. Saltando este juego.")
        except json.JSONDecodeError:
            logger.error(
                f"Error al decodificar JSON en: {json_file_path}. El archivo podría estar corrupto. Saltando este juego.")
        except Exception as e:
            logger.error(
                f"Ocurrió un error inesperado al procesar el juego '{game_dir_name}' ({json_file_path}): {e}. Saltando este juego.")


def main(args) -> None:
    # configurando descarga de SoccerNet
    file_video = ["1_720p.mkv", "2_720p.mkv"] if args.calidad_video == "720p" else ["1_224p.mkv", "2_224p.mkv"]
    # file_label = ["Labels.json", "Labels-v2.json", "Labels-cameras.json", "Labels-v3.json"]
    file_label = "Labels-v2.json"
    list_games = getListGames(
        split=[
            "train",
            "valid",
            "test",
            # "challenge",  # dataset de retos para nuevos releases
        ]
    )
    result_list = [list_games[index] for index in args.partidos_indice]

    files = [
        file_label,
        file_video[0],
        file_video[1],
    ]

    ut.write_soccernet_games_in_txt(
        list_games,
        os.path.join(SOCCERNET_RESULTS, "SoccerNet_list_games.txt"),
    )
    ut.write_soccernet_games_in_txt(
        result_list,
        os.path.join(SOCCERNET_RESULTS, f"SoccerNet_list_games_requested_{time.time()}.txt"),
    )

    if args.omitir_descarga == 0:
        t_start = time.time()

        for game in result_list:
            # 'downloadGame' solo crea el directorio del 'game' actual
            downloader.downloadGame(game=game, files=files)

        ut.get_time_employed(t_start, "Descarga de archivos desde SoccerNet.")

    if args.omitir_recorte == 0:
        t_start = time.time()

        for game in result_list:
            # time.sleep(4.0)
            for video in file_video:
                # for action in actions:
                saved_clips = cut_video_segments_by_label(
                    directory=os.path.join(DS_SOCCERNET_RAW, str(game).replace('\\', '/')),
                    video_filename=video,
                    json_filename=file_label,
                    # label_to_find=action,
                    mas_menos=1.5,
                )

        ut.get_time_employed(t_start, "Generación de videoclips por cada acción.")

    # crear archivos csv para EDA
    procesar_csv(
        list_games=list_games,
        partidos_indice=args.partidos_indice,
        json_filename=file_label,
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Descargar archivos de SoccerNet y recortar acciones etiquetadas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--partidos_indice", default=[0], type=ut.non_negative_int, nargs="+",
                        help="Índice de cada partido que desea descargar, valores enteros entre 0 y 500.")
    parser.add_argument("--calidad_video", default="224p", type=str, choices=["224p", "720p"],
                        help="Tamaño del video a utilizar.")
    parser.add_argument("--omitir_descarga", default=0, type=int, choices=[0, 1],
                        help="Omite el proceso de descarga en SoccerNet. 0: apagado, 1: prendido.")
    parser.add_argument("--omitir_recorte", default=0, type=int, choices=[0, 1],
                        help="Omite el proceso de recortar las acciones en los videos de SoccerNet. 0: apagado, 1: prendido.")

    return parser.parse_args()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        args = parse_arguments()
    else:
        # Valores por defecto si no se proporcionan argumentos desde la línea de comandos
        class Args:
            partidos_indice = ut.obtener_numeros(PARTIDOS_INDICE_LOTE[1])
            # calidad_video = "224p"
            calidad_video = "720p"
            omitir_descarga = 0
            omitir_recorte = 0


        args = Args()

    config_log()
    ut.verify_system()
    main(args)
