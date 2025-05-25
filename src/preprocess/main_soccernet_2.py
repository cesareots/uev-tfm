import argparse
import glob
import json
import logging
import sys
import os

from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.utils import getListGames
from moviepy import VideoFileClip # Asegúrate de que moviepy esté instalado (pip install moviepy)

from src.utils import utils as ut # Asumo que src/utils/utils.py existe y es correcto
from src.utils.constants import * # Asumo que src/utils/constants.py existe y es correcto

logger = logging.getLogger(__name__)
downloader = SoccerNetDownloader(LocalDirectory=DS_SOCCERNET_RAW)
downloader.password = SOCCERNET_PASSWORD

FILE_VIDEO_224p = ["1_224p.mkv", "2_224p.mkv"]
FILE_VIDEO_720p = ["1_720p.mkv", "2_720p.mkv"]
file_labels = ["Labels.json", "Labels-v2.json", "Labels-cameras.json", "Labels-v3.json"]

# Define tus acciones aquí (como lo tenías en tu script original)
SOCCERNET_LABELS = {
    "Goal": 0,
    "Yellow card": 1,
    "Shots on target": 2,
}

ACCIONES_RECORTAR = ut.extraer_claves_ordenadas(SOCCERNET_LABELS)

FILES = [
    file_labels[0],
    file_labels[1],
    file_labels[2],
    FILE_VIDEO_224p[0],
    FILE_VIDEO_224p[1],
    FILE_VIDEO_720p[0],
    FILE_VIDEO_720p[1],
]

SPLIT = [
    "train",
    "valid",
    "test",
]


def config_log() -> None:
    ut.verify_directory(LOG_DIR)

    logging.basicConfig(
        filename=LOG_SOCCERNET,
        filemode="a",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO, # Mantener en INFO, pero los mensajes ahora serán más detallados
        encoding="utf-8",
    )


def cut_video_segments_by_label(
        directory: str,
        video_filename: str,
        json_filename: str,
        mas_menos: float = 2.0,
) -> list:
    """
    Busca anotaciones en un archivo JSON para todas las ACCIONES_RECORTAR,
    calcula los frames correspondientes en un video y extrae segmentos
    de video alrededor de esos puntos (+- 2 segundos).
    Si un 'gameTime' se repite para un mismo tiempo-partido (sin importar la etiqueta),
    solo se considera la primera aparición.

    Args:
        directory (str): Ruta al directorio que contiene el video y el JSON.
        video_filename (str): Nombre del archivo de video (formato mkv).
        json_filename (str): Nombre del archivo JSON de anotaciones.
        mas_menos (float): +- N segundos para recortar el video a partir de 'label_to_find'

    Returns:
        list: Una lista de las rutas completas de los archivos de video cortados guardados.
              Retorna una lista vacía si no se encontraron anotaciones con los labels
              o si ocurrió un error.
    """
    video_path = os.path.join(directory, video_filename)
    json_path = os.path.join(directory, json_filename)
    saved_clip_paths = []

    logger.info(f"--- Iniciando procesamiento para video: '{video_path}' con JSON: '{json_path}' ---")

    # Validaciones iniciales
    if not os.path.isdir(directory):
        logger.error(f"ERROR: Directorio no encontrado: '{directory}'")
        return []

    if not os.path.exists(video_path):
        logger.error(f"ERROR: Video no encontrado: '{video_path}'")
        return []

    if not os.path.exists(json_path):
        logger.error(f"ERROR: JSON no encontrado: '{json_path}'")
        return []

    # cargar y parsear JSON
    annotations = []
    try:
        with open(json_path, 'r', encoding='utf-8') as f: # Añadido encoding
            data = json.load(f)
        if not isinstance(data, dict) or \
           "annotations" not in data or \
           not isinstance(data["annotations"], list):
            logger.error(f"ERROR: El archivo JSON '{json_path}' no tiene el formato esperado (debe contener una clave 'annotations' que sea una lista).")
            return []
        annotations = data["annotations"]
        logger.info(f"JSON '{json_path}' cargado exitosamente. Se encontraron {len(annotations)} anotaciones.")
    except json.JSONDecodeError as e:
        logger.error(f"ERROR: JSON no válido en '{json_path}': {e}")
        return []
    except Exception as e:
        logger.error(f"ERROR inesperado al cargar o parsear JSON '{json_path}': {e}")
        return []

    gt_prefix = video_filename.split("_")[0] # '1' o '2' para el primer o segundo tiempo

    # Set para llevar un registro de los gameTime (parte - MM:SS) ya procesados para CUALQUIER label
    processed_gametimes_for_video = set()
    all_matching_annotations = [] # Para almacenar todas las anotaciones únicas encontradas, sin importar el label

    # Ahora, iteramos por cada label que queremos recortar
    for label_to_find in ACCIONES_RECORTAR:
        for annotation in annotations:
            # Solo procesar anotaciones que coincidan con el label actual del bucle
            if isinstance(annotation, dict) and annotation.get("label") == label_to_find:
                
                game_time_full_str = annotation.get("gameTime")
                if not game_time_full_str:
                    logger.warning(f"ADVERTENCIA: Anotación sin 'gameTime' para label '{label_to_find}'. Saltando: {annotation}")
                    continue

                game_time_parts = game_time_full_str.split(" - ")
                
                if len(game_time_parts) < 2 or game_time_parts[0] != gt_prefix:
                    # Esto es un filtro importante: asegura que la anotación corresponde al video (primer o segundo tiempo)
                    # No es un error, es un filtro esperado para que solo las anotaciones de la parte del video actual sean consideradas.
                    continue 

                time_part_mm_ss = game_time_parts[-1] # MM:SS
                unique_identifier = f"{gt_prefix}-{time_part_mm_ss}" # Ej: '1-07:15'

                if unique_identifier in processed_gametimes_for_video:
                    logger.warning(
                        f"ADVERTENCIA: GameTime '{game_time_full_str}' (label '{label_to_find}') ya procesado para video '{video_filename}'. IGNORANDO esta ocurrencia."
                    )
                    print(
                        f"IGNORANDO (gameTime duplicado): GameTime='{game_time_full_str}', Label='{label_to_find}', Video='{video_filename}'"
                    )
                    continue
                
                all_matching_annotations.append(annotation)
                processed_gametimes_for_video.add(unique_identifier)
                logger.info(f"Anotación ÚNICA encontrada y añadida: '{label_to_find}' en '{game_time_full_str}'.")

    # Ordenar las anotaciones por gameTime para asegurar un orden consistente en los clips
    all_matching_annotations.sort(key=lambda x: (int(x["gameTime"].split(" - ")[0]),
                                                 int(x["gameTime"].split(" - ")[1].split(":")[0]),
                                                 int(x["gameTime"].split(" - ")[1].split(":")[1])))


    if not all_matching_annotations:
        logger.info(f"No se encontraron anotaciones únicas y válidas para recortar en el video '{video_filename}'.")
        return []

    logger.info(f"Se encontraron {len(all_matching_annotations)} anotaciones únicas para el video '{video_filename}' después del filtrado.")

    # cargar el video para obtener su duración
    # video_clip = None # Esta línea ya no es necesaria con 'with'
    try:
        # Usar un contexto manager para asegurar el cierre del video_clip
        with VideoFileClip(video_path) as video_clip:
            video_duration = video_clip.duration  # Duración en segundos
            logger.info(f"Video '{video_filename}' cargado. Duración: {video_duration:.2f} segundos.")
            
            # Mueve el resto del procesamiento de clips dentro de este 'with' block
            # para que 'video_clip' esté disponible.

            # procesar cada anotación coincidente y cortar el video
            for annotation in all_matching_annotations:
                label_for_clip = annotation.get("label")
                game_time_str = annotation.get("gameTime")

                # Crear el directorio de salida para la acción actual
                output_dir = os.path.join(DS_SOCCERNET_ACTIONS, label_for_clip)
                os.makedirs(output_dir, exist_ok=True)

                existing_files = glob.glob(os.path.join(output_dir, "*.mkv"))
                segment_index = len(existing_files) + 1

                time_part = game_time_str.split(' - ')[-1]

                try:
                    minutes, seconds = map(int, time_part.split(':'))
                    total_seconds = minutes * 60 + seconds
                except ValueError:
                    logger.warning(
                        f"ADVERTENCIA: Formato de 'gameTime' inesperado '{game_time_str}' (label '{label_for_clip}'). Saltando anotación.")
                    continue

                cut_start_time = max(0.0, total_seconds - mas_menos)
                cut_end_time = min(video_duration, total_seconds + mas_menos)

                if cut_end_time <= cut_start_time:
                    logger.warning(
                        f"ADVERTENCIA: Ventana de corte para '{label_for_clip}' en '{game_time_str}' es inválida ({cut_start_time:.2f}s a {cut_end_time:.2f}s). Saltando.")
                    continue

                logger.info(f"Cortando '{label_for_clip}' en '{game_time_str}'. Ventana: {cut_start_time:.2f}s a {cut_end_time:.2f}s")

                try:
                    subclip = video_clip.subclipped(cut_start_time, cut_end_time)
                    
                    # Nombre de archivo más descriptivo para depuración
                    game_id = os.path.basename(directory) # Obtiene solo el nombre del partido
                    output_filename = f"{label_for_clip}_{segment_index:04d}_{game_id}_{game_time_str.replace(' - ', '_').replace(':', '')}.mkv"
                    output_path = os.path.join(output_dir, output_filename)

                    subclip.write_videofile(output_path, codec='libx264', audio_codec='aac', logger=None) # logger=None para moviepy para no saturar la salida
                    
                    logger.info(f"Segmento guardado exitosamente: '{output_path}'")
                    saved_clip_paths.append(output_path)
                    
                    # Cierra el subclip explícitamente si MoviePy no lo hace automáticamente
                    # Esto podría ayudar a liberar recursos de FFMPEG
                    subclip.close()

                except Exception as e:
                    logger.error(f"ERROR: Al procesar o guardar el segmento para '{label_for_clip}' en '{game_time_str}': {e}", exc_info=True)
                    # Aquí es crucial el exc_info=True para ver el traceback de MoviePy/FFMPEG

            # video_clip.close() # Esta línea ya no es necesaria gracias al 'with' statement
            logger.info(f"--- Procesamiento completado para video '{video_filename}'. Total clips guardados: {len(saved_clip_paths)} ---")

    except Exception as e:
        logger.error(f"ERROR: No se pudo cargar el video '{video_path}' con moviepy (o un error general durante el procesamiento del video): {e}", exc_info=True)
        return []

    return saved_clip_paths


def download_files(
        list_games: list,
        cantidad: int = 1,
) -> None:
    ut.write_soccernet_games_in_txt(list_games, os.path.join(SOCCERNET_RESULTS, "SoccerNet_list_games.txt"))

    for i in range(cantidad):
        game_to_download = list_games[i]
        logger.info(f"Intentando descargar partido (cantidad): {game_to_download}")
        try:
            downloader.downloadGame(game=game_to_download, files=FILES)
            logger.info(f"Descarga exitosa para {game_to_download}")
        except Exception as e:
            logger.error(f"ERROR al descargar partido {game_to_download}: {e}")


def download_files_by_index(
        list_games: list,
        partidos_seleccionados: list = [0],
) -> None:
    ut.write_soccernet_games_in_txt(list_games, os.path.join(SOCCERNET_RESULTS, "SoccerNet_list_games.txt"))
    result_list = [list_games[index] for index in partidos_seleccionados]
    ut.write_soccernet_games_in_txt(result_list,
                                    os.path.join(SOCCERNET_RESULTS, "SoccerNet_list_games_downloaded_by_index.txt"))

    for game in result_list:
        logger.info(f"Intentando descargar partido (por índice): {game}")
        try:
            downloader.downloadGame(game=game, files=FILES)
            logger.info(f"Descarga exitosa para {game}")
        except Exception as e:
            logger.error(f"ERROR al descargar partido {game}: {e}")


def cut_video_loop(
        games: list,
        videos: list,
) -> None:
    # recorro todos los directorios
    for game_idx, game_path_suffix in enumerate(games):
        game_full_path = os.path.join(DS_SOCCERNET_RAW, str(game_path_suffix).replace('\\', '/'))
        
        logger.info(f"--- Procesando partido {game_idx+1}/{len(games)}: '{game_path_suffix}' ---")

        if not os.path.exists(game_full_path):
            logger.warning(f"ADVERTENCIA: Directorio del partido no encontrado, saltando: '{game_full_path}'")
            continue # Salta a la siguiente iteración si el directorio del partido no existe

        for video_filename in videos:
            # cut_video_segments_by_label maneja todas las ACCIONES_RECORTAR internamente
            try:
                saved_clips = cut_video_segments_by_label(
                    directory=game_full_path,
                    video_filename=video_filename,
                    json_filename=file_labels[1], # Asumiendo que Labels-v2.json es el que usas
                )
                logger.info(f"Finalizado el procesamiento del video '{video_filename}' para el partido '{game_path_suffix}'. Clips guardados: {len(saved_clips)}")
            except Exception as e:
                logger.critical(f"ERROR CRÍTICO: Falló el procesamiento de '{video_filename}' en '{game_path_suffix}'. Error: {e}", exc_info=True)
                # exc_info=True para imprimir el traceback completo
                # No hacemos 'continue' aquí, la función interna ya lo maneja y devuelve una lista vacía.
                # Si una excepción se propaga hasta aquí, es un fallo mayor.


def main(args) -> None:
    list_games = getListGames(split=SPLIT)

    if args.omitir_descarga == 0:
        if args.partidos_cantidad > 0:
            download_files(list_games, args.partidos_cantidad)
        else:
            # Usar los índices del archivo de texto adjunto para probar
            # Asegúrate que tus índices coincidan con los de getListGames()
            selected_indices = args.partidos_indice 
            # Los indices que me diste en el txt son:
            # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499

            # Para propósitos de depuración, puedes leer tu archivo adjunto de indices:
            # with open('SoccerNet_list_games_downloaded_by_index.txt', 'r') as f:
            #     downloaded_game_paths = [line.strip() for line in f if line.strip()]
            #     # Ahora necesitas mapear estas rutas de vuelta a sus índices en la lista completa
            #     # Esto puede ser complicado si getListGames() no mantiene el orden exacto.
            #     # La forma más segura es tener una lista de los *índices* que realmente se descargaron.
            #     # Por ejemplo, si tus indices_ejemplo fueron 0, 1, 29, 58, etc., asegúrate de que esa lista sea `args.partidos_indice`.
            #     # Por ahora, usaré tu lista de índices del default args.
            
            download_files_by_index(list_games, selected_indices) # Aquí deberías usar los índices de los 68 partidos descargados.

    if args.omitir_recorte == 0:
        # En la lista de `list_games`, solo los partidos que tienen sus directorios descargados
        # serán realmente procesados por `cut_video_loop`.
        # La función `cut_video_loop` ahora tiene una verificación para `os.path.exists(game_full_path)`.
        
        # Filtrar list_games para solo incluir los partidos que esperas tener descargados
        # Basándome en los indices que me diste en el main_soccernet.py.
        # Ajusta esta lista `games_to_process` si tus partidos_indice varían o si quieres procesar un subconjunto específico.
        games_to_process = [list_games[idx] for idx in args.partidos_indice] 
        
        if args.calidad_video == "224p":
            cut_video_loop(games_to_process, FILE_VIDEO_224p)
        elif args.calidad_video == "720p":
            cut_video_loop(games_to_process, FILE_VIDEO_720p)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Descargar archivos de SoccerNet y recortar acciones etiquetadas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--omitir_descarga", default=0, type=int, choices=[0, 1],
                        help="Omite el proceso de descarga en SoccerNet. 0: apagado, 1: prendido.")
    parser.add_argument("--partidos_cantidad", default=0, type=ut.non_negative_int,
                        help="Descarga los primeros N partidos, valor entero entre 0 y 500.")
    parser.add_argument("--partidos_indice", default=[
                0, 1, 2, 3,
                4, 5, 6, 7,
                29, 30, 31, 32,
                58, 59, 60, 61,
                80, 81, 82, 83,
                109, 110, 111, 112,
                120, 121, 122, 339, # Aquí está el indice 339
                123, 124, 125, 126,
                147, 148, 149, 150,
                152, 153, 154, 155,
                162, 163, 164, 165,
                178, 179, 180, 181,
                185, 186, 187, 188, # Italia
                190, 191, 192, 193, # Italia (los 4 últimos de tu lista adjunta)
                234, 235, 236, 237,
                240, 241, 242, 243,
                258, 259, 260, 261,
            ], type=ut.non_negative_int, nargs="+",
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
        class Args:
            omitir_descarga = 1 # Asume que ya están descargados
            partidos_cantidad = 0
            partidos_indice = [
                # Copia los índices exactos de los 68 partidos que tienes descargados.
                # Esta es la lista que me pasaste en tu main_soccernet.py
                0, 1, 2, 3,  # england_epl/2014-2015/
                4, 5, 6, 7,  # england_epl/2015-2016/
                29, 30, 31, 32,  # england_epl/2016-2017/
                58, 59, 60, 61,  # europe_uefa-champions-league/2014-2015/
                80, 81, 82, 83,  # europe_uefa-champions-league/2015-2016/
                109, 110, 111, 112,  # europe_uefa-champions-league/2016-2017/
                120,  # france_ligue-1/2014-2015/
                121, 122, 339,  # france_ligue-1/2015-2016/
                123, 124, 125, 126,  # france_ligue-1/2016-2017/
                147, 148, 149, 150,  # germany_bundesliga/2014-2015/
                152, 153, 154, 155,  # germany_bundesliga/2015-2016/
                162, 163, 164, 165,  # germany_bundesliga/2016-2017/
                178, 179, 180, 181,  # italy_serie-a/2014-2015/
                185, 186, 187, 188,  # italy_serie-a/2015-2016/
                190, 191, 192, 193,  # italy_serie-a/2016-2017/
                234, 235, 236, 237,  # spain_laliga/2014-2015/
                240, 241, 242, 243,  # spain_laliga/2015-2016/
                258, 259, 260, 261,  # spain_laliga/2016-2017/
            ]
            omitir_recorte = 0
            calidad_video = "224p"
        args = Args()

    config_log()
    ut.verify_system()
    main(args)
