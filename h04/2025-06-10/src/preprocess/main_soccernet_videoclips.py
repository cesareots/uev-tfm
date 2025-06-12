import argparse
import logging
import sys
from typing import List

from SoccerNet.utils import getListGames
from moviepy import VideoFileClip

from src.utils import utils as ut
from src.utils.constants import *

logger = logging.getLogger(__name__)

SPLIT = [
    "train",
    "valid",
    "test",
    # "challenge",  # dataset de retos para nuevos releases
]


def config_log() -> None:
    ut.verify_directory(LOG_DIR)

    logging.basicConfig(
        filename=LOG_SOCCERNET_CLIPS,
        filemode="a",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        # https://docs.python.org/3/library/logging.html
        level=logging.INFO,  # NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL
        encoding="utf-8",
    )


def cut(
        directory: str,
        corte_inicial: float = 0.0,
        longitud: float = 10.0,
        output_suffix: str = "_trimmed",
) -> List[str]:
    """
    Corta todos los videos (.mkv) encontrados en un directorio desde un tiempo inicial
    y con una longitud deseada, guardándolos con un nuevo nombre.

    Args:
        directory (str): Ruta al directorio donde buscar los videos.
        corte_inicial (float): Tiempo de inicio del corte en minutos. Por defecto es 0.0 (inicio del video).
        longitud (float): Longitud del segmento a cortar en MINUTOS. Por defecto es 10.0.
        output_suffix (str): Sufijo a añadir al nombre del archivo original para el video cortado.
                             Por defecto es '_trimmed'.

    Returns:
        List[str]: Una lista de las rutas completas de los videos cortados.
    """
    if not os.path.isdir(directory):
        # logger.error(f"Directorio no encontrado: '{directory}'")
        return []

    if corte_inicial < 0.0:
        logger.error(f"Corte inicial ({corte_inicial} minuto) del clip no puede ser negativo.")
        return []

    if longitud < 0.0:
        logger.error(f"La longitud ({longitud} minutos) del clip no puede ser negativo.")
        return []

    corte_inicial_seconds = corte_inicial * 60.0
    time_len_seconds = longitud * 60.0
    cut_video_paths = []

    for filename in os.listdir(directory):
        if filename.lower().endswith(".mkv"):
            original_filepath = os.path.join(directory, filename)

            # Crear el nuevo nombre de archivo
            name_without_ext = os.path.splitext(filename)[0]
            output_filename = f"{name_without_ext}{output_suffix}.mkv"
            output_filepath = os.path.join(directory, output_filename)

            logger.info(f"Procesando '{original_filepath}'")
            logger.info(f"Corte desde el minuto: {corte_inicial:.2f}, Duración: {longitud:.2f} minutos")
            # logger.info(f"Se guardará como '{output_filename}'")

            try:
                with VideoFileClip(original_filepath) as video:
                    if video.duration < (45 * 60.0):
                        logger.warning(
                            f"Al parecer '{filename}' NO es un video de soccernet. Su duración es menor a 45.00 minutos. Saltando.")
                        continue

                    if corte_inicial_seconds >= video.duration:
                        logger.warning(
                            f"El tiempo de corte inicial ({corte_inicial:.2f}) excede la duración del video ({(video.duration / 60.0):.2f} minutos). Saltando.")
                        continue

                    # Calcular el tiempo final del corte
                    end_time = corte_inicial_seconds + time_len_seconds

                    # Asegurarse de no exceder la duración original del video
                    if end_time > video.duration:
                        logger.warning(
                            f"La duración deseada ({longitud:.2f} minutos) excede el final del video. Cortando hasta el final (minuto {(video.duration / 60.0):.2f}).")
                        end_time = video.duration

                    # Realizar el corte
                    cut_clip = video.subclipped(corte_inicial_seconds, end_time)
                    cut_clip.write_videofile(
                        output_filepath,
                        codec="libx264",
                        audio_codec="aac",
                    )
                    cut_video_paths.append(output_filepath)
                    logger.info(f"'{output_filename}' generado.")
            except Exception as e:
                logger.error(f"Al cortar '{filename}': {e}")
                logger.error(f"Asegúrate de que el video no esté corrupto y que ffmpeg esté instalado y en el PATH.")

    return cut_video_paths


def main(args) -> None:
    list_games = getListGames(split=SPLIT)
    result_list = [list_games[index] for index in args.partidos_indice]

    for game in result_list:
        # print(game)
        # videos = glob.glob(os.path.join(DS_SOCCERNET, game, "*.mkv"))
        # print(videos)
        saved_clips = cut(
            directory=os.path.join(DS_SOCCERNET_RAW, str(game).replace('\\', '/')),
            corte_inicial=args.corte_inicial,
            longitud=args.longitud,
        )
        # print(f"\nsaved_clips:\n{saved_clips}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Genera clips de cada video-raw, a partir de cualquier minuto y con una longitud deseada.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--corte_inicial", default=0.0, type=ut.non_negative_float,
                        help="Corte incial en el minuto X del video-raw.")
    parser.add_argument("--longitud", default=10.0, type=ut.non_negative_float,
                        help="Duración en minutos del clip.")
    parser.add_argument("--partidos_indice", default=[0], type=ut.non_negative_int, nargs="+",
                        help="Índice de cada partido que desea generar los clips, valores enteros entre 0 y 500.")

    return parser.parse_args()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        args = parse_arguments()
    else:
        # Valores por defecto si no se proporcionan argumentos desde la línea de comandos
        class Args:
            corte_inicial = 1.0
            longitud = 1.0
            partidos_indice = [
                0,  # 1,  # england_epl/2014-2015/
                # 4, 5,  # england_epl/2015-2016/
                # 29, 30,  # england_epl/2016-2017/
                # 58, 59,  # europe_uefa-champions-league/2014-2015/
                # 80, 81,  # europe_uefa-champions-league/2015-2016/
                # 109, 110,  # europe_uefa-champions-league/2016-2017/
                # 120,  # france_ligue-1/2014-2015/
                # 121, 122,  # france_ligue-1/2015-2016/
                # 123, 124,  # france_ligue-1/2016-2017/
                # 147, 148,  # germany_bundesliga/2014-2015/
                # 152, 153,  # germany_bundesliga/2015-2016/
                # 162, 163,  # germany_bundesliga/2016-2017/
                # 178, 179,  # italy_serie-a/2014-2015/
                # 185, 186,  # italy_serie-a/2015-2016/
                # 190, 191,  # italy_serie-a/2016-2017/
                # 234, 235,  # spain_laliga/2014-2015/
                # 240, 241,  # spain_laliga/2015-2016/
                # 258, 259,  # spain_laliga/2016-2017/
            ]


        args = Args()

    config_log()
    ut.verify_system()
    main(args)
