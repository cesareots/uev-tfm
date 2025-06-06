import argparse
import logging
import sys
from pathlib import Path

from src.utils import utils as ut
from src.utils.constants import *

logger = logging.getLogger(__name__)


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


def borrar_videos_en_directorios(
        dir_root: str,
        lista_directorios_games: list,
):
    if not lista_directorios_games:
        logger.warning("La lista de directorios de juegos está vacía. No se borrará nada.")
        return

    logger.info("Iniciando proceso de borrado de archivos .mkv...")
    logger.warning("ESTA ACCIÓN ES IRREVERSIBLE Y BORRARÁ LOS ARCHIVOS PERMANENTEMENTE.")

    archivos_borrados_count = 0
    archivos_fallidos_count = 0

    for game_dir_str in lista_directorios_games:
        game_dir = Path(dir_root) / game_dir_str

        if not game_dir.is_dir():
            logger.warning(f"El directorio '{game_dir_str}' no existe o no es un directorio. Saltando.")
            continue

        logger.info(f"Procesando directorio: '{game_dir_str}'")

        mkv_files_encontrados = list(game_dir.glob('*.mkv'))

        if not mkv_files_encontrados:
            logger.info(f"No se encontraron archivos .mkv en '{game_dir_str}'.")
            continue

        for mkv_file_path in mkv_files_encontrados:
            try:
                mkv_file_path.unlink()  # Borrado definitivo del archivo
                logger.info(f"Archivo borrado: '{mkv_file_path}'")
                archivos_borrados_count += 1
            except FileNotFoundError:
                logger.error(f"Archivo no encontrado: '{mkv_file_path}'")
                archivos_fallidos_count += 1
            except PermissionError:
                logger.error(f"Permiso denegado: '{mkv_file_path}'")
                archivos_fallidos_count += 1
            except Exception as e:
                logger.error(f"'{mkv_file_path}': {str(e)}")
                archivos_fallidos_count += 1

    logger.info("Resumen del Proceso de Borrado")
    logger.info(f"Directorios procesados: {len(lista_directorios_games)}")
    logger.info(f"Archivos .mkv borrados exitosamente: {archivos_borrados_count}")

    if archivos_fallidos_count > 0:
        logger.error(f"Archivos .mkv que no se pudieron borrar: {archivos_fallidos_count}")

    log_con = "Proceso de borrado finalizado."
    print(log_con)
    logger.info(log_con)


def main(args):
    if args.dir is None:
        logger.warning(f"Falta indicar el archivo .txt conteniendo las rutas de los partidos que desea eliminar.")
        sys.exit(1)

    # Confirmación antes de ejecutar
    respuesta = input(
        f"Se van a borrar archivos .mkv de los directorios. Esta acción es IRREVERSIBLE. ¿Estás seguro? (escribe 'SI' para confirmar): ")

    if respuesta == "SI":
        dir = Path(SOCCERNET_RESULTS) / args.dir
        # print(dir)
        juegos_a_eliminar = ut.leer_registros_txt(dir)
        # print(len(juegos_a_eliminar))
        # print(juegos_a_eliminar)
        borrar_videos_en_directorios(DS_SOCCERNET_RAW, juegos_a_eliminar)
    else:
        print("Proceso de borrado cancelado por el usuario.")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Eliminar videos .mkv desde una lista de directorios.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dir",
        default=None,
        # default="SoccerNet_list_games_requested_1748594477.9772956.txt",  # TODO lote 0
        # default="SoccerNet_list_games_requested_1748596402.292449.txt",  # TODO lote 1
        # default="SoccerNet_list_games_requested_1749074829.8473907.txt",  # TODO lote 2
        # default="SoccerNet_list_games_requested_1749088675.5337903.txt",  # TODO lote 3
        # default="SoccerNet_list_games_requested_1749128377.2351687.txt",  # TODO lote 4
        # default="SoccerNet_list_games_requested_1749137328.8384361.txt",  # TODO lote 5
        # default="SoccerNet_list_games_requested_1749163068.419272.txt",  # TODO lote 6
        # default="SoccerNet_list_games_requested_1749176683.4895058.txt",  # TODO lote 7
        # default="SoccerNet_list_games_requested_1749188322.544813.txt",  # TODO lote 8
        # default="SoccerNet_list_games_requested_1749192169.2504919.txt",  # TODO lote 9
        # default="SoccerNet_list_games_requested_1749195880.9138103.txt",  # TODO lote 10
        # default="SoccerNet_list_games_requested_1749196097.900653.txt",  # TODO lote 11
        # default="SoccerNet_list_games_requested_1749199284.1203258.txt",  # TODO lote 12
        # default="SoccerNet_list_games_requested_1749199921.2614012.txt",  # TODO lote 13
        # default="SoccerNet_list_games_requested_1749212399.5182917.txt",  # TODO lote 14
        # default="SoccerNet_list_games_requested_1749213361.2971253.txt",  # TODO lote 15
        type=str,
        help="Archivo .txt que contiene los directorios a recorrer.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    config_log()
    ut.verify_system()
    main(args)
