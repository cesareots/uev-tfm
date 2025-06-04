import argparse
import logging
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
    # Confirmación antes de ejecutar
    respuesta = input(f"Se van a borrar archivos .mkv de los directorios. Esta acción es IRREVERSIBLE. ¿Estás seguro? (escribe 'SI' para confirmar): ")
    
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
        #default="prueba_de_borrado.txt",
        #default="SoccerNet_list_games_requested_1748594477.9772956.txt",  # TODO
        default="SoccerNet_list_games_requested_1748596402.292449.txt",  # TODO
        type=str,
        help="Archivo .txt que contiene los directorios a recorrer.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    config_log()
    ut.verify_system()
    main(args)
