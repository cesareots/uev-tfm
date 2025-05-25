import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.preprocess.dataset_soccernet import DatasetSoccernet, CLIP_DURATION_SEC,FRAMES_PER_CLIP,TARGET_FPS
from src.utils import utils as ut
from src.utils.constants import *

logger = logging.getLogger(__name__)

# reproducibilidad
# random.seed(SEMILLA)  # TODO en que casos se usa?
torch.manual_seed(SEMILLA)


def config_log() -> None:
    ut.verify_directory(LOG_DIR)

    logging.basicConfig(
        filename=LOG_SOCCERNET_TENSORS,
        filemode="a",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        # https://docs.python.org/3/library/logging.html
        level=logging.INFO,  # NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL
        encoding="utf-8",
    )


# Función para Preprocesar y Guardar Tensores
# Esta función NO DEBERIA aplicar DATA AUGMENTATION porque guardaria datos aumentados fijos.
# Solo aplica las transformaciones deterministas (muestreo, redimensionamiento fijo).
def preprocess_and_save_dataset(
        args,
        input_root_dir: str,
        output_root_dir: str,
        label_map: dict,
        frames_per_clip: int,
        target_fps: float,
        target_size: tuple = None
) -> None:
    """
    Carga videos MKV, los preprocesa a tensores PyTorch y los guarda en un nuevo directorio.
    Esta función NO aplica transformaciones aleatorias de Data Augmentation.

    Args:
        input_root_dir (str): Directorio raíz de los videos MKV originales.
        output_root_dir (str): Directorio donde se guardarán los tensores PyTorch preprocesados.
        label_map (dict): Mapeo de nombres de acción a IDs numéricos.
        frames_per_clip (int): Número deseado de frames por clip después del muestreo.
        target_fps (float): FPS al que se decodificarán los videos (referencia para muestreo).
        target_size (tuple, optional): (height, width) deseado para los frames. Si es None, se usa el original.
    """
    logger.info(f"Iniciando el preprocesamiento y guardado de tensores")
    logger.info(f"args.video_size: {args.video_size}")  # TODO implementar logica? para procesar clips de 240p o 720p
    logger.info(f"Videos de entrada: {input_root_dir}")
    logger.info(f"Tensores de salida: {output_root_dir}")

    # Aquí, el dataset se inicializa SIN transformaciones aleatorias.
    # Solo se aplica el redimensionamiento si 'target_size' es diferente al tamaño original.
    source_dataset = DatasetSoccernet(
        root_dir=input_root_dir,
        label_map=label_map,
        frames_per_clip=frames_per_clip,
        target_fps=target_fps,
        target_size=target_size,  # Pasa el target_size para el redimensionamiento determinista
        transform=None,  # TODO NO debería aplicar data augmentation aqui
    )

    if len(source_dataset) == 0:
        logger.error("No se encontraron videos para preprocesar. Finalizando...")
        return

    source_dataloader = DataLoader(
        source_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )

    os.makedirs(output_root_dir, exist_ok=True)
    processed_count = 0
    skipped_count = 0

    for i, (video_tensor, label, original_path_list) in enumerate(source_dataloader):
        original_path = Path(original_path_list[0])

        if label.item() == -1:
            logger.error(f"Saltando video debido a error de carga/procesamiento: '{original_path.name}'")
            skipped_count += 1
            continue

        action_name_str = original_path.parent.name
        output_class_dir = Path(output_root_dir) / action_name_str
        os.makedirs(output_class_dir, exist_ok=True)

        tensor_filename = original_path.stem + "_processed.pt"
        output_filepath = output_class_dir / tensor_filename

        try:
            torch.save(video_tensor.squeeze(0), output_filepath)
            processed_count += 1
            logger.info(f"Tensor guardado: '{output_filepath}'")
        except Exception as e:
            logger.error(f"Al guardar '{output_filepath}': {e}")
            skipped_count += 1

    logger.info(f"Videos procesados y guardados: {processed_count}")
    logger.info(f"Videos saltados (errores): {skipped_count}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Genera tensores en formato PyTorch a partir de los videoclips.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--video_size", default="224p", type=str, choices=["224p", "720p"],
                        help="Tamaño de los clips que se utilizarán.")

    return parser.parse_args()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        args = parse_arguments()
    else:
        # Valores por defecto si no se proporcionan argumentos desde la línea de comandos
        class Args:
            #video_size = "224p"
            video_size = "720p"


        args = Args()

    config_log()
    ut.verify_system()
    preprocess_and_save_dataset(
        args=args,
        input_root_dir=DS_SOCCERNET_ACTIONS,
        output_root_dir=DS_SOCCERNET_TENSORS,
        label_map=SOCCERNET_LABELS,
        frames_per_clip=FRAMES_PER_CLIP,
        target_fps=TARGET_FPS,
        target_size=None  # None para usar la resolución original definida en 'dataset_soccernet'
        # Si tus videos de 720p se procesarán, podrías poner (128, 128)
        # o (224, 224) si quieres un redimensionamiento determinista.
        # Por ejemplo: target_size=(224, 224)
    )
