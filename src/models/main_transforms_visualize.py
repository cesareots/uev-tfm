import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.models.video import R2Plus1D_18_Weights

from src.models.main_cnn3d import TARGET_SIZE_DATASET
from src.models.transforms import get_transforms_resnet, get_transforms_cnn3d_rgb, get_transforms_cnn3d_grayscale
from src.preprocess.dataset_soccernet import DatasetSoccernet, SOCCERNET_LABELS, INV_LABEL_MAP
from src.utils import utils as ut
from src.utils.constants import DS_SOCCERNET_ACTIONS
from src.utils.constants import LOG_DIR, LOG_TRANSFORMS_VISUALIZE, RESULTS

logger = logging.getLogger(__name__)


def config_log() -> None:
    ut.verify_directory(LOG_DIR)

    logging.basicConfig(
        filename=LOG_TRANSFORMS_VISUALIZE,
        filemode="a",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        # https://docs.python.org/3/library/logging.html
        level=logging.INFO,  # NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL
        encoding="utf-8",
    )


def unnormalize_tensor(tensor: torch.Tensor, mean, std):
    """
    Revierte la normalización de un tensor para poder visualizarlo.
    """
    # Clonar para no modificar el tensor original
    tensor = tensor.clone()
    # La fórmula inversa es: original = (normalizado * std) + mean
    # Iteramos sobre cada canal
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)

    return tensor


def visualize_and_save_clip(
        video_tensor: torch.Tensor,
        save_path: Path,
        mean=None,
        std=None,
        num_frames_to_show: int = 4,
        title: str = "Frames Transformados"
):
    """
    Toma un tensor de vídeo (C, T, H, W), lo desnormaliza (si es necesario),
    y guarda una visualización de varios de sus frames en disco.
    """
    logger.info(f"Visualizando y guardando clip en '{save_path}'")

    # Desnormalizar si se proporcionan la media y la desviación estándar
    if mean is not None and std is not None:
        video_tensor = unnormalize_tensor(video_tensor, mean, std)

    # Asegurar que el tensor esté en el rango [0, 1] para la visualización
    video_tensor = torch.clamp(video_tensor, 0, 1)

    # Seleccionar frames para mostrar
    total_frames = video_tensor.shape[1]
    # Asegurarse de no mostrar más frames de los que hay
    num_frames_to_show = min(total_frames, num_frames_to_show)
    indices = np.linspace(0, total_frames - 1, num_frames_to_show, dtype=int)

    # Crear la figura con Matplotlib
    fig, axs = plt.subplots(1, num_frames_to_show, figsize=(15, 5))
    fig.suptitle(title, fontsize=16)

    for i, frame_idx in enumerate(indices):
        # Obtener el frame y permutar de (C, H, W) a (H, W, C) para imshow
        frame = video_tensor[:, frame_idx, :, :]
        # Si es escala de grises, necesitamos quitar la dimensión de canal
        if frame.shape[0] == 1:
            frame_to_show = frame.squeeze(0).cpu().numpy()
            axs[i].imshow(frame_to_show, cmap='gray')
        else:
            frame_to_show = frame.permute(1, 2, 0).cpu().numpy()
            axs[i].imshow(frame_to_show)

        axs[i].set_title(f"Frame {frame_idx}")
        axs[i].axis('off')

    # Guardar la figura
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)  # Cerrar la figura para liberar memoria
    logger.info("Visualización guardada exitosamente.")


def visualizar_cada_clase(
        ind: int,
        mod: str,
):
    logger.info(f"Cargando muestra de vídeo con transformaciones tipo: '{mod}'")

    # Parámetros comunes
    frames_per_clip = 16  # Debe coincidir con el entrenamiento del modelo
    target_fps = 16 / 3.0

    if "resnet" in mod:
        weights = R2Plus1D_18_Weights.KINETICS400_V1
        train_transforms, val_transforms = get_transforms_resnet(weights)
        transform_to_apply = train_transforms if 'train' in mod else val_transforms
        mean, std = weights.transforms().mean, weights.transforms().std
    else:  # cnn3d
        if "rgb" in mod:
            train_transforms, val_transforms, _ = get_transforms_cnn3d_rgb(TARGET_SIZE_DATASET)
        else:
            train_transforms, val_transforms, _ = get_transforms_cnn3d_grayscale(TARGET_SIZE_DATASET)

        transform_to_apply = train_transforms if 'train' in mod else val_transforms
        mean, std = None, None  # No hay normalización Z-score en este caso

    # Crear una instancia del dataset para obtener una muestra
    # No es necesario cargar todos los datos, solo creamos el objeto
    dataset = DatasetSoccernet(
        root_dir=DS_SOCCERNET_ACTIONS,
        label_map=SOCCERNET_LABELS,
        frames_per_clip=frames_per_clip,
        target_fps=target_fps,
        transform=transform_to_apply,
    )

    # Obtener una muestra y su etiqueta
    video_tensor, label_idx, video_path = dataset[ind]
    label_name = INV_LABEL_MAP.get(label_idx, "Desconocido")

    output_filename = f"visualized_sample_{ind}_{mod}.png"
    output_path = Path(RESULTS) / "transform_visuals" / output_filename

    visualize_and_save_clip(
        video_tensor=video_tensor,
        save_path=output_path,
        mean=mean,
        std=std,
        num_frames_to_show=4,
        #title=f"Muestra de '{label_name}' después de transformaciones '{mod}'\n{Path(video_path).name}"
        title=f"Transformaciones '{mod}'\n{Path(video_path).name}"
    )


def main(args):
    lista_indice_clip = [0, 1631, 3637, 6378]
    lista_model_type = ["cnn3d_gray", "cnn3d_rgb", "resnet_train", "resnet_val"]

    for ind in lista_indice_clip:
        for mod in lista_model_type:
            visualizar_cada_clase(ind, mod)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="...",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--nada",
        default=None,
        type=str,
        help="...",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    config_log()
    ut.verify_system()
    main(args)
