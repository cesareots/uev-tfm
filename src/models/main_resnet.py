import argparse
import logging
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as T_v2
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

from src.models.engine_training import train_model, evaluate_model, extras
from src.preprocess.dataset_soccernet import NUM_CLASSES, CLIP_DURATION_SEC, crear_dividir_dataset_t_v_t
from src.utils import utils as ut
from src.utils.constants import *

logger = logging.getLogger(__name__)

# Configuración para Transfer Learning
BATCH_SIZE = 8  # Ajustar según VRAM
INITIAL_LEARNING_RATE = 0.001  # Tasa de aprendizaje para la nueva capa clasificadora
# LEARNING_RATE_FINETUNE = 0.0001 # Tasa de aprendizaje más baja si se hace fine-tuning de todo el modelo después

# Para R(2+1)D_18 pre-entrenado en Kinetics, usualmente se usan 16 frames.
# Tu DatasetSoccernet se adaptará para muestrear esta cantidad.
TRANSFER_MODEL_FRAMES_PER_CLIP = 16

# Este TARGET_FPS ahora es un valor conceptual para el muestreo manual, no un parámetro directo de decodificación en torchvision.io.read_video
# servirá para 'torchcodec'
TARGET_FPS = float(TRANSFER_MODEL_FRAMES_PER_CLIP / CLIP_DURATION_SEC)

# Las transformaciones de los pesos pre-entrenados definirán el tamaño espacial (ej. 112x112)
# No necesitamos TARGET_SIZE_DATASET, ya que las transformaciones lo dictarán.

# Checkpointing
EPOCAS_CHECKPOINT_SAVE_INTERVAL = 1  # TODO ¿Guardar cada época es bueno para transfer learning?
SAVE_BEST_METRIC_TYPE = "loss"

# LR Scheduler
LR_SCHEDULER_PATIENCE = 2  # Paciencia para reducir el LR (en épocas)
LR_SCHEDULER_FACTOR = 0.5  # Factor por el cual se reduce el LR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Usando dispositivo: {device}")

torch.manual_seed(SEMILLA)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEMILLA)
np.random.seed(SEMILLA)
random.seed(SEMILLA)


def config_log() -> None:
    ut.verify_directory(LOG_DIR)

    logging.basicConfig(
        filename=LOG_MODEL_RESNET,
        filemode="a",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        encoding="utf-8",
    )


def get_pretrained_r2plus1d_model(
        num_classes_output: int,
        freeze_backbone: bool = True,
):
    logger.info("Cargando modelo R(2+1)D_18 pre-entrenado con pesos Kinetics-400...")
    weights = R2Plus1D_18_Weights.KINETICS400_V1
    model = r2plus1d_18(weights=weights)

    if freeze_backbone:
        logger.info("Congelando pesos del backbone pre-entrenado.")
        for param in model.parameters():
            param.requires_grad = False

    # Reemplazar la cabeza de clasificación (model.fc)
    num_original_features = model.fc.in_features
    model.fc = nn.Linear(num_original_features, num_classes_output)
    logger.info(f"Cabeza de clasificación reemplazada: {num_original_features} -> {num_classes_output} clases.")

    # return model, weights.transforms()
    return model, weights


def main(args):
    logger.info("Iniciando arquitectura  de modelo RESNET - Transfer Learning con R(2+1)D_18.")
    logger.info(f"Frames por clip: {TRANSFER_MODEL_FRAMES_PER_CLIP}, Duración clip: {CLIP_DURATION_SEC}s")
    # logger.info(f"Modelo esperará frames de tamaño HxW: {TARGET_SIZE_DATASET[0]}x{TARGET_SIZE_DATASET[1]}")

    start_epoch = 0
    initial_best_val_metric = None

    # actual_checkpoint_to_load: ruta definitiva del checkpoint
    actual_checkpoint_to_load, run_name_to_use, checkpoint_dir_run = extras(
        args.resume_checkpoint_file,
        M_RESNET,
    )

    #
    model, weights = get_pretrained_r2plus1d_model(
        num_classes_output=NUM_CLASSES,
        freeze_backbone=True,
    )
    model = model.to(device)
    logger.info("Transformaciones deterministas que acepta el modelo:")
    logger.info(weights.transforms())

    # pipeline de transformaciones para ENTRENAMIENTO
    train_transforms = T_v2.Compose([
        # Los modelos de torchvision esperan uint8 en el rango [0, 255] al inicio
        T_v2.ToDtype(
            torch.uint8,
            scale=False,
        ),
        T_v2.RandomHorizontalFlip(p=0.5),
        T_v2.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.05,
        ),
        T_v2.RandomRotation(
            degrees=5,  # grados
            expand=False,  # evita cambios en el tamaño del frame.
        ),
        # Redimensionar el lado más corto a 128
        T_v2.Resize(
            size=(128),
            antialias=True,
        ),
        # En lugar de CenterCrop, para entrenamiento se suele usar RandomCrop para que el modelo vea diferentes partes de la imagen.
        T_v2.RandomCrop(size=(112, 112)),
        # Convertir a float y escalar a [0,1] antes de normalizar
        T_v2.ToDtype(
            torch.float32,
            scale=True,
        ),
        # Normalizar con la media y std específicas del modelo
        T_v2.Normalize(
            mean=weights.transforms().mean,
            std=weights.transforms().std,
        ),
    ])

    # pipeline de transformaciones para VALIDACIÓN, solo contiene el pre-procesamiento determinista.
    val_transforms = T_v2.Compose([
        # Los modelos de torchvision esperan uint8 en el rango [0, 255] al inicio
        T_v2.ToDtype(
            torch.uint8,
            scale=False,
        ),
        # Redimensionar el lado más corto a 128 (ejemplo de torchvision)
        T_v2.Resize(
            size=(128),
            antialias=True,
        ),
        # Recortar el centro a 112x112
        T_v2.CenterCrop(size=(112, 112)),
        # Convertir a float y escalar a [0,1] antes de normalizar
        T_v2.ToDtype(
            torch.float32,
            scale=True,
        ),
        # Normalizar con la media y std específicas del modelo
        T_v2.Normalize(
            mean=weights.transforms().mean,
            std=weights.transforms().std,
        ),
    ])

    # Creación y División del Dataset
    train_dataloader, val_dataloader, test_dataloader = crear_dividir_dataset_t_v_t(
        frames_per_clip=TRANSFER_MODEL_FRAMES_PER_CLIP,
        target_fps=TARGET_FPS,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        batch_size=BATCH_SIZE,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
    )

    # Optimizador y Scheduler
    # Entrenar solo la cabeza clasificadora nueva si freeze_backbone=True
    criterion = nn.CrossEntropyLoss()
    params_to_optimize = model.fc.parameters() if model.fc.weight.requires_grad else model.parameters()
    optimizer = optim.Adam(params_to_optimize, lr=INITIAL_LEARNING_RATE)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
    )
    logger.info(f"Optimizando parámetros de: {'model.fc' if model.fc.weight.requires_grad else 'todo el modelo'}")

    # Cargar Checkpoint si se especifica
    if actual_checkpoint_to_load and actual_checkpoint_to_load.exists():
        checkpoint = torch.load(actual_checkpoint_to_load, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        # Solo cargar optimizer y scheduler si los parámetros que optimizan coinciden
        # (ej. si se guardó optimizando solo fc, y ahora también solo fc)
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint.get('scheduler_state_dict'):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("Estado del optimizador y scheduler cargados.")
        except ValueError as e:
            logger.warning(
                f"No se pudo cargar el estado del optimizador/scheduler, posiblemente por cambio en parámetros a optimizar: {e}. "
                "Se reiniciará el optimizador/scheduler.")
            # Re-inicializar el optimizador para los parámetros correctos si falla la carga
            optimizer = optim.Adam(params_to_optimize, lr=INITIAL_LEARNING_RATE)
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=LR_SCHEDULER_FACTOR,
                patience=LR_SCHEDULER_PATIENCE,
            )

        start_epoch = checkpoint['epoch']
        initial_best_val_metric = checkpoint.get('best_val_metric_value')
        logger.info(
            f"Reanudando entrenamiento desde época {start_epoch + 1}. LR: {optimizer.param_groups[0]['lr']:.7f}")
    else:
        logger.info(f"Iniciando nuevo entrenamiento desde época {start_epoch + 1}.")

    model_name = "model_RESNET"
    train_model(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        criterion=criterion,
        num_epochs=args.epocas,
        device=device,
        checkpoint_dir_run=checkpoint_dir_run,
        save_every_n_epochs=EPOCAS_CHECKPOINT_SAVE_INTERVAL,
        save_best_metric_type=SAVE_BEST_METRIC_TYPE,
        start_epoch=start_epoch,
        initial_best_val_metric=initial_best_val_metric,
        checkpoint_base_name=model_name,
    )

    # Cargar el MEJOR modelo guardado durante el entrenamiento
    best_model_path = checkpoint_dir_run / f"{model_name}_best.pth"

    if best_model_path.exists():
        logger.info(f"Mejor modelo '{best_model_path}' para evaluación final, en 'TEST'.")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        logger.warning(
            f"No se encontró archivo del mejor modelo. Evaluando con el último estado del modelo, en 'TEST'.")

    logger.info("Iniciando evaluación final, en 'TEST' (NO VISTO)")
    evaluate_model(
        model=model,
        dataloader=test_dataloader,
        criterion=criterion,
        device=device,
        per_epoch_eval=False,  # Para obtener el log completo y detallado
    )

    logger.info(f"Checkpoints de '{run_name_to_use}' se encuentran en '{checkpoint_dir_run}'")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Modelo RESNET - Transfer Learning con R(2+1)D_18 - entrenamiento y guardado, directamente desde videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--epocas",
        default=1,  # TODO
        type=ut.non_negative_int,
        help="Número de épocas para el entrenamiento.",
    )
    parser.add_argument(
        "--resume_checkpoint_file",
        # default="20250605-233152/model_RESNET_best.pth",  # TODO
        default=None,  # empezará un nuevo entrenamiento
        type=str,
        help="Ruta relativa o absoluta para reanudar entrenamiento desde un .pth.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    config_log()
    ut.verify_system()
    main(args)
