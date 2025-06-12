import argparse
import logging
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

from src.models.engine_training import train_model, evaluate_model, extras
from src.models.transforms import get_transforms_resnet
from src.preprocess.dataset_soccernet import NUM_CLASSES, CLIP_DURATION_SEC, get_output_size_from_transforms, \
    crear_dividir_dataset_t_v_t
from src.utils import utils as ut
from src.utils.constants import *

logger = logging.getLogger(__name__)

# Hiperparámetros y configuración para Transfer Learning
BATCH_SIZE = 16  # según VRAM (en cuda)
# BATCH_SIZE = 32
INITIAL_LEARNING_RATE = 0.001  # Tasa de aprendizaje para la nueva capa clasificadora

# Para R(2+1)D_18 pre-entrenado en Kinetics, usualmente se usan 16 frames.
# Tu DatasetSoccernet se adaptará para muestrear esta cantidad.
TRANSFER_MODEL_FRAMES_PER_CLIP = 16

# Este TARGET_FPS ahora es un valor conceptual para el muestreo manual, no un parámetro directo de decodificación en torchvision.io.read_video
# servirá para 'torchcodec'
TARGET_FPS = float(TRANSFER_MODEL_FRAMES_PER_CLIP / CLIP_DURATION_SEC)

# Las transformaciones de los pesos pre-entrenados definirán el tamaño espacial (ej. 112x112)
# No necesitamos TARGET_SIZE_DATASET, ya que las transformaciones lo dictarán.

# Checkpointing
EPOCAS_CHECKPOINT_SAVE_INTERVAL = 1  # para tener todas las metricas y poder graficarlas a gusto
SAVE_BEST_METRIC_TYPE = "loss"

# LR Scheduler
LR_SCHEDULER_PATIENCE = 5  # Paciencia para reducir el LR (en épocas)
LR_SCHEDULER_FACTOR = 0.5  # Factor por el cual se reduce el LR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        # https://docs.python.org/3/library/logging.html
        level=logging.INFO,  # NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL
        encoding="utf-8",
    )


def model_r2plus1d_fine_tuning_granular(
        num_classes_output: int,
        granular: bool = True,
):
    logger.info("Cargando modelo R(2+1)D_18 pre-entrenado con pesos Kinetics-400.")
    weights = R2Plus1D_18_Weights.KINETICS400_V1
    model = r2plus1d_18(weights=weights)

    logger.info("Congelando pesos del backbone pre-entrenado.")
    for param in model.parameters():
        param.requires_grad = False

    # Reemplazar la cabeza de clasificación (model.fc)
    num_original_features = model.fc.in_features
    model.fc = nn.Linear(num_original_features, num_classes_output)
    logger.info(f"Cabeza de clasificación reemplazada: {num_original_features} -> {num_classes_output} clases.")

    if granular:
        logger.info("Descongelando el último bloque (BasicBlock) de model.layer4 y model.fc")

        for name, param in model.named_parameters():
            # Descongelar si el nombre del parámetro contiene "layer4.1." (el segundo BasicBlock)
            # if "layer4.1." in name or "fc." in name:
            if "layer4.1." in name:
                param.requires_grad = True
                logger.info(f"Descongelando: {name}")  # ver qué se descongela
            else:
                param.requires_grad = False

        # Configurar Optimizador con Tasa de Aprendizaje Diferencial
        learning_rate_backbone_finetune = 0.00001  # Tasa de aprendizaje 'muy baja' para las capas de ResNet
        learning_rate_head_finetune = INITIAL_LEARNING_RATE  # Tasa de aprendizaje 'media' para la capa clasificadora

        params_to_optimize = [
            {"params": model.layer4[1].parameters(), "lr": learning_rate_backbone_finetune},
            {"params": model.fc.parameters(), "lr": learning_rate_head_finetune},
        ]

        logger.info(
            f"Optimizador configurado con LR diferencial: último bloque de layer4={learning_rate_backbone_finetune}, fc={learning_rate_head_finetune}")
    else:
        params_to_optimize = model.fc.parameters()

    return model, weights, params_to_optimize


def main(args):
    logger.info("Iniciando arquitectura  de modelo RESNET - Transfer Learning con R(2+1)D_18.")
    logger.info(f"Usando dispositivo: {device}")
    logger.info(f"Frames por clip: {TRANSFER_MODEL_FRAMES_PER_CLIP}, Duración clip: {CLIP_DURATION_SEC}s")

    start_epoch = 0
    initial_best_val_metric = None

    # actual_checkpoint_to_load: ruta definitiva del checkpoint
    actual_checkpoint_to_load, checkpoint_dir_run = extras(M_RESNET, args.resume_checkpoint_file)

    #
    model, weights, params_to_optimize = model_r2plus1d_fine_tuning_granular(
        num_classes_output=NUM_CLASSES,
        granular=True,  # TODO sera granular o solo la capa clasificadora?
    )
    model = model.to(device)

    logger.info("Transformaciones deterministas que acepta el modelo:")
    logger.info(weights.transforms())
    train_transforms, val_transforms = get_transforms_resnet(weights)

    # es mejor usar val_transforms para el tamaño de tensores dummy, ya que es más simple y determinista.
    expected_size = get_output_size_from_transforms(val_transforms)

    if expected_size is None:
        # logger.warning("No se pudo determinar el tamaño de salida de las transformaciones. En este modelo 'r2plus1d_18' se conoce el tamaño esperado (112, 112)")
        # expected_size=(112, 112)
        logger.error("No se pudo determinar el tamaño de salida de las transformaciones. Saliendo.")
        sys.exit(1)

    # Creación y División del Dataset
    train_dataloader, val_dataloader, test_dataloader = crear_dividir_dataset_t_v_t(
        frames_per_clip=TRANSFER_MODEL_FRAMES_PER_CLIP,
        target_fps=TARGET_FPS,
        train_transforms=train_transforms,
        validation_test_transforms=val_transforms,
        expected_output_size=expected_size,
        batch_size=BATCH_SIZE,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
    )

    # Optimizador y Scheduler
    # Entrenar solo la cabeza clasificadora nueva si freeze_backbone=True
    criterion = nn.CrossEntropyLoss()
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

    logger.info("Iniciando evaluación final, en 'TEST' (datos no vistos)")
    evaluate_model(
        model=model,
        dataloader=test_dataloader,
        criterion=criterion,
        device=device,
        per_epoch_eval=False,
    )

    logger.info(f"Proceso total finalizado... Checkpoints se encuentran en '{checkpoint_dir_run}'")


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
        default=None,  # empezará un nuevo entrenamiento
        #default="20250607-032043/model_RESNET_best.pth",  # TODO
        type=str,
        help="Ruta relativa o absoluta para reanudar entrenamiento desde un .pth.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    config_log()
    ut.verify_system()
    main(args)
