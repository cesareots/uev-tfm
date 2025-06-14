import argparse
import logging
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.engine_training import train_model, evaluate_model, extras
from src.models.transforms import get_transforms_cnn3d_grayscale
from src.preprocess.dataset_soccernet import NUM_CLASSES, CLIP_DURATION_SEC, get_output_size_from_transforms, \
    crear_dividir_dataset_t_v_t
from src.utils import utils as ut
from src.utils.constants import *

logger = logging.getLogger(__name__)

# BATCH_SIZE = 16
# BATCH_SIZE = 32
BATCH_SIZE = 64
# BATCH_SIZE = 128
INITIAL_LEARNING_RATE = 0.001
# INITIAL_LEARNING_RATE = 0.0001

# frames deseados por clip para el modelo (muestreados)
FRAMES_PER_CLIP = 16  # grandes modelos preentrenados lo utilizan
# FRAMES_PER_CLIP = 32

# Este TARGET_FPS ahora es un valor conceptual para el muestreo manual, no un parámetro directo de decodificación en torchvision.io.read_video
# servirá para 'torchcodec'
TARGET_FPS = float(FRAMES_PER_CLIP / CLIP_DURATION_SEC)

# Tamaño final de los frames que entrarán al modelo después de las transformaciones
TARGET_SIZE_DATASET = (128, 128)
# TARGET_SIZE_DATASET = (256, 256)

# Guardar checkpoint de época cada N épocas
EPOCAS_CHECKPOINT_SAVE_INTERVAL = 1  # para tener todas las metricas y poder graficarlas a gusto
# Métrica para 'model_best.pth': 'loss' o 'accuracy'
SAVE_BEST_METRIC_TYPE = "loss"

# Parámetros para el LR Scheduler (ReduceLROnPlateau)
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
        filename=LOG_MODEL_CNN3D,
        filemode="a",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        # https://docs.python.org/3/library/logging.html
        level=logging.INFO,  # NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL
        encoding="utf-8",
    )


class SimpleCNN3D(nn.Module):
    def __init__(self, num_classes, input_channels=1, input_frames=None, input_size=None):
        super(SimpleCNN3D, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Dropout3d(p=0.1),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),

            nn.Conv3d(32, 128, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Dropout3d(p=0.1),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),

            nn.Conv3d(128, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Dropout3d(p=0.1),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),
        )

        # Calcular la forma de salida de self.features para verificación y para obtener num_features_for_fc
        # (Batch=1, C, T, H, W)
        dummy_input_size_tuple = (1, input_channels, input_frames, input_size[0], input_size[1])
        dummy_tensor = torch.zeros(dummy_input_size_tuple)

        with torch.no_grad():
            try:
                output_shape_features = self.features(dummy_tensor).shape
            except RuntimeError as e:
                logger.error(
                    f"Error al calcular la salida de self.features con input_channels={input_channels}, input_frames={input_frames}, input_size={input_size}: {e}. "
                    f"Las dimensiones de entrada pueden ser demasiado pequeñas para la arquitectura.")
                raise ValueError(
                    f"Error al calcular el tamaño aplanado con input_size={input_size}. Ajusta el modelo o el tamaño de entrada.") from e

        logger.info(
            f"Dimensiones de entrada al modelo (C, T, H, W): ({input_channels}, {input_frames}, {input_size[0]}, {input_size[1]})")
        logger.info(f"Dimensiones tras self.features (antes de adaptive_pool): {output_shape_features}")

        # Comprobar si alguna dimensión (T, H, W) colapsó a cero después de self.features
        # output_shape_features es (B, C, T_out, H_out, W_out)
        if any(d == 0 for d in output_shape_features[2:]):  # Comprobar T_out, H_out, W_out
            raise ValueError(
                f"Una o más dimensiones (T, H, W) se han reducido a 0 después de self.features: {output_shape_features}. "
                f"Dimensiones de entrada (T, H, W): ({input_frames}, {input_size[0]}, {input_size[1]}). "
                f"Esto puede ocurrir si la entrada es demasiado pequeña para la profundidad/strides de la red convolucional.")

        # Pooling adaptativo global para reducir cada canal a un tamaño de 1x1x1
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # El número de características para la capa FC es el número de canales de salida de self.features
        num_features_for_fc = output_shape_features[1]  # Dimensión de canales

        # Después de adaptive_pool, la forma será (Batch, num_features_for_fc, 1, 1, 1)
        # Al aplanar, esto se convierte en (Batch, num_features_for_fc)
        logger.info(f"Después de adaptive_pool, las dimensiones espaciales/temporales se reducen a (1,1,1).")
        logger.info(f"Número de características para la capa lineal (FC): {num_features_for_fc}")

        if num_features_for_fc == 0:  # No debería ocurrir si las capas convolucionales están definidas correctamente
            raise ValueError(
                f"El número de características para la capa FC es 0. Salida de self.features: {output_shape_features}")

        self.fc = nn.Sequential(
            nn.Linear(num_features_for_fc, 16),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        # Input shape: (Batch, C, T, H, W)
        x = self.features(x)
        # Aplicar pooling adaptativo
        x = self.adaptive_pool(x)
        # Aplanar: la salida de adaptive_pool es (Batch, Channels, 1, 1, 1)
        x = x.view(x.size(0), -1)
        # capa 'fully connected'
        x = self.fc(x)

        return x


def main(args):
    logger.info("Iniciando arquitectura de modelo CNN3D.")
    logger.info(f"Usando dispositivo: {device}")
    logger.info(f"Frames por clip: {FRAMES_PER_CLIP}, Duración clip: {CLIP_DURATION_SEC}s")
    logger.info(f"Modelo esperará frames de tamaño HxW: {TARGET_SIZE_DATASET}")

    start_epoch = 0
    initial_best_val_metric = None

    # actual_checkpoint_to_load: ruta definitiva del checkpoint
    actual_checkpoint_to_load, checkpoint_dir_run = extras(M_BASIC, args.resume_checkpoint_file)

    # train_transforms, val_transforms, input_channels = get_transforms_cnn3d_rgb(TARGET_SIZE_DATASET)
    # logger.info("Usando 3 canales (RGB).")
    train_transforms, val_transforms, input_channels = get_transforms_cnn3d_grayscale(TARGET_SIZE_DATASET)
    logger.info("Usando 1 canal (escala de grises).")

    expected_size = get_output_size_from_transforms(val_transforms)

    if expected_size is None:
        logger.error("No se pudo determinar el tamaño de salida de las transformaciones. Saliendo.")
        sys.exit(1)

    # Creación y División del Dataset
    train_dataloader, val_dataloader, test_dataloader = crear_dividir_dataset_t_v_t(
        frames_per_clip=FRAMES_PER_CLIP,
        target_fps=TARGET_FPS,
        train_transforms=train_transforms,
        validation_test_transforms=val_transforms,
        expected_output_size=expected_size,
        batch_size=BATCH_SIZE,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
    )

    # Las dimensiones C, T, H, W se definen por DatasetSoccernet y las transformaciones.
    model = SimpleCNN3D(
        num_classes=NUM_CLASSES,
        input_channels=input_channels,
        input_frames=FRAMES_PER_CLIP,
        input_size=expected_size,  # Tamaño final tras transformaciones
    ).to(device)
    logger.info("Arquitectura del modelo:")
    logger.info(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LEARNING_RATE)
    scheduler = ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
        # verbose=False,  # 'False' para no duplicar logs
    )
    logger.info(
        f"Scheduler ReduceLROnPlateau activado: paciencia={LR_SCHEDULER_PATIENCE}, factor={LR_SCHEDULER_FACTOR}, monitoreando val_loss.")

    # Cargar Checkpoint si actual_checkpoint_to_load está definido y existe
    if actual_checkpoint_to_load and actual_checkpoint_to_load.exists():
        # logger.info(f"Cargando checkpoint desde: '{actual_checkpoint_to_load}'")
        checkpoint = torch.load(
            actual_checkpoint_to_load,
            map_location=device,
            # weights_only=False,
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch']
        initial_best_val_metric = checkpoint.get('best_val_metric_value')
        logger.info(
            f"Reanudando entrenamiento desde época {start_epoch + 1}. LR actual: {optimizer.param_groups[0]['lr']:.7f}")
    else:
        logger.info(f"Iniciando nuevo entrenamiento desde época {start_epoch + 1}.")

    model_name = "model_CNN3D"
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
        checkpoint = torch.load(
            best_model_path,
            map_location=device,
            # weights_only=False,
        )
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
        class_names=SOCCERNET_LABELS.keys(),
    )

    logger.info(f"Proceso total finalizado... Checkpoints se encuentran en '{checkpoint_dir_run}'")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Modelo CNN3D - entrenamiento y guardado, directamente desde videos.",
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
        # default="20250603-233152/model_CNN3D_best.pth",  # TODO
        type=str,
        help="Reanudar entrenamiento desde un checkpoint.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    config_log()
    ut.verify_system()
    main(args)
