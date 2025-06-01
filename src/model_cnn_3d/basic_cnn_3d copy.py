import argparse
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as T_v2
from torch.utils.data import DataLoader, random_split

from src.preprocess.dataset_soccernet import DatasetSoccernet, FRAMES_PER_CLIP, NUM_CLASSES, TARGET_FPS, \
    CLIP_DURATION_SEC
from src.utils import utils as ut
from src.utils.constants import *

logger = logging.getLogger(__name__)

# Hiperparámetros de entrenamiento
BATCH_SIZE = 8  # segun VRAM
LEARNING_RATE = 0.001

# Tamaño final de los frames que entrarán al modelo después de las transformaciones
# Esto debe coincidir con lo que producen tus transformaciones (ej. RandomResizedCrop)
TARGET_SIZE_DATASET = (256, 256)

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
        filename=LOG_MODEL_CNN3D,
        filemode="a",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        # https://docs.python.org/3/library/logging.html
        level=logging.INFO,  # NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL
        encoding="utf-8",
    )


class SimpleCNN3D(nn.Module):
    def __init__(self, num_classes, input_channels=3, input_frames=FRAMES_PER_CLIP, input_size=TARGET_SIZE_DATASET):
        super(SimpleCNN3D, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),  # (B, 32, T/2, H/2, W/2)

            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),  # (B, 64, T/4, H/4, W/4)

            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),  # (B, 128, T/8, H/8, W/8)
        )

        # Calcular el tamaño de la entrada a la capa lineal después de aplanar
        # Podemos usar un tensor dummy para que PyTorch nos diga el tamaño
        # (Batch=1, C, T, H, W)
        dummy_input_size = (1, input_channels, input_frames, input_size[0], input_size[1])
        dummy_tensor = torch.zeros(dummy_input_size)

        with torch.no_grad():
            try:
                size_after_features = self.features(dummy_tensor).shape
            except RuntimeError as e:
                logger.error(
                    f"Calculando el tamaño a aplanar: {e}. Las dimensiones de entrada {input_size} pueden ser demasiado pequeñas.")
                # Considera ajustar la arquitectura o el tamaño de entrada si esto ocurre.
                # Para que el código no falle, podrías lanzar una excepción o usar un tamaño de emergencia y advertir.
                raise ValueError(
                    f"Error al calcular el tamaño aplanado con input_size={input_size}. Ajusta el modelo o el tamaño de entrada.") from e

        # Tamaño a aplanar: canales * dim_temporal * dim_alto * dim_ancho
        flattened_size = size_after_features[1] * size_after_features[2] * size_after_features[3] * size_after_features[
            4]
        logger.info(f"Tamaño a aplanar calculado: {size_after_features} -> {flattened_size}")

        if flattened_size == 0:
            raise ValueError(
                f"Flattened size es 0. Output de features: {size_after_features}. Input H,W: {input_size}, T: {input_frames}")

        logger.info(f"Tamaño de entrada al modelo HxW: {input_size}")
        logger.info(f"Dimensiones tras capas convolucionales: {size_after_features}")
        logger.info(f"Tamaño a aplanar calculado: {flattened_size}")

        self.fc = nn.Linear(flattened_size, num_classes)

    def forward(self, x):
        # Input x shape: (Batch, C, T, H, W)
        x = self.features(x)
        # Flatten
        x = x.view(x.size(0), -1)
        # Fully connected layer
        x = self.fc(x)

        return x


def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
    t_start = time.time()
    model.train()
    logger.info(f"Entrenamiento iniciado... Número de épocas: {num_epochs}")

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for i, batch_data in enumerate(dataloader):
            inputs, labels, video_paths = batch_data  # Asumiendo que el dataloader ahora devuelve paths

            # Filtrar muestras que tuvieron errores (etiqueta -1)
            valid_indices = labels != -1
            inputs = inputs[valid_indices].to(device)
            labels = labels[valid_indices].to(device)

            if inputs.size(0) == 0:  # Si todos los items en el batch eran inválidos
                logger.warning(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}]: Batch vacío después de filtrar errores. Saltando.")
                continue

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Estadísticas
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            # Imprimir estadísticas cada cierto número de batches
            if (i + 1) % 10 == 0:
                log_con = f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}"
                print(log_con)
                logger.info(log_con)

        epoch_loss = running_loss / total_samples if total_samples > 0 else 0
        epoch_acc = correct_predictions / total_samples if total_samples > 0 else 0

        log_con = f"Epoch [{epoch + 1}/{num_epochs}] finalizada. Pérdida: {epoch_loss:.4f}, Precisión: {epoch_acc:.4f}"
        print(log_con)
        logger.info(log_con)

    ut.get_time_employed(t_start, "Entrenamiento.")


def evaluate_model(model, dataloader, criterion, device):
    t_start = time.time()
    model.eval()
    logger.info("Evaluación iniciada...")
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for i, batch_data in enumerate(dataloader):
            inputs, labels, video_paths = batch_data

            valid_indices = labels != -1
            inputs = inputs[valid_indices].to(device)
            labels = labels[valid_indices].to(device)

            if inputs.size(0) == 0:
                continue

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Estadísticas
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    avg_loss = running_loss / total_samples if total_samples > 0 else 0
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0

    ut.get_time_employed(t_start, "Evaluación.")
    logger.info(f"Pérdida promedio en validación: {avg_loss:.4f}, Precisión en validación: {accuracy:.4f}")

    return avg_loss, accuracy


def main(args):
    logger.info("Iniciando carga de datos directamente desde vídeos.")
    logger.info(f"Frames por clip: {FRAMES_PER_CLIP}, Duración clip: {CLIP_DURATION_SEC}s")
    logger.info(f"Modelo esperará frames de tamaño HxW: {TARGET_SIZE_DATASET[0]}x{TARGET_SIZE_DATASET[1]}")

    train_transforms = T_v2.Compose([
        T_v2.RandomHorizontalFlip(p=0.5),  # volteo horizontal aleatorio
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
        T_v2.RandomResizedCrop(
            size=(TARGET_SIZE_DATASET[0], TARGET_SIZE_DATASET[1]),
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            antialias=True,
        ),
    ])
    val_transforms = T_v2.Compose([
        T_v2.Resize(
            size=(TARGET_SIZE_DATASET[0], TARGET_SIZE_DATASET[1]),
            antialias=True,
        ),
    ])

    # --- Creación y División del Dataset ---
    logger.info("1. Creando dataset base para obtener la lista de todos los vídeos.")
    # Instancia base solo para recolectar todos los video_items. Sin transformaciones aquí.
    base_dataset_for_items = DatasetSoccernet(
        root_dir=DS_SOCCERNET_ACTIONS,
        label_map=SOCCERNET_LABELS,
        frames_per_clip=FRAMES_PER_CLIP,
        target_fps=TARGET_FPS,
        transform=None,  # No aplicar transformaciones
    )

    if len(base_dataset_for_items) == 0:
        logger.error(
            "El dataset base (DatasetSoccernet) está vacío. Verifica la ruta y los archivos en DS_SOCCERNET_ACTIONS.")
        sys.exit(1)

    # Esta es la lista de (path, label, action_name)
    all_video_items = base_dataset_for_items.video_items

    logger.info(f"2. Dividiendo {len(all_video_items)} vídeos en conjuntos de entrenamiento y validación (80/20)")
    total_items = len(all_video_items)
    train_size = int(0.8 * total_items)
    val_size = total_items - train_size

    indices = list(range(total_items))
    # Asegura que la división sea la misma cada vez
    generator = torch.Generator().manual_seed(SEMILLA)
    train_indices, val_indices = random_split(
        indices,
        [train_size, val_size],
        generator=generator,
    )

    # Seleccionar los items para cada conjunto usando los índices obtenidos
    train_video_items_subset = [all_video_items[i] for i in train_indices]
    val_video_items_subset = [all_video_items[i] for i in val_indices]

    logger.info(f"3. Creando dataset de entrenamiento ({len(train_video_items_subset)} videos) con train_transforms.")
    # Crear el DatasetSoccernet para ENTRENAMIENTO, pasándole solo su subconjunto de vídeos y sus transformaciones
    train_dataset = DatasetSoccernet(
        root_dir=None,
        label_map=SOCCERNET_LABELS,
        frames_per_clip=FRAMES_PER_CLIP,
        target_fps=TARGET_FPS,
        transform=train_transforms,
        video_items_list=train_video_items_subset,
    )

    logger.info(f"4. Creando dataset de validación ({len(val_video_items_subset)} vídeos) con val_transforms")
    # Crear el DatasetSoccernet para VALIDACIÓN
    val_dataset = DatasetSoccernet(
        root_dir=None,
        label_map=SOCCERNET_LABELS,
        frames_per_clip=FRAMES_PER_CLIP,
        target_fps=TARGET_FPS,
        transform=val_transforms,
        video_items_list=val_video_items_subset,
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        logger.error("Uno de los datasets (train o val) está vacío después del split. Verificar.")
        sys.exit(1)

    logger.info("5. Creando DataLoaders.")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Las dimensiones C, T, H, W se definen por DatasetSoccernet y las transformaciones.
    model = SimpleCNN3D(
        num_classes=NUM_CLASSES,
        input_channels=3,  # RGB
        input_frames=FRAMES_PER_CLIP,
        input_size=(TARGET_SIZE_DATASET[0], TARGET_SIZE_DATASET[1])  # Tamaño final tras transformaciones
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_model(model, train_dataloader, criterion, optimizer, args.epocas, device)
    evaluate_model(model, val_dataloader, criterion, device)

    # Guardar estado del modelo entrenado
    model_filename = f"model_torch_CNN3D_{args.epocas}epochs_{int(time.time())}.pth"
    model_save_path = Path(MODELS_DIR) / model_filename
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"Modelo guardado en '{model_save_path}'")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Mdelo CNN3D - entrenamiento y guardado, directamente desde videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--epocas",
        default=2,  # TODO ajustar
        type=ut.non_negative_int,
        help="Número de épocas para el entrenamiento."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    config_log()
    ut.verify_system()
    main(args)
