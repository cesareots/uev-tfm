import argparse
import logging
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.preprocess.dataset_soccernet import ORIGINAL_SIZE, FRAMES_PER_CLIP, NUM_CLASSES, \
    preprocessed_dataset
from src.utils import utils as ut
from src.utils.constants import *

logger = logging.getLogger(__name__)

# Hiperparámetros de entrenamiento, ajustar basado en la memoria de la GPU
BATCH_SIZE = 8
LEARNING_RATE = 0.001

# si hay disponibilidad de GPU, si no CPU
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
    def __init__(self, num_classes, input_channels=3, input_frames=FRAMES_PER_CLIP,
                 input_size=(ORIGINAL_SIZE[1], ORIGINAL_SIZE[0])):
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
        dummy_tensor = torch.autograd.Variable(torch.zeros(dummy_input_size))

        # Ajuste para evitar el error si la entrada es muy pequeña para MaxPool3d
        # Calcula el tamaño de la salida de features para determinar el flattened_size
        with torch.no_grad():
            try:
                size_after_features = self.features(dummy_tensor).shape
            except RuntimeError as e:
                logger.error(
                    f"Calculando el tamaño a aplanar: {e}. Las dimensiones de entrada ({input_size}) pueden ser demasiado pequeñas para las capas de MaxPool3d.")
                # Proporciona un tamaño de respaldo o ajusta la arquitectura del modelo
                # si las dimensiones son demasiado pequeñas.
                # Por ahora, usaré un tamaño fijo si falla el cálculo, pero lo ideal es ajustar el modelo.
                # Esto es solo para la inicialización.
                size_after_features = (1, 128, 1, 1, 1)  # Un valor por defecto muy pequeño si falla el cálculo

        # Tamaño a aplanar: canales * dim_temporal * dim_alto * dim_ancho
        flattened_size = size_after_features[1] * size_after_features[2] * size_after_features[3] * size_after_features[
            4]
        logger.info(f"Tamaño a aplanar calculado: {size_after_features} -> {flattened_size}")

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
        total_predictions = 0

        for i, (inputs, labels) in enumerate(dataloader):
            valid_indices = labels != -1
            inputs = inputs[valid_indices].to(device)
            labels = labels[valid_indices].to(device)

            if inputs.size(0) == 0:
                continue

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass y optimización
            loss.backward()
            optimizer.step()

            # Estadísticas
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            # Imprimir estadísticas cada cierto número de batches
            if (i + 1) % 10 == 0:
                log_con = f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}"
                print(log_con)
                logger.info(log_con)

        epoch_loss = running_loss / total_predictions if total_predictions > 0 else 0
        epoch_acc = correct_predictions / total_predictions if total_predictions > 0 else 0

        log_con = f"Epoch [{epoch + 1}/{num_epochs}] finalizada. Pérdida: {epoch_loss:.4f}, Precisión: {epoch_acc:.4f}"
        print(log_con)
        logger.info(log_con)

    t_end = time.time() - t_start
    min = np.round(t_end / 60.0, decimals=2)
    logger.info(f"Entrenamiento finalizado... Tiempo total: {min} minutos.")


def evaluate_model(model, dataloader, criterion, device):
    t_start = time.time()
    model.eval()
    logger.info("Evaluación iniciada...")
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            valid_indices = labels != -1
            inputs = inputs[valid_indices].to(device)
            labels = labels[valid_indices].to(device)

            if inputs.size(0) == 0:
                continue

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Estadísticas
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    avg_loss = running_loss / total_predictions if total_predictions > 0 else 0
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    t_end = time.time() - t_start
    min = np.round(t_end / 60.0, decimals=2)
    logger.info(f"Evaluación finalizada... Tiempo total: {min} minutos.")
    logger.info(f"Pérdida promedio: {avg_loss:.4f}, Precisión: {accuracy:.4f}")

    return avg_loss, accuracy


def main(args):
    logger.info("Iniciando carga de tensores preprocesados para entrenamiento.")
    #logger.info(f"Tamaño del video en tensores: {args.video_size}")  # TODO sera posible cargar tensores de un tamaño especifico? entiendo que deberia tener los 2 tamaños de tensores en disco duro
    preprocessed_train_dataset = preprocessed_dataset()

    if len(preprocessed_train_dataset) == 0:
        logger.error("El dataset preprocesado está vacío.")
        exit()

    # División de datos (ejemplo: 80% entrenamiento, 20% validación)
    # Es común dividir los datos ANTES de pasar transformaciones si las transformaciones son complejas o si el dataset se usa para múltiples fines.
    # Si se está dividiendo el dataset completo preprocesado, asegurar que los índices sean consistentes.
    # Para simplicidad aquí, dividimos el dataset "base" sin transformaciones aún.

    # Dado que no tenemos un split train/val para los archivos .mkv, haremos el split después del preprocesamiento.
    # Es mejor crear un split inicial de archivos .mkv y luego preprocesar por separado TRAIN y VAL.
    # Pero si todo está en una sola carpeta preprocesada, se puede dividir así:
    total_preprocessed_items = len(preprocessed_train_dataset)
    train_size_p = int(0.8 * total_preprocessed_items)
    val_size_p = total_preprocessed_items - train_size_p

    # RandomSplit creará sub-datasets. Necesitamos aplicar las transformaciones al dataset completo y luego dividir, o crear dos instancias de 'DatasetTensors', cada una con su transform y apuntando a rutas de train/val.
    # Para este ejemplo, haremos el split después de instanciar el dataset, y las transformaciones se aplicarán en __getitem__
    train_dataset_p, val_dataset_p = torch.utils.data.random_split(
        dataset=preprocessed_train_dataset,
        lengths=[train_size_p, val_size_p],
    )

    train_dataloader_p = DataLoader(
        train_dataset_p,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,  # TODO que es?
        pin_memory=True,  # TODO que es?
    )
    val_dataloader_p = DataLoader(
        val_dataset_p,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    try:
        # Determinar las dimensiones de entrada para el modelo
        # Se asume que el primer tensor válido tendrá las dimensiones correctas
        dummy_video_tensor_p = None

        for i in range(len(preprocessed_train_dataset)):
            temp_video_p, temp_label_p = preprocessed_train_dataset[i]
            if temp_label_p != -1:  # Asegurarse de que no sea un dummy tensor
                dummy_video_tensor_p = temp_video_p
                break

        if dummy_video_tensor_p is None:
            logger.error(
                "No se pudo cargar ninguna muestra válida del dataset preprocesado para determinar el tamaño de entrada del modelo.")
            exit()

        C_in_p, T_in_p, H_in_p, W_in_p = dummy_video_tensor_p.shape

        logger.info(
            f"Dimensiones de entrada al modelo (desde tensores): (Batch, {C_in_p}, {T_in_p}, {H_in_p}, {W_in_p})")

        model_p = SimpleCNN3D(
            num_classes=NUM_CLASSES,
            input_channels=C_in_p,
            input_frames=T_in_p,
            input_size=(H_in_p, W_in_p),
        ).to(device)

        # Definir la función de pérdida y el optimizador
        criterion_p = nn.CrossEntropyLoss()
        optimizer_p = optim.Adam(model_p.parameters(), lr=LEARNING_RATE)

        train_model(model_p, train_dataloader_p, criterion_p, optimizer_p, args.epocas, device)
        evaluate_model(model_p, val_dataloader_p, criterion_p, device)

        # Guardar estado del modelo entrenado
        model_name = f"models/model_torch_CNN3D_{args.epocas}_epochs_{time.time()}.pth"
        torch.save(model_p.state_dict(), model_name)
        logger.info(f"Modelo guardado en '{model_name}'.")
    except Exception as e:
        logger.error(str(e))


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Modelo CNN3D - entrenamiento y guardado.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--epocas", default=1, type=ut.non_negative_int,
                        help="Númrero de épocas para el entrenamiento.")
    parser.add_argument("--video_size", default="224p", type=str, choices=["224p", "720p"],
                        help="Tamaño de los tensores que se utilizarán.")

    return parser.parse_args()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        args = parse_arguments()
    else:
        # Valores por defecto si no se proporcionan argumentos desde la línea de comandos
        class Args:
            epocas = 1
            video_size = "224p"
            # video_size = "720p"


        args = Args()

    config_log()
    ut.verify_system()
    main(args)
