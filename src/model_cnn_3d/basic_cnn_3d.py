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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split

from src.preprocess.dataset_soccernet import DatasetSoccernet, FRAMES_PER_CLIP, NUM_CLASSES, TARGET_FPS, \
    CLIP_DURATION_SEC
from src.utils import utils as ut
from src.utils.constants import *

logger = logging.getLogger(__name__)

# Hiperparámetros de entrenamiento
BATCH_SIZE = 8  # segun VRAM
INITIAL_LEARNING_RATE = 0.001

# Tamaño final de los frames que entrarán al modelo después de las transformaciones
# Esto debe coincidir con lo que producen tus transformaciones (ej. RandomResizedCrop)
TARGET_SIZE_DATASET = (256, 256)

# Guardar checkpoint de época cada N épocas
EPOCAS_CHECKPOINT_SAVE_INTERVAL = 3
# Métrica para 'model_best.pth': 'loss' o 'accuracy'
SAVE_BEST_METRIC_TYPE = "loss"

# Parámetros para el LR Scheduler (ReduceLROnPlateau)
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


def train_model(
        model,
        optimizer,
        scheduler,
        train_dataloader,
        val_dataloader,
        criterion,
        num_epochs,
        device,
        checkpoint_dir_run,  # Directorio para una ejecución específica
        save_every_n_epochs,  # Frecuencia para guardar checkpoints de época
        save_best_metric_type,  # 'loss' o 'accuracy' para el mejor modelo
        start_epoch: int = 0,
        # run_name="run",
        initial_best_val_metric=None,  # Para reanudar el mejor valor
):
    # model.train()
    t_start_training = time.time()
    logger.info(
        f"Entrenamiento iniciado... Número de épocas: {num_epochs - start_epoch} (desde la época {start_epoch + 1} hasta {num_epochs})")

    # lleva el registro del mejor valor de la métrica de validación encontrado hasta el momento (para decidir cuándo guardar model_CNN3D_best.pth).
    best_val_metric_value = initial_best_val_metric if initial_best_val_metric is not None \
        else (float('inf') if save_best_metric_type == "loss" else float('-inf'))

    # Crear un subdirectorio para los checkpoints de esta ejecución específica
    # checkpoint_dir = Path(M_BASIC) / run_name
    # checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Guardando checkpoints en: {checkpoint_dir_run}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        current_epoch_display = epoch + 1
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        epoch_t_start = time.time()

        for i, batch_data in enumerate(train_dataloader):
            inputs, labels, video_paths = batch_data

            # Filtrar muestras que tuvieron errores (etiqueta -1)
            valid_indices = labels != -1
            inputs = inputs[valid_indices].to(device)
            labels = labels[valid_indices].to(device)

            if inputs.size(0) == 0:  # Si todos los items en el batch eran inválidos
                logger.warning(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_dataloader)}]: Batch vacío después de filtrar errores. Saltando.")
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
            if (i + 1) % 15 == 0:
                log_con = f"Epoch [{current_epoch_display}/{num_epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}"
                print(log_con)
                logger.info(log_con)

        epoch_train_loss = running_loss / total_samples if total_samples > 0 else 0
        epoch_train_acc = correct_predictions / total_samples if total_samples > 0 else 0
        epoch_duration = time.time() - epoch_t_start

        log_con_epoch = f"Epoch [{current_epoch_display}/{num_epochs}] (Entrenamiento) finalizada en {epoch_duration:.2f}s. Pérdida: {epoch_train_loss:.4f}, Precisión: {epoch_train_acc:.4f}"
        print(log_con_epoch)
        logger.info(log_con_epoch)

        # Validación al final de cada época
        val_loss, val_acc = evaluate_model(
            model,
            val_dataloader,
            criterion,
            device,
            per_epoch_eval=True,
        )
        logger.info(
            f"Epoch [{current_epoch_display}/{num_epochs}] (Validación). Pérdida: {val_loss:.4f}, Precisión: {val_acc:.4f}")

        # Learning Rate Scheduler
        current_lr_before_step = optimizer.param_groups[0]['lr']
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)  # ReduceLROnPlateau necesita la métrica
            else:
                scheduler.step()  # Otros schedulers solo se llaman

            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < current_lr_before_step:
                logger.info(
                    f"Tasa de aprendizaje reducida de {current_lr_before_step:.7f} a {new_lr:.7f} por el scheduler.")
        logger.info(f"LR actual para la próxima época: {optimizer.param_groups[0]['lr']:.7f}")

        # Checkpointing
        checkpoint_data = {
            'epoch': current_epoch_display,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,  # Guardar estado del scheduler
            'train_loss': epoch_train_loss,
            'train_acc': epoch_train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'best_val_metric_value': best_val_metric_value,  # Para reanudar el guardado del mejor
        }

        # Guardar el modelo más reciente
        # latest_checkpoint_path = checkpoint_dir / "model_latest.pth"
        # torch.save(checkpoint_data, latest_checkpoint_path)
        # logger.info(f"Checkpoint guardado (último): '{latest_checkpoint_path}'")

        # Guardar periódicamente por época
        if save_every_n_epochs > 0 and current_epoch_display % save_every_n_epochs == 0:
            epoch_checkpoint_path = checkpoint_dir_run / f"model_CNN3D_epoch_{current_epoch_display}.pth"
            torch.save(checkpoint_data, epoch_checkpoint_path)
            logger.info(f"Checkpoint guardado (época {current_epoch_display}): '{epoch_checkpoint_path}'")

        # Guardar el mejor modelo
        current_metric_for_best = val_loss if save_best_metric_type == "loss" else val_acc
        is_better = (current_metric_for_best < best_val_metric_value) if save_best_metric_type == "loss" \
            else (current_metric_for_best > best_val_metric_value)

        if is_better:
            best_val_metric_value = current_metric_for_best
            # Actualizar el valor en checkpoint_data antes de guardarlo como el mejor
            checkpoint_data['best_val_metric_value'] = best_val_metric_value
            best_checkpoint_path = checkpoint_dir_run / "model_CNN3D_best.pth"
            torch.save(checkpoint_data, best_checkpoint_path)
            logger.info(
                f"Nuevo mejor checkpoint guardado (Val {save_best_metric_type}: {best_val_metric_value:.4f}): '{best_checkpoint_path}'")

    ut.get_time_employed(t_start_training, "Entrenamiento.")


def evaluate_model(
        model,
        dataloader,
        criterion,
        device,
        per_epoch_eval=False,
):
    model.eval()

    if not per_epoch_eval:
        t_start = time.time()
        logger.info("Evaluación final iniciada...")

    # logger.info("Evaluación iniciada...")
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

    if not per_epoch_eval:
        ut.get_time_employed(t_start, "Evaluación final.")
        log_con = f"Pérdida promedio en validación: {avg_loss:.4f}, Precisión en validación: {accuracy:.4f}"
        print(log_con)
        logger.info(log_con)

    return avg_loss, accuracy


def main(args):
    logger.info("Iniciando carga de datos directamente desde vídeos.")
    logger.info(f"Frames por clip: {FRAMES_PER_CLIP}, Duración clip: {CLIP_DURATION_SEC}s")
    logger.info(f"Modelo esperará frames de tamaño HxW: {TARGET_SIZE_DATASET[0]}x{TARGET_SIZE_DATASET[1]}")

    start_epoch = 0
    initial_best_val_metric = None
    run_name_to_use = None
    resume_checkpoint_file_path_from_arg = args.resume_checkpoint_file
    actual_checkpoint_to_load = None  # para la ruta definitiva del checkpoint

    if resume_checkpoint_file_path_from_arg is not None:
        path_obj = Path(resume_checkpoint_file_path_from_arg)

        if path_obj.is_absolute():
            # Si el path proporcionado ya es absoluto, lo usamos directamente.
            actual_checkpoint_to_load = path_obj
            logger.info(f"Se proporcionó una ruta absoluta para reanudar: '{actual_checkpoint_to_load}'")
        else:
            actual_checkpoint_to_load = Path(M_BASIC) / resume_checkpoint_file_path_from_arg
            logger.info(f"Se proporcionó una ruta relativa, resolviendo a: '{actual_checkpoint_to_load}'")

        if actual_checkpoint_to_load.exists():
            # Esto asegura que los nuevos checkpoints continúen en el mismo directorio de la ejecución anterior.
            run_name_to_use = actual_checkpoint_to_load.parent.name
            logger.info(
                f"Se reanudará desde el checkpoint: '{actual_checkpoint_to_load}'. El nombre de la ejecución será: '{run_name_to_use}'.")
        else:
            logger.error(f"No existe checkpoint especificado: '{resume_checkpoint_file_path_from_arg}')")
            sys.exit(1)
    else:
        # Si no se especifica un checkpoint para reanudar, es una nueva ejecución.
        run_name_to_use = time.strftime("%Y%m%d-%H%M%S")
        logger.info(f"Iniciando nuevo entrenamiento: {run_name_to_use}")  # actual_checkpoint_to_load permanece None

    # print(run_name_to_use)
    checkpoint_dir_run = Path(M_BASIC) / run_name_to_use
    checkpoint_dir_run.mkdir(parents=True, exist_ok=True)

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

    # Creación y División del Dataset
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
        checkpoint = torch.load(actual_checkpoint_to_load, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch']
        initial_best_val_metric = checkpoint.get('best_val_metric_value')
        logger.info(
            f"Reanudando entrenamiento desde la época {start_epoch + 1}. LR actual: {optimizer.param_groups[0]['lr']:.7f}")

    # current_time_str = time.strftime("%Y%m%d-%H%M%S")
    # run_name = f"cnn3d_{args.epocas}_epochs_{current_time_str}"

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
    )

    # La evaluación final después del bucle de entrenamiento ahora es opcional, ya que guardamos el mejor modelo y el último durante el entrenamiento.
    # Pero puede ser útil para una última verificación del estado final del modelo.
    best_model_path = checkpoint_dir_run / "model_CNN3D_best.pth"
    if best_model_path.exists():
        logger.info(f"Cargando el mejor modelo desde '{best_model_path}' para la evaluación final.")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        logger.warning(f"No se encontró el archivo del mejor modelo. Evaluando el último estado del modelo.")

    evaluate_model(
        model=model,
        dataloader=val_dataloader,
        criterion=criterion,
        device=device,
        per_epoch_eval=False,
    )

    logger.info(f"Los checkpoints de la ejecución '{run_name_to_use}' se encuentran en '{checkpoint_dir_run}'")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Modelo CNN3D - entrenamiento y guardado, directamente desde videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--epocas",
        default=5,  # TODO
        type=ut.non_negative_int,
        help="Número de épocas para el entrenamiento.",
    )
    parser.add_argument(
        "--resume_checkpoint_file",
        #default="20250603-233152/model_CNN3D_best.pth",  # TODO
        default=None,  # TODO
        type=str,
        help="Reanudar entrenamiento desde un checkpoint.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    config_log()
    ut.verify_system()
    main(args)
