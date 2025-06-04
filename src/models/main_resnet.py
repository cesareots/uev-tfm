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
# Importar el modelo y los pesos específicos
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split

# Asumiendo que DatasetSoccernet y las constantes están en las rutas correctas
from src.preprocess.dataset_soccernet import DatasetSoccernet, NUM_CLASSES as ACTUAL_NUM_CLASSES, TARGET_FPS, \
    CLIP_DURATION_SEC # Usaremos ACTUAL_NUM_CLASSES para la cabeza del modelo
from src.utils import utils as ut
from src.utils.constants import * # Asegúrate que M_BASIC, SEMILLA, LOG_DIR, etc., estén aquí

logger = logging.getLogger(__name__)

# --- Parámetros de Configuración para Transfer Learning ---
# Parámetros de Entrenamiento Fijos
BATCH_SIZE = 8  # Ajustar según VRAM
INITIAL_LEARNING_RATE = 0.001 # Tasa de aprendizaje para la nueva capa clasificadora
# LEARNING_RATE_FINETUNE = 0.0001 # Tasa de aprendizaje más baja si se hace fine-tuning de todo el modelo después

# Para R(2+1)D_18 pre-entrenado en Kinetics, usualmente se usan 16 frames.
# Tu DatasetSoccernet se adaptará para muestrear esta cantidad.
TRANSFER_MODEL_FRAMES_PER_CLIP = 16

# Las transformaciones de los pesos pre-entrenados definirán el tamaño espacial (ej. 112x112)
# No necesitamos TARGET_SIZE_DATASET como antes, ya que las transformaciones lo dictarán.

# Checkpointing
EPOCAS_CHECKPOINT_SAVE_INTERVAL = 1 # Guardar cada época es bueno para transfer learning
SAVE_BEST_METRIC_TYPE = "loss"

# LR Scheduler
LR_SCHEDULER_PATIENCE = 2
LR_SCHEDULER_FACTOR = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Usando dispositivo: {device}")

torch.manual_seed(SEMILLA)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEMILLA)
np.random.seed(SEMILLA)
random.seed(SEMILLA)


def config_log() -> None:
    ut.verify_directory(LOG_DIR)
    # Asumimos que LOG_MODEL_CNN3D se puede renombrar o tener uno nuevo para transfer learning
    logging.basicConfig(
        filename=LOG_MODEL_CNN3D, # Considera un nuevo archivo de log, ej. LOG_MODEL_TRANSFER
        filemode="a",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        encoding="utf-8",
    )

def get_pretrained_r2plus1d_model(num_classes_output: int, freeze_backbone: bool = True):
    """
    Carga el modelo R(2+1)D_18 pre-entrenado en Kinetics-400,
    congela el backbone (opcionalmente) y reemplaza la cabeza de clasificación.
    """
    logger.info("Cargando modelo R(2+1)D_18 pre-entrenado con pesos Kinetics-400...")
    weights = R2Plus1D_18_Weights.KINETICS400_V1
    model = r2plus1d_18(weights=weights)

    if freeze_backbone:
        logger.info("Congelando pesos del backbone pre-entrenado.")
        for param in model.parameters():
            param.requires_grad = False
        # Solo los parámetros de la nueva capa 'fc' tendrán requires_grad = True por defecto.

    # Reemplazar la cabeza de clasificación (model.fc)
    num_original_features = model.fc.in_features
    model.fc = nn.Linear(num_original_features, num_classes_output)
    logger.info(f"Cabeza de clasificación reemplazada: {num_original_features} -> {num_classes_output} clases.")

    return model, weights.transforms() # Devolver también las transformaciones recomendadas


# Las funciones train_model y evaluate_model pueden ser muy similares
# a las de tu main_basic_cnn_3d.py, ya que la lógica del bucle es la misma.
# Asegúrate de que manejan correctamente el scheduler y el checkpointing.
# (Aquí se asume que tienes esas funciones ya definidas como en tu script anterior)

def train_model(
    model, optimizer, scheduler,
    train_dataloader, val_dataloader, criterion,
    num_epochs, device,
    checkpoint_dir_run, save_every_n_epochs, save_best_metric_type,
    start_epoch: int = 0, initial_best_val_metric=None
):
    t_start_training = time.time()
    logger.info(
        f"Entrenamiento iniciado... Épocas: {num_epochs - start_epoch} (desde {start_epoch + 1} hasta {num_epochs})")
    best_val_metric_value = initial_best_val_metric if initial_best_val_metric is not None \
        else (float('inf') if save_best_metric_type == "loss" else float('-inf'))
    logger.info(f"Guardando checkpoints en: {checkpoint_dir_run}")

    for epoch in range(start_epoch, num_epochs):
        current_epoch_display = epoch + 1
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        epoch_t_start = time.time()

        for i, batch_data in enumerate(train_dataloader):
            inputs, labels, _ = batch_data # Ignoramos video_paths aquí
            valid_indices = labels != -1
            inputs = inputs[valid_indices].to(device)
            labels = labels[valid_indices].to(device)
            if inputs.size(0) == 0: continue

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            if (i + 1) % 20 == 0: # Loguear cada 20 batches
                logger.info(f"Epoch [{current_epoch_display}/{num_epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

        epoch_train_loss = running_loss / total_samples if total_samples > 0 else 0
        epoch_train_acc = correct_predictions / total_samples if total_samples > 0 else 0
        epoch_duration = time.time() - epoch_t_start
        print(f"Epoch [{current_epoch_display}/{num_epochs}] (Train) Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} Dur: {epoch_duration:.2f}s")
        logger.info(f"Epoch [{current_epoch_display}/{num_epochs}] (Train) Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} Dur: {epoch_duration:.2f}s")

        val_loss, val_acc = evaluate_model(model, val_dataloader, criterion, device, per_epoch_eval=True)
        print(f"Epoch [{current_epoch_display}/{num_epochs}] (Val)   Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        logger.info(f"Epoch [{current_epoch_display}/{num_epochs}] (Val)   Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        current_lr_before_step = optimizer.param_groups[0]['lr']
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau): scheduler.step(val_loss)
            else: scheduler.step()
            if optimizer.param_groups[0]['lr'] < current_lr_before_step:
                 logger.info(f"LR reducida de {current_lr_before_step:.7f} a {optimizer.param_groups[0]['lr']:.7f}")
        logger.info(f"LR para próxima época: {optimizer.param_groups[0]['lr']:.7f}")

        checkpoint_data = {
            'epoch': current_epoch_display,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_loss': epoch_train_loss, 'train_acc': epoch_train_acc,
            'val_loss': val_loss, 'val_acc': val_acc,
            'best_val_metric_value': best_val_metric_value,
        }

        if save_every_n_epochs > 0 and current_epoch_display % save_every_n_epochs == 0:
            path = checkpoint_dir_run / f"r2plus1d_epoch_{current_epoch_display}.pth"
            torch.save(checkpoint_data, path)
            logger.info(f"Checkpoint guardado: '{path}'")

        current_metric_for_best = val_loss if save_best_metric_type == "loss" else val_acc
        is_better = (current_metric_for_best < best_val_metric_value) if save_best_metric_type == "loss" \
            else (current_metric_for_best > best_val_metric_value)
        if is_better:
            best_val_metric_value = current_metric_for_best
            checkpoint_data['best_val_metric_value'] = best_val_metric_value
            path = checkpoint_dir_run / "r2plus1d_best.pth"
            torch.save(checkpoint_data, path)
            logger.info(f"Nuevo mejor checkpoint (Val {save_best_metric_type}: {best_val_metric_value:.4f}): '{path}'")
            
    ut.get_time_employed(t_start_training, "Entrenamiento total.")


def evaluate_model(model, dataloader, criterion, device, per_epoch_eval=False):
    if not per_epoch_eval:
        t_start = time.time()
        logger.info("Evaluación final estándar iniciada...")
    
    model.eval()
    running_loss, correct_predictions, total_samples = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            valid_indices = labels != -1
            inputs = inputs[valid_indices].to(device)
            labels = labels[valid_indices].to(device)
            if inputs.size(0) == 0: continue
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
    avg_loss = running_loss / total_samples if total_samples > 0 else 0
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    
    if not per_epoch_eval:
        ut.get_time_employed(t_start, "Evaluación final estándar.")
        print(f"Evaluación Final - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")
        logger.info(f"Evaluación Final - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")
    return avg_loss, accuracy


def main(args):
    logger.info("Iniciando Transfer Learning con R(2+1)D_18.")
    
    # --- Configuración de Ejecución y Reanudación ---
    start_epoch = 0
    initial_best_val_metric = None
    run_name_to_use = None
    actual_checkpoint_to_load = None

    if args.resume_checkpoint_file:
        path_obj = Path(args.resume_checkpoint_file)
        if path_obj.is_absolute(): actual_checkpoint_to_load = path_obj
        else: actual_checkpoint_to_load = Path(M_BASIC) / args.resume_checkpoint_file # Asume que M_BASIC es el dir base de runs
        
        if actual_checkpoint_to_load.exists():
            run_name_to_use = actual_checkpoint_to_load.parent.name
            logger.info(f"Reanudando desde: '{actual_checkpoint_to_load}'. Run name: '{run_name_to_use}'.")
        else:
            logger.error(f"No existe checkpoint: '{actual_checkpoint_to_load}'")
            sys.exit(1)
    else:
        current_time_str = time.strftime("%Y%m%d-%H%M%S")
        run_name_to_use = f"r2plus1d_e{args.epocas}_lr{INITIAL_LEARNING_RATE}_{current_time_str}"
        logger.info(f"Nueva ejecución: {run_name_to_use}")

    checkpoint_dir_run = Path(M_BASIC) / run_name_to_use # M_BASIC debe ser tu directorio base para modelos/runs
    checkpoint_dir_run.mkdir(parents=True, exist_ok=True)

    # --- Modelo ---
    # Usar ACTUAL_NUM_CLASSES de tu dataset_soccernet o constants
    model, video_transforms = get_pretrained_r2plus1d_model(num_classes_output=ACTUAL_NUM_CLASSES, freeze_backbone=True)
    model = model.to(device)
    
    logger.info("Transformaciones de vídeo para el modelo pre-entrenado:")
    logger.info(video_transforms)

    # --- Carga de Datos ---
    # Usar TRANSFER_MODEL_FRAMES_PER_CLIP para DatasetSoccernet
    # No es necesario target_size en DatasetSoccernet, las transformaciones lo manejan.
    base_dataset = DatasetSoccernet(
        root_dir=DS_SOCCERNET_ACTIONS, label_map=SOCCERNET_LABELS,
        frames_per_clip=TRANSFER_MODEL_FRAMES_PER_CLIP, # Importante: frames que espera el modelo
        target_fps=TARGET_FPS, # Esto es conceptual para tu DatasetSoccernet
        transform=None # Las transformaciones se aplican después de crear los subconjuntos
    )
    if not base_dataset: sys.exit("Dataset base vacío.")
    
    all_items = base_dataset.video_items
    train_size = int(0.8 * len(all_items))
    val_size = len(all_items) - train_size
    generator = torch.Generator().manual_seed(SEMILLA)
    train_indices, val_indices = random_split(range(len(all_items)), [train_size, val_size], generator=generator)

    train_items = [all_items[i] for i in train_indices]
    val_items = [all_items[i] for i in val_indices]

    # Aplicar las transformaciones específicas del modelo pre-entrenado
    # Nota: DatasetSoccernet aplica su 'transform' internamente.
    # Si video_transforms necesita ser aplicado al OUTPUT de DatasetSoccernet (el tensor)
    # o si DatasetSoccernet debe tomar una pipeline de T_v2.Compose directamente, hay que ajustar.
    # Asumimos que video_transforms es una T_v2.Compose que DatasetSoccernet puede usar.
    train_dataset = DatasetSoccernet(video_items_list=train_items, label_map=SOCCERNET_LABELS,
                                     frames_per_clip=TRANSFER_MODEL_FRAMES_PER_CLIP, target_fps=TARGET_FPS,
                                     transform=video_transforms) # Aplicar transformaciones aquí
    val_dataset = DatasetSoccernet(video_items_list=val_items, label_map=SOCCERNET_LABELS,
                                   frames_per_clip=TRANSFER_MODEL_FRAMES_PER_CLIP, target_fps=TARGET_FPS,
                                   transform=video_transforms) # Usar las mismas para validación (sin aumento aleatorio)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # --- Optimizador y Scheduler ---
    # Entrenar solo la cabeza clasificadora nueva si freeze_backbone=True
    params_to_optimize = model.fc.parameters() if model.fc.weight.requires_grad else model.parameters()
    optimizer = optim.Adam(params_to_optimize, lr=INITIAL_LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE, verbose=False)
    logger.info(f"Optimizando parámetros de: {'model.fc' if model.fc.weight.requires_grad else 'todo el modelo'}")


    # --- Cargar Checkpoint si se especifica ---
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
            logger.warning(f"No se pudo cargar el estado del optimizador/scheduler, posiblemente por cambio en parámetros a optimizar: {e}. Se reiniciará el optimizador/scheduler.")
            # Re-inicializar el optimizador para los parámetros correctos si falla la carga
            optimizer = optim.Adam(params_to_optimize, lr=INITIAL_LEARNING_RATE)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE, verbose=False)


        start_epoch = checkpoint['epoch']
        initial_best_val_metric = checkpoint.get('best_val_metric_value')
        logger.info(f"Reanudando desde época {start_epoch + 1}. LR: {optimizer.param_groups[0]['lr']:.7f}")
    else:
        logger.info(f"Iniciando nuevo entrenamiento desde época {start_epoch + 1}.")

    # --- Entrenamiento ---
    train_model(
        model=model, optimizer=optimizer, scheduler=scheduler,
        train_dataloader=train_dataloader, val_dataloader=val_dataloader, criterion=nn.CrossEntropyLoss(),
        num_epochs=args.epocas, device=device,
        checkpoint_dir_run=checkpoint_dir_run,
        save_every_n_epochs=EPOCAS_CHECKPOINT_SAVE_INTERVAL,
        save_best_metric_type=SAVE_BEST_METRIC_TYPE,
        start_epoch=start_epoch, initial_best_val_metric=initial_best_val_metric
    )
    
    logger.info("Entrenamiento finalizado.")
    
    best_model_path = checkpoint_dir_run / "r2plus1d_best.pth"
    if best_model_path.exists():
        logger.info(f"Cargando mejor modelo desde '{best_model_path}' para evaluación final.")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict']) # Cargar solo los pesos del modelo
    else:
        logger.warning("No se encontró el mejor modelo. Evaluando el último estado.")

    evaluate_model(model, val_dataloader, nn.CrossEntropyLoss(), device, per_epoch_eval=False)
    logger.info(f"Checkpoints de '{run_name_to_use}' en '{checkpoint_dir_run}'")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Transfer Learning con R(2+1)D_18.")
    parser.add_argument("--epocas", default=10, type=ut.non_negative_int, help="Número de épocas.")
    parser.add_argument("--resume_checkpoint_file", default=None, type=str,
                        help="Ruta a checkpoint (.pth) para reanudar (relativa a M_BASIC o absoluta).")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    config_log() # Configurar el logging
    ut.verify_system() # Verificar sistema si es necesario
    main(args)
