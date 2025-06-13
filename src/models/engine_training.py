import logging
import sys
import time
from pathlib import Path

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.utils import utils as ut

logger = logging.getLogger(__name__)


def train_model(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        num_epochs: int,
        device: torch.device,
        checkpoint_dir_run: Path,  # Directorio para una ejecución específica
        save_every_n_epochs: int,  # Frecuencia para guardar checkpoints de época
        save_best_metric_type: str,  # 'loss' o 'accuracy' para el mejor modelo
        start_epoch: int = 0,
        initial_best_val_metric=None,  # Para reanudar el mejor valor
        checkpoint_base_name: str = "model_name",
):
    # model.train()
    t_start_training = time.time()
    logger.info(
        f"Entrenamiento iniciado ({checkpoint_base_name})... Número de épocas: {num_epochs - start_epoch} (desde {start_epoch + 1} hasta {num_epochs})")

    # lleva el registro del mejor valor de la métrica de validación encontrado hasta el momento (para decidir cuándo guardar {checkpoint_base_name}_best.pth).
    best_val_metric_value = initial_best_val_metric if initial_best_val_metric is not None \
        else (float('inf') if save_best_metric_type == "loss" else float('-inf'))

    logger.info(f"Se guardarán los checkpoints en: '{checkpoint_dir_run}'")

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
            if (i + 1) % 25 == 0:
                log_con = f"Epoch [{current_epoch_display}/{num_epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}"
                print(log_con)
                logger.info(log_con)

        epoch_train_loss = running_loss / total_samples if total_samples > 0 else 0
        epoch_train_acc = correct_predictions / total_samples if total_samples > 0 else 0
        epoch_duration = ut.time_in_minutes(epoch_t_start)

        log_con_epoch = f"Epoch [{current_epoch_display}/{num_epochs}] ({checkpoint_base_name} - Entrenamiento de época actual), Loss: {epoch_train_loss:.4f}, Accuracy: {epoch_train_acc:.4f}, Duración {epoch_duration:.2f} minutos"
        print(log_con_epoch)
        logger.info(log_con_epoch)

        # Validación al final de cada época
        epoch_v_start = time.time()
        val_loss, val_acc = evaluate_model(
            model,
            val_dataloader,
            criterion,
            device,
            per_epoch_eval=True,
        )
        epoch_v_duration = ut.time_in_minutes(epoch_v_start)
        val_log_msg = f"Epoch [{current_epoch_display}/{num_epochs}] ({checkpoint_base_name} - Validación de época actual), Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, Duración {epoch_v_duration:.2f} minutos"
        print(val_log_msg)
        logger.info(val_log_msg)

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
                    f"LR reducida de {current_lr_before_step:.7f} a {new_lr:.7f}")

        logger.info(f"LR para la próxima época ({checkpoint_base_name}): {optimizer.param_groups[0]['lr']:.7f}")

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
            # Usar checkpoint_base_name para el nombre del archivo
            path = checkpoint_dir_run / f"{checkpoint_base_name}_epoch_{current_epoch_display}.pth"
            torch.save(checkpoint_data, path)
            logger.info(f"Checkpoint guardado: '{path}'")

        # Guardar el mejor modelo
        current_metric_for_best = val_loss if save_best_metric_type == "loss" else val_acc
        is_better = (current_metric_for_best < best_val_metric_value) if save_best_metric_type == "loss" \
            else (current_metric_for_best > best_val_metric_value)

        if is_better:
            best_val_metric_value = current_metric_for_best
            # Actualizar el valor en checkpoint_data antes de guardarlo como el mejor
            checkpoint_data['best_val_metric_value'] = best_val_metric_value
            # Usar checkpoint_base_name para el nombre del archivo
            path = checkpoint_dir_run / f"{checkpoint_base_name}_best.pth"
            torch.save(checkpoint_data, path)
            logger.info(
                f"Nuevo mejor checkpoint guardado (Val {save_best_metric_type}: {best_val_metric_value:.4f}): '{path}'")

    ut.get_time_employed(t_start_training, f"Entrenamiento finalizado ({checkpoint_base_name}).")


def evaluate_model(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        device: torch.device,
        per_epoch_eval: bool = False,
):
    if not per_epoch_eval:
        t_start = time.time()
        logger.info("Evaluación final iniciada...")

    model.eval()
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
        log_con = f"Evaluación final - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}"
        print(log_con)
        logger.info(log_con)

    return avg_loss, accuracy


def extras(
        model_name_dir: str,
        resume_checkpoint_file_path_from_arg: str,
):
    actual_checkpoint_to_load = None

    if resume_checkpoint_file_path_from_arg is not None:
        path_obj = Path(resume_checkpoint_file_path_from_arg)

        if path_obj.is_absolute():
            # Si el path proporcionado ya es absoluto, lo usamos directamente.
            actual_checkpoint_to_load = path_obj
            logger.info(f"Se proporcionó una ruta absoluta para reanudar: '{actual_checkpoint_to_load}'")
        else:
            actual_checkpoint_to_load = Path(model_name_dir) / resume_checkpoint_file_path_from_arg
            logger.info(f"Se proporcionó una ruta relativa, resolviendo a: '{actual_checkpoint_to_load}'")

        if actual_checkpoint_to_load.exists():
            # Esto asegura que los nuevos checkpoints continúen en el mismo directorio de la ejecución anterior.
            run_name_to_use = actual_checkpoint_to_load.parent.name
            logger.info(f"Reanudando desde checkpoint: '{actual_checkpoint_to_load}'. "
                        f"Nombre de la ejecución: '{run_name_to_use}'.")
        else:
            logger.error(f"No existe checkpoint especificado: '{actual_checkpoint_to_load}')")
            sys.exit(1)
    else:
        # Si no se especifica un checkpoint para reanudar, significa una nueva ejecución.
        run_name_to_use = time.strftime("%Y%m%d-%H%M%S")
        logger.info(f"Iniciando nuevo entrenamiento: {run_name_to_use}")  # actual_checkpoint_to_load permanece None

    # print(run_name_to_use)
    checkpoint_dir_run = Path(model_name_dir) / run_name_to_use
    checkpoint_dir_run.mkdir(parents=True, exist_ok=True)

    return actual_checkpoint_to_load, checkpoint_dir_run
