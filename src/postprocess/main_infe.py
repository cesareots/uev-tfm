import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.io
import torchvision.transforms.v2 as T_v2
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from tqdm import tqdm

from src.models.main_cnn3d import SimpleCNN3D, FRAMES_PER_CLIP, TARGET_SIZE_DATASET
from src.models.transforms import get_transforms_cnn3d, get_transforms_resnet
from src.preprocess.dataset_soccernet import INV_LABEL_MAP
from src.preprocess.dataset_soccernet import get_output_size_from_transforms
from src.utils import utils as ut
from src.utils.constants import SOCCERNET_LABELS, LOG_DIR, LOG_INFERENCE

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Usando dispositivo: {device}")


def config_log() -> None:
    ut.verify_directory(LOG_DIR)

    logging.basicConfig(
        filename=LOG_INFERENCE,
        filemode="a",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        encoding="utf-8",
    )


def load_model_and_transforms(
        checkpoint_path: Path,
        model_type: str,
        num_classes: int,
        device: torch.device,
):
    """
    Carga un modelo entrenado desde un checkpoint y define las transformaciones de inferencia.
    """
    logger.info(f"Cargando checkpoint desde '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    transforms = None

    if model_type == 'resnet':
        logger.info("Instanciando modelo RESNET.")
        weights = R2Plus1D_18_Weights.KINETICS400_V1
        train_transforms, val_transforms = get_transforms_resnet(weights)
        transforms = val_transforms
        model = r2plus1d_18(num_classes=num_classes)
    elif model_type == 'cnn3d':
        logger.info("Instanciando modelo CNN3D.")
        train_transforms, val_transforms = get_transforms_cnn3d(TARGET_SIZE_DATASET)
        transforms = val_transforms
        expected_size = get_output_size_from_transforms(transforms)
        model = SimpleCNN3D(
            num_classes=num_classes,
            input_frames=FRAMES_PER_CLIP,
            input_size=expected_size,
        )
    else:
        log_con = f"Tipo de modelo '{model_type}' no soportado. Elija 'resnet' o 'cnn3d'."
        logger.error(log_con)
        raise ValueError(log_con)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, transforms


def run_sliding_window(
        model: nn.Module,
        video_path: Path,
        transforms: T_v2.Compose,
        device: torch.device,
        clip_duration: float,
        frames_per_clip: int,
        stride: float,
):
    """
    Fase 2: Ejecuta el modelo sobre un vídeo largo usando una ventana deslizante.
    """
    logger.info(f"Iniciando Fase 2: Análisis con ventana deslizante (stride: {stride} seg)")
    info = torchvision.io.video_info(str(video_path))
    video_duration = info.duration
    logger.info(f"Duración del vídeo: {video_duration:.2f} segundos.")
    raw_predictions = []

    # 'tqdm' para una barra de progreso
    pbar_title = "Procesando vídeo"

    with tqdm(total=int(video_duration / stride), desc=pbar_title) as pbar:
        for start_sec in np.arange(0, video_duration - clip_duration, stride):
            end_sec = start_sec + clip_duration

            try:
                # Cargar solo el clip actual para ser eficiente con la memoria
                clip, audio, info = torchvision.io.read_video(
                    str(video_path),
                    start_pts=start_sec,
                    end_pts=end_sec,
                    pts_unit='sec',
                    output_format="TCHW"
                )
            except Exception as e:
                logger.warning(f"No se pudo leer el clip de {start_sec:.2f}s a {end_sec:.2f}s. Saltando. {e}")
                pbar.update(1)
                continue

            # Muestreo de frames (similar a DatasetSoccernet)
            num_total_frames = clip.shape[0]
            if num_total_frames > 0:
                indices = np.linspace(0, num_total_frames - 1, frames_per_clip, dtype=int)
                sampled_clip = clip[indices]
            else:
                pbar.update(1)
                continue

            # Pre-procesamiento
            processed_clip = transforms(sampled_clip)
            processed_clip = processed_clip.permute(1, 0, 2, 3)  # (C, T, H, W)

            # Inferencia
            with torch.no_grad():
                input_tensor = processed_clip.unsqueeze(0).to(device)  # Añadir dimensión de batch
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)

                prediction = {
                    "timestamp": start_sec,
                    "prediction_idx": predicted_idx.item(),
                    "class_name": INV_LABEL_MAP.get(predicted_idx.item(), "Desconocido"),
                    "confidence": confidence.item()
                }
                raw_predictions.append(prediction)

            pbar.update(1)

    logger.info(f"Fase 2 completada. Se generaron {len(raw_predictions)} predicciones crudas.")

    return raw_predictions


def post_process_predictions(
        raw_predictions: list,
        confidence_threshold: float,
        min_event_duration: float,
        stride: float
):
    """
    Fase 3: Filtra y agrupa las predicciones crudas para generar una lista de eventos.
    """
    logger.info(f"Iniciando Fase 3: Post-procesamiento (umbral={confidence_threshold}, dur_min={min_event_duration}s)")
    # Filtrar por confianza
    confident_preds = [p for p in raw_predictions if p['confidence'] >= confidence_threshold]

    if not confident_preds:
        logger.warning("No se encontraron predicciones por encima del umbral de confianza.")
        return []

    # Agrupar predicciones consecutivas de la misma clase en eventos
    events = []

    if confident_preds:
        current_event = {
            "evento": confident_preds[0]['class_name'],
            "inicio": confident_preds[0]['timestamp'],
            "fin": confident_preds[0]['timestamp'] + stride,
            "confianzas": [confident_preds[0]['confidence']]
        }

        for i in range(1, len(confident_preds)):
            pred = confident_preds[i]
            prev_pred = confident_preds[i - 1]

            # Si la clase es la misma y el tiempo es consecuente (dentro de stride * 1.5 para ser flexible)
            if pred['class_name'] == current_event['evento'] and (pred['timestamp'] - prev_pred['timestamp']) < (
                    stride * 1.5):
                current_event['fin'] = pred['timestamp'] + stride
                current_event['confianzas'].append(pred['confidence'])
            else:
                # Calcular la confianza promedio y guardar el evento anterior
                current_event['confianza_promedio'] = np.mean(current_event['confianzas'])
                del current_event['confianzas']  # Limpiar la lista de confianzas
                events.append(current_event)

                # Iniciar un nuevo evento
                current_event = {
                    "evento": pred['class_name'],
                    "inicio": pred['timestamp'],
                    "fin": pred['timestamp'] + stride,
                    "confianzas": [pred['confidence']]
                }

        # Añadir el último evento
        current_event['confianza_promedio'] = np.mean(current_event['confianzas'])
        del current_event['confianzas']
        events.append(current_event)

    # Filtrar eventos que son demasiado cortos
    final_events = [e for e in events if (e['fin'] - e['inicio']) >= min_event_duration]
    logger.info(f"Se detectaron {len(final_events)} eventos significativos.")

    return final_events


def main(args):
    model, transforms = load_model_and_transforms(
        checkpoint_path=Path(args.checkpoint_path),
        model_type=args.model_type,
        num_classes=len(SOCCERNET_LABELS),
        device=device,
    )

    raw_predictions = run_sliding_window(
        model=model,
        video_path=Path(args.video_path),
        transforms=transforms,
        device=device,
        clip_duration=args.clip_duration,
        frames_per_clip=args.frames_per_clip,
        stride=args.stride,
    )

    final_events = post_process_predictions(
        raw_predictions=raw_predictions,
        confidence_threshold=args.confidence_threshold,
        min_event_duration=args.min_event_duration,
        stride=args.stride
    )

    print("\n" + "=" * 50)
    print("RESULTADOS FINALES")
    print("=" * 50)

    if not final_events:
        print("No se detectaron eventos significativos que cumplan con los criterios.")
    else:
        for event in final_events:
            start_time_str = time.strftime('%H:%M:%S', time.gmtime(event['inicio']))
            end_time_str = time.strftime('%H:%M:%S', time.gmtime(event['fin']))
            print(f"Evento: {event['evento']:<15} | "
                  f"Inicio: {start_time_str} ({event['inicio']:.2f}s) | "
                  f"Fin: {end_time_str} ({event['fin']:.2f}s) | "
                  f"Confianza Promedio: {event['confianza_promedio']:.2f}")

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_events, f, indent=4)

        logger.info(f"Resultados guardados en: {output_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Inferencia sobre un vídeo completo para detectar eventos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video_path",
        default="inference/england_epl/2016-2017/2017-05-06 - 17-00 Leicester 3 - 0 Watford/1_720p.mkv",  # TODO
        required=True,
        type=str,
        help="Ruta al archivo de vídeo del partido completo.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default="models/RESNET/20250607-032043/model_RESNET_best.pth",  # TODO
        required=True,
        type=str,
        help="Ruta al archivo de checkpoint (.pth) del modelo entrenado.",
    )
    parser.add_argument(
        "--model_type",
        default="resnet",  # TODO
        required=True,
        type=str,
        choices=["resnet", "cnn3d"],
        help="Tipo de arquitectura del modelo cargado.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Ruta opcional para guardar la lista de eventos en formato JSON.",
    )
    # Parámetros de inferencia
    parser.add_argument(
        "--stride",
        type=float,
        default=1.0,
        help="Segundos a deslizar la ventana en cada paso.",
    )
    parser.add_argument(
        "--clip_duration",
        type=float,
        default=3.0,  # TODO deberia ser igual a la duración de los clips de entrenamiento?
        help="Duración en segundos de cada clip a analizar.",
    )
    parser.add_argument(
        "--frames_per_clip",
        type=int,
        default=16,  # TODO
        help="Número de frames a muestrear por clip (debe coincidir con el entrenamiento si es posible).",
    )
    # Parámetros de post-procesamiento
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.7,  # TODO
        help="Umbral de confianza para considerar una predicción válida.",
    )
    parser.add_argument(
        "--min_event_duration",
        type=float,
        default=2.0,
        help="Duración mínima en segundos para que un evento sea considerado válido.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    config_log()
    ut.verify_system()
    main(args)
