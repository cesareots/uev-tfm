import argparse
import json
import logging
import time
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn as nn
import torchvision.io
import torchvision.transforms.v2 as T_v2
from tqdm import tqdm

from src.models.main_cnn3d import FRAMES_PER_CLIP
from src.models.main_resnet import TRANSFER_MODEL_FRAMES_PER_CLIP
from src.postprocess.load_models import load_model_and_transforms
from src.postprocess.video_resumen_engine import video_resumen_moviepy
from src.preprocess.dataset_soccernet import INV_LABEL_MAP
from src.utils import utils as ut
from src.utils.constants import SOCCERNET_LABELS, LOG_DIR, LOG_INFERENCE, MAS_MENOS_CLIPS, UMBRALES, BUFFER_SECONDS, \
    DURACION_MIN_EVENTO

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def run_sliding_window(
        model: nn.Module,
        video_path: Path,
        transforms: T_v2.Compose,
        device: torch.device,
        clip_duration: float,
        frames_per_clip: int,
        stride: float,
):
    logger.info(f"Análisis con ventana deslizante (stride: {stride} seg)")

    try:
        with av.open(str(video_path)) as container:
            video_stream = container.streams.video[0]
            # Usar la duración del stream (puede ser None)
            stream_duration_in_pts = video_stream.duration

            if stream_duration_in_pts is not None:
                # Si existe, la convertimos a segundos usando su 'time_base'
                video_duration = float(stream_duration_in_pts * video_stream.time_base)
            else:
                # Esta duración suele estar en microsegundos, por lo que la dividimos por 1,000,000.
                logger.warning(
                    f"La duración del stream no está disponible para '{video_path.name}'. Usando la duración del contenedor como fallback.")
                video_duration = float(container.duration / 1_000_000)

            if video_duration <= 0:
                raise ValueError("La duración detectada del vídeo es cero, negativa o no se pudo determinar.")
    except Exception as e:
        logger.error(f"No se pudo obtener la información de duración del vídeo '{video_path.name}' con PyAV: {e}")
        return []

    logger.info(f"Duración del vídeo detectada: {video_duration:.2f} segundos.")
    raw_predictions = []
    pbar_title = "Procesando vídeo"

    with tqdm(total=int((video_duration - clip_duration) / stride) + 1, desc=pbar_title) as pbar:
        for start_sec in np.arange(0, video_duration - clip_duration, stride):
            end_sec = start_sec + clip_duration

            try:
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

            num_total_frames = clip.shape[0]
            if num_total_frames > 0:
                indices = np.linspace(0, num_total_frames - 1, frames_per_clip, dtype=int)
                sampled_clip = clip[indices]
            else:
                pbar.update(1)
                continue

            processed_clip = transforms(sampled_clip)
            processed_clip = processed_clip.permute(1, 0, 2, 3)

            with torch.no_grad():
                input_tensor = processed_clip.unsqueeze(0).to(device)
                logits = model(input_tensor)
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)

                prediction = {
                    "timestamp": start_sec,
                    "prediction_idx": predicted_idx.item(),
                    "class_name": INV_LABEL_MAP.get(predicted_idx.item(), "Desconocido"),
                    "confidence": confidence.item()
                }
                raw_predictions.append(prediction)

            pbar.update(1)

    logger.info(f"Se generaron {len(raw_predictions)} predicciones crudas.")

    return raw_predictions


def post_process_predictions(
        raw_predictions: list,
        confidence_thresholds: dict,
        default_threshold: float,
        min_event_duration: float,
        stride: float,
        output_file: Path,
):
    # Filtrar por confianza usando los umbrales por clase
    confident_preds = []

    for p in raw_predictions:
        class_name = p['class_name']
        # Obtener el umbral específico para esta clase, o usar el por defecto si no está definido
        threshold = confidence_thresholds.get(class_name, default_threshold)

        if p['confidence'] >= threshold:
            confident_preds.append(p)

    if not confident_preds:
        logger.warning("No se encontraron predicciones que superaran sus respectivos umbrales de confianza.")
        return []

    logger.info(f"Se encontraron {len(confident_preds)} predicciones de alta confianza después del filtrado inicial.")

    # Agrupar predicciones consecutivas de la misma clase en eventos
    events = []

    if confident_preds:
        # primer evento
        current_event = {
            "evento": confident_preds[0]['class_name'],
            "inicio": confident_preds[0]['timestamp'],
            "fin": confident_preds[0]['timestamp'] + stride,
            "confianzas": [confident_preds[0]['confidence']]
        }

        for i in range(1, len(confident_preds)):
            pred = confident_preds[i]
            prev_pred = confident_preds[i - 1]

            # Si la clase es la misma y el tiempo es consecuente
            if pred['class_name'] == current_event['evento'] and (pred['timestamp'] - prev_pred['timestamp']) < (
                    stride * 1.5):
                current_event['fin'] = pred['timestamp'] + stride
                current_event['confianzas'].append(pred['confidence'])
            else:
                # Guardar el evento anterior y empezar uno nuevo
                current_event['confianza_promedio'] = np.mean(current_event['confianzas'])
                del current_event['confianzas']
                events.append(current_event)

                current_event = {
                    "evento": pred['class_name'],
                    "inicio": pred['timestamp'],
                    "fin": pred['timestamp'] + stride,
                    "confianzas": [pred['confidence']]
                }

        # Añadir el último evento de todos
        current_event['confianza_promedio'] = np.mean(current_event['confianzas'])
        del current_event['confianzas']
        events.append(current_event)

    # Filtrar eventos que son demasiado cortos
    final_events = [e for e in events if (e['fin'] - e['inicio']) >= min_event_duration]
    logger.info(
        f"Se detectaron {len(final_events)} eventos significativos. Para que un evento sea significativo debe durar al menos: {min_event_duration:.2f} segundos")

    if not final_events:
        logger.info("No se detectaron eventos significativos que cumplan con los criterios.")
    else:
        for event in final_events:
            # start_time_str = time.strftime("%H:%M:%S", time.gmtime(event['inicio']))
            start_time_str = time.strftime("%M:%S", time.gmtime(event["inicio"]))
            end_time_str = time.strftime("%M:%S", time.gmtime(event["fin"]))
            log_con = f"Evento: {event['evento']:<15} | Inicio: {start_time_str} ({event['inicio']:.2f}s) | Fin: {end_time_str} ({event['fin']:.2f}s) | Confianza promedio: {event['confianza_promedio']:.2f}"
            # print(log_con)
            logger.info(log_con)

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_events, f, indent=4)

    logger.info(f"Eventos detectados (json) guardados en: '{output_file}'")

    return final_events


def puente(
        frames_per_clip,
        ruta_video,
        stride,
        raw_predictions_json,
) -> list:
    model, transforms = load_model_and_transforms(
        checkpoint_path=Path(args.checkpoint_path),
        model_type=args.model_type,
        num_classes=len(SOCCERNET_LABELS),
        device=device,
        frames_per_clip=frames_per_clip,
    )
    raw_predictions = run_sliding_window(
        model=model,
        video_path=ruta_video,
        transforms=transforms,
        device=device,
        clip_duration=MAS_MENOS_CLIPS * 2,
        frames_per_clip=frames_per_clip,
        stride=stride,
    )
    ut.guardar_predicciones_json(
        predicciones=raw_predictions,
        ruta_archivo=raw_predictions_json,
    )

    return raw_predictions


def main(args):
    logger.info(f"Usando dispositivo: {device}")

    # configuracion previa
    ruta_video = Path(args.video_path)
    raw_predictions_json = ruta_video.parent / f"{ruta_video.stem}_predictions_raw.json"
    output_file = ruta_video.parent / f"{ruta_video.stem}_predictions_post_process.json"  # ruta para guardar la lista de eventos en formato JSON
    frames_per_clip = FRAMES_PER_CLIP if args.model_type == "CNN3D" else TRANSFER_MODEL_FRAMES_PER_CLIP
    # los clips duran 3.0 segundos; primera inferencia: [0-3], segunda inferencia: [2-5], tercera inferencia: [4-7], por tanto ningun limite de tiempo se escapa
    stride = 2.0
    umbral_defecto = 0.75
    logger.info(f"Usando umbrales de confianza por clase: {UMBRALES}")

    t_start = time.time()
    raw_predictions = puente(frames_per_clip, ruta_video, stride, raw_predictions_json)
    # raw_predictions = ut.leer_predicciones_json(raw_predictions_json)  # TODO solo para debug

    final_events = post_process_predictions(
        raw_predictions=raw_predictions,
        confidence_thresholds=UMBRALES,
        default_threshold=umbral_defecto,
        min_event_duration=DURACION_MIN_EVENTO,
        stride=stride,
        output_file=output_file,
    )
    # print(final_events)

    video_resumen_moviepy(
        final_events=final_events,
        original_video_path=ruta_video,
        buffer_seconds=BUFFER_SECONDS,
    )
    ut.get_time_employed(t_start, f"Inferencia final. Video-resumen generado.")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Inferencia sobre un vídeo completo para detectar eventos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_type",
        default="CNN3D",  # TODO
        # default="RESNET",  # TODO
        type=str,
        choices=["CNN3D", "RESNET"],
        help="Tipo de arquitectura del modelo a utilizar.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default="models/CNN3D/20250614-132942/model_CNN3D_best.pth",  # TODO
        # default="models/RESNET/20250607-032043/model_RESNET_best.pth",  # TODO
        type=str,
        help="Ruta al archivo de checkpoint (.pth) del modelo entrenado y elegido.",
    )
    parser.add_argument(
        "--video_path",
        default="inference/spain_laliga/2016-2017/2017-04-26 - 20-30 Barcelona 7 - 1 Osasuna/11 minutos/2_720p_recortado.mkv",
        # TODO
        type=str,
        help="Ruta del vídeo a inferir.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    config_log()
    ut.verify_system()
    main(args)
