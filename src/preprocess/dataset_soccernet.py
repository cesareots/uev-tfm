import warnings
from pathlib import Path

import numpy as np
import torch
import torchvision.io
import torchvision.transforms.v2 as T_v2
from torch.utils.data import Dataset
import logging

from src.utils.constants import *

logger = logging.getLogger(__name__)

# ignorar el warning de la deprecación de video de torchvision.io
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torchvision.io._video_deprecation_warning"
)

NUM_CLASSES = len(SOCCERNET_LABELS)
# Inverso del mapeo para interpretar las predicciones
INV_LABEL_MAP = {v: k for k, v in SOCCERNET_LABELS.items()}

# Alto x Ancho original
ORIGINAL_SIZE = (720, 1280)

# Duración de los clips
CLIP_DURATION_SEC = 3.0

# frames deseados por clip para el modelo (muestreados)
# FRAMES_PER_CLIP = 16
FRAMES_PER_CLIP = 32
# FRAMES_PER_CLIP = 64

# Este TARGET_FPS ahora es un valor conceptual para el muestreo manual, no un parámetro directo de decodificación en torchvision.io.read_video
# servirá para 'torchcodec'
TARGET_FPS = float(FRAMES_PER_CLIP / CLIP_DURATION_SEC)


class DatasetSoccernet(Dataset):
    def __init__(self, root_dir, label_map, frames_per_clip, target_fps, target_size=None, transform=None,
                 video_items_list=None):
        self.label_map = label_map
        self.frames_per_clip = frames_per_clip
        self.target_fps = target_fps
        self.target_size = target_size
        self.transform = transform

        if video_items_list is not None:
            self.video_items = video_items_list
            self.root_dir = None  # No es estrictamente necesario si los paths en video_items_list son absolutos
            logger.info(f"DatasetSoccernet inicializado con {len(self.video_items)} videos predefinidos.")
        else:
            self.root_dir = Path(root_dir)
            self.video_items = []
            self._find_videos()  # Escanea el root_dir como antes
            logger.info(f"DatasetSoccernet encontró {len(self.video_items)} videos en {root_dir}.")

    def _find_videos(self):
        for action_name_str in self.label_map.keys():
            action_dir = self.root_dir / action_name_str

            if not action_dir.is_dir():
                logger.warning(f"Directorio '{action_dir}' no encontrado. Saltando.")
                continue

            label_id = self.label_map[action_name_str]

            for video_file in action_dir.glob('*.mkv'):
                self.video_items.append((str(video_file), label_id, action_name_str))

    def __len__(self):
        return len(self.video_items)

    def __getitem__(self, idx):
        video_path_str, label, action_name_str = self.video_items[idx]
        video_path = Path(video_path_str)

        try:
            # Decodifica el video. Aca no podemos pasar target_fps.
            video_tensor_raw, audio, info = torchvision.io.read_video(
                str(video_path),
                pts_unit='sec',
                # output_format="THWC",  # por defecto
                output_format="TCHW",  # Tiempo, Canales, Alto, Ancho
            )

            # TODO 'torchvision.io.read_video' pronto estará DEPRECATED
            # a la fecha 2025-05-22 pip no instala correctamente torchcodec==0.4.0
            # instala esta version: 0.0.0.dev0
            # video_tensor_raw, audio, info = torchcodec.read_video(
            #    str(video_path),
            #    fps=self.target_fps,  # Ahora se puede especificar FPS directamente
            #    output_format="CTHW",
            # )
            # video_tensor_raw = video_tensor_raw.permute(1, 0, 2, 3)

            # Muestreo Manual de Frames
            num_total_frames = video_tensor_raw.shape[0]

            if num_total_frames == 0:
                raise ValueError(f"No se pudieron decodificar frames del video: {video_path}")

            # Si hay menos frames de los que necesitamos, rellenar (padding)
            if num_total_frames < self.frames_per_clip:
                # Rellenar con ceros si el video es más corto
                padding_needed = self.frames_per_clip - num_total_frames
                # video_tensor_raw.shape[1:] para obtener (C, H, W)
                padding_shape = (padding_needed, *video_tensor_raw.shape[1:])
                zero_padding = torch.zeros(padding_shape, dtype=video_tensor_raw.dtype)
                video_tensor_processed = torch.cat([video_tensor_raw, zero_padding], dim=0)
            else:
                # Muestreo uniforme: seleccionar 'FRAMES_PER_CLIP' frames equidistantes
                indices = np.linspace(0, num_total_frames - 1, self.frames_per_clip, dtype=int)
                video_tensor_processed = video_tensor_raw[indices]
            # video_tensor_processed shape: (FRAMES_PER_CLIP, C, H_original, W_original)

            # Aplicar redimensionamiento INTERMEDIO si se especificó target_size aquí y no hay transformaciones que lo hagan primero
            # O si las transformaciones esperan una entrada de un tamaño específico.
            if self.target_size is not None:
                # Comprobar si las dimensiones ya son las deseadas (podría haber sido hecho por una transformacion previa si 'transform' se aplicara antes)
                if (video_tensor_processed.shape[2], video_tensor_processed.shape[3]) != self.target_size:
                    # T_v2.Resize espera (C,H,W) o (B,C,H,W) o (T,C,H,W) y las aplica
                    resize_intermediate = T_v2.Resize(self.target_size, antialias=True)
                    video_tensor_processed = resize_intermediate(video_tensor_processed)
            # Ahora video_tensor_processed tiene la forma (FRAMES_PER_CLIP, C, H_target_size, W_target_size) o (FRAMES_PER_CLIP, C, H_original, W_original)

            # APLICAR TRANSFORMACIONES DE AUMENTO DE DATOS / FINALES
            # Estas transformaciones deben llevar el frame al tamaño final que espera el modelo.
            if self.transform:
                video_tensor_processed = self.transform(video_tensor_processed)
            # video_tensor_processed shape: (FRAMES_PER_CLIP, C, H_final_model, W_final_model)

            video_tensor_final = video_tensor_processed.float() / 255.0
            video_tensor_final = video_tensor_final.permute(1, 0, 2, 3)  # (C, T, H, W)
            label_tensor = torch.tensor(label, dtype=torch.long)

            return video_tensor_final, label_tensor, video_path_str  # Devolver video_path_str es útil para debugging
        except Exception as e:
            logger.error(f"Al cargar o procesar el video {video_path_str}: {e}")
            # Si se produce un error, devuelve un tensor dummy y una etiqueta -1
            # El tamaño del dummy tensor debe coincidir con lo que el modelo espera después de las transformaciones.
            # Esto es difícil de determinar a priori sin ejecutar las transformaciones.
            # Considera obtener H_final_model y W_final_model de la configuración de transformaciones si es posible.
            # Por ahora, usamos valores genéricos o basados en una configuración esperada.
            # Para mayor robustez, el tamaño dummy debería ser el tamaño de salida de self.transform
            dummy_c = 3
            dummy_t = self.frames_per_clip

            if self.transform:
                # ADVERTENCIA: Si hay transformaciones, el tamaño dummy idealmente debería ser el tamaño de SALIDA de esas transformaciones (ej. TARGET_SIZE_DATASET).
                # Como DatasetSoccernet no conoce ese valor directamente desde basic_cnn_3d.py, por ahora, usaremos un tamaño conocido (ej. 256x256 si es tu estándar) o el original.
                # Usar el original es más seguro que un valor hardcodeado incorrecto si TARGET_SIZE_DATASET cambia.
                logger.warning("Creando tensor dummy con tamaño post-transformación (256,256) estimado.")
                dummy_h, dummy_w = (256, 256) # Asumimos que esto coincide con TARGET_SIZE_DATASET (H,W)
                # Si prefieres el original como fallback general:
                # dummy_h, dummy_w = (ORIGINAL_SIZE[0], ORIGINAL_SIZE[1])
            elif self.target_size: # Si se proveyó un target_size intermedio (y es (H,W))
                dummy_h, dummy_w = self.target_size
            else: # Si no hay transformaciones ni target_size, usar el tamaño original
                dummy_h, dummy_w = (ORIGINAL_SIZE[0], ORIGINAL_SIZE[1])

            dummy_video = torch.zeros((dummy_c, dummy_t, dummy_h, dummy_w), dtype=torch.float)
            dummy_label = torch.tensor(-1, dtype=torch.long)  # Etiqueta inválida

            return dummy_video, dummy_label, video_path_str
