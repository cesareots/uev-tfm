import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torchvision.io
import torchvision.transforms.v2 as T_v2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset

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


class DatasetSoccernet(Dataset):
    def __init__(self, root_dir, label_map, frames_per_clip, target_fps, transform=None, expected_output_size=None,
                 video_items_list=None):
        self.label_map = label_map
        self.frames_per_clip = frames_per_clip
        self.target_fps = target_fps
        self.transform = transform
        self.expected_output_size = expected_output_size

        if video_items_list is not None:
            self.video_items = video_items_list
            self.root_dir = None  # No es estrictamente necesario si los paths en video_items_list son absolutos
            #logger.info(f"DatasetSoccernet inicializado con {len(self.video_items)} videos predefinidos.")
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

            # Estas transformaciones deben llevar el frame al tamaño final que espera el modelo.
            if self.transform:
                video_tensor_processed = self.transform(video_tensor_processed)
            # video_tensor_processed shape: (FRAMES_PER_CLIP, C, H_final_model, W_final_model)

            video_tensor_final = video_tensor_processed.permute(1, 0, 2, 3)  # (C, T, H, W)
            label_tensor = torch.tensor(label, dtype=torch.long)

            return video_tensor_final, label_tensor, video_path_str  # video_path_str es útil para debugging
        except Exception as e:
            logger.error(f"Al cargar o procesar el video '{video_path_str}': {e}")
            # Si se produce un error, devuelve un tensor dummy y una etiqueta -1
            # El tamaño del dummy tensor debe coincidir con lo que el modelo espera después de las transformaciones.
            dummy_c = 3
            dummy_t = self.frames_per_clip
            dummy_h, dummy_w = self.expected_output_size
            #logger.warning(f"Creando tensor dummy: ({dummy_c}, {dummy_t}, {dummy_h}, {dummy_w})")
            dummy_video = torch.zeros((dummy_c, dummy_t, dummy_h, dummy_w), dtype=torch.float)
            dummy_label = torch.tensor(-1, dtype=torch.long)  # Etiqueta inválida

            return dummy_video, dummy_label, video_path_str


def get_output_size_from_transforms(transforms_pipeline: T_v2.Compose) -> tuple | None:
    """
    Inspecciona una pipeline de T_v2.Compose para encontrar el tamaño de salida (H, W).
    Itera en reversa a través de las transformaciones para encontrar la última instancia de Resize, CenterCrop, RandomCrop, o RandomResizedCrop que defina un tamaño explícito (H, W).

    Args:
        transforms_pipeline (T_v2.Compose): La pipeline de transformaciones.

    Returns:
        tuple | None: Una tupla (Alto, Ancho) si se encuentra, o None si no.
    """
    output_size = None

    # Iteramos en reversa porque la última transformación que define el tamaño es la que cuenta.
    for transform in reversed(transforms_pipeline.transforms):
        # Verificar si la transformación tiene un atributo 'size'
        if hasattr(transform, 'size'):
            size_attr = transform.size
            # 'size' puede ser un int o una tupla/lista. Solo interesa cuando es una tupla (H, W) que define explícitamente el tamaño de salida.
            if isinstance(size_attr, (list, tuple)) and len(size_attr) == 2:
                output_size = tuple(size_attr)
                logger.info(
                    f"Tamaño de salida detectado de la transformación '{type(transform).__name__}': {output_size}")

                return output_size

    logger.warning("No se pudo detectar un tamaño de salida explícito (H, W) en la pipeline de transformaciones.")

    return None


def crear_dividir_dataset(
        frames_per_clip: int,
        target_fps: float,
        train_transforms: T_v2.Compose,
        val_transforms: T_v2.Compose,
        expected_output_size: tuple,
        batch_size: int,
):
    logger.info("1. Creando dataset base para obtener la lista de todos los vídeos.")
    # Instancia base solo para recolectar todos los video_items. Sin transformaciones aquí.
    base_dataset_for_items = DatasetSoccernet(
        root_dir=DS_SOCCERNET_ACTIONS,
        label_map=SOCCERNET_LABELS,
        frames_per_clip=frames_per_clip,
        target_fps=target_fps,
        transform=None,  # No aplicar transformaciones
        expected_output_size=expected_output_size,
    )

    if len(base_dataset_for_items) == 0:
        logger.error(f"'{DS_SOCCERNET_ACTIONS}' está vacío.")
        sys.exit(1)

    # lista (path, label, action_name)
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

    # Aplicar las transformaciones específicas del modelo
    logger.info(f"3. Creando dataset de entrenamiento ({len(train_video_items_subset)} videos).")
    # Crear el DatasetSoccernet para ENTRENAMIENTO, pasándole solo su subconjunto de vídeos y sus transformaciones
    train_dataset = DatasetSoccernet(
        root_dir=None,
        label_map=SOCCERNET_LABELS,
        frames_per_clip=frames_per_clip,
        target_fps=target_fps,
        transform=train_transforms,
        expected_output_size=expected_output_size,
        video_items_list=train_video_items_subset,
    )

    logger.info(f"4. Creando dataset de validación ({len(val_video_items_subset)} vídeos) con val_transforms")
    # Crear el DatasetSoccernet para VALIDACIÓN, pasándole solo su subconjunto de vídeos y sus transformaciones
    val_dataset = DatasetSoccernet(
        root_dir=None,
        label_map=SOCCERNET_LABELS,
        frames_per_clip=frames_per_clip,
        target_fps=target_fps,
        transform=val_transforms,
        expected_output_size=expected_output_size,
        video_items_list=val_video_items_subset,
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        logger.error("Uno de los datasets (train o val) está vacío después del split. Verificar.")
        sys.exit(1)

    logger.info("5. Creando DataLoaders.")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_dataloader, val_dataloader


def crear_dividir_dataset_t_v_t(
        frames_per_clip: int,
        target_fps: float,
        train_transforms: T_v2.Compose,
        validation_test_transforms: T_v2.Compose,
        expected_output_size: tuple,
        batch_size: int,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
):
    """
    Crea y divide el dataset en entrenamiento, validación y prueba, asegurando
    la estratificación de clases, y devuelve sus respectivos DataLoaders.
    """
    # Validar que los porcentajes suman 1.0
    if not np.isclose(train_split + val_split + test_split, 1.0):
        raise ValueError("Los porcentajes de train, val, y test deben sumar 1.0")

    logger.info("1. Creando dataset base para obtener la lista de todos los vídeos.")
    base_dataset = DatasetSoccernet(
        root_dir=DS_SOCCERNET_ACTIONS,
        label_map=SOCCERNET_LABELS,
        frames_per_clip=frames_per_clip,
        target_fps=target_fps,
        transform=None,  # No aplicar transformaciones
        expected_output_size=expected_output_size,
    )
    if len(base_dataset) == 0:
        logger.error(f"'{DS_SOCCERNET_ACTIONS}' está vacío.")
        sys.exit(1)

    # lista (path, label, action_name)
    all_video_items = base_dataset.video_items

    logger.info(f"2. Extraer los paths (X) y las etiquetas (y) para la estratificación.")
    paths = [item[0] for item in all_video_items]
    labels = [item[1] for item in all_video_items]

    logger.info(f"3. Primera división: separar el conjunto de Test")
    # Creamos un conjunto de entrenamiento temporal (train+val) y el conjunto de test final
    train_val_indices, test_indices = train_test_split(
        range(len(paths)),  # Dividir sobre los índices
        test_size=test_split,
        stratify=labels,  # Mantiene la proporción de clases
        random_state=SEMILLA,
    )

    logger.info(f"4. Segunda división: separar Train y Validation del conjunto temporal")
    # Calculamos el tamaño de la validación como proporción del conjunto train_val restante
    val_size_proportion = val_split / (train_split + val_split)

    train_indices, val_indices = train_test_split(
        train_val_indices,  # Dividir el conjunto de índices que no son de test
        test_size=val_size_proportion,
        stratify=[labels[i] for i in train_val_indices],  # Estratificar sobre las etiquetas de este subconjunto
        random_state=SEMILLA,
    )

    # Reconstruir las listas de items para cada conjunto
    train_items = [all_video_items[i] for i in train_indices]
    val_items = [all_video_items[i] for i in val_indices]
    test_items = [all_video_items[i] for i in test_indices]

    # Crear los tres Datasets
    train_dataset = DatasetSoccernet(
        root_dir=None,
        label_map=SOCCERNET_LABELS,
        frames_per_clip=frames_per_clip,
        target_fps=target_fps,
        transform=train_transforms,
        expected_output_size=expected_output_size,
        video_items_list=train_items,
    )
    val_dataset = DatasetSoccernet(
        root_dir=None,
        label_map=SOCCERNET_LABELS,
        frames_per_clip=frames_per_clip,
        target_fps=target_fps,
        transform=validation_test_transforms,
        expected_output_size=expected_output_size,
        video_items_list=val_items,
    )
    test_dataset = DatasetSoccernet(
        root_dir=None,
        label_map=SOCCERNET_LABELS,
        frames_per_clip=frames_per_clip,
        target_fps=target_fps,
        transform=validation_test_transforms,
        expected_output_size=expected_output_size,
        video_items_list=test_items,
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        logger.error("Uno de los datasets (train/validation/test) está vacío después del split. Verificar.")
        sys.exit(1)

    # Crear los tres DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        # num_workers=2,  # Número de subprocesos para la carga de datos.
        num_workers=0,
        pin_memory=True,  # Optimización para transferencias a GPU.
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        # num_workers=2,
        num_workers=0,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        # num_workers=2,
        num_workers=0,
        pin_memory=True,
    )

    logger.info(f"5. Datasets y DataLoaders creados. "
                f"Train ({len(train_items)} videos). Validation ({len(val_items)} videos). Test ({len(test_items)} videos).")

    return train_dataloader, val_dataloader, test_dataloader
