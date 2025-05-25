from pathlib import Path
import warnings
import numpy as np
import torch
import torchvision.io
import torchvision.transforms.v2 as T_v2
from torch.utils.data import Dataset
from src.utils.constants import * # Asegúrate de que SOCCERNET_LABELS esté definido aquí

# ignorar el warning de la deprecación de video de torchvision.io
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torchvision.io._video_deprecation_warning"
)

# TODO: Asegúrate de que SOCCERNET_LABELS esté definido en src/utils/constants.py
# O define el LABEL_MAP aquí si es más conveniente para este script.
# Por ejemplo:
# SOCCERNET_LABELS = {
#     "Goal": 0,
#     "Yellow card": 1,
#     "Shots on target": 2,
#     # ... otras acciones
# }

NUM_CLASSES = len(SOCCERNET_LABELS) # Extraer automáticamente
# Inverso del mapeo para interpretar las predicciones
INV_LABEL_MAP = {v: k for k, v in SOCCERNET_LABELS.items()}

# Ancho x Alto original de los frames (W, H), soccernet establece estos tamaños para sus videos
ORIGINAL_SIZE = (398, 224)
# ORIGINAL_SIZE = (1280, 720) # Para videos 720p

# Duración de los clips
CLIP_DURATION_SEC = 3

# frames deseados por clip para el modelo (muestreados)
FRAMES_PER_CLIP = 16
# FRAMES_PER_CLIP = 32
# FRAMES_PER_CLIP = 64

# FPS efectivo al muestrear 16 frames en 3 segundos
# Este TARGET_FPS ahora es un valor conceptual para el muestreo manual, no un parámetro directo de decodificación en torchvision.io.read_video
# servirá para 'torchvision.io.read_video' para saber cuántos frames leer por segundo
TARGET_FPS = FRAMES_PER_CLIP / CLIP_DURATION_SEC


# Dimensiones de los tensores preprocesados (si usas DatasetTensors) o objetivo para el modelo (para DatasetVideos)
# Asegúrate que estas correspondan a lo que tu modelo espera
TARGET_HEIGHT_MODEL = 224 # Ejemplo: 128
TARGET_WIDTH_MODEL = 398 # Ejemplo: 128

# --- Dataset para cargar tensores preprocesados ---
class DatasetTensors(Dataset):
    def __init__(self, root_dir, inv_label_map, transform=None):
        self.root_dir = Path(root_dir)
        self.inv_label_map = inv_label_map
        self.transform = transform
        self.tensor_files = []
        self.labels = []

        # Recorrer los directorios de acciones y cargar los nombres de archivos y etiquetas
        for label_id, action_name in inv_label_map.items():
            action_dir = self.root_dir / action_name
            if action_dir.is_dir():
                for tensor_file in action_dir.glob("*.pt"):
                    self.tensor_files.append(tensor_file)
                    self.labels.append(label_id)
            else:
                warnings.warn(f"Directorio de acción no encontrado: {action_dir}. Saltando.")

        if not self.tensor_files:
            raise RuntimeError(f"No se encontraron archivos de tensor en '{root_dir}'. Verifica la ruta y la estructura.")

    def __len__(self):
        return len(self.tensor_files)

    def __getitem__(self, idx):
        tensor_path = self.tensor_files[idx]
        label = self.labels[idx]

        try:
            video_tensor = torch.load(tensor_path)
            # Aplicar transformaciones si están definidas
            if self.transform:
                video_tensor = self.transform(video_tensor)

            # Convertir a float32 y normalizar a [0, 1] si no lo está ya
            # Los tensores cargados ya deberían estar normalizados si los guardaste así.
            # Si no, o si quieres aplicar otra normalización, hazlo aquí:
            # video_tensor = video_tensor.to(torch.float32) / 255.0 # Si los valores son 0-255

            return video_tensor, label
        except Exception as e:
            warnings.warn(f"Error al cargar o procesar el tensor {tensor_path}: {e}. Retornando None.")
            return None, -1 # Retorna None y una etiqueta inválida para manejo posterior

# --- NUEVA CLASE: Dataset para cargar videos directamente ---
class DatasetVideos(Dataset):
    def __init__(self, root_dir, inv_label_map, transform=None):
        self.root_dir = Path(root_dir) # Asumo que este es el DS_SOCCERNET_ACTIONS de main_soccernet
        self.inv_label_map = inv_label_map
        self.transform = transform
        self.video_files = []
        self.labels = []

        # Recorrer los directorios de acciones y cargar los nombres de archivos de video y etiquetas
        for label_id, action_name in inv_label_map.items():
            action_dir = self.root_dir / action_name
            if action_dir.is_dir():
                for video_file in action_dir.glob("*.mkv"): # Asume que tus videos están en .mkv
                    self.video_files.append(video_file)
                    self.labels.append(label_id)
            else:
                warnings.warn(f"Directorio de acción no encontrado: {action_dir}. Saltando.")
        
        if not self.video_files:
            raise RuntimeError(f"No se encontraron archivos de video en '{root_dir}'. Verifica la ruta y la estructura.")

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        label = self.labels[idx]

        try:
            # Leer el video completo
            # torchvision.io.read_video devuelve (video_tensor, audio_tensor, info)
            # video_tensor shape: (T, H, W, C) - frames, height, width, channels
            # audio_tensor shape: (A, C) - samples, channels
            # info: dict con metadata
            video_tensor, audio, info = torchvision.io.read_video(
                str(video_path), 
                pts_unit='sec', # Unidades para 'start_pts' y 'end_pts'
                # Puedes especificar start_pts y end_pts si quieres leer solo una porción del video
                # El video ya está recortado a 3 segundos, así que leemos todo
                output_format="TCHW" # Convertir a (T, C, H, W) que es más común en PyTorch para video
            )
            
            # Convertir a float32 y normalizar a [0, 1]
            # torchvision.io.read_video devuelve uint8 por defecto. Normalizamos aquí.
            video_tensor = video_tensor.to(torch.float32) / 255.0

            # Aplicar transformaciones si están definidas
            if self.transform:
                video_tensor = self.transform(video_tensor)
            
            # Asegurarse de que el número de frames sea el esperado (FRAMES_PER_CLIP)
            # Esto es crucial si tu modelo espera un número fijo de frames
            current_frames = video_tensor.shape[0]
            if current_frames > FRAMES_PER_CLIP:
                # Si hay más frames de los deseados, seleccionar FRAMES_PER_CLIP uniformemente
                # Por ejemplo, si tienes 75 frames y quieres 16, seleccionas 16 frames espaciados
                indices = torch.linspace(0, current_frames - 1, FRAMES_PER_CLIP).long()
                video_tensor = video_tensor[indices]
            elif current_frames < FRAMES_PER_CLIP:
                # Si hay menos frames, rellenar (padding)
                # Puedes usar zero-padding o repetir frames. Aquí un simple zero-padding.
                padding_needed = FRAMES_PER_CLIP - current_frames
                # Ajusta padding a las dimensiones (T, C, H, W)
                padding = torch.zeros(padding_needed, video_tensor.shape[1], video_tensor.shape[2], video_tensor.shape[3], dtype=video_tensor.dtype)
                video_tensor = torch.cat((video_tensor, padding), dim=0)

            return video_tensor, label
        except Exception as e:
            warnings.warn(f"Error al cargar o procesar el video {video_path}: {e}. Retornando None.", UserWarning)
            return None, -1 # Retorna None y una etiqueta inválida para manejo posterior

# --- Transformaciones de datos ---
# Para entrenar: aumentación de datos (aleatoria) y redimensionamiento
train_transforms_videos = T_v2.Compose([
    # Convertir a torch.float32 y normalizar a [0,1] ya se hace en DatasetVideos
    T_v2.Resize(size=(TARGET_HEIGHT_MODEL, TARGET_WIDTH_MODEL), antialias=True), # Redimensionar los frames
    # T_v2.RandomCrop(size=(TARGET_HEIGHT_MODEL, TARGET_WIDTH_MODEL)), # Si quieres un recorte aleatorio
    T_v2.RandomHorizontalFlip(p=0.5), # Volteo horizontal aleatorio
    # Puedes añadir más transformaciones aquí si lo deseas (e.g., color jitter, rotación)
    # T_v2.ToDtype(torch.float32, scale=True), # Esto ya está en DatasetVideos
    # T_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Si deseas normalización ImageNet
])

# Para validación/test: solo redimensionamiento (determinista)
val_transforms_videos = T_v2.Compose([
    T_v2.Resize(size=(TARGET_HEIGHT_MODEL, TARGET_WIDTH_MODEL), antialias=True), # Redimensionar los frames
    # T_v2.ToDtype(torch.float32, scale=True), # Esto ya está en DatasetVideos
    # T_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Si deseas normalización ImageNet
])


# --- Funciones para obtener los datasets ---
# Esta es la función original para tensores preprocesados
def preprocessed_dataset(dataset_root, split, video_size):
    """
    Carga el dataset de tensores preprocesados.
    """
    if video_size == "224p":
        # Asumiendo que tus tensores 224p están en DS_SOCCERNET_TENSORS
        root_dir = Path(DS_SOCCERNET_TENSORS) / split
    elif video_size == "720p":
        # Ajusta esta ruta si tus tensores 720p están en un subdirectorio diferente
        root_dir = Path(DS_SOCCERNET_TENSORS_720P) / split 
    else:
        raise ValueError("video_size debe ser '224p' o '720p'")

    # Las transformaciones para los tensores precargados, si son necesarias.
    # Si los tensores ya están en el formato final (H, W, C), estas podrían ser mínimas.
    # Ejemplo:
    train_transforms_tensors = T_v2.Compose([
        # Si tus tensores guardados no son del tamaño final deseado, puedes añadir aquí un T_v2.Resize determinista.
        # T_v2.Resize(size=(TARGET_HEIGHT_MODEL, TARGET_WIDTH_MODEL), antialias=True),
        T_v2.RandomHorizontalFlip(p=0.5), # Aumentación de datos
    ])

    val_transforms_tensors = T_v2.Compose([
        # T_v2.Resize(size=(TARGET_HEIGHT_MODEL, TARGET_WIDTH_MODEL), antialias=True),
    ])

    if split == "train":
        return DatasetTensors(root_dir=root_dir, inv_label_map=INV_LABEL_MAP, transform=train_transforms_tensors)
    elif split == "valid":
        return DatasetTensors(root_dir=root_dir, inv_label_map=INV_LABEL_MAP, transform=val_transforms_tensors)
    elif split == "test":
        return DatasetTensors(root_dir=root_dir, inv_label_map=INV_LABEL_MAP, transform=val_transforms_tensors)
    else:
        raise ValueError(f"Split '{split}' no soportado.")


# --- NUEVA FUNCIÓN: Para cargar videos directamente ---
def direct_video_dataset(dataset_root_actions, split, video_size):
    """
    Carga el dataset de videos directamente desde los archivos .mkv.
    """
    # El root_dir aquí es donde están tus videos recortados por acción (DS_SOCCERNET_ACTIONS)
    # y los subdirectorios 'train', 'valid', 'test' si los tienes.
    # Si tus videos están directamente en DS_SOCCERNET_ACTIONS/Goal, DS_SOCCERNET_ACTIONS/Yellow card, etc.
    # entonces el 'split' deberá manejarse fuera de la función o dentro de los DatasetVideos
    # para apuntar a la subcarpeta correcta si tus videos están distribuidos por split.

    # Asumiré que tus videos recortados están estructurados así:
    # DS_SOCCERNET_ACTIONS/train/Goal/*.mkv
    # DS_SOCCERNET_ACTIONS/valid/Goal/*.mkv
    # DS_SOCCERNET_ACTIONS/test/Goal/*.mkv
    # Si no es así, ajusta la ruta 'root_dir'.

    root_dir = Path(dataset_root_actions) / split # Asume que DS_SOCCERNET_ACTIONS contiene subcarpetas de split

    # No se necesita el video_size aquí directamente para la ruta si todos los recortes
    # de 224p/720p van a la misma DS_SOCCERNET_ACTIONS.
    # Sin embargo, el TARGET_HEIGHT_MODEL y TARGET_WIDTH_MODEL se usarán en las transformaciones.
    
    if split == "train":
        return DatasetVideos(root_dir=root_dir, inv_label_map=INV_LABEL_MAP, transform=train_transforms_videos)
    elif split == "valid":
        return DatasetVideos(root_dir=root_dir, inv_label_map=INV_LABEL_MAP, transform=val_transforms_videos)
    elif split == "test":
        return DatasetVideos(root_dir=root_dir, inv_label_map=INV_LABEL_MAP, transform=val_transforms_videos)
    else:
        raise ValueError(f"Split '{split}' no soportado.")