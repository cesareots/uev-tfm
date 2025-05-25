from pathlib import Path
import warnings
import numpy as np
import torch
import torchvision.io
import torchvision.transforms.v2 as T_v2
from torch.utils.data import Dataset, Subset # Importamos Subset para el split
from src.utils.constants import * # Asegúrate de que SOCCERNET_LABELS y otras constantes estén definidas aquí

# ignorar el warning de la deprecación de video de torchvision.io
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torchvision.io._video_deprecation_warning"
)

# TODO: Asegúrate de que SOCCERNET_LABELS esté definido en src/utils/constants.py
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
                warnings.warn(f"Directorio de acción no encontrado: {action_dir}. Saltando.", UserWarning)

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
            # Si los tensores guardados ya están en float32 y normalizados [0,1], esta línea puede ser redundante.
            # Asegúrate de la escala de tus tensores guardados. Si son 0-255, normaliza:
            # video_tensor = video_tensor.to(torch.float32) / 255.0 

            return video_tensor, label
        except Exception as e:
            warnings.warn(f"Error al cargar o procesar el tensor {tensor_path}: {e}. Retornando None.", UserWarning)
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
                warnings.warn(f"Directorio de acción no encontrado: {action_dir}. Saltando.", UserWarning)
        
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
            # output_format="TCHW" lo convierte a (T, C, H, W)
            video_tensor, audio, info = torchvision.io.read_video(
                str(video_path), 
                pts_unit='sec', # Unidades para 'start_pts' y 'end_pts'
                output_format="TCHW" 
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
                indices = torch.linspace(0, current_frames - 1, FRAMES_PER_CLIP).long()
                video_tensor = video_tensor[indices]
            elif current_frames < FRAMES_PER_CLIP:
                # Si hay menos frames, rellenar (padding)
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
    T_v2.Resize(size=(TARGET_HEIGHT_MODEL, TARGET_WIDTH_MODEL), antialias=True), # Redimensionar los frames
    T_v2.RandomHorizontalFlip(p=0.5), # Volteo horizontal aleatorio
    # T_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Si deseas normalización ImageNet
])

# Para validación/test: solo redimensionamiento (determinista)
val_transforms_videos = T_v2.Compose([
    T_v2.Resize(size=(TARGET_HEIGHT_MODEL, TARGET_WIDTH_MODEL), antialias=True), # Redimensionar los frames
    # T_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Si deseas normalización ImageNet
])


# --- Funciones para obtener los datasets (sin split interno) ---
def get_full_preprocessed_dataset(dataset_root, video_size):
    """
    Carga el dataset completo de tensores preprocesados sin split.
    """
    if video_size == "224p":
        root_dir = Path(dataset_root) # DS_SOCCERNET_TENSORS
    elif video_size == "720p":
        root_dir = Path(DS_SOCCERNET_TENSORS_720P) 
    else:
        raise ValueError("video_size debe ser '224p' o '720p'")

    # Las transformaciones para los tensores precargados, si son necesarias.
    # Aquí aplicamos las transformaciones de entrenamiento para el dataset completo
    # si se va a dividir después. Si no hay aumentación en los tensores, el transform puede ser None
    # y la aumentación se hace solo en el entrenamiento.
    # Pero si la aumentación (como el flip) se aplica a todo el dataset y luego se divide,
    # el 'val_dataset' también la tendría. Es mejor aplicar solo las deterministas aquí.
    # La aumentación real se hace solo en el DataLoader de entrenamiento o en una clase de DatasetTrain específica.
    
    # Para la carga inicial, podemos usar una transformación que solo garantice el tamaño correcto
    # si los tensores no tienen el tamaño final.
    # Si los tensores ya tienen TARGET_HEIGHT_MODEL y TARGET_WIDTH_MODEL, set transform=None.
    # Ejemplo: Si tus tensores están ya en el tamaño final, estas transformaciones pueden ser None.
    tensor_transforms = T_v2.Compose([
        T_v2.Resize(size=(TARGET_HEIGHT_MODEL, TARGET_WIDTH_MODEL), antialias=True) 
        # Asegúrate de que tus tensores preprocesados tengan las dimensiones correctas.
        # Si ya las tienen, esta línea es opcional o puedes usar transform=None.
    ])

    return DatasetTensors(root_dir=root_dir, inv_label_map=INV_LABEL_MAP, transform=tensor_transforms)


def get_full_direct_video_dataset(dataset_root_actions, video_size):
    """
    Carga el dataset completo de videos directamente desde los archivos .mkv sin split.
    """
    # El root_dir aquí es donde están tus videos recortados por acción (DS_SOCCERNET_ACTIONS)
    # y los subdirectorios como 'Goal', 'Yellow card', etc.
    root_dir = Path(dataset_root_actions)

    # Las transformaciones de video, incluyendo redimensionamiento
    # Aquí usamos las transformaciones de validación (deterministas)
    # ya que la aumentación se aplica solo a los datos de entrenamiento después del split.
    video_transforms = T_v2.Compose([
        T_v2.Resize(size=(TARGET_HEIGHT_MODEL, TARGET_WIDTH_MODEL), antialias=True),
        # Si deseas normalización ImageNet, añádela aquí.
        # T_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return DatasetVideos(root_dir=root_dir, inv_label_map=INV_LABEL_MAP, transform=video_transforms)