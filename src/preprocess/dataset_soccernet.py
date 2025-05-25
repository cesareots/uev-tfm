from pathlib import Path
import warnings
import numpy as np
import torch
import torchvision.io
import torchvision.transforms.v2 as T_v2
from torch.utils.data import Dataset
from src.utils.constants import *

# ignorar el warning de la deprecación de video de torchvision.io
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torchvision.io._video_deprecation_warning"
)

NUM_CLASSES = 3  # TODO extraer automaticamente de 'SOCCERNET_LABELS'
# Inverso del mapeo para interpretar las predicciones
INV_LABEL_MAP = {v: k for k, v in SOCCERNET_LABELS.items()}

# Ancho x Alto original de los frames (W, H), soccernet establece estos tamaños para sus videos
ORIGINAL_SIZE = (398, 224)
#ORIGINAL_SIZE = (1280, 720)

# Duración de los clips
CLIP_DURATION_SEC = 3

# frames deseados por clip para el modelo (muestreados)
FRAMES_PER_CLIP = 16
# FRAMES_PER_CLIP = 32
# FRAMES_PER_CLIP = 64

# FPS efectivo al muestrear 16 frames en 3 segundos
# Este TARGET_FPS ahora es un valor conceptual para el muestreo manual, no un parámetro directo de decodificación en torchvision.io.read_video
# servirá para 'torchcodec'
TARGET_FPS = float(FRAMES_PER_CLIP / CLIP_DURATION_SEC)

class DatasetSoccernet(Dataset):
    def __init__(self, root_dir, label_map, frames_per_clip, target_fps, target_size=None, transform=None):
        self.root_dir = Path(root_dir)
        self.label_map = label_map
        self.frames_per_clip = frames_per_clip
        self.target_fps = target_fps
        self.target_size = target_size
        self.transform = transform
        self.video_items = []
        self._find_videos()

    def _find_videos(self):
        #print(f"Buscando videos en: {self.root_dir}")
        for action_name_str in self.label_map.keys():
            action_dir = self.root_dir / action_name_str
            if not action_dir.is_dir():
                print(f"Advertencia: Directorio '{action_dir}' no encontrado. Saltando.")
                continue

            label_id = self.label_map[action_name_str]
            for video_file in action_dir.glob('*.mkv'):
                self.video_items.append((str(video_file), label_id, action_name_str))

        print(f"Encontrados {len(self.video_items)} videos.")

    def __len__(self):
        return len(self.video_items)

    def __getitem__(self, idx):
        video_path_str, label, action_name_str = self.video_items[idx]
        video_path = Path(video_path_str)

        try:
            # Usando torchvision.io.read_video
            # Decodifica el video. En tu versión, no podemos pasar target_fps aquí.
            # Por defecto, devuelve un tensor de forma (T, C, H, W) para 'output_format=TCHW'.
            video_tensor_raw, audio, info = torchvision.io.read_video(
                str(video_path),
                pts_unit='sec',
                #output_format="THWC",  # por defecto
                output_format="TCHW",  # Tiempo, Canales, Alto, Ancho
            )
            # video_tensor_raw shape: (num_total_frames_original, C, H, W)
            
            # TODO 'torchvision.io.read_video' pronto estara DEPRECATED
            # a la fecha 2025-05-22 pip no instala correctamente torchcodec==0.4.0
            # instala esta version: 0.0.0.dev0
            #video_tensor_raw, audio, info = torchcodec.read_video(
            #    str(video_path),
            #    fps=self.target_fps,  # Ahora se puede especificar FPS directamente
            #    output_format="CTHW"  # Channels, Time, Height, Width
            #)
            #video_tensor_raw = video_tensor_raw.permute(1, 0, 2, 3)

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
            # Ahora video_tensor_processed tiene la forma (FRAMES_PER_CLIP, C, H, W)

            # APLICAR TRANSFORMACIONES, deben aplicarse antes de la normalización y la permutación final.
            # T_v2.Compose puede aceptar transformaciones que operan en tensores (T, C, H, W)
            if self.transform:
                video_tensor_processed = self.transform(video_tensor_processed)

            # --- Redimensionamiento FINAL si se especifica y no lo hizo ya una transformación ---
            # Si target_size es None, usamos las dimensiones originales de los frames muestreados.
            # Si target_size está definido, nos aseguramos de que el tensor tenga esas dimensiones.
            # Esto se hace aquí para asegurar que el tensor tenga el tamaño correcto ANTES de la normalización
            # y la permutación final (C, T, H, W) si las transformaciones no lo garantizaron.
            final_h = self.target_size[0] if self.target_size is not None else video_tensor_processed.shape[2]
            final_w = self.target_size[1] if self.target_size is not None else video_tensor_processed.shape[3]

            if (video_tensor_processed.shape[2], video_tensor_processed.shape[3]) != (final_h, final_w):
                resize_transform = T_v2.Resize((final_h, final_w), antialias=True)
                # T_v2.Resize opera por defecto en (C, H, W) o (H, W).
                # Para (T, C, H, W), necesitamos que se aplique a cada frame.
                # Afortunadamente, T_v2.Compose puede manejar esto si las transformaciones
                # están diseñadas para secuencias. Si no, habría que iterar.
                # Aquí, asumo que las transformaciones de T_v2 como Resize están bien con (T,C,H,W)
                # (aunque internamente pueden aplicar a cada frame o usar modos que lo soporten).
                # La forma más segura para T_v2.Resize con TCHW es aplicarla como parte de una composición.
                # Alternativa robusta si T_v2.Resize no opera bien en TCHW directamente:
                # resized_frames = []
                # for frame in video_tensor_processed: # frame shape: (C, H, W)
                #     resized_frames.append(resize_transform(frame))
                # video_tensor_processed = torch.stack(resized_frames, dim=0)

                # Sin embargo, para mantenerlo conciso y aprovechando T_v2:
                # T_v2.Resize y otros operan en la última dimensión. Si input es TCHW, Resize espera CHW,
                # pero T_v2.Compose puede manejar el envío a los frames individuales si la transformación es 'per-frame'.
                # A menudo, las transformaciones de imagen estándar (como Resize, RandomHorizontalFlip, ColorJitter)
                # aplicadas a un tensor TCHW, si no son conscientes del tiempo, se aplican a cada frame.
                # La forma segura si la transformación no es "video-aware" es hacer T.Compose([T.ToImage(), T.Resize(...), T.ToDtype(...)])
                # o iterar sobre la dimensión T. Pero T_v2 está diseñado para esto.
                video_tensor_processed = resize_transform(video_tensor_processed)

            # Normalizar a [0, 1] y convertir a float
            video_tensor_final = video_tensor_processed.float() / 255.0

            # Permutar a (C, T, H, W), PyTorch espera este orden para Conv3d
            # Las transformaciones de T_v2 mantienen el orden de las dimensiones, por lo que esto sigue siendo válido.
            video_tensor_final = video_tensor_final.permute(1, 0, 2, 3)  # Channels, Time, Height, Width

            label_tensor = torch.tensor(label, dtype=torch.long)

            return video_tensor_final, label_tensor, video_path_str
        except Exception as e:
            print(f"Error al cargar o procesar el video {video_path_str}: {e}")
            # Usar las dimensiones originales si no se especificó target_size
            dummy_height = self.target_size[0] if self.target_size is not None else ORIGINAL_SIZE[1]
            dummy_width = self.target_size[1] if self.target_size is not None else ORIGINAL_SIZE[0]
            dummy_video = torch.zeros((3, self.frames_per_clip, dummy_height, dummy_width), dtype=torch.float)
            dummy_label = torch.tensor(-1, dtype=torch.long)

            return dummy_video, dummy_label, video_path_str

class DatasetTensors(Dataset):
    def __init__(self, root_dir, inv_label_map, transform=None):
        self.root_dir = Path(root_dir)
        self.inv_label_map = inv_label_map
        self.tensor_files = []
        self.labels = []
        self.transform = transform
        self._find_tensors()

    def _find_tensors(self):
        print(f"Buscando tensores .pt en: {self.root_dir}")

        for label_id, action_name in self.inv_label_map.items():
            action_dir = self.root_dir / action_name

            if not action_dir.is_dir():
                print(f"Advertencia: Directorio de tensor '{action_dir}' no encontrado. Saltando.")
                continue

            for tensor_file in action_dir.glob('*.pt'):
                self.tensor_files.append(str(tensor_file))
                self.labels.append(label_id)

        print(f"{len(self.tensor_files)} tensores encontrados.")

    def __len__(self):
        return len(self.tensor_files)

    def __getitem__(self, idx):
        tensor_path = self.tensor_files[idx]
        label = self.labels[idx]

        try:
            video_tensor = torch.load(tensor_path)  # shape: (C, T, H, W)

            # Las transformaciones 2D (RandomResizedCrop, Flip, ColorJitter) esperan entrada (C, H, W).
            # Para aplicar transformaciones 2D a cada frame, necesitamos:
            # 1. Permutar a (T, C, H, W) para iterar sobre frames.
            # 2. Aplicar la transformación a cada frame (que será (C, H, W)).
            # 3. Volver a apilar los frames.
            # 4. Permutar de nuevo a (C, T, H, W) para el modelo.

            # Aplicar data augmentation
            if self.transform:
                # Permutar para que el tiempo sea la primera dimensión para la iteración
                video_tensor = video_tensor.permute(1, 0, 2, 3) # (T, C, H, W)
                
                transformed_frames = []
                for frame in video_tensor: # Cada 'frame' es un tensor (C, H, W)
                    transformed_frames.append(self.transform(frame))
                
                # Apilar los frames de nuevo en un tensor (T, C, H, W)
                video_tensor = torch.stack(transformed_frames, dim=0)
                
                # Permutar de vuelta a (C, T, H, W) para el modelo
                video_tensor = video_tensor.permute(1, 0, 2, 3)

            label_tensor = torch.tensor(label, dtype=torch.long)

            return video_tensor, label_tensor
        except Exception as e:
            print(f"Error al cargar tensor {tensor_path}: {e}")
            # Usar las dimensiones originales si no se especificó target_size
            dummy_height = ORIGINAL_SIZE[1]
            dummy_width = ORIGINAL_SIZE[0]
            # Si se redimensionó el dataset al guardarlo, necesitaríamos saber ese target_size aquí.
            # Para mayor robustez, podrías guardar las dimensiones junto con el tensor, o
            # simplemente asumir el tamaño estándar que se usó al preprocesar.
            dummy_video = torch.zeros((3, FRAMES_PER_CLIP, dummy_height, dummy_width), dtype=torch.float)
            dummy_label = torch.tensor(-1, dtype=torch.long)

            return dummy_video, dummy_label


def preprocessed_dataset():
    TARGET_HEIGHT_MODEL = 224
    TARGET_WIDTH_MODEL = 398
    # Define las transformaciones para la AUMENTACIÓN DE DATOS durante el entrenamiento
    # Se aplican al tensor (C, T, H, W)
    train_transforms = T_v2.Compose([
        T_v2.RandomHorizontalFlip(p=0.5),  # Volteo horizontal aleatorio
        T_v2.ColorJitter(  # Ajustes de color
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.05,
        ),
        T_v2.RandomRotation(
            degrees=5,  # grados
            expand=False,  # evita cambios en el tamaño del frame.
        ),
        # Redimensionamiento aleatorio y recorte. Importante: debe estar después de flip/color jitter.
        # Aquí asumimos una salida cuadrada de (224, 224) después de las transformaciones.
        # Si tus videos son rectangulares y quieres mantener esa proporción, ajusta esto.
        T_v2.RandomResizedCrop(
            # TODO deberia recibir dinamicamente? ya que habrian 2 tamaños de tensores (224p y 720p); o es que aca se redimensiona al 'size' indicado (sin importar tamaño del tensor)?
            #size=(224, 398),
            size=(TARGET_HEIGHT_MODEL, TARGET_WIDTH_MODEL), # Tamaño de salida deseado
            scale=(0.8, 1.0),
            # ratio=(0.9, 1.1),
            #ratio=(16 / 9, 16 / 9),  # sin cambiar la relación de aspecto
            ratio=(float(TARGET_WIDTH_MODEL/TARGET_HEIGHT_MODEL), float(TARGET_WIDTH_MODEL/TARGET_HEIGHT_MODEL)),
            #antialias=True,
        ),  # Recorte y redimensionamiento aleatorio
        # Nota: La normalización a [0,1] ya está hecha al guardar los tensores.
        # Si quieres normalización por media/desviación estándar, añádela aquí DESPUÉS de T_v2.ToDtype(torch.float32, scale=True).
        # T_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Ejemplo para ImageNet
    ])

    # Para la validación, solo transformaciones deterministas (redimensionamiento si es necesario, sin aumentación)
    #val_transforms = T_v2.Compose([
        # Si tus tensores guardados no son del tamaño final deseado, puedes añadir aquí un T_v2.Resize determinista.
        # Por ejemplo, si preprocesaste a (224,398) y quieres entrenar con (128,128):
        # T_v2.Resize(size=(128, 128), antialias=True)
    #])
    
    preprocessed_train_dataset = DatasetTensors(
        root_dir=DS_SOCCERNET_TENSORS,
        inv_label_map=INV_LABEL_MAP,
        transform=train_transforms  # Aplicar aumentación solo en el dataset de entrenamiento
    )
    
    # preprocessed_val_dataset = DatasetTensors(
    #    root_dir=DS_SOCCERNET_TENSORS,  # Asumiendo que todo el dataset preprocesado se divide
    #    inv_label_map=INV_LABEL_MAP,
    #    transform=val_transforms  # Sin aumentación en el dataset de validación
    # )
    
    return preprocessed_train_dataset
