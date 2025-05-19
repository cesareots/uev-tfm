import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.io # Para leer videos eficientemente
import torchvision.transforms as T # Para transformaciones, si las necesitas
import numpy as np
import random
from pathlib import Path # Útil para manejar rutas de archivos

# --- Configuración ---
# Define el directorio raíz donde están tus carpetas de acciones (Goal, Yellow card, Shots on target)
DATASET_ROOT = 'dataset/soccernet_final/'
NUM_CLASSES = 3
# Mapeo de nombres de acciones a índices numéricos
LABEL_MAP = {"Goal": 0, "Yellow card": 1, "Shots on target": 2}
# Inverso del mapeo para interpretar las predicciones
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# Configuración de los videos
ORIGINAL_FPS = 25
ORIGINAL_SIZE = (398, 224) # Ancho x Alto
CLIP_DURATION_SEC = 4
# Frames deseados por clip para el modelo (muestreados)
FRAMES_PER_CLIP = 16
# FPS efectivo al muestrear 16 frames en 4 segundos (16 / 4 = 4)
# torchvision.io.read_video puede decodificar directamente a este FPS
TARGET_FPS = FRAMES_PER_CLIP / CLIP_DURATION_SEC # Esto es 4.0

# Hiperparámetros de entrenamiento
BATCH_SIZE = 8 # Puedes necesitar ajustar esto basado en la memoria de tu GPU
LEARNING_RATE = 0.001
NUM_EPOCHS = 10 # Un número pequeño para un ejemplo rápido

# Configurar dispositivo (GPU si está disponible, si no CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Fijar semillas para reproducibilidad (opcional pero recomendado)
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

np.random.seed(42)
random.seed(42)

# --- Dataset Personalizado ---
class ActionVideoDataset(Dataset):
    def __init__(self, root_dir, label_map, frames_per_clip, target_fps, target_size=None):
        self.root_dir = Path(root_dir)
        self.label_map = label_map
        self.frames_per_clip = frames_per_clip
        self.target_fps = target_fps
        self.target_size = target_size # (height, width) si se desea redimensionar

        self.video_files = []
        self.labels = []

        # Construir la lista de archivos y etiquetas recorriendo directorios
        self._find_videos()

    def _find_videos(self):
        print(f"Buscando videos en: {self.root_dir}")
        # Itera sobre cada subdirectorio (que son los nombres de las acciones)
        for action_name in self.label_map.keys():
            action_dir = self.root_dir / action_name
            if not action_dir.is_dir():
                print(f"Advertencia: Directorio '{action_dir}' no encontrado. Saltando.")
                continue

            # Itera sobre los archivos .mkv dentro del directorio de acción
            for video_file in action_dir.glob('*.mkv'):
                self.video_files.append(str(video_file)) # Almacena la ruta como string
                self.labels.append(self.label_map[action_name]) # Almacena la etiqueta numérica

        print(f"Encontrados {len(self.video_files)} videos.")

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        label = self.labels[idx]

        try:
            # Lee el video usando torchvision.io
            # target_fps le indica a ffmpeg que decodifique a 4 FPS efectivo,
            # dándonos aproximadamente 16 frames en 4 segundos.
            # output_format='TCHW' -> Time, Channels, Height, Width
            video_tensor, audio, info = torchvision.io.read_video(
                video_path,
                pts_unit='sec', # Unidades para start_pts y end_pts (por defecto es 'sec')
                output_format='TCHW',
                target_fps=self.target_fps # Decodificar al FPS deseado para muestreo
            )
            # video_tensor shape: (num_read_frames, 3, height, width)

            # --- Manejar número de frames y dimensiones ---

            # Asegurarse de que tenemos exactamente FRAMES_PER_CLIP frames
            num_read_frames = video_tensor.shape[0]
            if num_read_frames < self.frames_per_clip:
                # Rellenar con ceros si el video es más corto de lo esperado (o si la decodificación dio menos frames)
                padding_needed = self.frames_per_clip - num_read_frames
                padding_shape = (padding_needed, video_tensor.shape[1], video_tensor.shape[2], video_tensor.shape[3])
                zero_padding = torch.zeros(padding_shape, dtype=video_tensor.dtype)
                video_tensor = torch.cat([video_tensor, zero_padding], dim=0)
                # print(f"Advertencia: Video corto {video_path}. Leídos {num_read_frames} frames, rellenado con ceros.")
            elif num_read_frames > self.frames_per_clip:
                 # Esto no debería pasar si target_fps está bien calculado para 4s,
                 # pero como seguridad, tomamos los primeros FRAMES_PER_CLIP
                 video_tensor = video_tensor[:self.frames_per_clip]


            # Asegurarse de que la resolución es la esperada si target_size está definido
            if self.target_size is not None and (video_tensor.shape[2], video_tensor.shape[3]) != self.target_size:
                 # Aplica redimensionamiento espacial a cada frame
                 resize_transform = T.Resize(self.target_size, antialias=True) # antialias=True es recomendado
                 # torchvision.transforms trabaja en (C, H, W) o (H, W)
                 # Permutamos temporalmente para redimensionar, luego volvemos a TCHW
                 video_tensor_permuted = video_tensor.permute(1, 0, 2, 3) # C, T, H, W
                 # Aplicar Resize a (C, T, H, W) trata T como parte de la dimensión batch,
                 # lo cual es correcto para Resize espacial frame a frame
                 video_tensor_resized = resize_transform(video_tensor_permuted)
                 video_tensor = video_tensor_resized.permute(1, 0, 2, 3) # Volver a T, C, H, W


            # Convertir a float y normalizar [0, 1]
            video_tensor = video_tensor.float() / 255.0

            # Cambiar el orden de las dimensiones a (C, T, H, W) que PyTorch espera para Conv3d
            video_tensor = video_tensor.permute(1, 0, 2, 3) # Channels, Time, Height, Width

            # Convertir la etiqueta numérica a tensor
            label_tensor = torch.tensor(label, dtype=torch.long)

            return video_tensor, label_tensor

        except Exception as e:
            print(f"Error al cargar o procesar el video {video_path}: {e}")
            # En caso de error, puedes devolver un tensor dummy para no romper el batch
            # O simplemente omitir este índice (requiere un DataLoader con error_handling=True o manejarlo manualmente)
            # Para simplicidad en el ejemplo, devolvemos un tensor de ceros y una etiqueta dummy (-1)
            # En un caso real, podrías querer registrar el error y omitir.
            dummy_video = torch.zeros((3, self.frames_per_clip, ORIGINAL_SIZE[1], ORIGINAL_SIZE[0]), dtype=torch.float)
            if self.target_size is not None:
                 dummy_video = torch.zeros((3, self.frames_per_clip, self.target_size[0], self.target_size[1]), dtype=torch.float)
            dummy_label = torch.tensor(-1, dtype=torch.long) # Etiqueta inválida
            return dummy_video, dummy_label


# --- Modelo CNN 3D Simple ---

class SimpleCNN3D(nn.Module):
    def __init__(self, num_classes, input_channels=3, input_frames=FRAMES_PER_CLIP, input_size=(ORIGINAL_SIZE[1], ORIGINAL_SIZE[0])):
        super(SimpleCNN3D, self).__init__()

        # Tamaño de entrada: (Batch, input_channels, input_frames, input_size[0], input_size[1])
        # Ejemplo con tus datos: (Batch, 3, 16, 224, 398)

        # Capa 1: Conv3d -> ReLU -> MaxPool3d
        # Kernel Conv: (tiempo, alto, ancho) -> (3, 3, 3)
        # Padding Conv: mantiene dimensiones espaciales y temporales si es 1
        # Kernel Pool: (tiempo, alto, ancho) -> (2, 2, 2) con stride 2 reduce a la mitad en las 3 dims
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # Después de pool1: (Batch, 32, input_frames/2, input_size[0]/2, input_size[1]/2)
        # Ejemplo: (Batch, 32, 8, 112, 199)

        # Capa 2: Conv3d -> ReLU -> MaxPool3d
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # Después de pool2: (Batch, 64, input_frames/4, input_size[0]/4, input_size[1]/4)
        # Ejemplo: (Batch, 64, 4, 56, 99) (nota: 199/2/2 = 49.75, MaxPool redondea, 99 está mal calculado aquí, sería floor((199-2)/2)+1 = 99.5 -> 99 con padding 1. floor((99-2)/2)+1 = 49.5 -> 49. Pero con padding y kernel 2, stride 2, es división exacta. (199+2*1-3)/2 + 1 = 99.5. Ah, MaxPool3d default padding is 0. Let's assume stride=2, kernel=2, padding=0 for simplicity in dimension calculation for the example). Let's recalculate assuming kernel=2, stride=2, padding=0 for pooling.
        # Correct MaxPool3d calculation with kernel=(2,2,2), stride=(2,2,2), padding=(0,0,0): Output size = floor((Input_dim - Kernel_dim)/Stride_dim) + 1
        # After Conv1 (padding 1, k=3): Temporal = (16+2*1-3)/1+1=16. Spatial = (224+2*1-3)/1+1=224, (398+2*1-3)/1+1=398 -> (B, 32, 16, 224, 398)
        # After Pool1 (k=2, s=2, p=0): Temporal = floor((16-2)/2)+1=8. Spatial = floor((224-2)/2)+1=112, floor((398-2)/2)+1=199 -> (B, 32, 8, 112, 199)
        # After Conv2 (padding 1, k=3): Temporal = (8+2*1-3)/1+1=8. Spatial = (112+2*1-3)/1+1=112, (199+2*1-3)/1+1=199 -> (B, 64, 8, 112, 199)
        # After Pool2 (k=2, s=2, p=0): Temporal = floor((8-2)/2)+1=4. Spatial = floor((112-2)/2)+1=56, floor((199-2)/2)+1=99 -> (B, 64, 4, 56, 99)

        # Capa 3: Conv3d -> ReLU -> MaxPool3d
        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # Después de pool3: (Batch, 128, input_frames/8, input_size[0]/8, input_size[1]/8)
        # Temporal = floor((4-2)/2)+1=2. Spatial = floor((56-2)/2)+1=27. floor((99-2)/2)+1=49 -> (B, 128, 2, 27, 49) # Hmm, calculations are getting tricky. Let's simplify spatial padding in convs to 0, and kernels/strides slightly. Or trust PyTorch to calculate it and use a dummy tensor.
        # Let's redefine slightly simpler layers for calculation certainty or use a dummy tensor to get the shape. Using a dummy tensor is safer.

        # --- Revised Model Architecture with Dummy Tensor for Flatten Size ---
        # Use padding=1 for conv, kernel=2, stride=2 for pool
        self.features = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2), # (B, 32, T/2, H/2, W/2)

            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2), # (B, 64, T/4, H/4, W/4)

            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)  # (B, 128, T/8, H/8, W/8)
        )

        # Calcular el tamaño de la entrada a la capa lineal después de aplanar
        # Podemos usar un tensor dummy para que PyTorch nos diga el tamaño
        dummy_input_size = (1, input_channels, input_frames, input_size[0], input_size[1]) # (Batch=1, C, T, H, W)
        dummy_tensor = torch.autograd.Variable(torch.zeros(dummy_input_size))
        size_after_features = self.features(dummy_tensor).shape

        # Tamaño a aplanar: canales * dim_temporal * dim_alto * dim_ancho
        flattened_size = size_after_features[1] * size_after_features[2] * size_after_features[3] * size_after_features[4]
        print(f"Tamaño a aplanar calculado: {size_after_features} -> {flattened_size}")


        self.fc = nn.Linear(flattened_size, num_classes)


    def forward(self, x):
        # Input x shape: (Batch, C, T, H, W)
        x = self.features(x)
        # Flatten
        x = x.view(x.size(0), -1) # x.size(0) es el tamaño del batch
        # Fully connected layer
        x = self.fc(x)
        return x

# --- Función de Entrenamiento ---

def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
    model.train() # Poner el modelo en modo entrenamiento
    print("\n--- Empezando entrenamiento ---")

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for i, (inputs, labels) in enumerate(dataloader):
            # Omitir entradas con etiquetas inválidas (-1) si ocurrieron errores en el Dataset
            valid_indices = labels != -1
            inputs = inputs[valid_indices].to(device)
            labels = labels[valid_indices].to(device)

            if inputs.size(0) == 0: # Saltar si no hay entradas válidas en el batch
                 continue

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass y optimización
            loss.backward()
            optimizer.step()

            # Estadísticas
            running_loss += loss.item() * inputs.size(0) # Multiplicar por tamaño del batch real
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            # Imprimir estadísticas cada cierto número de batches (opcional)
            if (i + 1) % 10 == 0: # Imprime cada 10 batches
                print(f"  Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        epoch_loss = running_loss / total_loader_size(dataloader, -1) # Usar la suma total de elementos válidos
        epoch_acc = correct_predictions / total_predictions if total_predictions > 0 else 0

        print(f"Epoch [{epoch+1}/{num_epochs}] finalizada. Pérdida: {epoch_loss:.4f}, Precisión: {epoch_acc:.4f}")

    print("--- Entrenamiento finalizado ---")

# Helper para obtener el tamaño total de elementos válidos en el DataLoader
def total_loader_size(dataloader, invalid_label):
    total_valid = 0
    for _, labels in dataloader:
        total_valid += (labels != invalid_label).sum().item()
    return total_valid


# --- Función de Evaluación (Opcional, pero recomendado) ---

def evaluate_model(model, dataloader, criterion, device):
    model.eval() # Poner el modelo en modo evaluación
    print("\n--- Empezando evaluación ---")

    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    # Desactivar cálculo de gradientes para evaluación
    with torch.no_grad():
        for inputs, labels in dataloader:
            valid_indices = labels != -1
            inputs = inputs[valid_indices].to(device)
            labels = labels[valid_indices].to(device)

            if inputs.size(0) == 0:
                 continue

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Estadísticas
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()


    avg_loss = running_loss / total_loader_size(dataloader, -1) if total_loader_size(dataloader, -1) > 0 else 0
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    print(f"--- Evaluación finalizada ---")
    print(f"Pérdida promedio: {avg_loss:.4f}, Precisión: {accuracy:.4f}")
    return avg_loss, accuracy


# --- Ejecución Principal ---

if __name__ == "__main__":
    # Asegúrate de que DATASET_ROOT existe y contiene las subcarpetas goal, Yellow card, red card
    # con archivos .mkv dentro.
    # Para un test rápido, puedes crear directorios vacíos:
    # os.makedirs(os.path.join(DATASET_ROOT, 'goal'), exist_ok=True)
    # os.makedirs(os.path.join(DATASET_ROOT, 'Yellow card'), exist_ok=True)
    # os.makedirs(os.path.join(DATASET_ROOT, 'red card'), exist_ok=True)
    # Y colocar algunos archivos .mkv (aunque sean dummies o clips cortos) para que el Dataset los encuentre.
    # moviepy o ffmpeg necesitan poder leer estos archivos.

    # Crear el dataset
    # Puedes pasar target_size=(128, 128) si quieres redimensionar los frames
    dataset = ActionVideoDataset(
        root_dir=DATASET_ROOT,
        label_map=LABEL_MAP,
        frames_per_clip=FRAMES_PER_CLIP,
        target_fps=TARGET_FPS,
        target_size=None # Deja None para usar la resolución original (224, 398)
                         # o cambia a (alto, ancho) ej: (128, 128) para redimensionar
    )

    # Verificar si se encontraron videos
    if len(dataset) == 0:
        print(f"Error: No se encontraron videos en '{DATASET_ROOT}' con las subcarpetas esperadas.")
        print(f"Asegúrate de tener la estructura '{DATASET_ROOT}/[accion]/[video].mkv'")
        print(f"Y que las acciones en las carpetas coincidan con {list(LABEL_MAP.keys())}")
        exit()

    # Dividir el dataset en conjuntos de entrenamiento y validación (ej. 80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Crear DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2) # num_workers > 0 acelera la carga de datos
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Determinar las dimensiones de entrada finales al modelo (después de posible redimensionamiento)
    # Tomamos un item dummy del dataset para saber el shape exacto
    try:
        dummy_video_tensor, dummy_label = dataset[0]
        # Si el primer item da error, intenta el siguiente válido
        i = 0
        while dummy_label == -1 and i < len(dataset):
             i += 1
             dummy_video_tensor, dummy_label = dataset[i]

        if dummy_label == -1:
             print("Error: No se pudo cargar ninguna muestra válida del dataset para determinar el tamaño de entrada del modelo.")
             exit()

        _, C_in, T_in, H_in, W_in = dummy_video_tensor.shape # Shape es (C, T, H, W)

        print(f"\nDimensiones de entrada al modelo: (Batch, {C_in}, {T_in}, {H_in}, {W_in})")

        # Crear el modelo
        model = SimpleCNN3D(
            num_classes=NUM_CLASSES,
            input_channels=C_in,
            input_frames=T_in,
            input_size=(H_in, W_in)
        ).to(device)

        # Definir la función de pérdida y el optimizador
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Entrenar el modelo
        train_model(model, train_dataloader, criterion, optimizer, NUM_EPOCHS, device)

        # Evaluar el modelo (opcional después de entrenar)
        evaluate_model(model, val_dataloader, criterion, device)

        # Opcional: Guardar el modelo entrenado
        # torch.save(model.state_dict(), 'action_classification_model.pth')
        # print("\nModelo guardado en 'action_classification_model.pth'")

    except Exception as e:
        print(f"\nOcurrió un error durante la inicialización o entrenamiento: {e}")
