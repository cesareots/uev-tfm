import argparse
import logging
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Importa ambas funciones para cargar datasets
from src.preprocess.dataset_soccernet import ORIGINAL_SIZE, FRAMES_PER_CLIP, NUM_CLASSES, \
    preprocessed_dataset, direct_video_dataset # Importa la nueva función
from src.utils import utils as ut
from src.utils.constants import * # Asegúrate de que DS_SOCCERNET_ACTIONS esté definido aquí

logger = logging.getLogger(__name__)

# Hiperparámetros de entrenamiento, ajustar basado en la memoria de la GPU
BATCH_SIZE = 8
LEARNING_RATE = 0.001

# si hay disponibilidad de GPU, si no CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Usando dispositivo: {device}")

torch.manual_seed(SEMILLA)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEMILLA)

np.random.seed(SEMILLA)
random.seed(SEMILLA)


def config_log() -> None:
    ut.verify_directory(LOG_DIR)

    logging.basicConfig(
        filename=LOG_MODEL_CNN3D,
        filemode="a",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO, # O logging.DEBUG para más verbosidad
        encoding="utf-8",
    )


# --- Modelo Simple CNN 3D ---
# Asumo que esta clase ya la tienes definida y es correcta
class SimpleCNN3D(nn.Module):
    def __init__(self, num_classes, input_channels, input_frames, input_size):
        super(SimpleCNN3D, self).__init__()
        # input_size es (H, W)
        H_in, W_in = input_size 
        
        # Capa convolucional 1 (3D)
        # In: (Batch, C_in, T_in, H_in, W_in)
        # Out: (Batch, 32, T_out1, H_out1, W_out1)
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2)) # No reduce la dimensión temporal (T), solo H y W

        # Capa convolucional 2
        # In: (Batch, 32, T_out1, H_out1, W_out1)
        # Out: (Batch, 64, T_out2, H_out2, W_out2)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2)) # No reduce la dimensión temporal (T), solo H y W

        # Calculamos las dimensiones de salida después de las capas convolucionales y de pooling
        # Esto es un cálculo estimado, asumiendo stride=1 y padding=1 para conv, y kernel_size para pool
        # T_out = T_in (porque kernel_size[0] y stride[0] en pool son 1)
        # H_out = H_in // (2*2) = H_in // 4
        # W_out = W_in // (2*2) = W_in // 4
        
        # Puedes pasar un dummy tensor para calcular el tamaño exacto
        # dummy_tensor = torch.zeros(1, input_channels, input_frames, H_in, W_in)
        # with torch.no_grad():
        #     x = self.pool1(self.relu1(self.conv1(dummy_tensor)))
        #     x = self.pool2(self.relu2(self.conv2(x)))
        # final_flat_features = x.numel() # Esto daría el tamaño total

        # Para un cálculo manual simplificado:
        # H_final = H_in // 4
        # W_final = W_in // 4
        # self.fc_input_features = 64 * input_frames * H_final * W_final # Ajustar si tu pool reduce T
        # Es más seguro calcularlo dinámicamente si no estás 100% seguro de las dimensiones

        # --- Calcular el tamaño de la capa lineal dinámicamente ---
        # Esto es más robusto a cambios en el modelo o tamaños de entrada
        self._set_fc_input_features(input_channels, input_frames, H_in, W_in)


        # Capa lineal (clasificador)
        self.fc = nn.Linear(self.fc_input_features, num_classes)

    def _set_fc_input_features(self, input_channels, input_frames, H_in, W_in):
        # Crear un tensor dummy para calcular el tamaño de la capa lineal
        dummy_tensor = torch.zeros(1, input_channels, input_frames, H_in, W_in)
        with torch.no_grad():
            x = self.pool1(self.relu1(self.conv1(dummy_tensor)))
            x = self.pool2(self.relu2(self.conv2(x)))
        self.fc_input_features = x.numel()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        
        # Aplanar el tensor para la capa lineal
        x = x.view(-1, self.fc_input_features) # -1 para el tamaño del batch
        x = self.fc(x)
        return x


# --- Función collate_fn para manejar elementos None en el DataLoader ---
def collate_fn(batch):
    # Filtra las muestras inválidas (donde video_tensor es None)
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return torch.tensor([]), torch.tensor([]) # Retorna tensores vacíos si el batch está vacío
    
    # 'default_collate' es lo que DataLoader usaría normalmente.
    # Tenemos que importarlo explícitamente.
    from torch.utils.data._utils.collate import default_collate
    return default_collate(batch)


# --- Funciones de entrenamiento y evaluación (sin cambios) ---
def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        logger.info(f"Época {epoch + 1}/{num_epochs}")
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        for i, (videos, labels) in enumerate(dataloader):
            # Asegúrate de que videos y labels no estén vacíos por el collate_fn
            if videos.numel() == 0: # Si no hay elementos válidos en el batch
                logger.warning(f"Batch vacío en la época {epoch+1}, iteración {i}. Saltando.")
                continue

            videos = videos.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            if (i + 1) % 10 == 0:
                logger.info(f"  Batch {i + 1}, Pérdida: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct_predictions / total_samples
        logger.info(f"  Pérdida media de la época: {avg_loss:.4f}, Precisión: {accuracy:.2f}%")


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for i, (videos, labels) in enumerate(dataloader):
            if videos.numel() == 0:
                logger.warning(f"Batch vacío en evaluación, iteración {i}. Saltando.")
                continue
            
            videos = videos.to(device)
            labels = labels.to(device)

            outputs = model(videos)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct_predictions / total_samples
    logger.info(f"\nResultados de evaluación:")
    logger.info(f"  Pérdida media: {avg_loss:.4f}, Precisión: {accuracy:.2f}%")


def main(args) -> None:
    try:
        # Configurar el logger
        config_log()

        # --- Bandera para elegir el tipo de dataset ---
        LOAD_FROM_TENSORS = args.load_from_tensors # Se manejará con un argumento argparse

        if LOAD_FROM_TENSORS:
            logger.info("Cargando dataset de tensores preprocesados...")
            # Cargar los datasets
            train_dataset = preprocessed_dataset(DS_SOCCERNET_TENSORS, "train", args.video_size)
            val_dataset = preprocessed_dataset(DS_SOCCERNET_TENSORS, "valid", args.video_size)
        else:
            logger.info("Cargando dataset de videos directamente desde archivos...")
            # Asegúrate que DS_SOCCERNET_ACTIONS apunta al directorio raíz de tus videos recortados
            train_dataset = direct_video_dataset(DS_SOCCERNET_ACTIONS, "train", args.video_size)
            val_dataset = direct_video_dataset(DS_SOCCERNET_ACTIONS, "valid", args.video_size)

        # Usar la función collate_fn personalizada en DataLoader
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count() // 2, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count() // 2, collate_fn=collate_fn)

        # Obtener las dimensiones del tensor de entrada (para el modelo)
        # Iterar para encontrar la primera muestra válida
        dummy_video_tensor = None
        dummy_label = -1
        i = 0
        while dummy_video_tensor is None and i < len(train_dataset):
            dummy_video_tensor, dummy_label = train_dataset[i]
            i += 1

        if dummy_label == -1:
            logger.error("Error: No se pudo cargar ninguna muestra válida del dataset para determinar el tamaño de entrada del modelo.")
            exit()

        # C_in, T_in, H_in, W_in = dummy_video_tensor.shape # Desempaqueta 4 valores en 4 variables
        # La forma de los tensores de video en PyTorch es (N, C, T, H, W) o (C, T, H, W) para una sola muestra.
        # torchvision.io.read_video con output_format="TCHW" da (T, C, H, W).
        # Nuestros DatasetVideos lo convierte a (T, C, H, W) y luego el collate_fn lo convierte a (N, T, C, H, W).
        # El modelo espera (N, C, T, H, W). Necesitamos permutar.
        
        # dummy_video_tensor del dataset es (T, C, H, W)
        T_in, C_in, H_in, W_in = dummy_video_tensor.shape
        
        # Para el modelo 3D CNN, la entrada esperada es (Batch, C, T, H, W)
        print(f"\nDimensiones de entrada esperadas para el modelo: (Batch, {C_in}, {T_in}, {H_in}, {W_in})")

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
        train_model(model, train_dataloader, criterion, optimizer, args.epocas, device)

        # Evaluar el modelo (opcional después de entrenar)
        evaluate_model(model, val_dataloader, criterion, device)

        # Guardar estado del modelo entrenado
        model_name = f"models/model_torch_CNN3D_{args.epocas}_epochs_{time.time()}.pth"
        torch.save(model.state_dict(), model_name)
        logger.info(f"Modelo guardado en '{model_name}'.")
    except Exception as e:
        logger.error(str(e), exc_info=True) # Añadir exc_info=True para ver el traceback completo


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Modelo CNN3D - entrenamiento y guardado.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--epocas", default=1, type=ut.non_negative_int,
                        help="Número de épocas para el entrenamiento.")
    parser.add_argument("--video_size", default="224p", type=str, choices=["224p", "720p"],
                        help="Tamaño de los tensores/videos que se utilizarán (ej: '224p' o '720p').")
    parser.add_argument("--load_from_tensors", default=1, type=int, choices=[0, 1],
                        help="0: Cargar videos directamente de archivos. 1: Cargar tensores preprocesados.")

    return parser.parse_args()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        args = parse_arguments()
    else:
        # Valores por defecto para depuración si no se proporcionan argumentos
        class Args:
            epocas = 1
            video_size = "224p" # O "720p"
            load_from_tensors = 0 # Cambia a 0 para cargar videos directamente, 1 para tensores

        args = Args()

    main(args)