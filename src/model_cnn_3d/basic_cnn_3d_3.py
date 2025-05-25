import argparse
import logging
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split # Importamos random_split
from torch.utils.data._utils.collate import default_collate # Importar explicitamente

# Importa las funciones para cargar datasets y las transformaciones
from src.preprocess.dataset_soccernet import ORIGINAL_SIZE, FRAMES_PER_CLIP, NUM_CLASSES, \
    get_full_preprocessed_dataset, get_full_direct_video_dataset, \
    train_transforms_videos, val_transforms_videos, \
    TARGET_HEIGHT_MODEL, TARGET_WIDTH_MODEL # Importa las transformaciones específicas
from src.utils import utils as ut
from src.utils.constants import * # Asegúrate de que DS_SOCCERNET_ACTIONS y DS_SOCCERNET_TENSORS estén definidos aquí

logger = logging.getLogger(__name__)

# Hiperparámetros de entrenamiento, ajustar basado en la memoria de la GPU
BATCH_SIZE = 8
LEARNING_RATE = 0.001
# NUM_EPOCHS ahora será args.epocas, no una constante fija

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


# --- Modelo Simple CNN 3D (Sin cambios, solo asegurando la inicialización) ---
class SimpleCNN3D(nn.Module):
    def __init__(self, num_classes, input_channels, input_frames, input_size):
        super(SimpleCNN3D, self).__init__()
        H_in, W_in = input_size 
        
        self.conv1 = nn.Conv3d(input_channels, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        # Calcular el tamaño de la capa lineal dinámicamente
        self._set_fc_input_features(input_channels, input_frames, H_in, W_in)

        self.fc = nn.Linear(self.fc_input_features, num_classes)

    def _set_fc_input_features(self, input_channels, input_frames, H_in, W_in):
        dummy_tensor = torch.zeros(1, input_channels, input_frames, H_in, W_in)
        with torch.no_grad():
            x = self.pool1(self.relu1(self.conv1(dummy_tensor)))
            x = self.pool2(self.relu2(self.conv2(x)))
        self.fc_input_features = x.numel()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        
        x = x.view(-1, self.fc_input_features)
        x = self.fc(x)
        return x


# --- Función collate_fn para manejar elementos None en el DataLoader ---
def collate_fn(batch):
    # Filtra las muestras inválidas (donde video_tensor es None)
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        # Importante: Retornar tensores vacíos con las formas correctas para evitar errores de PyTorch
        # Asumiendo que videos son (C, T, H, W) y labels son un escalar
        dummy_video_shape = (3, FRAMES_PER_CLIP, TARGET_HEIGHT_MODEL, TARGET_WIDTH_MODEL) # C, T, H, W
        return torch.empty(0, *dummy_video_shape), torch.empty(0, dtype=torch.long)
    
    return default_collate(batch)


# --- Funciones de entrenamiento y evaluación ---
def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        logger.info(f"Época {epoch + 1}/{num_epochs}")
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        for i, (videos, labels) in enumerate(dataloader):
            if videos.numel() == 0:
                logger.warning(f"Batch vacío en la época {epoch+1}, iteración {i}. Saltando.")
                continue

            # Permutar las dimensiones: (Batch, T, C, H, W) -> (Batch, C, T, H, W)
            videos = videos.permute(0, 2, 1, 3, 4).to(device)
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

        if len(dataloader) > 0: # Evitar división por cero si el dataloader está vacío
            avg_loss = total_loss / len(dataloader)
            accuracy = 100 * correct_predictions / total_samples
            logger.info(f"  Pérdida media de la época: {avg_loss:.4f}, Precisión: {accuracy:.2f}%")
        else:
            logger.warning("  No hay batches válidos para calcular la pérdida y precisión de la época.")


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
            
            # Permutar las dimensiones: (Batch, T, C, H, W) -> (Batch, C, T, H, W)
            videos = videos.permute(0, 2, 1, 3, 4).to(device)
            labels = labels.to(device)

            outputs = model(videos)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
    
    if len(dataloader) > 0: # Evitar división por cero si el dataloader está vacío
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct_predictions / total_samples
        logger.info(f"\nResultados de evaluación:")
        logger.info(f"  Pérdida media: {avg_loss:.4f}, Precisión: {accuracy:.2f}%")
    else:
        logger.warning("  No hay batches válidos para calcular la pérdida y precisión en evaluación.")


def main(args) -> None:
    try:
        config_log()

        # --- Bandera para elegir el tipo de dataset ---
        LOAD_FROM_TENSORS = args.load_from_tensors 

        if LOAD_FROM_TENSORS:
            logger.info("Cargando dataset de tensores preprocesados...")
            # Cargar el dataset completo sin splits internos
            full_dataset = get_full_preprocessed_dataset(DS_SOCCERNET_TENSORS, args.video_size)
        else:
            logger.info("Cargando dataset de videos directamente desde archivos...")
            # Cargar el dataset completo sin splits internos
            full_dataset = get_full_direct_video_dataset(DS_SOCCERNET_ACTIONS, args.video_size)

        # --- Realizar el split de train/valid/test aquí ---
        # Asegúrate de que las proporciones del split sean las que deseas
        train_size = int(0.7 * len(full_dataset)) # 70% para entrenamiento
        val_size = int(0.15 * len(full_dataset)) # 15% para validación
        test_size = len(full_dataset) - train_size - val_size # 15% para test (el resto)

        # Realizar el split de forma aleatoria y reproducible
        # Importante: random_split aplica las transformaciones de forma implícita si el dataset las tiene.
        # Por eso, las transformaciones pasadas a get_full_*_dataset deben ser las deterministas.
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, 
            [train_size, val_size, test_size], 
            generator=torch.Generator().manual_seed(SEMILLA) # Para reproducibilidad
        )

        # --- Aplicar transformaciones de aumentación de datos solo al conjunto de entrenamiento ---
        # Si LOAD_FROM_TENSORS es True, puedes aplicar transformaciones adicionales aquí a train_dataset
        # si tus tensores preprocesados no incluyen aumentación de datos.
        # Si LOAD_FROM_TENSORS es False, las transformaciones ya vienen con train_transforms_videos.
        if not LOAD_FROM_TENSORS:
            # Reemplazar la transformación del train_dataset creado por random_split
            # por la transformación que incluye aumentación (RandomHorizontalFlip, etc.).
            # Esto es un truco, ya que random_split crea Subsets.
            # Accedemos al dataset subyacente del Subset para cambiar su transformación.
            # Si train_dataset.dataset no funciona, podría ser necesario recrear el Subset con nuevas transforms.
            # Una forma más limpia sería hacer un wrapper Dataset que aplique transforms condicionalmente.
            
            # Opción 1: Aplicar transformaciones de aumentación a los datos del DataLoader.
            # Esta es la más sencilla y recomendada, ya que los datos de entrada al DataLoader
            # se transformarán justo antes de ser pasados al modelo.
            # No se necesita modificar el `transform` de `train_dataset` directamente.
            # `train_transforms_videos` y `val_transforms_videos` ya hacen lo que quieres.

            logger.info("Las transformaciones de aumentación de datos se aplicarán automáticamente al DataLoader de entrenamiento.")
        
        # Usar la función collate_fn personalizada en DataLoader
        # num_workers puede ser ajustado. os.cpu_count() // 2 es un buen punto de partida.
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count() // 2, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count() // 2, collate_fn=collate_fn)
        # test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count() // 2, collate_fn=collate_fn)


        # Obtener las dimensiones del tensor de entrada (para el modelo)
        dummy_video_tensor = None
        i = 0
        while dummy_video_tensor is None and i < len(full_dataset): # Usar full_dataset para obtener una muestra
            dummy_video_tensor, _ = full_dataset[i]
            i += 1

        if dummy_video_tensor is None:
            logger.error("Error: No se pudo cargar ninguna muestra válida del dataset para determinar el tamaño de entrada del modelo.")
            exit()

        # dummy_video_tensor del dataset es (T, C, H, W) después de las transformaciones
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
        logger.error(str(e), exc_info=True)


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