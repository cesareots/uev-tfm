import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

# --- Cargar un modelo R(2+1)D pre-entrenado en Kinetics-400 ---
# Usaremos la enumeración de pesos para obtener la versión más reciente y recomendada
weights = R2Plus1D_18_Weights.KINETICS400_V1
model_r2plus1d = r2plus1d_18(weights=weights)

# El "núcleo" o extractor de características es la mayor parte del modelo, excluyendo la capa final de clasificación (fully connected layer).
# En los modelos de torchvision, la capa de clasificación se llama a menudo 'fc'.

print("Arquitectura original de R(2+1)D_18:")
print(model_r2plus1d)
print("-" * 50)

# Para usarlo como extractor de características para transfer learning:
# 1. Observa la capa final. Para r2plus1d_18, es model_r2plus1d.fc
#    Esta capa lineal toma las características aplanadas y las proyecta a las clases originales (ej. 400 para Kinetics).
original_fc_in_features = model_r2plus1d.fc.in_features
print(f"Características de entrada a la capa fc original: {original_fc_in_features}")

# 2. Reemplaza la capa 'fc' con una nueva para tu número de clases.
#    Supongamos que NUM_CLASSES es el número de clases de tu dataset Soccernet.
NUM_CLASSES_SOCEURNET = 4 # Ejemplo, reemplaza con tu NUM_CLASSES real
model_r2plus1d.fc = nn.Linear(original_fc_in_features, NUM_CLASSES_SOCEURNET)

# Ahora, model_r2plus1d tiene el núcleo pre-entrenado y una nueva capa de clasificación.
# Puedes proceder a congelar capas y entrenar.
print("\nArquitectura R(2+1)D_18 con capa 'fc' modificada:")
print(model_r2plus1d.fc)
print("-" * 50)

# El "núcleo" como tal serían todas las capas ANTES de model_r2plus1d.fc:
# model_r2plus1d.stem
# model_r2plus1d.layer1
# model_r2plus1d.layer2
# model_r2plus1d.layer3
# model_r2plus1d.layer4
# Y luego un pooling adaptativo antes de la capa fc.

# Si quisieras pasar un tensor de vídeo a través del núcleo para obtener características:
# (Asumiendo un tensor de entrada dummy)
# batch_size = 1
# num_channels = 3
# clip_frames = 16 # Ejemplo
# height = 112 # Tamaño de entrada esperado por r2plus1d_18
# width = 112
# dummy_video = torch.randn(batch_size, num_channels, clip_frames, height, width)

# feature_extractor = nn.Sequential(
#     model_r2plus1d.stem,
#     model_r2plus1d.layer1,
#     model_r2plus1d.layer2,
#     model_r2plus1d.layer3,
#     model_r2plus1d.layer4,
#     # nn.AdaptiveAvgPool3d((1,1,1)) # El modelo internamente ya hace pooling antes de fc
# )
# features = feature_extractor(dummy_video)
# print(f"\nForma de las características extraídas (antes de aplanar para fc): {features.shape}")
# Después de esto, se aplanaría y pasaría a la capa fc.
# El modelo r2plus1d_18 ya incluye un AveragePool3d antes de la capa fc en su `forward`.
