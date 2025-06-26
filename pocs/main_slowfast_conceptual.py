import torch
import torch.nn as nn

# --- Cargar un modelo SlowFast pre-entrenado (ej. slowfast_r50) ---
# Esto descargará el modelo y los pesos si es la primera vez.
# Necesitas tener conexión a internet.
try:
    model_slowfast = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
except Exception as e:
    print(f"Error al cargar el modelo SlowFast desde torch.hub: {e}")
    print("Asegúrate de tener conexión a internet y que el repositorio de PyTorchVideo sea accesible.")
    model_slowfast = None

if model_slowfast:
    print("Arquitectura original de SlowFast (solo la parte de bloques, el clasificador es 'blocks[6]'):")
    # La estructura de SlowFast es más compleja de imprimir directamente que un Sequential.
    # Generalmente tiene una sección para el 'multipathway_transform' (preparación de entradas),
    # y luego 'blocks' que contienen las etapas de los caminos Slow y Fast y su fusión.
    # El clasificador final suele estar al final del último bloque (ej. blocks[6] para slowfast_r50).
    
    # Para inspeccionar, puedes iterar sobre los módulos principales:
    # for name, module in model_slowfast.named_children():
    #    print(f"Componente: {name}, Tipo: {type(module)}")
    # print(model_slowfast.blocks[6]) # Muestra la cabeza de clasificación original

    # Para usarlo para transfer learning:
    # El clasificador en los modelos SlowFast de PyTorchVideo suele estar en el último
    # bloque, a menudo como una proyección lineal y un softmax/log_softmax.
    # Para slowfast_r50, la cabeza de proyección está en model_slowfast.blocks[6].proj
    
    # Necesitas saber el número de características de entrada a esa proyección.
    # Esto puede variar, pero para slowfast_r50 es 2048 (del slow pathway) + 256 (del fast pathway) = 2304
    # O puedes obtenerlo inspeccionando la capa:
    if hasattr(model_slowfast.blocks[6], 'proj'):
        original_proj_in_features = model_slowfast.blocks[6].proj.in_features
        print(f"\nCaracterísticas de entrada a la proyección original: {original_proj_in_features}")

        NUM_CLASSES_SOCEURNET = 4 # Ejemplo
        # Reemplazar la capa de proyección
        model_slowfast.blocks[6].proj = nn.Linear(original_proj_in_features, NUM_CLASSES_SOCEURNET)
        
        # El output_pool también puede necesitar ajuste si esperas un formato específico.
        # Generalmente, los modelos SlowFast terminan con un pooling antes de la proyección.
        # La capa de activación final (ej. Softmax) también se manejaría externamente o
        # se añadiría después de esta nueva proyección si es necesario para tu pérdida.

        print("\nProyección de la cabeza de SlowFast modificada:")
        print(model_slowfast.blocks[6].proj)
    else:
        print("No se encontró la capa 'proj' esperada en model_slowfast.blocks[6]. La estructura puede haber cambiado.")
    print("-" * 50)

    # El "núcleo" aquí son los bloques que componen los caminos Slow y Fast y sus conexiones.
    # Es decir, model_slowfast.blocks[0] hasta model_slowfast.blocks[5] (aproximadamente).

    # Para pasar un tensor de vídeo:
    # SlowFast espera una LISTA de tensores como entrada: uno para el Slow pathway y otro para el Fast pathway.
    # Debes preprocesar tus clips para generar estas dos vistas con diferentes tasas de muestreo.
    # clip_frames_slow = 8 # Ejemplo
    # clip_frames_fast = 32 # Ejemplo
    # height = 256 # Ejemplo
    # width = 320 # Ejemplo
    # num_channels = 3
    # batch_size = 1
    
    # input_slow = torch.randn(batch_size, num_channels, clip_frames_slow, height, width)
    # input_fast = torch.randn(batch_size, num_channels, clip_frames_fast, height, width)
    # list_of_inputs = [input_slow, input_fast]
    
    # Para obtener características (requiere un forward parcial o modificar la clase):
    # No hay un '.features' simple. Tendrías que modificar la clase o ejecutar hasta
    # un punto intermedio de la función forward del modelo.
    # features = model_slowfast(list_of_inputs) # Esto daría las logits de la clase con el modelo modificado
    # print(f"\nForma de la salida del modelo SlowFast modificado: {features.shape}")
    