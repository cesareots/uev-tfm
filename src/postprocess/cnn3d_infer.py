# import torch
# import torchvision.io
# import numpy as np
from src.models.main_cnn3d import *

# La dependencia de torchvision.io.read_video es para leer todo el video.
# Para grandes videos, puede ser más eficiente usar un iterador de frames como cv2.VideoCapture
# o decodificar por chunks, pero para 10 minutos está bien.

# Ajustar esto si cambias la duración del clip en el entrenamiento
CLIP_DURATION_SEC_INFERENCE = CLIP_DURATION_SEC
INFERENCE_FPS = ORIGINAL_FPS  # Leer el video a su FPS original para tener todos los frames


def infer_full_video(
        model,
        video_path,
        device,
        frames_per_clip,
        clip_duration_sec,
        original_fps,
        target_fps_model_input,
        original_size,
        inv_label_map,
        stride_sec: float = 1.0,
):
    """
    Realiza inferencia sobre un video completo usando un enfoque de ventana deslizante.
    Genera series temporales de probabilidades para cada acción.
    Args:
        model (nn.Module): El modelo 3D CNN entrenado.
        video_path (str): Ruta al archivo de video del partido completo.
        device (torch.device): El dispositivo donde ejecutar el modelo.
        frames_per_clip (int): Número de frames por clip que el modelo espera (ej. 16).
        clip_duration_sec (int): Duración en segundos de cada clip (ej. 4 segundos).
        original_fps (int): FPS original del video (ej. 25).
        target_fps_model_input (float): FPS efectivo con el que se muestrean los frames para el modelo (ej. 4.0).
        original_size (tuple): (alto, ancho) original de los frames.
        inv_label_map (dict): Mapeo inverso de etiquetas para referencia.
        stride_sec (float): El paso de la ventana deslizante en segundos.
    Returns:
        dict: Un diccionario donde las claves son los nombres de las acciones
              y los valores son listas de tuplas (timestamp_sec, probability)
              para cada acción a lo largo del video.
    """
    model.eval()
    all_probabilities_over_time = {label: [] for label in inv_label_map.values()}

    # Decodificar el video completo a su FPS original
    print(f"Decodificando video completo: '{video_path}'")
    try:
        # Se decodifica a FPS original para luego muestrear correctamente cada ventana
        full_video_tensor, audio, info = torchvision.io.read_video(
            video_path,
            pts_unit='sec',
            output_format='TCHW'  # Time, Channels, Height, Width
        )
        # full_video_tensor shape: (Num_total_frames, C, H, W)
        print(f"Video decodificado. Shape: {full_video_tensor.shape}")

        num_total_frames = full_video_tensor.shape[0]
        video_fps = info['video_fps'] if 'video_fps' in info else ORIGINAL_FPS  # Usar FPS real si disponible

        # Normalizar a float y permutar a (C, T, H, W) en el bucle para cada clip
        full_video_tensor = full_video_tensor.float() / 255.0
    except Exception as e:
        print(f"Error al decodificar el video completo {video_path}: {e}")
        return all_probabilities_over_time  # Devolver vacío

    # Calcular cuántos frames hay en un clip completo a `video_fps`
    frames_in_full_clip = int(clip_duration_sec * video_fps)
    # Convertir el stride de segundos a número de frames
    stride_frames = int(stride_sec * video_fps)

    # Iterar con ventana deslizante
    current_frame_start = 0

    while current_frame_start + frames_in_full_clip <= num_total_frames:
        t_start_sec = current_frame_start / video_fps
        t_center_sec = t_start_sec + (clip_duration_sec / 2)  # Timestamp central del clip

        # Extraer los frames para la ventana actual
        clip_frames = full_video_tensor[current_frame_start: current_frame_start + frames_in_full_clip]

        # Muestreo uniforme: seleccionar `frames_per_clip` frames equidistantes del clip extraído
        # Esto asegura que el input al modelo tiene el número de frames esperado (ej. 16)
        if clip_frames.shape[0] < frames_per_clip:  # Esto debería ser raro si frames_in_full_clip > frames_per_clip
            # Si el clip es demasiado corto (ej. al final del video), rellena o salta
            print(f"Advertencia: Clip en {t_start_sec:.2f}s es más corto de lo esperado. Saltando.")
            current_frame_start += stride_frames
            continue

        indices = np.linspace(0, clip_frames.shape[0] - 1, frames_per_clip, dtype=int)
        sampled_clip_tensor = clip_frames[indices]

        # Asegurar el formato (C, T, H, W) para el modelo
        sampled_clip_tensor = sampled_clip_tensor.permute(1, 0, 2, 3)  # Channels, Time, Height, Width

        with torch.no_grad():
            # Añadir dimensión de batch
            input_tensor = sampled_clip_tensor.unsqueeze(0).to(device)
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze(0).cpu().numpy()

        # Almacenar probabilidades para cada acción
        for idx, prob in enumerate(probabilities):
            action_name = inv_label_map[idx]
            all_probabilities_over_time[action_name].append((t_center_sec, prob))

        current_frame_start += stride_frames

        # Opcional: Imprimir progreso
        if current_frame_start % (video_fps * 60) == 0:  # Cada minuto
            print(f"Inferencia en {current_frame_start / video_fps:.2f} segundos...")

    print("Inferencia del video completada.")
    
    return all_probabilities_over_time


# --- Cargar el modelo entrenado antes de la inferencia ---
# Asegúrate de que los parámetros del modelo coincidan con cómo fue entrenado
# Dimensiones de entrada al modelo (C, T, H, W)
# Debes obtener esto del dummy_video_tensor.shape del dataset
# Por ejemplo, si tus videos son (224, 398) y usas 16 frames:
C_MODEL_IN = 3
T_MODEL_IN = FRAMES_PER_CLIP
H_MODEL_IN = ORIGINAL_SIZE[1]  # 224
W_MODEL_IN = ORIGINAL_SIZE[0]  # 398

# Cargar el modelo
model_for_inference = SimpleCNN3D(
    num_classes=NUM_CLASSES,
    input_channels=C_MODEL_IN,
    input_frames=T_MODEL_IN,
    input_size=(H_MODEL_IN, W_MODEL_IN)
).to(device)

model_for_inference.load_state_dict(torch.load("z_models/model_torch_cnn_3d_10_epochs.pth", map_location=device))
model_for_inference.eval()  # Modo evaluación

# Ejemplo de uso:
# Asegúrate de tener un archivo de video para probar
TEST_VIDEO_PATH = "dataset/soccernet/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/1_224p_trimmed.mkv"

if os.path.exists(TEST_VIDEO_PATH):
    all_probs = infer_full_video(
        model_for_inference,
        TEST_VIDEO_PATH,
        device,
        FRAMES_PER_CLIP,
        CLIP_DURATION_SEC,
        ORIGINAL_FPS,
        TARGET_FPS,
        ORIGINAL_SIZE,
        INV_LABEL_MAP,
        #stride_sec=0.5  # Ejemplo: paso de 0.5 segundos
    )
    print("\nSeries temporales de probabilidades generadas:")
    #print(all_probs)
else:
    print(f"Error: No se encontró el video de prueba en {TEST_VIDEO_PATH}")


from scipy.signal import find_peaks # Necesitarás instalar scipy: pip install scipy

def post_process_probabilities(
    all_probabilities_over_time,
    inv_label_map,
    confidence_thresholds,
    NMS_window_sec,
):
    """
    Realiza post-procesamiento para extraer eventos discretos.
    Args:
        all_probabilities_over_time (dict): Salida de infer_full_video.
        inv_label_map (dict): Mapeo inverso de etiquetas.
        confidence_thresholds (dict): Diccionario de umbrales de confianza para cada acción.
                                      Ej: {"Goal": 0.8, "Yellow card": 0.75, ...}
                                      Si no se especifica para una acción, se puede usar un umbral general.
        NMS_window_sec (float): Ventana de Supresión No Máxima en segundos.
    Returns:
        list: Lista de eventos detectados, cada uno como un diccionario.
              Ej: [{'event_type': 'Goal', 'timestamp_sec': 19.5, 'confidence': 0.98}]
    """
    detected_events = []
    
    # Convertir NMS_window_sec a número de "pasos" del stride
    # Asumimos que la lista de probabilidades se generó con un stride_sec consistente
    # Necesitamos el stride_sec de la función infer_full_video
    # Por simplicidad aquí, lo calcularemos asumiendo el primer elemento de una lista de prob.
    # En un código real, la función infer_full_video debería devolver también el stride_sec real.
    
    # Para estimar el stride_frame_idx (asumiendo que las listas de all_probabilities_over_time
    # están ordenadas por timestamp y tienen un espaciado regular)
    if not all_probabilities_over_time:
        return []
    
    # Tomar la primera acción para estimar el stride_sec_used_in_inference
    first_action_probs = next(iter(all_probabilities_over_time.values()))
    
    if len(first_action_probs) > 1:
        # Asume que el `stride_sec` usado en `infer_full_video` es constante.
        stride_sec_used_in_inference = first_action_probs[1][0] - first_action_probs[0][0]
        # Calcular la ventana NMS en número de índices (aproximado)
        nms_window_indices = max(1, int(NMS_window_sec / stride_sec_used_in_inference))
    else:
        stride_sec_used_in_inference = 1.0 # Default si no hay suficientes puntos
        nms_window_indices = 1

    for action_name, prob_list in all_probabilities_over_time.items():
        if not prob_list:
            continue
            
        timestamps = np.array([p[0] for p in prob_list])
        probabilities = np.array([p[1] for p in prob_list])
        
        # Obtener umbral específico para esta acción o usar un umbral general
        threshold = confidence_thresholds.get(action_name, 0.7) # Umbral general de 0.7 si no específico
        
        # 1. Umbralización y Detección de Picos
        # find_peaks busca picos en la señal. `height` es el umbral mínimo.
        # 'distance' es la distancia horizontal mínima entre picos (para NMS inicial)
        # Convertimos la distancia en segundos a número de índices
        
        # indices_of_peaks, _ = find_peaks(probabilities, height=threshold, distance=nms_window_indices)
        # Un enfoque más robusto para NMS es hacerlo después de identificar todos los picos.
        
        # Identificar todos los puntos por encima del umbral
        candidate_indices = np.where(probabilities >= threshold)[0]
        
        if len(candidate_indices) == 0:
            continue

        # Ordenar los candidatos por probabilidad descendente
        sorted_candidates = sorted(candidate_indices, key=lambda i: probabilities[i], reverse=True)
        
        suppressed = np.zeros_like(probabilities, dtype=bool)
        
        for idx in sorted_candidates:
            if suppressed[idx]:
                continue
            
            # Este es un evento detectado
            detected_events.append({
                'event_type': action_name,
                'timestamp_sec': timestamps[idx],
                'confidence': probabilities[idx]
            })
            
            # Suprimir picos cercanos usando NMS
            # Calcular la ventana de supresión en índices alrededor del pico actual
            start_suppress_idx = max(0, idx - nms_window_indices)
            end_suppress_idx = min(len(probabilities), idx + nms_window_indices + 1)
            suppressed[start_suppress_idx:end_suppress_idx] = True

    # Ordenar todos los eventos detectados por tiempo
    detected_events.sort(key=lambda x: x['timestamp_sec'])
    
    return detected_events

# --- Ejemplo de uso del post-procesamiento ---
# Estos umbrales son de ejemplo, necesitarás ajustarlos experimentalmente
# para cada acción basándote en el rendimiento de tu modelo.
CONFIDENCE_THRESHOLDS = {
    "Goal": 0.8,
    "Yellow card": 0.8,
    "Shots on target": 0.9,
    # Añadir umbrales para el resto de las 17 acciones
}
NMS_WINDOW_SEC = 2.0 # Si dos detecciones de la misma acción están a menos de 2 segundos, quédate con la más alta

# all_probs sería el resultado de la función infer_full_video
detected_final_events = post_process_probabilities(all_probs, INV_LABEL_MAP, CONFIDENCE_THRESHOLDS, NMS_WINDOW_SEC)
print("\nEventos finales detectados:")
for event in detected_final_events:
    # Formatear el timestamp para mayor legibilidad
    minutes = int(event['timestamp_sec'] // 60)
    seconds = int(event['timestamp_sec'] % 60)
    print(f"  {minutes:02d}:{seconds:02d} - {event['event_type']} (Confianza: {event['confidence']:.4f})")
