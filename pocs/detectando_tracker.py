import cv2
from ultralytics import YOLO

import utils as ut
from constants import *

# Cargar tu modelo YOLO. Asegúrate de que pueda detectar balones, jugadores, etc.
# Si tu modelo no fue entrenado originalmente para estos objetos, quizás necesites afinarlo.
model = YOLO("yolo11n.pt")  # Ajusta según tu modelo

# --- Configuración del video, tiempo a procesar y rastreador ---
video_path = os.path.join(DS_SOCCERNET, DS_SOCCERNET_V)
tracker_config_path = "bytetrack.yaml"  # <--- Nombre del archivo de configuración del rastreador (si está en carpeta default de ultralytics/cfg/trackers)
# O usa la ruta completa si es un archivo personalizado: "/ruta/a/mi/bytetrack_custom.yaml"

duration_seconds = 10

# --- Configuración para guardar el video resultante (Opcional) ---
output_dir = "runs/track/video_timed_tracking_output"
os.makedirs(output_dir, exist_ok=True)
output_filename = os.path.join(output_dir, "video_tracking_result.mp4")
video_writer = None  # Se inicializará después del primer fotograma

# --- Inicializar historial de seguimiento (para tu lógica de eventos) ---
# Puedes usar un diccionario donde la clave es el track_id y el valor es una lista de estados pasados
track_history = {}  # {track_id: [{'frame': N, 'time': T, 'bbox_xywh': [x,y,w,h], 'conf': c, 'class_id': cls_id}, ...], ...}

# --- Cargar el video SOLO para obtener FPS y total de fotogramas ---
# Esto es necesario para calcular el número de fotogramas objetivo para la duración.
cap_info = cv2.VideoCapture(video_path)
if not cap_info.isOpened():
    print(f"Error: No se pudo abrir el archivo de video en {video_path} para obtener información.")
    exit()  # Salir si no se puede abrir el video

fps = cap_info.get(cv2.CAP_PROP_FPS)
frame_count_total = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
duration_total_seconds = frame_count_total / fps
cap_info.release()  # Liberar el objeto de captura una vez obtenida la información

print(f"Video cargado: {video_path}")
print(f"FPS del video: {fps}")
print(f"Duración Total del video: {duration_total_seconds:.2f} segundos")

# --- Calcular el número de fotogramas objetivo ---
target_frame_count = min(int(duration_seconds * fps), frame_count_total)
print(
    f"Se realizará seguimiento en los primeros {duration_seconds} segundos, equivalente a {target_frame_count} fotogramas.")

# --- Iniciar el Seguimiento (Tracking) ---
# Usamos model.track() en lugar de model.predict()
# stream=True para obtener un generador
# show=False y save=False porque manejaremos la visualización y guardado manualmente
# tracker=... especifica el archivo de configuración del rastreador
results_generator = model.track(
    source=video_path,
    tracker=tracker_config_path,
    stream=True,
    show=False,  # Nosotros controlamos cv2.imshow
    save=False,  # Nosotros controlamos VideoWriter
    verbose=False  # Menos mensajes en consola
)

frame_number = 0  # Contador para el fotograma actual

print("Iniciando seguimiento de objetos...")

# --- Bucle para procesar fotograma a fotograma los resultados del tracking ---
try:
    # Iteramos sobre el generador que nos da resultados CON tracking IDs
    for result in results_generator:
        current_time_in_video = frame_number / fps

        # --- ¡La condición para detener por tiempo! ---
        if frame_number >= target_frame_count:
            print(f"Alcanzado el número de fotogramas objetivo ({target_frame_count}). Deteniendo seguimiento.")
            break  # Salir del bucle

        # --- Obtener el fotograma con detecciones y tracking IDs dibujados ---
        annotated_frame = result.plot()  # Dibuja las cajas, clases, confianzas Y los IDs de seguimiento

        # --- Inicializar el VideoWriter en el primer fotograma (si vas a guardar) ---
        if video_writer is None:
            frame_height, frame_width, _ = annotated_frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Códec para MP4
            video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
            print(f"Guardando video con seguimiento en: {output_filename}")

        # --- Escribir el fotograma anotado en el archivo de video (si vas a guardar) ---
        if video_writer is not None:
            video_writer.write(annotated_frame)

        # --- Mostrar el fotograma en tiempo real ---
        cv2.imshow("YOLOv8 Tracking (Procesando X segundos)", annotated_frame)

        # --- Permitir la detención manual con tecla 'q' (opcional) ---
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Detención manual solicitada.")
            break

        # --- Acceder a las detecciones CON Tracking IDs y guardarlas en el historial ---
        if result.boxes.is_track and result.boxes is not None:  # Asegurarse de que hay tracking IDs y detecciones
            for box in result.boxes:
                # Verifica si box.id no es None (aunque is_track=True, siempre verifica)
                if box.id is not None:
                    track_id_value = int(box.id[0])  # El ID de seguimiento
                    # Obtén otras propiedades que necesites
                    x_center, y_center, w, h = box.xywh[0].cpu().numpy()
                    conf_value = float(box.conf[0])
                    class_id_value = int(box.cls[0])
                    class_name = model.names[class_id_value]

                    # Almacenar el estado actual de este objeto rastreado
                    if track_id_value not in track_history:
                        track_history[track_id_value] = []

                    track_history[track_id_value].append({
                        'frame': frame_number,
                        'time': current_time_in_video,
                        'class_name': class_name,
                        'confidence': conf_value,
                        'bbox_xywh': [float(x_center), float(y_center), float(w), float(h)],  # Guardar como flotantes
                        # Puedes añadir 'bbox_xyxy' si lo necesitas también
                    })

        # --- ¡¡¡Aquí va tu lógica de detección de eventos!!! ---
        # Ahora tienes:
        # - El fotograma actual: frame
        # - El número de fotograma: frame_number
        # - El tiempo en segundos: current_time_in_video
        # - Las detecciones CON Tracking IDs: result.boxes (y puedes acceder a 'box.id')
        # - El historial de seguimiento de objetos: track_history (con ID persistentes)

        # Tu lógica aquí analizaría la información en 'result.boxes' para el fotograma actual,
        # y quizás consultaría 'track_history' para ver la trayectoria o el estado pasado de un objeto.
        # Por ejemplo:
        # - Buscar si el balón (clase 'ball') está siendo rastreado (tiene ID).
        # - Consultar la posición del balón en los últimos N fotogramas desde track_history[ball_id].
        # - Ver si la trayectoria del balón rastreado cruza la línea de gol (necesitas definir la línea).
        # - Buscar jugadores rastreados ('person') cerca de la línea de fuera de juego y comparar sus IDs con el balón y el pasador.
        # - Si detectas una tarjeta (si tu modelo detecta tarjetas o árbitros haciendo gestos) asociada a un jugador rastreado.

        # if result.boxes.is_track: # Asegúrate de que estás en modo tracking
        #     for box in result.boxes:
        #         if box.id is not None:
        #             track_id = int(box.id[0])
        #             class_name = model.names[int(box.cls[0])]
        #             x_c, y_c, w, h = box.xywh[0].cpu().numpy()

        #             # Ejemplo MUY BÁSICO: Imprimir algo si se detecta un jugador rastreado
        #             # if class_name == 'person':
        #             #     print(f"  -> Jugador rastreado ID {track_id} detectado en fotograma {frame_number}")

        #             # Ejemplo BÁSICO (concept): Detectar si el balón entra en una zona
        #             # if class_name == 'ball':
        #             #     # Necesitas definir las coordenadas de la "zona de gol" o "zona de saque de banda"
        #             #     if zona_de_gol_contiene(x_c, y_c): # Implementa esta función
        #             #         print(f"  -> ¡BALÓN rastreado ID {track_id} en ZONA DE GOL en fotograma {frame_number}!")
        #                         # Aquí añadirías lógica para CONFIRMAR si es un gol (mirar trayectoria, etc.)

        # Incrementar el contador de fotogramas
        frame_number += 1
        ut.write_list_in_txt(lista=track_history, file="detectando_tracker_gemini_track_history.txt")

except Exception as e:
    print(f"\nOcurrió un error durante el seguimiento: {e}")
    import traceback

    traceback.print_exc()  # Imprime detalles del error

finally:
    # --- Limpieza al finalizar (siempre se ejecuta) ---
    if video_writer is not None:
        video_writer.release()  # Asegura que el archivo de video se cierre correctamente
        print(f"\nArchivo de video con seguimiento guardado en: {output_filename}")
    cv2.destroyAllWindows()  # Cierra cualquier ventana de OpenCV abierta
    print("Proceso de seguimiento finalizado.")

    # --- Opcional: Puedes guardar el historial de seguimiento completo si lo necesitas ---
    # import json
    # with open(os.path.join(output_dir, "tracking_history.json"), 'w') as f:
    #     json.dump(track_history, f, indent=4)
    # print(f"Historial de seguimiento guardado en: {os.path.join(output_dir, 'tracking_history.json')}")
