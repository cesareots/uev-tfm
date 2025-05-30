import logging
import time

import cv2
from ultralytics import YOLO

import utils as ut
from constants import *

logger = logging.getLogger(__name__)


def config_log() -> None:
    ut.verify_directory(LOG_DIR)

    logging.basicConfig(
        filename=LOG_MAIN,
        filemode="a",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        # https://docs.python.org/3/library/logging.html
        level=logging.INFO,  # NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL
    )


def get_model_yolo() -> YOLO:
    model = YOLO(model=YOLO_MODEL)
    model.info()

    # el modelo yolo se carga en cpu/gpu
    print(f"\n--- model.device: {model.device}")

    # capacidad del modelo para detectar: jugadores, balones, etc.
    # print(f"\n--- model.names:\n{model.names}")
    ut.write_dictionary_in_txt(
        diccionario=model.names,
        archivo_salida=f"{YOLO_MODEL_NAMES}{YOLO_MODEL.split(".")[0]}{EXT_TXT}",
    )

    return model


def main() -> None:
    model = get_model_yolo()

    # procesar un video
    video_path = os.path.join(DS_SOCCERNET, DS_SOCCERNET_V)
    cap = ut.load_video(video_path)
    # cargar labels
    labels_path = os.path.join(DS_SOCCERNET, DS_SOCCERNET_L_V2)
    video_labels = ut.load_annotations(labels_path)

    time_user = 5.0  # cantidad segundos
    # time_user = 4 / 25  # cantidad frames
    # time_user = 0.0  # video completo
    name_unique = time.time()

    # Obtener información del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = frame_count / fps
    frame_number = 0

    if time_user > 0.0:
        # para asegurar no procesar más fotogramas de los que tiene el video
        time_fps = min(int(time_user * fps), frame_count)
    else:
        time_fps = frame_count

    print(f"\n--- Procesando  : {video_path}")
    print(f"--- FPS         : {fps}")
    print(f"--- Total Frames: {frame_count}")
    print(f"--- Duración    : {duration_seconds:.2f} segundos")
    print(f"--- Se procesarán los primeros {time_user} segundos, equivalente a {time_fps} frames.")
    t_start = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        frame_number += 1

        if not success:
            break

        # stream=True para procesar vídeos largos o grandes conjuntos de datos y gestionar la memoria de forma eficiente.
        # Cuando stream=False los resultados de todos los fotogramas o puntos de datos se almacenan en memoria, lo que puede dar lugar a errores de memoria en el caso de entradas de gran tamaño.
        # Por el contrario, stream=True utiliza un generador, que sólo mantiene en memoria los resultados del fotograma o punto de datos actual, lo que reduce significativamente el consumo de memoria y evita problemas de falta de memoria.
        results = model.predict(
            source=frame,
            stream=True,
            conf=0.3,
            # imgsz=(a,b)
            show=False,
            save=False,
            verbose=False
        )

        # TODO uso de tracker?
        # model.track(
        #    source="video.mp4",
        #    tracker="/ruta/a/mi/archivo_personalizado_bytetrack.yaml"
        # )

        # Los resultados son un generador, iteramos sobre él (normalmente 1 item por fotograma)
        for result in results:
            # Aquí 'result' contiene las detecciones para este fotograma
            # Puedes acceder a las cajas, clases, confianzas, etc.
            # result.boxes, result.masks, result.keypoints, etc.
            print(f"\n--- result:\n{result}")
            print(f"\n--- result.boxes:\n{result.boxes}")
            print(f"\n--- result.masks:\n{result.masks}")
            print(f"\n--- result.keypoints:\n{result.keypoints}")

            # --- Ejemplo: Acceder a las detecciones de balones y jugadores ---
            detections = []
            detections.append(f"frame_number: {frame_number}")

            if result.boxes is not None:
                for box in result.boxes:
                    box.is_track = True
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    bbox = box.xyxy[0].cpu().numpy().astype(int).tolist()  # [x1, y1, x2, y2]
                    # print(f"\n--- bbox:\n{bbox}")
                    class_name = model.names[class_id]
                    # print(f"\n--- model.names[class_id]:\n{class_name}")

                    # Guarda solo las detecciones que te interesan para los eventos
                    # Asegúrate de que tu modelo YOLO detecte "ball" y "person"
                    if class_name in ["ball", "person"]:
                        detections.append({
                            "class": class_name,
                            "confidence": confidence,
                            "bbox": bbox
                        })

            # print(f"\n--- detections:\n{detections}")
            # TODO verificar si crea automaticamente los directorios
            ut.write_list_in_txt(
                lista=detections,
                file=os.path.join(YOLO_PATH_RESULTS, f"{FILE_DETECTION_SIMPLE}{name_unique}{EXT_TXT}"),
            )

            # --- Aquí puedes empezar a añadir la lógica para detectar eventos ---
            # Tienes:
            # - El número de fotograma actual: frame_number
            # - El tiempo actual en el video: current_time_in_video = frame_number / fps
            # - Las detecciones de objetos en este fotograma: detections (lista de diccionarios)

            # Lógica de eventos (EJEMPLO CONCEPTUAL - ¡La parte difícil!)
            # if "ball" in [d["class"] for d in detections] and algún_jugador_esta_cerca_de_porteria(detecciones, frame):
            #     print(f"Posible evento de gol en el tiempo: {current_time_in_video:.2f}s (Fotograma: {frame_number})")

            # result.save(filename=os.path.join(PRE_RUTA, f"result_frame_{frame_number}.jpg"))

            # --- Opcional: Mostrar el fotograma con detecciones ---
            annotated_frame = result.plot()  # Dibuja las detecciones
            cv2.imshow("Detección en Video", annotated_frame)

        # frame_number += 1

        # para no procesar el video completo
        if frame_number >= time_fps:
            print(f"--- Tiempo alcanzado, frames procesados: {frame_number}")
            break

        # Detener con 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            logger.info("--- Detención manual solicitada.")
            break

    ut.get_time_employed(t_start=t_start, message=f"procesando {frame_number} frames de '{video_path}'")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    config_log()
    ut.verify_system()
    main()
