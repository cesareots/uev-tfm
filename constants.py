import os

import dotenv

dotenv.load_dotenv()

# temporales, son de prueba
FILE_DETECTION_SIMPLE = "deteccion_por_frame_"
DS_SOCCERNET_V = os.getenv("DS_SOCCERNET_V")
DS_SOCCERNET_L_V2 = os.getenv("DS_SOCCERNET_L_V2")

#
LOG_DIR = "log/"
LOG_MAIN = f"{LOG_DIR}main.log"
LOG_YOLO_EXPORT = f"{LOG_DIR}yolo_export.log"
LOG_SOCCERNET = f"{LOG_DIR}soccernet.log"

#
EXT_TXT = ".txt"
EXT_JSON = ".json"

# debug
DEBUG = int(os.getenv("DEBUG"))

# soccernet
SOCCERNET_PASSWORD = os.getenv("SOCCERNET_PASSWORD")

# ultralytics - yolo
YOLO_MODEL = os.getenv("YOLO_MODEL")
YOLO_MODEL_NAMES = "model_names_"
YOLO_PATH_RESULTS = "results"

# path
DS_SOCCERNET = os.getenv("DS_SOCCERNET")
DS_SOCCERNET_FINAL = os.getenv("DS_SOCCERNET_FINAL")
