import os

import dotenv

dotenv.load_dotenv()

#
RESULTS = "results/"
MODELS_DIR = "models/"
M_BASIC = f"{MODELS_DIR}CNN3D/"
M_RESNET = f"{MODELS_DIR}RESNET/"

# soccernet
SOCCERNET_PASSWORD = os.getenv("SOCCERNET_PASSWORD")
SOCCERNET_LABELS = {
    "Goal": 0,
    "Yellow card": 1,
    "Substitution": 2,
    "Corner": 3,
}
PARTIDOS_INDICE_LOTE = [
    [0, 49],
    [50, 99],
    [100, 149],
    [150, 199],
    [200, 249],
    [250, 299],
    [300, 349],
    [350, 399],
    [400, 416],  # 417 para inferencia, england_epl
    [418, 435],  # 436 para inferencia, europe_uefa-champions-league
    [437, 437],  # 438 para inferencia, france_ligue-1
    [439, 449],
    [450, 451],  # 452 para inferencia, germany_bundesliga
    [453, 473],  # 474 para inferencia, italy_serie-a
    [475, 498],  # 499 para inferencia, spain_laliga
]
SOCCERNET_RESULTS = f"{RESULTS}soccernet"
DS_SOCCERNET_RAW = os.getenv("DS_SOCCERNET_RAW")
DS_SOCCERNET_ACTIONS = os.getenv("DS_SOCCERNET_ACTIONS")
DS_SOCCERNET_TENSORS = os.getenv("DS_SOCCERNET_TENSORS")

# temporales, son de prueba
FILE_DETECTION_SIMPLE = "deteccion_por_frame_"
DS_SOCCERNET_V = os.getenv("DS_SOCCERNET_V")
DS_SOCCERNET_L_V2 = os.getenv("DS_SOCCERNET_L_V2")

#
SEMILLA = 34

# logs
LOG_DIR = "log/"
LOG_MAIN = f"{LOG_DIR}main.log"
LOG_YOLO_EXPORT = f"{LOG_DIR}yolo_export.log"
LOG_SOCCERNET = f"{LOG_DIR}soccernet.log"
LOG_SOCCERNET_CLIPS = f"{LOG_DIR}soccernet_clips.log"
LOG_SOCCERNET_TENSORS = f"{LOG_DIR}soccernet_tensors.log"
LOG_MODEL_CNN3D = f"{LOG_DIR}model_cnn3d.log"
LOG_MODEL_RESNET = f"{LOG_DIR}model_resnet.log"

#
EXT_TXT = ".txt"
EXT_JSON = ".json"

# debug
DEBUG = 1

# ultralytics - yolo
YOLO_MODEL = os.getenv("YOLO_MODEL")
YOLO_MODEL_NAMES = "model_names_"
YOLO_PATH_RESULTS = f"{RESULTS}yolo"
