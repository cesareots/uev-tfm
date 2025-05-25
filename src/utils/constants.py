import os

import dotenv

dotenv.load_dotenv()

#
RESULTS = "results/"

# soccernet
SOCCERNET_PASSWORD = os.getenv("SOCCERNET_PASSWORD")
SOCCERNET_LABELS = {  # TODO que acciones utilizaré? (preprocesamiento)
    "Goal": 0,
    "Yellow card": 1,
    "Shots on target": 2,
}
# no tomo en cuenta el split que hace soccernet, simplemente descargo la misma cantidad de partidos por [liga-temporada]
PARTIDOS_INDICE_LOTE_1 = [
    0, 1, 2, 3,  # england_epl/2014-2015/
    4, 5, 6, 7,  # england_epl/2015-2016/
    29, 30, 31, 32,  # england_epl/2016-2017/
    58, 59, 60, 61,  # europe_uefa-champions-league/2014-2015/
    80, 81, 82, 83,  # europe_uefa-champions-league/2015-2016/
    109, 110, 111, 112,  # europe_uefa-champions-league/2016-2017/
]
PARTIDOS_INDICE_LOTE_2 = [
    120,  # france_ligue-1/2014-2015/
    121, 122, 339,  # france_ligue-1/2015-2016/
    123, 124, 125, 126,  # france_ligue-1/2016-2017/
    147, 148, 149, 150,  # germany_bundesliga/2014-2015/
    152, 153, 154, 155,  # germany_bundesliga/2015-2016/
    162, 163, 164, 165,  # germany_bundesliga/2016-2017/
    178, 179, 180, 181,  # italy_serie-a/2014-2015/
]
PARTIDOS_INDICE_LOTE_3 = [
    185, 186, 187, 188,  # italy_serie-a/2015-2016/
    190, 191, 192, 193,  # italy_serie-a/2016-2017/
    234, 235, 236, 237,  # spain_laliga/2014-2015/
    240, 241, 242, 243,  # spain_laliga/2015-2016/
    258, 259, 260, 261,  # spain_laliga/2016-2017/
]
PARTIDOS_INDICE_LOTE_4 = [
    
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

#
EXT_TXT = ".txt"
EXT_JSON = ".json"

# debug
DEBUG = 1

# ultralytics - yolo
YOLO_MODEL = os.getenv("YOLO_MODEL")
YOLO_MODEL_NAMES = "model_names_"
YOLO_PATH_RESULTS = f"{RESULTS}yolo"
