import logging
import sys
import time

from ultralytics import YOLO

import utils as ut
from constants import *

logger = logging.getLogger(__name__)


def config_log() -> None:
    logging.basicConfig(
        filename=LOG_YOLO_EXPORT,
        filemode="a",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        # https://docs.python.org/3/library/logging.html
        level=logging.INFO,  # NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL
    )


def main(model_name: str) -> None:
    try:
        model = YOLO(model=model_name)
        model.info()
        model.export(
            format="openvino",
            # TODO
        )

        logger.info(f"Modelo YOLO '{model_name}' exportado correctamente a OpenVINO...")
    except Exception as e:
        logger.error(e)
        sys.exit(1)


def bucle_export() -> None:
    # models = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
    models = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]

    for m in models:
        main(model_name=m)


def inference() -> None:
    name = "yolo11n_openvino_model/"
    # name = "yolo11s_openvino_model/"
    # name = "yolo11m_openvino_model/"
    # name = "yolo11l_openvino_model/"
    # name = "yolo11x_openvino_model/"  # error:
    # RuntimeError: Exception from src\inference\src\cpp\infer_request.cpp:223:
    # Exception from src\plugins\intel_npu\src\utils\src\zero\zero_wrappers.cpp:182:
    # L0 zeCommandQueueExecuteCommandLists result: ZE_RESULT_ERROR_DEVICE_LOST, code 0x70000001 - device hung, reset, was removed, or driver update occurred

    # Load the exported OpenVINO model
    ov_model = YOLO(name)

    # video_path = os.path.join(PATH_DATASET, PATH_DATASET_ONE_VIDEO)
    # video_path = "z_pruebas/img_vid/futbol_1.mkv"
    video_path = "z_pruebas/img_vid/futbol_2.mkv"

    t_start = time.time()

    device = "intel:npu"
    # device = "intel:gpu"
    # device = "intel:cpu"

    results = ov_model(
        source=video_path,
        device=device,
        save=True,
        show=True,
    )

    total_frames = sum(1 for result in results)
    ut.get_time_employed(t_start=t_start,
                         message=f"infiriendo {total_frames} frames de '{video_path}' en '{name}' con '{device}'")


if __name__ == "__main__":
    ut.verify_system()
    config_log()
    # main(model_name=YOLO_MODEL)
    # bucle_export()
    inference()
