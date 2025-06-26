import logging
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

from src.models.main_cnn3d import SimpleCNN3D, TARGET_SIZE_DATASET
from src.models.transforms import get_transforms_cnn3d_grayscale, get_transforms_resnet
from src.preprocess.dataset_soccernet import get_output_size_from_transforms

logger = logging.getLogger(__name__)


def load_model_and_transforms(
        checkpoint_path: Path,
        model_type: str,
        num_classes: int,
        device: torch.device,
        frames_per_clip: int,
):
    try:
        logger.info(f"Cargando checkpoint desde '{checkpoint_path}'")
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            # weights_only=False,
        )
        transforms = None

        if model_type == "CNN3D":
            logger.info(f"Instanciando modelo {model_type}.")
            train_transforms, val_transforms, input_channels = get_transforms_cnn3d_grayscale(TARGET_SIZE_DATASET)
            transforms = val_transforms
            expected_size = get_output_size_from_transforms(transforms)
            model = SimpleCNN3D(
                num_classes=num_classes,
                input_channels=input_channels,
                input_frames=frames_per_clip,
                input_size=expected_size,
            )
        elif model_type == "RESNET":
            logger.info(f"Instanciando modelo {model_type}.")
            weights = R2Plus1D_18_Weights.KINETICS400_V1
            model = r2plus1d_18(weights=weights)
            num_original_features = model.fc.in_features
            model.fc = nn.Linear(num_original_features, num_classes)
            train_transforms, transforms = get_transforms_resnet(weights)
        else:
            log_con = f"Tipo de modelo '{model_type}' no soportado."
            logger.error(log_con)
            raise ValueError(log_con)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        return model, transforms
    except RuntimeError as e:
        logger.error(f"En RuntimeError: {str(e)}")
    except Exception as e:
        logger.error(f"En Exception: {str(e)}")
