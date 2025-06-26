import logging

import torch
import torchvision.transforms.v2 as T_v2

logger = logging.getLogger(__name__)


def get_transforms_cnn3d_rgb(target_size_dataset: tuple):
    train_transforms = T_v2.Compose([
        T_v2.RandomHorizontalFlip(p=0.5),  # volteo horizontal aleatorio
        T_v2.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.05,
        ),
        T_v2.RandomRotation(
            degrees=5,  # grados
            expand=False,  # evita cambios en el tamaño del frame.
        ),
        T_v2.RandomResizedCrop(
            size=target_size_dataset,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            antialias=True,
        ),
        # Convertir a float y escalar a [0,1] antes de normalizar
        T_v2.ToDtype(
            torch.float32,
            scale=True,
        ),
    ])

    val_transforms = T_v2.Compose([
        T_v2.Resize(
            size=target_size_dataset,
            antialias=True,
        ),
        # Convertir a float y escalar a [0,1] antes de normalizar
        T_v2.ToDtype(
            torch.float32,
            scale=True,
        ),
    ])

    return train_transforms, val_transforms, 3


def get_transforms_cnn3d_grayscale(target_size_dataset: tuple):
    train_transforms = T_v2.Compose([
        T_v2.Grayscale(num_output_channels=1),
        T_v2.RandomHorizontalFlip(p=0.5),  # volteo horizontal aleatorio
        T_v2.RandomRotation(
            degrees=5,  # grados
            expand=False,  # evita cambios en el tamaño del frame.
        ),
        T_v2.RandomResizedCrop(
            size=target_size_dataset,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            antialias=True,
        ),
        # Convertir a float y escalar a [0,1] antes de normalizar
        T_v2.ToDtype(
            torch.float32,
            scale=True,
        ),
    ])

    val_transforms = T_v2.Compose([
        T_v2.Grayscale(num_output_channels=1),
        T_v2.Resize(
            size=target_size_dataset,
            antialias=True,
        ),
        # Convertir a float y escalar a [0,1] antes de normalizar
        T_v2.ToDtype(
            torch.float32,
            scale=True,
        ),
    ])

    return train_transforms, val_transforms, 1


def get_transforms_resnet(weights):
    # pipeline de transformaciones para ENTRENAMIENTO
    train_transforms = T_v2.Compose([
        # Los modelos de torchvision esperan uint8 en el rango [0, 255] al inicio
        T_v2.ToDtype(
            torch.uint8,
            scale=False,
        ),
        T_v2.RandomHorizontalFlip(p=0.5),
        T_v2.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.05,
        ),
        T_v2.RandomRotation(
            degrees=5,  # grados
            expand=False,  # evita cambios en el tamaño del frame.
        ),
        # Redimensionar el lado más corto a 128
        T_v2.Resize(
            size=(128),
            antialias=True,
        ),
        # En lugar de CenterCrop, para entrenamiento se suele usar RandomCrop para que el modelo vea diferentes partes de la imagen.
        T_v2.RandomCrop(size=(112, 112)),
        # Convertir a float y escalar a [0,1] antes de normalizar
        T_v2.ToDtype(
            torch.float32,
            scale=True,
        ),
        # Normalizar con la media y std específicas del modelo
        T_v2.Normalize(
            mean=weights.transforms().mean,
            std=weights.transforms().std,
        ),
    ])

    # pipeline de transformaciones para VALIDACIÓN, solo contiene el pre-procesamiento determinista.
    val_transforms = T_v2.Compose([
        # Los modelos de torchvision esperan uint8 en el rango [0, 255] al inicio
        T_v2.ToDtype(
            torch.uint8,
            scale=False,
        ),
        # Redimensionar el lado más corto a 128 (ejemplo de torchvision)
        T_v2.Resize(
            size=(128),
            antialias=True,
        ),
        # Recortar el centro a 112x112
        T_v2.CenterCrop(size=(112, 112)),
        # Convertir a float y escalar a [0,1] antes de normalizar
        T_v2.ToDtype(
            torch.float32,
            scale=True,
        ),
        # Normalizar con la media y std específicas del modelo
        T_v2.Normalize(
            mean=weights.transforms().mean,
            std=weights.transforms().std,
        ),
    ])

    return train_transforms, val_transforms
