#!/usr/bin/env python3
"""
Degradation-Aware Augmentation Pipeline (Task 2)

Implements stochastic degradation augmentations for robust deepfake detection
training. Applied with per-transform probability to simulate real-world
image degradations (compression, resolution loss, blur, noise, etc.).
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple


# Model-specific normalization parameters
MODEL_NORMALIZATION = {
    'efficientnet': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
    },
    'mesonet': {
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
    },
}


def get_degradation_augmentations(
    prob: float = 0.5,
    jpeg_quality_range: Tuple[int, int] = (30, 95),
    downscale_range: Tuple[float, float] = (0.25, 1.0),
    blur_sigma_range: Tuple[float, float] = (0.1, 2.0),
    noise_std_range: Tuple[int, int] = (2, 15),
    brightness_contrast_limit: float = 0.2,
    hue_sat_val_limits: Tuple[int, int, int] = (10, 20, 10),
    rotate_limit: int = 5,
) -> A.Compose:
    """
    Create stochastic degradation augmentation pipeline.

    Each transform is applied independently with probability `prob`.
    This preserves some clean samples while exposing the model to
    diverse real-world degradations.

    Args:
        prob: Per-transform application probability
        jpeg_quality_range: JPEG compression quality range
        downscale_range: Resolution downscale factor range
        blur_sigma_range: Gaussian/motion blur sigma range
        noise_std_range: Gaussian noise std range (0-255 scale)
        brightness_contrast_limit: Brightness/contrast jitter limit
        hue_sat_val_limits: HSV jitter limits (hue, sat, val)
        rotate_limit: Rotation limit in degrees

    Returns:
        Albumentations Compose with degradation transforms
    """
    return A.Compose([
        # Compression artifacts - simulates social media re-compression
        A.ImageCompression(
            quality_lower=jpeg_quality_range[0],
            quality_upper=jpeg_quality_range[1],
            p=prob
        ),

        # Resolution degradation - simulates low-res source footage
        A.OneOf([
            A.Downscale(
                scale_min=downscale_range[0],
                scale_max=downscale_range[1],
                interpolation=cv2.INTER_AREA,
                p=1.0
            ),
            A.NoOp(p=1.0),
        ], p=prob),

        # Blur - simulates camera/motion blur
        A.OneOf([
            A.GaussianBlur(
                blur_limit=(3, 7),
                sigma_limit=blur_sigma_range,
                p=1.0
            ),
            A.MotionBlur(
                blur_limit=(3, 7),
                p=1.0
            ),
            A.NoOp(p=1.0),
        ], p=prob),

        # Noise - simulates sensor noise, compression artifacts
        A.OneOf([
            A.GaussNoise(
                var_limit=(noise_std_range[0]**2, noise_std_range[1]**2),
                p=1.0
            ),
            A.ISONoise(
                color_shift=(0.01, 0.05),
                intensity=(0.1, 0.5),
                p=1.0
            ),
            A.NoOp(p=1.0),
        ], p=prob),

        # Color/lighting jitter - simulates varying lighting conditions
        A.RandomBrightnessContrast(
            brightness_limit=brightness_contrast_limit,
            contrast_limit=brightness_contrast_limit,
            p=prob
        ),
        A.HueSaturationValue(
            hue_shift_limit=hue_sat_val_limits[0],
            sat_shift_limit=hue_sat_val_limits[1],
            val_shift_limit=hue_sat_val_limits[2],
            p=prob
        ),

        # Mild geometric - keep conservative for face alignment
        A.Rotate(
            limit=rotate_limit,
            border_mode=cv2.BORDER_CONSTANT,
            p=prob
        ),
        A.HorizontalFlip(p=0.5),
    ], p=1.0)


def get_training_transforms(
    model_name: str,
    input_size: int,
    degradation_prob: float = 0.5,
    degradation_config: Optional[Dict] = None
) -> A.Compose:
    """
    Get complete training transforms combining degradations + model-specific normalization.

    Degradations are applied FIRST (on [0,255] image), then resize + normalization.

    Args:
        model_name: 'efficientnet' or 'mesonet'
        input_size: Model input size (224 or 256)
        degradation_prob: Probability for each degradation transform
        degradation_config: Optional dict with custom degradation parameters

    Returns:
        Albumentations Compose pipeline
    """
    if model_name not in MODEL_NORMALIZATION:
        raise ValueError(f"Unknown model: {model_name}")

    norm_params = MODEL_NORMALIZATION[model_name]

    # Use config if provided, otherwise defaults
    if degradation_config is None:
        degradation_config = {}

    degradation = get_degradation_augmentations(
        prob=degradation_prob,
        jpeg_quality_range=degradation_config.get('jpeg_quality_range', (30, 95)),
        downscale_range=degradation_config.get('downscale_range', (0.25, 1.0)),
        blur_sigma_range=degradation_config.get('blur_sigma_range', (0.1, 2.0)),
        noise_std_range=degradation_config.get('noise_std_range', (2, 15)),
        brightness_contrast_limit=degradation_config.get('brightness_contrast_limit', 0.2),
        hue_sat_val_limits=degradation_config.get('hue_sat_val_limits', (10, 20, 10)),
        rotate_limit=degradation_config.get('rotate_limit', 5),
    )

    # Combine: degradations -> resize -> normalize -> tensor
    return A.Compose([
        *degradation.transforms,  # Apply degradations on raw image
        A.Resize(input_size, input_size),
        A.Normalize(mean=norm_params['mean'], std=norm_params['std']),
        ToTensorV2(),
    ])


def get_albumentations_transforms(
    model_name: str,
    input_size: int,
    is_train: bool = True
) -> A.Compose:
    """
    Get standard Albumentations transforms (without degradations).

    Used for validation/test or when degradations are disabled.
    """
    if model_name not in MODEL_NORMALIZATION:
        raise ValueError(f"Unknown model: {model_name}")

    norm_params = MODEL_NORMALIZATION[model_name]

    if is_train:
        return A.Compose([
            A.Resize(input_size, input_size),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=5, p=0.3
            ),
            A.Normalize(mean=norm_params['mean'], std=norm_params['std']),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(input_size, input_size),
            A.Normalize(mean=norm_params['mean'], std=norm_params['std']),
            ToTensorV2(),
        ])


def create_fixed_severity_augmentation(
    severity: str,
    model_name: str,
    input_size: int
) -> A.Compose:
    """
    Create fixed-severity augmentation for robustness evaluation.

    Args:
        severity: 'mild', 'moderate', or 'severe'
        model_name: 'efficientnet' or 'mesonet'
        input_size: Model input size

    Returns:
        Albumentations Compose with fixed degradation parameters
    """
    severity_configs = {
        'mild': {
            'jpeg_quality': 85,
            'downscale': 0.8,
            'blur_sigma': 1.0,
            'noise_std': 5,
            'brightness_contrast': 0.1,
        },
        'moderate': {
            'jpeg_quality': 50,
            'downscale': 0.5,
            'blur_sigma': 2.0,
            'noise_std': 10,
            'brightness_contrast': 0.2,
        },
        'severe': {
            'jpeg_quality': 30,
            'downscale': 0.25,
            'blur_sigma': 3.0,
            'noise_std': 15,
            'brightness_contrast': 0.3,
        },
    }

    if severity not in severity_configs:
        raise ValueError(f"Unknown severity: {severity}. Choose from: mild, moderate, severe")

    cfg = severity_configs[severity]
    norm_params = MODEL_NORMALIZATION[model_name]

    # Build fixed transforms
    transforms_list = [
        # Fixed JPEG compression
        A.ImageCompression(
            quality_lower=cfg['jpeg_quality'],
            quality_upper=cfg['jpeg_quality'],
            p=1.0
        ),
        # Fixed downscale + upscale
        A.Downscale(
            scale_min=cfg['downscale'],
            scale_max=cfg['downscale'],
            interpolation=cv2.INTER_AREA,
            p=1.0
        ),
        # Fixed blur
        A.GaussianBlur(
            blur_limit=(3, 7),
            sigma_limit=(cfg['blur_sigma'], cfg['blur_sigma']),
            p=1.0
        ),
        # Fixed noise
        A.GaussNoise(
            var_limit=(cfg['noise_std']**2, cfg['noise_std']**2),
            p=1.0
        ),
        # Fixed color jitter
        A.RandomBrightnessContrast(
            brightness_limit=cfg['brightness_contrast'],
            contrast_limit=cfg['brightness_contrast'],
            p=1.0
        ),
        # Resize and normalize
        A.Resize(input_size, input_size),
        A.Normalize(mean=norm_params['mean'], std=norm_params['std']),
        ToTensorV2(),
    ]

    return A.Compose(transforms_list)


def apply_degradation_to_image(
    image: np.ndarray,
    severity: str = 'moderate'
) -> np.ndarray:
    """
    Apply fixed degradation to a single image (numpy array).

    Useful for creating degraded test sets.

    Args:
        image: Input image as numpy array (H, W, 3) in RGB, range [0, 255]
        severity: 'mild', 'moderate', or 'severe'

    Returns:
        Degraded image as numpy array (H, W, 3) in RGB, range [0, 255]
    """
    severity_configs = {
        'mild': {'quality': 85, 'downscale': 0.8, 'blur': 1.0, 'noise': 5},
        'moderate': {'quality': 50, 'downscale': 0.5, 'blur': 2.0, 'noise': 10},
        'severe': {'quality': 30, 'downscale': 0.25, 'blur': 3.0, 'noise': 15},
    }

    cfg = severity_configs[severity]
    img = image.copy().astype(np.uint8)

    # JPEG compression
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), cfg['quality']]
    _, encimg = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR), encode_param)
    img = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Downscale + upscale
    h, w = img.shape[:2]
    new_h, new_w = int(h * cfg['downscale']), int(w * cfg['downscale'])
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    # Gaussian blur
    ksize = max(3, int(cfg['blur'] * 2) | 1)  # Ensure odd
    img = cv2.GaussianBlur(img, (ksize, ksize), cfg['blur'])

    # Gaussian noise
    noise = np.random.normal(0, cfg['noise'], img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img


def create_degraded_test_set(
    dataset,
    severity_levels: List[str] = ['mild', 'moderate', 'severe'],
    model_name: str = 'efficientnet',
    input_size: int = 224
) -> Dict[str, any]:
    """
    Create multiple degraded versions of a test dataset for robustness evaluation.

    Args:
        dataset: Dataset object with samples (path, label)
        severity_levels: List of severity levels to generate
        model_name: Model name for normalization
        input_size: Model input size

    Returns:
        Dict mapping severity -> (transform, degraded_samples)
    """
    degraded_sets = {}

    for severity in severity_levels:
        transform = create_fixed_severity_augmentation(severity, model_name, input_size)
        degraded_sets[severity] = {
            'transform': transform,
            'description': f'{severity.capitalize()} degradation'
        }

    return degraded_sets


def get_default_degradation_config() -> Dict:
    """Get default degradation configuration from dataset_config.yaml schema."""
    return {
        'enabled': True,
        'probability': 0.5,
        'jpeg_quality_range': [30, 95],
        'downscale_range': [0.25, 1.0],
        'blur_sigma_range': [0.1, 2.0],
        'noise_std_range': [2, 15],
        'brightness_contrast_limit': 0.2,
        'hue_sat_val_limits': [10, 20, 10],
        'rotate_limit': 5,
    }


if __name__ == '__main__':
    # Test the augmentation pipeline
    import torch

    # Test with a dummy image
    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    # Test degradation augmentations
    degradation = get_degradation_augmentations(prob=1.0)
    augmented = degradation(image=dummy_img)
    print(f"Degraded image shape: {augmented['image'].shape}")
    print(f"Degraded image range: [{augmented['image'].min()}, {augmented['image'].max()}]")

    # Test full training transforms
    train_transform = get_training_transforms('efficientnet', 224, degradation_prob=0.5)
    augmented = train_transform(image=dummy_img)
    print(f"Training transform output shape: {augmented['image'].shape}")
    print(f"Training transform output type: {type(augmented['image'])}")

    # Test fixed severity
    for sev in ['mild', 'moderate', 'severe']:
        fixed = create_fixed_severity_augmentation(sev, 'efficientnet', 224)
        aug = fixed(image=dummy_img)
        print(f"{sev.capitalize()} - shape: {aug['image'].shape}")

    print("\nAll tests passed!")