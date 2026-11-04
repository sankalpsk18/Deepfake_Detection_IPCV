"""Image preprocessing utilities for deepfake detection."""

import cv2
import numpy as np
import os
import re
from PIL import Image
from io import BytesIO
from typing import Tuple, Optional, Union
import logging

from app.config import settings

logger = logging.getLogger(__name__)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal and injection attacks.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename safe for use
    """
    # Remove path components
    filename = os.path.basename(filename)
    # Remove dangerous characters
    filename = re.sub(r'[^\w\s\-\.]', '', filename)
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255 - len(ext)] + ext
    # Ensure not empty
    if not filename or filename in ('.', '..'):
        filename = "upload"
    return filename


def validate_image_file(file_bytes: bytes, filename: str) -> Tuple[bool, Optional[str]]:
    """
    Validate uploaded image file.

    Args:
        file_bytes: Raw file bytes
        filename: Original filename

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check file size
    if len(file_bytes) > settings.MAX_FILE_SIZE:
        return False, f"File size exceeds maximum allowed ({settings.MAX_FILE_SIZE / 1024 / 1024:.0f}MB)"

    # Check extension
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if f".{ext}" not in settings.ALLOWED_EXTENSIONS:
        return False, f"Unsupported file format. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"

    # Try to decode as image
    try:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return False, "Invalid or corrupted image file"
        if img.size == 0:
            return False, "Empty image"
    except Exception as e:
        return False, f"Failed to decode image: {str(e)}"

    return True, None



def detect_and_crop_face(
    image: np.ndarray,
    confidence_threshold: float = 0.5,
    padding: float = 0.3
) -> Tuple[np.ndarray, bool, Optional[Tuple[int, int, int, int]]]:

    """
    Detect faces using MediaPipe short-range and full-range models.
    Returns cropped face, detection status and bounding box.
    """

    if image is None or image.size == 0:
        raise ValueError("Input image is empty")

    h, w = image.shape[:2]

    try:
        import mediapipe as mp

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        detections = []

        # Try both short-range and full-range detection
        for model_selection in (0, 1):
            with mp.solutions.face_detection.FaceDetection(
                model_selection=model_selection,
                min_detection_confidence=confidence_threshold
            ) as face_detector:

                results = face_detector.process(rgb_image)

                if results.detections:
                    detections.extend(results.detections)

        if detections:

            best_detection = max(
                detections,
                key=lambda d: float(d.score[0])
            )

            bbox = best_detection.location_data.relative_bounding_box

            # Convert relative coordinates to image coordinates
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            box_w = int(bbox.width * w)
            box_h = int(bbox.height * h)

            # Add padding
            pad_x = int(box_w * padding)
            pad_y = int(box_h * padding)

            # Clamp coordinates to image boundaries
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + box_w + pad_x)
            y2 = min(h, y + box_h + pad_y)

            # Validate bounding box
            if x2 > x1 and y2 > y1:

                cropped = image[y1:y2, x1:x2]

                if cropped.size > 0:
                    logger.info(
                        "Face detected",
                        extra={
                            "confidence": float(best_detection.score[0]),
                            "bbox": (x1, y1, x2 - x1, y2 - y1)
                        }
                    )

                    return cropped, True, (
                        x1, y1, x2 - x1, y2 - y1
                    )

    except ImportError:
        logger.exception("MediaPipe is not installed")

    except Exception as e:
        logger.exception("Face detection failed: %s", e)

    logger.warning("No face detected. Returning original image.")

    return image, False, None



def normalize_image(
    image: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    input_range: Tuple[float, float] = (0, 255)
) -> np.ndarray:
    """
    Normalize image to ImageNet statistics.

    Args:
        image: Input image (H, W, 3) in BGR or RGB, range 0-255
        mean: Normalization mean (RGB order)
        std: Normalization std (RGB order)
        input_range: Expected input range

    Returns:
        Normalized image (H, W, 3) float32
    """
    # Convert to float32 and scale to [0, 1]
    image = image.astype(np.float32)

    if input_range[1] > 1.0:
        image = image / 255.0

    # Ensure RGB (OpenCV uses BGR)
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    image = (image - mean) / std

    return image


def resize_with_padding(
    image: np.ndarray,
    target_size: Tuple[int, int],
    padding_color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    Resize image maintaining aspect ratio with padding.

    Args:
        image: Input image (H, W, 3)
        target_size: Target (height, width)
        padding_color: Color for padding (BGR)

    Returns:
        Resized and padded image
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size

    # Calculate scale to fit within target
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create padded canvas
    canvas = np.full((target_h, target_w, 3), padding_color, dtype=image.dtype)

    # Center the resized image
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas


def preprocess_for_efficientnet(
    image: np.ndarray,
    *,
    pre_cropped: Optional[np.ndarray] = None,
    face_detected: Optional[bool] = None
) -> Tuple[np.ndarray, bool]:
    """
    Preprocess image for EfficientNet-B0 (224x224, ImageNet norm).

    Args:
        image: Input image (H, W, 3) BGR (used if pre_cropped not provided)
        pre_cropped: Optional pre-cropped face image to use instead of running face detection
        face_detected: Optional pre-computed face detection result

    Returns:
        Tuple of (preprocessed_tensor, face_detected)
    """
    if pre_cropped is not None:
        cropped = pre_cropped
        detected = face_detected if face_detected is not None else False
    else:
        cropped, detected, _ = detect_and_crop_face(
            image,
            confidence_threshold=settings.FACE_DETECTION_CONFIDENCE,
            padding=settings.FACE_CROP_PADDING
        )

    resized = resize_with_padding(cropped, (settings.MODEL_INPUT_SIZE, settings.MODEL_INPUT_SIZE))
    normalized = normalize_image(resized)
    tensor = np.transpose(normalized, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0).astype(np.float32)

    return tensor, detected


def preprocess_for_mesonet(
    image: np.ndarray,
    *,
    pre_cropped: Optional[np.ndarray] = None,
    face_detected: Optional[bool] = None
) -> Tuple[np.ndarray, bool]:
    """
    Preprocess image for MesoNet (256x256, [0,1] range).

    Args:
        image: Input image (H, W, 3) BGR (used if pre_cropped not provided)
        pre_cropped: Optional pre-cropped face image to use instead of running face detection
        face_detected: Optional pre-computed face detection result

    Returns:
        Tuple of (preprocessed_tensor, face_detected)
    """
    if pre_cropped is not None:
        cropped = pre_cropped
        detected = face_detected if face_detected is not None else False
    else:
        cropped, detected, _ = detect_and_crop_face(
            image,
            confidence_threshold=settings.FACE_DETECTION_CONFIDENCE,
            padding=settings.FACE_CROP_PADDING
        )

    resized = resize_with_padding(cropped, (settings.MESONET_INPUT_SIZE, settings.MESONET_INPUT_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(normalized, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0).astype(np.float32)

    return tensor, detected


def preprocess_image(
    file_bytes: bytes,
    target_model: str = "efficientnet"
) -> Tuple[np.ndarray, bool]:
    """
    Full preprocessing pipeline from raw bytes to model input.

    Args:
        file_bytes: Raw image bytes
        target_model: "efficientnet" or "mesonet"

    Returns:
        Tuple of (preprocessed_tensor, face_detected)
    """
    # Decode image
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Failed to decode image")

    if target_model == "efficientnet":
        tensor, face_detected = preprocess_for_efficientnet(image)
    elif target_model == "mesonet":
        tensor, face_detected = preprocess_for_mesonet(image)
    else:
        raise ValueError(f"Unknown model: {target_model}")

    return tensor, face_detected


def load_image_from_bytes(file_bytes: bytes) -> np.ndarray:
    """Load image from bytes to numpy array (BGR)."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image")
    return image


def image_to_bytes(image: np.ndarray, format: str = ".jpg", quality: int = 90) -> bytes:
    """Convert numpy image to bytes."""
    if format.lower() in [".jpg", ".jpeg"]:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    elif format.lower() == ".png":
        encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
    else:
        encode_param = []

    success, encoded = cv2.imencode(format, image, encode_param)
    if not success:
        raise ValueError("Failed to encode image")
    return encoded.tobytes()