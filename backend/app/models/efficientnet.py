"""EfficientNet-B0 detector for deepfake detection."""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional

from app.config import settings
from app.models.base import BaseONNXDetector
from app.utils.preprocessing import preprocess_for_efficientnet

logger = logging.getLogger(__name__)


class EfficientNetDetector(BaseONNXDetector):
    """
    EfficientNet-B0 based deepfake detector using ONNX Runtime.

    Expected model: EfficientNet-B0 fine-tuned on FaceForensics++
    Input: (1, 3, 224, 224) float32, ImageNet normalized
    Output: (1, 2) logits or probabilities [real, fake]
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        providers: Optional[list[str]] = None
    ):
        """
        Initialize detector.

        Args:
            model_path: Path to ONNX model file
            device: "cpu" or "cuda"
            providers: ONNX Runtime execution providers
        """
        super().__init__(
            model_path=model_path or settings.EFFICIENTNET_MODEL_PATH,
            candidates=[
                "efficientnet_b0_deepfake.onnx",
                "onnxmodel.onnx",
                "model.onnx",
            ],
            device=device,
            providers=providers
        )

    @staticmethod
    def _fallback_probabilities(image: np.ndarray) -> np.ndarray:
        """Provide a stable, non-random fallback score when model weights are absent."""
        if image.size == 0:
            return np.array([0.5, 0.5], dtype=np.float32)

        arr = np.asarray(image)
        if arr.ndim == 4:
            arr = arr[0]

        if arr.ndim == 3:
            arr = np.transpose(arr, (1, 2, 0))

        rgb = np.clip(arr, -3.0, 3.0)
        gray = np.mean(rgb, axis=-1)
        lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
        sharpness = float(np.var(lap))
        contrast = float(np.std(gray))
        mean_intensity = float(np.mean(gray))
        face_score = 1.0 if np.abs(mean_intensity) < 1.5 else 0.5

        # Higher sharpness/contrast and moderate intensity usually suggest authentic imagery.
        fake_score = 0.35
        fake_score += min(0.25, max(0.0, (25.0 - sharpness) / 80.0))
        fake_score += min(0.20, max(0.0, (0.9 - contrast) / 2.0))
        fake_score += 0.20 if face_score < 0.8 else 0.0

        fake_score = float(np.clip(fake_score, 0.05, 0.9))
        real_score = 1.0 - fake_score
        return np.array([real_score, fake_score], dtype=np.float32)

    def _process_output(self, output: np.ndarray) -> np.ndarray:
        """Process model output to probabilities."""
        # Apply softmax to get probabilities.
        # Some exported binary classifiers return logits in fake-first order,
        # so normalize them to the app's canonical [real, fake] layout.
        probs = self._normalize_binary_probabilities(self._softmax(output))
        return probs

    def predict_from_bytes(self, file_bytes: bytes) -> Tuple[np.ndarray, float, bool]:
        """
        Full pipeline: preprocess bytes and predict.

        Args:
            file_bytes: Raw image bytes

        Returns:
            Tuple of (probabilities, inference_time_ms, face_detected)
        """
        return super().predict_from_bytes(file_bytes, preprocess_for_efficientnet)

    def _get_dummy_input(self) -> np.ndarray:
        """Get dummy input for warmup."""
        return np.random.randn(1, 3, 224, 224).astype(np.float32)


def create_efficientnet_detector() -> EfficientNetDetector:
    """Factory function to create EfficientNet detector with settings."""
    return EfficientNetDetector(
        device=settings.INFERENCE_DEVICE,
        providers=settings.ONNX_PROVIDERS
    )