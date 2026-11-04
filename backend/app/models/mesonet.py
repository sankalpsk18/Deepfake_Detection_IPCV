"""MesoNet detector for deepfake detection."""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional

from app.config import settings
from app.models.base import BaseONNXDetector
from app.utils.preprocessing import preprocess_for_mesonet

logger = logging.getLogger(__name__)


class MesoNetDetector(BaseONNXDetector):
    """
    MesoNet based deepfake detector using ONNX Runtime.

    MesoNet is a lightweight CNN designed specifically for deepfake detection.
    Input: (1, 3, 256, 256) float32, normalized to [0, 1]
    Output: (1, 2) probabilities [real, fake] or single sigmoid output
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
        self.output_type: str = "softmax"  # "softmax" or "sigmoid"
        super().__init__(
            model_path=model_path or settings.MESONET_MODEL_PATH,
            candidates=[
                "mesonet_deepfake.onnx",
                "onnxmodel.onnx",
                "model.onnx",
            ],
            device=device,
            providers=providers
        )

    def _post_load_init(self):
        """Determine output type by checking output shape after model load."""
        if self.session is not None:
            output_shape = self.session.get_outputs()[0].shape
            if len(output_shape) == 2 and output_shape[1] == 2:
                self.output_type = "softmax"
            elif len(output_shape) == 2 and output_shape[1] == 1:
                self.output_type = "sigmoid"
            else:
                self.output_type = "softmax"  # Default assumption
            logger.info(f"Output type: {self.output_type}, Shape: {output_shape}")

    @staticmethod
    def _fallback_probabilities(image: np.ndarray) -> np.ndarray:
        """Provide a stable fallback score when the ONNX model is unavailable."""
        if image.size == 0:
            return np.array([0.5, 0.5], dtype=np.float32)

        arr = np.asarray(image)
        if arr.ndim == 4:
            arr = arr[0]

        if arr.ndim == 3:
            arr = np.transpose(arr, (1, 2, 0))

        rgb = np.clip(arr, 0.0, 1.0)
        gray = np.mean(rgb, axis=-1)
        lap = cv2.Laplacian((gray * 255).astype(np.uint8), cv2.CV_32F)
        sharpness = float(np.var(lap))
        variance = float(np.std(gray))
        fake_score = 0.3 + min(0.25, max(0.0, (40.0 - sharpness) / 90.0))
        fake_score += min(0.2, max(0.0, (0.9 - variance) / 2.2))
        fake_score = float(np.clip(fake_score, 0.05, 0.85))
        real_score = 1.0 - fake_score
        return np.array([real_score, fake_score], dtype=np.float32)

    def _process_output(self, output: np.ndarray) -> np.ndarray:
        """Process model output to probabilities."""
        if self.output_type == "sigmoid":
            # Single output: probability of fake
            fake_prob = float(output)
            real_prob = 1.0 - fake_prob
            probs = np.array([real_prob, fake_prob], dtype=np.float32)
        else:
            # Two outputs: apply softmax.
            # Some exports are ordered as [fake, real] even when documented as [real, fake].
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
        return super().predict_from_bytes(file_bytes, preprocess_for_mesonet)

    def _get_dummy_input(self) -> np.ndarray:
        """Get dummy input for warmup."""
        if self.session is None:
            return np.random.randn(1, 3, 256, 256).astype(np.float32)

        input_shape = self.session.get_inputs()[0].shape
        height = 224
        width = 224

        if isinstance(input_shape, (list, tuple)):
            for idx, value in enumerate(input_shape):
                if isinstance(value, int) and value > 0 and idx in (2, 3):
                    if idx == 2:
                        height = value
                    if idx == 3:
                        width = value

        return np.random.randn(1, 3, height, width).astype(np.float32)


def create_mesonet_detector() -> MesoNetDetector:
    """Factory function to create MesoNet detector with settings."""
    return MesoNetDetector(
        device=settings.INFERENCE_DEVICE,
        providers=settings.ONNX_PROVIDERS
    )