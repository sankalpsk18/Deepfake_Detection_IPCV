"""Ensemble detector combining EfficientNet and MesoNet."""

import time
import uuid
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

import numpy as np

from app.config import settings
from app.models.efficientnet import EfficientNetDetector
from app.models.mesonet import MesoNetDetector
from app.models.exceptions import DetectionError
from app.utils.visualization import generate_gradcam, heatmap_to_bytes, save_heatmap
from app.utils.preprocessing import load_image_from_bytes
from app.utils.logging_config import get_logger
from app.utils.metrics import record_detection

logger = get_logger(__name__)

# Also keep standard logger for compatibility
import logging
_std_logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Structured detection result."""
    prediction: str  # "real" or "fake"
    confidence: float
    probabilities: Dict[str, float]  # {"real": 0.08, "fake": 0.92}
    model_used: str
    ensemble_active: bool  # Whether multiple distinct models were used
    models_used: list[str]  # List of model names that contributed
    processing_time_ms: float
    face_detected: bool
    heatmap_bytes: Optional[bytes] = None
    request_id: Optional[str] = None


class EnsembleDetector:
    """
    Ensemble detector combining EfficientNet-B0 and MesoNet.

    Uses weighted average of predictions:
    - EfficientNet weight: 0.7 (more accurate)
    - MesoNet weight: 0.3 (faster, complementary)
    """

    def __init__(
        self,
        efficientnet: Optional[EfficientNetDetector] = None,
        mesonet: Optional[MesoNetDetector] = None,
        efficientnet_weight: float = 0.7,
        mesonet_weight: float = 0.3
    ):
        """
        Initialize ensemble detector.

        Args:
            efficientnet: EfficientNet detector instance
            mesonet: MesoNet detector instance
            efficientnet_weight: Weight for EfficientNet predictions
            mesonet_weight: Weight for MesoNet predictions
        """
        self.efficientnet = efficientnet or EfficientNetDetector()
        self.mesonet = mesonet or MesoNetDetector()
        self.efficientnet_weight = efficientnet_weight
        self.mesonet_weight = mesonet_weight

        # Validate weights
        total = efficientnet_weight + mesonet_weight
        if abs(total - 1.0) > 1e-6:
            logger.warning("Weights sum to {total}, normalizing to 1.0", total=total)
            self.efficientnet_weight = efficientnet_weight / total
            self.mesonet_weight = mesonet_weight / total

        logger.info(
            "Ensemble initialized",
            efficientnet_weight=round(self.efficientnet_weight, 2),
            mesonet_weight=round(self.mesonet_weight, 2)
        )

    def _single_model_mode(self) -> bool:
        """Return True when only one real ONNX model is available for inference."""
        same_path = self.efficientnet.model_path == self.mesonet.model_path
        return same_path and self.efficientnet.session is not None and self.mesonet.session is not None

    def predict(self, file_bytes: bytes, request_id: Optional[str] = None) -> DetectionResult:
        """
        Run ensemble detection on image bytes.

        Args:
            file_bytes: Raw image bytes
            request_id: Optional request identifier

        Returns:
            DetectionResult with prediction and metadata
        """
        start_time = time.perf_counter()
        request_id = request_id or str(uuid.uuid4())[:8]

        try:
            # Determine if ensemble is active (two different model paths with loaded sessions)
            eff_loaded = self.efficientnet.session is not None
            mes_loaded = self.mesonet.session is not None
            different_paths = self.efficientnet.model_path != self.mesonet.model_path
            ensemble_active = eff_loaded and mes_loaded and different_paths

            models_used = []
            if eff_loaded:
                models_used.append("EfficientNet-B0")
            if mes_loaded:
                models_used.append("MesoNet")

            # Load and preprocess image once with face detection
            from app.utils.preprocessing import (
                load_image_from_bytes,
                detect_and_crop_face,
                preprocess_for_efficientnet,
                preprocess_for_mesonet
            )

            image = load_image_from_bytes(file_bytes)

            # Run face detection once at ensemble level
            cropped, face_detected, _ = detect_and_crop_face(
                image,
                confidence_threshold=settings.FACE_DETECTION_CONFIDENCE,
                padding=settings.FACE_CROP_PADDING
            )

            # If only one real model is present, use it directly rather than averaging
            # the same ONNX asset twice, which can reduce calibration and confidence.
            if self._single_model_mode():
                # Preprocess cropped image for EfficientNet
                eff_tensor, _ = preprocess_for_efficientnet(image, pre_cropped=cropped, face_detected=face_detected)
                eff_probs, eff_time = self.efficientnet.predict(eff_tensor)
                ensemble_probs = eff_probs.copy()
                mes_face = False
            else:
                # Preprocess cropped image for both models
                eff_tensor, _ = preprocess_for_efficientnet(image, pre_cropped=cropped, face_detected=face_detected)
                mes_tensor, _ = preprocess_for_mesonet(image, pre_cropped=cropped, face_detected=face_detected)

                eff_probs, eff_time = self.efficientnet.predict(eff_tensor)
                mes_probs, mes_time = self.mesonet.predict(mes_tensor)

                # Weighted ensemble
                ensemble_probs = (
                    self.efficientnet_weight * eff_probs +
                    self.mesonet_weight * mes_probs
                )

            # Final prediction.
            # Canonical probability order is [real, fake]. If the model export is fake-first,
            # it should already have been normalized upstream, so this logic stays stable.
            real_prob = float(ensemble_probs[0])
            fake_prob = float(ensemble_probs[1])
            prediction = "fake" if fake_prob > real_prob else "real"
            confidence = max(fake_prob, real_prob)

            # Face detection: true if face was detected at ensemble level
            face_detected = face_detected

            # Generate Grad-CAM heatmap (using EfficientNet as primary)
            heatmap_bytes = None
            if settings.GRADCAM_ENABLED:
                try:
                    heatmap_bytes = self._generate_heatmap(file_bytes, ensemble_probs)
                except Exception as e:
                    logger.warning("Heatmap generation failed", error=str(e))

            total_time = (time.perf_counter() - start_time) * 1000

            # Record detection metrics
            model_used_str = "+".join(models_used) if models_used else "unknown"
            record_detection(
                prediction=prediction,
                confidence=confidence,
                model_used=model_used_str,
                ensemble_active=ensemble_active,
                latency_seconds=total_time / 1000.0
            )

            logger.info(
                "Detection completed",
                prediction=prediction,
                confidence=confidence,
                model_used=model_used_str,
                ensemble_active=ensemble_active,
                processing_time_ms=round(total_time, 2),
                face_detected=face_detected,
                request_id=request_id
            )

            return DetectionResult(
                prediction=prediction,
                confidence=confidence,
                probabilities={"real": real_prob, "fake": fake_prob},
                model_used="ensemble",
                ensemble_active=ensemble_active,
                models_used=models_used,
                processing_time_ms=total_time,
                face_detected=face_detected,
                heatmap_bytes=heatmap_bytes,
                request_id=request_id
            )

        except Exception as e:
            logger.error("Ensemble detection failed", error=str(e), error_type=type(e).__name__)
            # Raise custom DetectionError with context
            raise DetectionError("Ensemble detection failed", original_error=e) from e

    def _generate_heatmap(self, file_bytes: bytes, probs: np.ndarray) -> Optional[bytes]:
        """Generate Grad-CAM heatmap for the prediction."""
        try:
            # Get EfficientNet session
            session = self.efficientnet.get_session()
            if session is None:
                return None

            # Preprocess for EfficientNet
            image = load_image_from_bytes(file_bytes)
            from app.utils.preprocessing import preprocess_for_efficientnet
            tensor, _ = preprocess_for_efficientnet(image)

            # Determine target class (fake=1, real=0)
            target_class = 1 if probs[1] > probs[0] else 0

            # Generate heatmap - pass file_bytes for fallback heatmap
            heatmap = generate_gradcam(
                session,
                self.efficientnet.model_path,
                tensor,
                target_layer=settings.GRADCAM_TARGET_LAYER,
                input_size=(settings.MODEL_INPUT_SIZE, settings.MODEL_INPUT_SIZE),
                class_idx=target_class,
                file_bytes=file_bytes
            )

            # Convert to bytes
            return heatmap_to_bytes(heatmap)

        except Exception as e:
            logger.error(f"Heatmap generation error: {e}")
            return None

    def predict_single_model(
        self,
        file_bytes: bytes,
        model_name: str
    ) -> Tuple[np.ndarray, float, bool]:
        """
        Run prediction with a single model.

        Args:
            file_bytes: Raw image bytes
            model_name: "efficientnet" or "mesonet"

        Returns:
            Tuple of (probabilities, inference_time_ms, face_detected)
        """
        if model_name == "efficientnet":
            return self.efficientnet.predict_from_bytes(file_bytes)
        elif model_name == "mesonet":
            return self.mesonet.predict_from_bytes(file_bytes)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            "ensemble": {
                "efficientnet_weight": self.efficientnet_weight,
                "mesonet_weight": self.mesonet_weight,
            },
            "efficientnet": {
                "loaded": self.efficientnet.session is not None,
                "model_path": self.efficientnet.model_path,
                "input_size": [224, 224],
            },
            "mesonet": {
                "loaded": self.mesonet.session is not None,
                "model_path": self.mesonet.model_path,
                "input_size": [256, 256],
            }
        }

    def warmup(self):
        """Warm up both models."""
        self.efficientnet.warmup()
        self.mesonet.warmup()
        logger.info("Ensemble detector warmed up")


def create_ensemble_detector() -> EnsembleDetector:
    """Factory function to create ensemble detector with settings."""
    return EnsembleDetector(
        efficientnet_weight=settings.ENSEMBLE_EFFICIENTNET_WEIGHT,
        mesonet_weight=settings.ENSEMBLE_MESONET_WEIGHT
    )