from pathlib import Path

import numpy as np

from app.config import settings
from app.models.detector import EnsembleDetector
from app.models.efficientnet import EfficientNetDetector
from app.models.mesonet import MesoNetDetector


def test_binary_probabilities_are_normalized_to_real_first_order():
    probs = np.array([0.92, 0.08], dtype=np.float32)

    # Config specifies fake_first: ONNX model outputs [fake, real]
    # _normalize_binary_probabilities converts to canonical [real, fake]
    assert settings.MODEL_OUTPUT_ORDER == "fake_first"

    # When output order is fake_first, input [fake_prob, real_prob] -> output [real_prob, fake_prob]
    assert np.allclose(
        EfficientNetDetector._normalize_binary_probabilities(probs),
        np.array([0.08, 0.92])
    )
    assert np.allclose(
        MesoNetDetector._normalize_binary_probabilities(probs),
        np.array([0.08, 0.92])
    )


def test_config_model_output_order_is_fake_first():
    """Verify config matches the ONNX model export order."""
    assert settings.MODEL_OUTPUT_ORDER == "fake_first"


def test_model_directory_points_to_backend_models_folder():
    expected_dir = Path(__file__).resolve().parents[1] / "models"
    assert settings.MODELS_DIR == expected_dir
    assert (settings.MODELS_DIR / "onnxmodel.onnx").exists()


def test_mesonet_warmup_does_not_crash_with_real_model():
    detector = MesoNetDetector()
    detector.warmup()
    assert detector.session is not None


def test_single_model_mode_avoids_doubled_same_model():
    detector = EnsembleDetector()

    # Simulate the real single-model case: same path, both sessions present.
    detector.efficientnet.model_path = detector.mesonet.model_path = 'models/onnxmodel.onnx'
    detector.efficientnet.session = object()
    detector.mesonet.session = object()

    assert detector._single_model_mode() is True

    # If both wrappers are pointing to the same real model, the ensemble should not
    # attempt to average them as if they were different detectors.
    same = detector.efficientnet.model_path == detector.mesonet.model_path
    assert same is True
