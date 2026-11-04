"""Pytest configuration and fixtures for API integration tests."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import io
from PIL import Image
from slowapi import Limiter


# PATCH SETTINGS FIRST - before any app modules are imported
# This ensures @limiter.limit(settings.RATE_LIMIT_DETECT) uses test values
with patch("app.config.settings.RATE_LIMIT_DETECT", "1000/minute"):
    with patch("app.config.settings.RATE_LIMIT_DEFAULT", "1000/minute"):
        with patch("app.config.settings.ENVIRONMENT", "development"):
            with patch("app.config.settings.API_KEYS", ""):  # Disable auth by default for tests
                from app.main import create_app
                from app.models.detector import EnsembleDetector, DetectionResult
                from app.config import settings


def _disable_rate_limiter():
    """Create a no-op limiter for testing."""
    limiter = Limiter(key_func=lambda: "test")

    def noop_limit(*args, **kwargs):
        def decorator(f):
            return f
        return decorator

    def noop_exempt(*args, **kwargs):
        def decorator(f):
            return f
        return decorator

    limiter.limit = noop_limit
    limiter.exempt = noop_exempt
    return limiter


# Apply limiter patches BEFORE importing app modules that use them
# (routes.py is already imported via create_app, but we patch it for the limiter instance)
_patcher_routes = patch("app.api.routes.limiter", _disable_rate_limiter())
_patcher_main = patch("app.main.limiter", _disable_rate_limiter())
_patcher_routes.start()
_patcher_main.start()


@pytest.fixture(scope="function")
def app():
    """Create FastAPI app for testing."""
    # Re-apply patches for each test (the limiter decorators were already applied at import)
    with patch("app.api.routes.limiter", _disable_rate_limiter()):
        with patch("app.main.limiter", _disable_rate_limiter()):
            with patch("app.config.settings.API_KEYS", ""):  # Disable auth by default
                yield create_app()


@pytest.fixture(scope="function")
def client(app):
    """Create TestClient for API testing."""
    return TestClient(app)


@pytest.fixture
def mock_detector():
    """Create a mock EnsembleDetector that returns predictable results."""
    detector = Mock(spec=EnsembleDetector)

    # Mock get_model_info
    detector.get_model_info.return_value = {
        "ensemble": {
            "efficientnet_weight": 0.7,
            "mesonet_weight": 0.3,
        },
        "efficientnet": {
            "loaded": True,
            "model_path": "efficientnet_b0_deepfake.onnx",
            "input_size": [224, 224],
        },
        "mesonet": {
            "loaded": True,
            "model_path": "mesonet_deepfake.onnx",
            "input_size": [256, 256],
        },
    }

    # Mock predict to return a fake detection result
    def mock_predict(file_bytes, request_id=None):
        return DetectionResult(
            prediction="fake",
            confidence=0.92,
            probabilities={"real": 0.08, "fake": 0.92},
            model_used="ensemble",
            ensemble_active=True,
            models_used=["EfficientNet-B0", "MesoNet"],
            processing_time_ms=145.2,
            face_detected=True,
            heatmap_bytes=b"fake_heatmap_png_bytes",
            request_id=request_id or "test-request-id",
        )

    detector.predict.side_effect = mock_predict
    detector.warmup = Mock()

    return detector


@pytest.fixture
def app_with_mock_detector(app, mock_detector):
    """Override app state with mock detector."""
    app.state.detector = mock_detector
    return app


@pytest.fixture
def client_with_mock(app_with_mock_detector):
    """TestClient with mocked detector."""
    return TestClient(app_with_mock_detector)


@pytest.fixture
def valid_jpeg_bytes():
    """Generate a small valid JPEG image for testing."""
    img = Image.new('RGB', (224, 224), color='red')
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=85)
    return buf.getvalue()


@pytest.fixture
def large_file_bytes():
    """Generate a file larger than MAX_FILE_SIZE (10MB)."""
    # Create 11MB of data
    return b"x" * (11 * 1024 * 1024)


@pytest.fixture
def empty_file_bytes():
    """Empty file bytes."""
    return b""


@pytest.fixture
def non_image_bytes():
    """Non-image file bytes (text file)."""
    return b"This is not an image file"


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset rate limiter between tests."""
    # slowapi uses in-memory storage; we can't easily reset it
    # but we can ensure each test gets a fresh client
    yield