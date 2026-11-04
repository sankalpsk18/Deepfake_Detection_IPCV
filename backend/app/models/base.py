"""Base detector class for ONNX-based deepfake detection models."""

import onnxruntime as ort
import numpy as np
import time
import logging
from pathlib import Path
from typing import Tuple, Optional, List

from app.config import settings

logger = logging.getLogger(__name__)


class BaseONNXDetector:
    """
    Base class for ONNX Runtime-based deepfake detectors.

    Provides shared functionality for model loading, inference, and utilities.
    Subclasses should implement model-specific preprocessing and output handling.
    """

    def __init__(
        self,
        model_path: Optional[str],
        candidates: List[str],
        device: str = "cpu",
        providers: Optional[List[str]] = None,
        session_options: Optional[ort.SessionOptions] = None
    ):
        """
        Initialize base detector.

        Args:
            model_path: Primary model path from settings
            candidates: List of alternative model filenames to search
            device: "cpu" or "cuda"
            providers: ONNX Runtime execution providers
            session_options: Optional custom session options
        """
        self.model_path = self._resolve_model_path(model_path, candidates)
        self.device = device
        self.providers = providers or settings.ONNX_PROVIDERS
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.output_name: Optional[str] = None
        self._session_options = session_options
        self._load_model()

    @staticmethod
    def _resolve_model_path(model_path: str, candidates: List[str]) -> str:
        """Resolve the first existing ONNX model path from a set of possible filenames."""
        search_paths = [Path(model_path)]
        for candidate in candidates:
            search_paths.append(Path(candidate))
            for base_dir in settings.MODEL_SEARCH_DIRS:
                search_paths.append(base_dir / candidate)

        for candidate in search_paths:
            if candidate.is_file():
                return str(candidate)

        return str(settings.MODELS_DIR / Path(model_path).name)

    def _create_session_options(self) -> ort.SessionOptions:
        """Create default session options. Override in subclass if needed."""
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_cpu_mem_arena = True
        sess_options.enable_mem_pattern = True
        return sess_options

    def _load_model(self):
        """Load ONNX model."""
        try:
            model_file = Path(self.model_path)
            if not model_file.exists():
                logger.warning(f"Model not found at {self.model_path}. Using dummy model.")
                self._create_dummy_session()
                return

            # Configure session options
            sess_options = self._session_options or self._create_session_options()

            # Add CUDA provider if requested
            providers = self.providers
            if self.device == "cuda":
                providers = ["CUDAExecutionProvider"] + providers

            self.session = ort.InferenceSession(
                str(model_file),
                sess_options=sess_options,
                providers=providers
            )

            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

            logger.info(f"{self.__class__.__name__} model loaded from {model_file}")
            logger.info(f"Input: {self.input_name}, Output: {self.output_name}")
            logger.info(f"Providers: {self.session.get_providers()}")

            # Allow subclass to do post-load initialization
            self._post_load_init()

        except Exception as e:
            logger.error(f"Failed to load {self.__class__.__name__} model: {e}")
            self._create_dummy_session()

    def _post_load_init(self):
        """Hook for subclass to perform post-load initialization."""
        pass

    def _create_dummy_session(self):
        """Create a dummy session for testing without model."""
        logger.warning(f"Using dummy {self.__class__.__name__} detector")
        self.session = None

    @staticmethod
    def _bytes_to_image(file_bytes: bytes) -> np.ndarray:
        """Convert bytes to OpenCV image."""
        import cv2
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")
        return image

    def get_session(self) -> Optional[ort.InferenceSession]:
        """Get ONNX session for Grad-CAM."""
        return self.session

    @staticmethod
    def _normalize_binary_probabilities(probs: np.ndarray) -> np.ndarray:
        """Normalize exported binary logits to canonical [real, fake] ordering."""
        probs = np.asarray(probs, dtype=np.float32)
        if probs.shape[-1] != 2:
            return probs
        if settings.MODEL_OUTPUT_ORDER == "fake_first":
            return probs[[1, 0]]
        return probs

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / exp_x.sum()

    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Run inference on preprocessed image.

        Args:
            image: Preprocessed image tensor

        Returns:
            Tuple of (probabilities, inference_time_ms)
            probabilities: (2,) array [real_prob, fake_prob]
        """
        start_time = time.perf_counter()

        if self.session is None:
            probs = self._fallback_probabilities(image)
            inference_time = (time.perf_counter() - start_time) * 1000
            return probs, inference_time

        try:
            # Run inference
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: image}
            )

            # Subclass handles output processing
            probs = self._process_output(outputs[0][0])

            inference_time = (time.perf_counter() - start_time) * 1000
            return probs, inference_time

        except Exception as e:
            logger.error(f"{self.__class__.__name__} inference failed: {e}")
            probs = np.array([0.5, 0.5], dtype=np.float32)
            inference_time = (time.perf_counter() - start_time) * 1000
            return probs, inference_time

    def _process_output(self, output: np.ndarray) -> np.ndarray:
        """Process model output to probabilities. Override in subclass."""
        raise NotImplementedError

    def _fallback_probabilities(self, image: np.ndarray) -> np.ndarray:
        """Provide fallback probabilities when model is unavailable. Override in subclass."""
        raise NotImplementedError

    def predict_from_bytes(self, file_bytes: bytes, preprocess_fn) -> Tuple[np.ndarray, float, bool]:
        """
        Full pipeline: preprocess bytes and predict.

        Args:
            file_bytes: Raw image bytes
            preprocess_fn: Preprocessing function for this model

        Returns:
            Tuple of (probabilities, inference_time_ms, face_detected)
        """
        # Preprocess
        tensor, face_detected = preprocess_fn(self._bytes_to_image(file_bytes))

        # Predict
        probs, inf_time = self.predict(tensor)

        return probs, inf_time, face_detected

    def warmup(self, num_runs: int = 3):
        """Warm up the model with dummy inputs."""
        if self.session is None:
            return

        dummy_input = self._get_dummy_input()
        for _ in range(num_runs):
            _ = self.session.run([self.output_name], {self.input_name: dummy_input})
        logger.info(f"{self.__class__.__name__} model warmed up")

    def _get_dummy_input(self) -> np.ndarray:
        """Get dummy input for warmup. Override in subclass."""
        raise NotImplementedError