"""Grad-CAM and heatmap visualization utilities."""

import cv2
import numpy as np
import onnxruntime
import tempfile
import os
import threading
from PIL import Image
from io import BytesIO
from typing import Optional, Tuple, Dict
import logging

from app.config import settings

logger = logging.getLogger(__name__)

# Module-level Grad-CAM cache to avoid recreating on every request
_gradcam_cache: Dict[str, "GradCAM"] = {}
_gradcam_cache_lock = threading.Lock()


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for ONNX models.

    Note: Full Grad-CAM requires access to gradients, which is not directly
    available in ONNX Runtime. This implementation uses a simplified approach
    by extracting feature maps and computing importance weights.
    """

    def __init__(
        self,
        model_session,
        model_path: str,
        target_layer: str = "features.7",
        input_size: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize Grad-CAM extractor.

        Args:
            model_session: ONNX Runtime InferenceSession
            model_path: Path to the ONNX model file (passed explicitly to avoid private attribute access)
            target_layer: Name of target convolutional layer
            input_size: Model input size (H, W)
        """
        self.session = model_session
        self.model_path = model_path
        self.target_layer = target_layer
        self.input_size = input_size
        self._temp_model_path: Optional[str] = None

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Try to get intermediate layer outputs
        # We need to create a modified session that outputs the target layer
        self._setup_feature_extractor()

    def _setup_feature_extractor(self):
        """Set up session to extract intermediate features."""
        logger.debug("Setting up Grad-CAM feature extractor")
        temp_path = None
        try:
            import onnx
            from onnx import helper

            logger.debug(f"Loading model from {self.model_path}")
            model = onnx.load(self.model_path)

            # Try multiple common layer names for EfficientNet-B0
            # The layer naming can vary depending on how the model was exported
            candidate_layers = [
                self.target_layer,  # e.g., "features.7"
                "features.7",
                "features.6",
                "features.5",
                "features.8",
                "features.7.block.2",  # Some exports use block naming
                "features.6.block.2",
                "Conv_7",  # Generic conv naming
                "Conv_6",
            ]

            target_output = None
            for candidate in candidate_layers:
                for node in model.graph.node:
                    if candidate in node.name or candidate in node.output[0]:
                        target_output = node.output[0]
                        logger.debug(f"Found target layer: {candidate} -> {target_output}")
                        break
                if target_output:
                    break

            # If still not found, try to find the last Conv layer before GlobalAveragePool
            if not target_output:
                conv_outputs = []
                for node in model.graph.node:
                    if node.op_type == "Conv":
                        conv_outputs.append(node.output[0])
                    elif node.op_type == "GlobalAveragePool" and conv_outputs:
                        # Use the last Conv before GlobalAveragePool
                        target_output = conv_outputs[-1]
                        logger.debug(f"Using last Conv before GlobalAveragePool: {target_output}")
                        break

            if target_output:
                logger.debug(f"Creating feature extractor for {target_output}")
                # Create new model with additional output
                model.graph.output.extend([
                    helper.make_tensor_value_info(
                        target_output,
                        onnx.TensorProto.FLOAT,
                        None  # Dynamic shape
                    )
                ])

                # Save modified model to temp file
                with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
                    temp_path = f.name
                    onnx.save(model, f.name)

                self.feature_session = onnxruntime.InferenceSession(
                    temp_path,
                    providers=settings.ONNX_PROVIDERS
                )
                self.feature_output_name = target_output
                self._temp_model_path = temp_path
                logger.debug(f"Grad-CAM feature extractor ready for layer: {target_output}")
            else:
                logger.warning(f"Could not find suitable layer for Grad-CAM, tried: {candidate_layers}")
                self.feature_session = None

        except Exception as e:
            logger.warning(f"Could not set up Grad-CAM feature extractor: {e}")
            self.feature_session = None
            # Clean up temp file if created
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass

    def __del__(self):
        """Clean up temporary model file on destruction."""
        if self._temp_model_path and os.path.exists(self._temp_model_path):
            try:
                os.unlink(self._temp_model_path)
            except Exception:
                pass

    def generate(
        self,
        input_tensor: np.ndarray,
        class_idx: int = 1,  # 1 for fake, 0 for real
        file_bytes: Optional[bytes] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: Preprocessed input (1, 3, H, W)
            class_idx: Target class index
            file_bytes: Optional raw image bytes for fallback

        Returns:
            Heatmap as numpy array (H, W) normalized to [0, 1]
        """
        logger.debug(f"Generating Grad-CAM, feature_session is None: {self.feature_session is None}")
        if self.feature_session is None:
            # Fallback: generate a simple attention map based on input gradients approximation
            # This varies with the input image content
            return self._input_based_heatmap(input_tensor, file_bytes=file_bytes)

        try:
            logger.debug(f"Running feature_session with input shape: {input_tensor.shape}")
            # Get features and predictions
            outputs = self.feature_session.run(
                [self.output_name, self.feature_output_name],
                {self.input_name: input_tensor}
            )

            logger.debug(f"Outputs length: {len(outputs)}, shapes: {[o.shape for o in outputs]}")

            predictions = outputs[0]  # (1, num_classes)
            features = outputs[1]     # (1, C, H', W')

            # Get gradient weights (simplified: use prediction as weight)
            # In true Grad-CAM, we'd compute gradients w.r.t. class score
            # Here we use a proxy: feature importance from weights
            weights = np.mean(features, axis=(2, 3))  # Global average pooling (1, C)

            # Weighted combination of feature maps
            cam = np.zeros(features.shape[2:], dtype=np.float32)
            for i, w in enumerate(weights[0]):
                cam += w * features[0, i, :, :]

            # ReLU
            cam = np.maximum(cam, 0)

            # Normalize
            cam = cam - cam.min()
            if cam.max() > 0:
                cam = cam / cam.max()

            # Resize to input size
            cam = cv2.resize(cam, self.input_size)

            return cam

        except Exception as e:
            logger.error(f"Grad-CAM generation failed: {e}")
            return self._input_based_heatmap(input_tensor)

    def _input_based_heatmap(self, input_tensor: np.ndarray, file_bytes: Optional[bytes] = None) -> np.ndarray:
        """
        Generate a heatmap based on input image content.
        This varies with the input image and provides a meaningful fallback.
        Uses image gradients and frequency analysis as a proxy for model attention.
        """
        try:
            # Use raw image bytes if available for more distinctive heatmaps
            if file_bytes is not None:
                # Decode raw image bytes
                nparr = np.frombuffer(file_bytes, np.uint8)
                img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img_bgr is not None:
                    # Convert to RGB
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    # Resize to target size
                    img_rgb = cv2.resize(img_rgb, self.input_size)
                    img = img_rgb.astype(np.float32) / 255.0
                else:
                    # Fallback to preprocessed tensor
                    img = input_tensor[0]  # (3, H, W)
                    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
                    img = img * std + mean
                    img = np.clip(img, 0, 1)
                    img = np.transpose(img, (1, 2, 0))  # (H, W, 3)
            else:
                # Fallback to preprocessed tensor
                img = input_tensor[0]  # (3, H, W)
                mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
                img = img * std + mean
                img = np.clip(img, 0, 1)
                img = np.transpose(img, (1, 2, 0))  # (H, W, 3)

            # Convert to grayscale
            gray = np.mean(img, axis=2)  # (H, W)

            # Compute gradients (edge detection) - areas with high gradients
            # often correspond to facial features and manipulation artifacts
            grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

            # Also compute Laplacian for second-order derivatives
            laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
            laplacian = np.abs(laplacian)

            # Combine gradient and laplacian
            attention = 0.7 * gradient_magnitude + 0.3 * laplacian

            # Apply Gaussian smoothing for smoother heatmap
            attention = cv2.GaussianBlur(attention, (15, 15), 5)

            # Normalize to [0, 1]
            attention = attention - attention.min()
            if attention.max() > 0:
                attention = attention / attention.max()

            # Resize to target input size
            attention = cv2.resize(attention, self.input_size)

            # Apply a subtle center bias (face is usually in center)
            h, w = self.input_size
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h // 2, w // 2
            center_bias = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (min(h, w) / 3)**2))
            attention = 0.8 * attention + 0.2 * center_bias

            # Final normalization
            attention = attention - attention.min()
            if attention.max() > 0:
                attention = attention / attention.max()

            return attention.astype(np.float32)

        except Exception as e:
            logger.error(f"Input-based heatmap generation failed: {e}")
            return self._dummy_heatmap()

    def _dummy_heatmap(self) -> np.ndarray:
        """Generate a dummy heatmap for fallback (center-focused)."""
        h, w = self.input_size
        # Create a subtle center-focused heatmap
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        heatmap = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (min(h, w) / 4)**2))
        return heatmap.astype(np.float32)


def _get_gradcam_cache_key(model_session, model_path: str, target_layer: str, input_size: Tuple[int, int]) -> str:
    """Generate a cache key for the GradCAM instance."""
    return f"{id(model_session)}:{model_path}:{target_layer}:{input_size[0]}:{input_size[1]}"


def generate_gradcam(
    model_session,
    model_path: str,
    input_tensor: np.ndarray,
    target_layer: str = "features.7",
    input_size: Tuple[int, int] = (224, 224),
    class_idx: int = 1,
    file_bytes: Optional[bytes] = None
) -> np.ndarray:
    """
    Generate Grad-CAM heatmap for a model prediction.

    Args:
        model_session: ONNX Runtime InferenceSession
        model_path: Path to the ONNX model file
        input_tensor: Preprocessed input (1, 3, H, W)
        target_layer: Target layer name
        input_size: Model input size
        class_idx: Target class (1 for fake)
        file_bytes: Optional raw image bytes for fallback heatmap

    Returns:
        Heatmap (H, W) normalized to [0, 1]
    """
    if not settings.GRADCAM_ENABLED:
        return np.zeros(input_size, dtype=np.float32)

    # Create cache key based on model session and config
    cache_key = _get_gradcam_cache_key(model_session, model_path, target_layer, input_size)

    with _gradcam_cache_lock:
        if cache_key not in _gradcam_cache:
            _gradcam_cache[cache_key] = GradCAM(model_session, model_path, target_layer, input_size)

        gradcam = _gradcam_cache[cache_key]

    return gradcam.generate(input_tensor, class_idx, file_bytes=file_bytes)


def overlay_heatmap(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay heatmap on original image.

    Args:
        original_image: Original image (H, W, 3) BGR
        heatmap: Heatmap (H, W) normalized [0, 1]
        alpha: Blending factor
        colormap: OpenCV colormap

    Returns:
        Overlaid image (H, W, 3) BGR
    """
    # Resize heatmap to match original image
    h, w = original_image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8),
        colormap
    )

    # Blend
    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)

    return overlay


def save_heatmap(
    heatmap: np.ndarray,
    output_path: str,
    original_image: Optional[np.ndarray] = None,
    alpha: float = 0.5
) -> str:
    """
    Save heatmap visualization to file.

    Args:
        heatmap: Heatmap (H, W) normalized [0, 1]
        output_path: Output file path
        original_image: Optional original image for overlay
        alpha: Blending factor for overlay

    Returns:
        Path to saved file
    """
    if original_image is not None:
        # Save overlay
        overlay = overlay_heatmap(original_image, heatmap, alpha)
        cv2.imwrite(output_path, overlay)
    else:
        # Save raw heatmap with colormap
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        cv2.imwrite(output_path, heatmap_colored)

    return output_path


def heatmap_to_bytes(heatmap: np.ndarray, format: str = ".png") -> bytes:
    """Convert heatmap to bytes for API response."""
    heatmap_colored = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    success, encoded = cv2.imencode(format, heatmap_colored)
    if not success:
        raise ValueError("Failed to encode heatmap")
    return encoded.tobytes()


def create_comparison_image(
    original: np.ndarray,
    heatmap: np.ndarray,
    prediction: str,
    confidence: float
) -> np.ndarray:
    """
    Create a side-by-side comparison image.

    Args:
        original: Original image (H, W, 3) BGR
        heatmap: Heatmap (H, W) normalized [0, 1]
        prediction: "real" or "fake"
        confidence: Confidence score

    Returns:
        Comparison image (H, 2*W, 3) BGR
    """
    h, w = original.shape[:2]

    # Create overlay
    overlay = overlay_heatmap(original, heatmap)

    # Combine side by side
    comparison = np.hstack([original, overlay])

    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "Original", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, f"Grad-CAM ({prediction}: {confidence:.1%})",
                (w + 10, 30), font, 1, (255, 255, 255), 2)

    return comparison