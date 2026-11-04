"""Application configuration using Pydantic Settings."""

import json
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field, model_validator
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    APP_NAME: str = "Deepfake Detector API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"

    # API
    API_PREFIX: str = "/api/v1"
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".webp"}

    # Paths
    # The backend project root is one level above app/, not the workspace root.
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    MODELS_DIR: Path = BASE_DIR / "models"
    TEMP_DIR: Path = BASE_DIR / "temp"
    LEGACY_MODELS_DIR: Path = BASE_DIR.parent / "models"
    MODEL_SEARCH_DIRS: list[Path] = [MODELS_DIR, LEGACY_MODELS_DIR]

    # Model Configuration
    # The shared ONNX asset in this project exports class logits in fake-first order.
    # Keep the app consistent with that asset to avoid reversed real/fake predictions.
    EFFICIENTNET_MODEL_PATH: str = "efficientnet_b0_deepfake.onnx"
    MESONET_MODEL_PATH: str = "mesonet_deepfake.onnx"
    MODEL_OUTPUT_ORDER: str = "fake_first"  # "real_first" or "fake_first" - ONNX model outputs [fake, real]
    MODEL_INPUT_SIZE: int = 224
    MESONET_INPUT_SIZE: int = 256

    # Ensemble weights (must sum to 1.0)
    ENSEMBLE_EFFICIENTNET_WEIGHT: float = 0.7
    ENSEMBLE_MESONET_WEIGHT: float = 0.3

    # Face Detection
    FACE_DETECTION_CONFIDENCE: float = 0.5
    FACE_CROP_PADDING: float = 0.3  # 30% padding around face

    # Inference
    INFERENCE_DEVICE: str = "cpu"  # "cpu" or "cuda"
    ONNX_PROVIDERS: list[str] = ["CPUExecutionProvider"]

    # Grad-CAM
    GRADCAM_ENABLED: bool = True
    GRADCAM_TARGET_LAYER: str = "features.7"  # EfficientNet-B0 last conv layer

    # Heatmap Cache
    HEATMAP_CACHE_MAX_SIZE: int = 100
    HEATMAP_CACHE_TTL_SECONDS: int = 600  # 10 minutes

    # Rate Limiting
    RATE_LIMIT_DETECT: str = "10/minute"  # POST /detect
    RATE_LIMIT_DEFAULT: str = "60/minute"  # Other endpoints

    # API Key Authentication (optional)
    # Comma-separated list of valid API keys. If empty, auth is disabled.
    API_KEYS: str = ""

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"

    @model_validator(mode="after")
    def validate_production_config(self) -> "Settings":
        """Validate required settings for production environment."""
        if self.ENVIRONMENT == "production":
            # Check that model files exist (or will be found via search dirs)
            eff_path = self._resolve_model_path(self.EFFICIENTNET_MODEL_PATH, [
                "efficientnet_b0_deepfake.onnx",
                "onnxmodel.onnx",
                "model.onnx",
            ])
            mes_path = self._resolve_model_path(self.MESONET_MODEL_PATH, [
                "mesonet_deepfake.onnx",
                "onnxmodel.onnx",
                "model.onnx",
            ])

            if not Path(eff_path).exists() and not Path(mes_path).exists():
                raise ValueError(
                    "Production mode requires at least one model file. "
                    f"Expected EfficientNet at {eff_path} or MesoNet at {mes_path}. "
                    "Place model files in the models/ directory or set EFFICIENTNET_MODEL_PATH/MESONET_MODEL_PATH."
                )

            # Check CORS origins are configured for production
            if self.CORS_ORIGINS == ["http://localhost:3000", "http://127.0.0.1:3000"]:
                raise ValueError(
                    "Production mode requires explicit CORS_ORIGINS configuration. "
                    "Set CORS_ORIGINS to your frontend domain(s)."
                )

        return self

    def _resolve_model_path(self, model_path: str, candidates: list[str]) -> str:
        """Resolve the first existing ONNX model path from a set of possible filenames."""
        search_paths = [Path(model_path)]
        for candidate in candidates:
            search_paths.append(Path(candidate))
            for base_dir in self.MODEL_SEARCH_DIRS:
                search_paths.append(base_dir / candidate)

        for candidate in search_paths:
            if candidate.is_file():
                return str(candidate)

        return str(self.MODELS_DIR / Path(model_path).name)

    def get_model_metadata(self) -> Optional[dict]:
        """Load model metadata from JSON file if it exists."""
        metadata_path = self.MODELS_DIR / "model_metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    return json.load(f)
            except Exception:
                return None
        return None


# Global settings instance
settings = Settings()

# Ensure directories exist
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)