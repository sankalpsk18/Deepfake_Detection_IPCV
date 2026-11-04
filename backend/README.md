# Deepfake Detector - Backend API

FastAPI-based backend for AI-generated/deepfake image detection.

## Features

- **Ensemble Detection**: Combines EfficientNet-B0 and MesoNet for robust detection
- **Face-Aware Preprocessing**: Uses MediaPipe for face detection and cropping
- **Explainability**: Grad-CAM heatmaps highlighting suspicious regions
- **Fast Inference**: ONNX Runtime for optimized CPU/GPU inference
- **RESTful API**: Clean API with automatic OpenAPI documentation

## Quick Start

### Prerequisites

- Python 3.11+
- (Optional) CUDA-enabled GPU for faster inference

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env

# Download model weights (see Model Weights section below)
# Place .onnx files in ./models/

# Run development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Using Docker

```bash
# Build and run
docker-compose up --build

# Or just backend
docker build -t deepfake-backend ./backend
docker run -p 8000:8000 -v ./backend/models:/app/models deepfake-backend
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/model/info` | Model information |
| POST | `/api/v1/detect` | Detect deepfake (multipart/form-data) |

### Example Request

```bash
curl -X POST "http://localhost:8000/api/v1/detect" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"
```

### Example Response

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2024-01-15T10:30:00Z",
  "result": {
    "prediction": "fake",
    "confidence": 0.92,
    "probabilities": {"real": 0.08, "fake": 0.92},
    "model_used": "ensemble",
    "processing_time_ms": 145.2,
    "face_detected": true,
    "heatmap_url": "/api/v1/heatmap/550e8400-e29b-41d4-a716-446655440000"
  },
  "filename": "test_image.jpg"
}
```

## Model Weights

The backend expects ONNX model files in the `./models/` directory:

- `efficientnet_b0_deepfake.onnx` - EfficientNet-B0 fine-tuned on FaceForensics++
- `mesonet_deepfake.onnx` - MesoNet for deepfake detection

### Obtaining Models

#### Option 1: Pre-trained from Hugging Face (Recommended)

```python
from huggingface_hub import hf_hub_download

# EfficientNet-B0
hf_hub_download(
    repo_id="deepfake-detection/efficientnet-b0-faceforensics",
    filename="model.onnx",
    local_dir="./models",
    local_dir_use_symlinks=False
)

# MesoNet
hf_hub_download(
    repo_id="deepfake-detection/mesonet-faceforensics",
    filename="model.onnx",
    local_dir="./models",
    local_dir_use_symlinks=False
)
```

#### Option 2: Convert from PyTorch

```bash
# Install conversion tools
pip install onnx torch torchvision

# Convert your trained models
python scripts/convert_to_onnx.py
```

#### Option 3: Train Your Own

See `scripts/train.py` for training scripts using FaceForensics++ dataset.

## Project Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration management
│   ├── models/
│   │   ├── detector.py      # Ensemble detector
│   │   ├── efficientnet.py  # EfficientNet wrapper
│   │   └── mesonet.py       # MesoNet wrapper
│   ├── schemas/
│   │   └── detection.py     # Pydantic models
│   ├── utils/
│   │   ├── preprocessing.py # Image preprocessing
│   │   └── visualization.py # Grad-CAM heatmaps
│   └── api/
│       └── routes.py        # API endpoints
├── models/                  # Model weights (.onnx)
├── requirements.txt
├── Dockerfile
└── README.md
```

## Configuration

Key settings in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_DEVICE` | `cpu` | `cpu` or `cuda` |
| `ENSEMBLE_EFFICIENTNET_WEIGHT` | `0.7` | EfficientNet weight in ensemble |
| `ENSEMBLE_MESONET_WEIGHT` | `0.3` | MesoNet weight in ensemble |
| `GRADCAM_ENABLED` | `true` | Enable heatmap generation |
| `MAX_FILE_SIZE` | `10485760` | Max upload size (10MB) |

## Development

### Code Quality

```bash
# Format code
black app/
isort app/

# Lint
ruff check app/
mypy app/

# Run tests
pytest
```

### Adding New Models

1. Create new model wrapper in `app/models/`
2. Implement `predict()` and `predict_from_bytes()` methods
3. Update `EnsembleDetector` to include new model
4. Adjust ensemble weights in config

## Performance

Typical inference times on CPU (Intel i7-11th gen):

| Model | Time (ms) |
|-------|-----------|
| EfficientNet-B0 | ~80-120 |
| MesoNet | ~10-20 |
| Ensemble (sequential) | ~100-150 |

With GPU (RTX 3060):
| Model | Time (ms) |
|-------|-----------|
| EfficientNet-B0 | ~10-15 |
| MesoNet | ~2-5 |
| Ensemble | ~15-25 |

## License

MIT License - See LICENSE file for details.