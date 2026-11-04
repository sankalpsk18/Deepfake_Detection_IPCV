# Dataset Guide for Deepfake Detection

## 🎯 Why Use Datasets?

Yes, **absolutely!** Using proper datasets for testing and training is **essential** for a production-quality deepfake detector. Here's why:

| Aspect | Without Dataset | With Dataset |
|--------|----------------|--------------|
| **Accuracy** | Random/dummy predictions | Measurable, verifiable performance |
| **Confidence** | Unreliable confidence scores | Calibrated probabilities |
| **Generalization** | Overfits to test images | Robust across lighting, poses, quality |
| **Benchmarking** | No comparison possible | Compare against SOTA (FaceForensics++, DFDC) |
| **Model Selection** | Can't compare architectures | Choose best model/ensemble weights |
| **Regression Testing** | Manual only | Automated CI/CD validation |

---

## 📊 Recommended Datasets

### 1. **FaceForensics++** (Most Popular)
- **Size**: 1,000 original + 4,000 manipulated videos
- **Methods**: Deepfakes, Face2Face, FaceSwap, NeuralTextures
- **Splits**: Train/Val/Test (720/140/140 videos)
- **Download**: ~100GB for full, ~20GB for compressed
- **Best for**: Main benchmark, paper comparisons

### 2. **DFDC (Deepfake Detection Challenge)**
- **Size**: 100,000+ videos, 3,500+ actors
- **Sources**: Multiple generation methods
- **Splits**: Train/Val/Test with hidden test labels
- **Download**: ~400GB
- **Best for**: Large-scale training, generalization

### 3. **Celeb-DF (v2)**
- **Size**: 590 real + 5,639 fake videos
- **Quality**: High-quality deepfakes
- **Best for**: Challenging test set, cross-dataset evaluation

### 4. **WildDeepfake**
- **Size**: ~7,000 videos from the wild
- **Quality**: Real-world artifacts, compression
- **Best for**: Real-world deployment testing

### 5. **KoDF (Korean Deepfake)**
- **Size**: 177 real + 2,007 fake
- **Best for**: Asian face representation, diversity

---

## 🚀 Quick Start: Test with FaceForensics++ (Small Subset)

### Option A: Use Pre-extracted Frames (Fastest)
```bash
# Download pre-extracted frames (~5GB)
wget https://github.com/ondyari/FaceForensics/releases/download/1.0/faceforensicspp_frames.zip
unzip faceforensicspp_frames.zip
```

### Option B: Download & Extract Yourself
```bash
# Install FaceForensics++ toolkit
git clone https://github.com/ondyari/FaceForensics
cd FaceForensics
pip install -r requirements.txt

# Download compressed videos (~20GB)
python download.py --dataset=faceforensicspp --compression=c23 --server=EU

# Extract frames
python extract_frames.py --dataset=faceforensicspp --compression=c23
```

---

## 🔧 Integrating Datasets into This Project

### 1. Create Dataset Structure
```
backend/
├── data/
│   ├── faceforensics/
│   │   ├── train/
│   │   │   ├── real/
│   │   │   └── fake/
│   │   ├── val/
│   │   │   ├── real/
│   │   │   └── fake/
│   │   └── test/
│   │       ├── real/
│   │       └── fake/
```

### 2. Training Script (`scripts/train.py`)
```python
#!/usr/bin/env python3
"""
Train deepfake detection models on FaceForensics++ dataset.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.samples = []
        
        for label, class_name in enumerate(['real', 'fake']):
            class_dir = self.root_dir / class_name
            for img_path in class_dir.glob('*.jpg'):
                self.samples.append((img_path, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def train_model(model, train_loader, val_loader, epochs=10, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        print(f'Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, Val Acc={val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        scheduler.step()
    
    return best_val_acc

if __name__ == '__main__':
    # Setup
    data_root = 'data/faceforensics'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = DeepfakeDataset(data_root, 'train', transform)
    val_dataset = DeepfakeDataset(data_root, 'val', transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # EfficientNet-B0
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model = model.to(device)
    
    # Train
    train_model(model, train_loader, val_loader, epochs=10, device=device)
    
    # Export to ONNX
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    torch.onnx.export(model, dummy_input, 'efficientnet_b0_deepfake.onnx',
                      input_names=['pixel_values'], output_names=['logits'],
                      dynamic_axes={'pixel_values': {0: 'batch_size'}, 'logits': {0: 'batch_size'}})
    print("Model exported to ONNX!")
```

### 3. Evaluation Script (`scripts/evaluate.py`)
```python
#!/usr/bin/env python3
"""
Evaluate trained models on test set.
"""
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np

def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # fake class prob
    
    # Metrics
    print(classification_report(all_labels, all_preds, target_names=['Real', 'Fake']))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print(f"ROC-AUC: {roc_auc_score(all_labels, all_probs):.4f}")
    
    return all_labels, all_preds, all_probs
```

---

## 📈 Evaluation Metrics to Track

| Metric | Target | Description |
|--------|--------|-------------|
| **Accuracy** | > 90% | Overall correctness |
| **AUC-ROC** | > 0.95 | Ranking quality |
| **Precision (Fake)** | > 85% | Low false alarms |
| **Recall (Fake)** | > 90% | Catch most fakes |
| **F1-Score** | > 87% | Balanced metric |
| **EER** | < 10% | Equal Error Rate |

---

## 🧪 Testing Strategy

### 1. Unit Tests (Backend)
```python
# backend/tests/test_detector.py
def test_ensemble_detector_on_known_samples():
    detector = create_ensemble_detector()
    
    # Test with known real image
    with open('test_data/real_face.jpg', 'rb') as f:
        result = detector.predict(f.read())
    assert result.prediction == 'real'
    assert result.confidence > 0.7
    
    # Test with known fake image
    with open('test_data/fake_face.jpg', 'rb') as f:
        result = detector.predict(f.read())
    assert result.prediction == 'fake'
    assert result.confidence > 0.7
```

### 2. Integration Tests
```bash
# Test API endpoints
pytest backend/tests/ -v

# Load test
locust -f backend/tests/load_test.py --host=http://localhost:8000
```

### 3. Cross-Dataset Evaluation
```python
# Evaluate on multiple datasets
datasets = ['faceforensics', 'celebdf', 'wilddeepfake', 'dfdc']
for ds in datasets:
    print(f"\n=== Evaluating on {ds} ===")
    evaluate_on_dataset(model, ds)
```

---

## 📁 Project Integration Checklist

- [ ] Download FaceForensics++ frames (or subset)
- [ ] Create `data/` directory structure
- [ ] Run `scripts/train.py` to train EfficientNet/MesoNet
- [ ] Export models to ONNX (`efficientnet_b0_deepfake.onnx`, `mesonet_deepfake.onnx`)
- [ ] Place ONNX files in `backend/models/`
- [ ] Run `scripts/evaluate.py` on test set
- [ ] Update `MODEL_OUTPUT_ORDER` in config based on export
- [ ] Add test images to `backend/test_data/`
- [ ] Run pytest for automated testing
- [ ] Set up CI/CD with dataset evaluation

---

## 💡 Pro Tips

1. **Start Small**: Use 10% of FaceForensics++ for initial experiments
2. **Data Augmentation**: Use Albumentations for robustness
   ```python
   import albumentations as A
   transform = A.Compose([
       A.HorizontalFlip(p=0.5),
       A.RandomBrightnessContrast(p=0.3),
       A.GaussianBlur(p=0.2),
       A.ImageCompression(quality_range=(50, 100), p=0.3),
       A.Resize(224, 224),
       A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
   ])
   ```
3. **Ensemble Weights**: Tune `ENSEMBLE_EFFICIENTNET_WEIGHT` based on validation
4. **Threshold Calibration**: Find optimal decision threshold per dataset
5. **Monitor Drift**: Track performance on holdout set over time

---

## 🔗 Resources

- **FaceForensics++**: https://github.com/ondyari/FaceForensics
- **DFDC**: https://ai.facebook.com/datasets/dfdc/
- **Celeb-DF**: https://github.com/yuezunli/celeb-deepfakeforensics
- **Paper**: "FaceForensics++: Learning to Detect Manipulated Facial Images" (ICCV 2019)
- **Benchmark**: https://github.com/ondyari/FaceForensics/tree/master/benchmark

---

## Summary

**Yes, datasets are crucial!** They transform your project from a demo into a production-ready system with measurable, verifiable performance. Start with FaceForensics++ (c23 compressed) - it's the standard benchmark and has good tooling.