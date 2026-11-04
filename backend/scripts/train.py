#!/usr/bin/env python3
"""
Train deepfake detection models (EfficientNet-B0, MesoNet) on the provided dataset.

Usage:
    # Original single-dataset training
    python scripts/train.py --model efficientnet --epochs 10 --batch-size 32
    python scripts/train.py --model mesonet --epochs 10 --batch-size 64
    python scripts/train.py --model both --epochs 10 --batch-size 32

    # Multi-dataset training with config (Task 1)
    python scripts/train.py --config dataset_config.yaml --model both --epochs 10

    # With degradation augmentations (Task 2)
    python scripts/train.py --config dataset_config.yaml --model both --epochs 10 --degradation-prob 0.5

    # With holdout validation (Task 1)
    python scripts/train.py --config dataset_config.yaml --model both --epochs 10 --holdout-validation
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, Subset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import timm

# Import new modules
try:
    from scripts.multi_dataset_loader import MultiDatasetLoader, create_dataloaders, get_albumentations_transforms
    from scripts.degradation_augmentations import get_degradation_augmentations, get_training_transforms
    MULTI_DATASET_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Multi-dataset modules not available: {e}")
    MULTI_DATASET_AVAILABLE = False

# Dataset paths - dataset is in project root
DATA_ROOT = Path(__file__).parent.parent.parent / "dataset"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class DeepfakeDataset(Dataset):
    """Dataset for deepfake detection with train/val/test splits (legacy single-dataset)."""

    def __init__(self, root_dir, split='train', transform=None, model_input_size=224):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.model_input_size = model_input_size
        self.samples = []

        # Class mapping: 0 = real, 1 = fake
        for label, class_name in enumerate(['Real', 'Fake']):
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                # Try lowercase
                class_dir = self.root_dir / class_name.lower()
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):
                    self.samples.append((img_path, label))
                for img_path in class_dir.glob('*.png'):
                    self.samples.append((img_path, label))
                for img_path in class_dir.glob('*.jpeg'):
                    self.samples.append((img_path, label))

        print(f"Loaded {len(self.samples)} samples from {split} split")
        if len(self.samples) > 0:
            real_count = sum(1 for _, l in self.samples if l == 0)
            fake_count = sum(1 for _, l in self.samples if l == 1)
            print(f"  Real: {real_count}, Fake: {fake_count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image as fallback
            if self.transform:
                dummy = Image.new('RGB', (self.model_input_size, self.model_input_size), (0, 0, 0))
                return self.transform(dummy), label
            return torch.zeros(3, self.model_input_size, self.model_input_size), label


def get_transforms(model_name, is_train=True, input_size=224):
    """Get transforms for the specified model (legacy torchvision transforms)."""
    if model_name == 'efficientnet':
        # EfficientNet uses ImageNet normalization
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif model_name == 'mesonet':
        # MesoNet uses [0, 1] normalization
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    if is_train:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


def get_albumentations_transforms_legacy(model_name, is_train=True, input_size=224):
    """Get Albumentations transforms (legacy wrapper for backward compatibility)."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import cv2

    if model_name == 'efficientnet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif model_name == 'mesonet':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    if is_train:
        return A.Compose([
            A.Resize(input_size, input_size),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5, p=0.3),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(input_size, input_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])


def create_efficientnet_model(num_classes=2, pretrained=True):
    """Create EfficientNet-B0 model for deepfake detection."""
    # Use timm for more flexibility
    model = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=num_classes)
    return model


def create_mesonet_model(num_classes=2):
    """Create MesoNet model for deepfake detection.

    MesoNet architecture from "MesoNet: a Compact Facial Video Forgery Detection Network"
    """
    class MesoNet(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            # MesoNet-4 architecture
            self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(8)
            self.pool1 = nn.MaxPool2d(2, 2)

            self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=2)
            self.bn2 = nn.BatchNorm2d(8)
            self.pool2 = nn.MaxPool2d(2, 2)

            self.conv3 = nn.Conv2d(8, 16, kernel_size=5, padding=2)
            self.bn3 = nn.BatchNorm2d(16)
            self.pool3 = nn.MaxPool2d(2, 2)

            self.conv4 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
            self.bn4 = nn.BatchNorm2d(16)
            self.pool4 = nn.MaxPool2d(4, 4)  # Global pooling to 1x1

            self.fc1 = nn.Linear(16, 16)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(16, num_classes)

            self.relu = nn.ReLU()
            self.leaky_relu = nn.LeakyReLU(0.1)

        def forward(self, x):
            x = self.pool1(self.leaky_relu(self.bn1(self.conv1(x))))
            x = self.pool2(self.leaky_relu(self.bn2(self.conv2(x))))
            x = self.pool3(self.leaky_relu(self.bn3(self.conv3(x))))
            x = self.pool4(self.leaky_relu(self.bn4(self.conv4(x))))

            x = x.view(x.size(0), -1)
            x = self.dropout(self.leaky_relu(self.fc1(x)))
            x = self.fc2(x)
            return x

    return MesoNet(num_classes)


def train_epoch(model, loader, criterion, optimizer, device, model_name):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    pbar = tqdm(loader, desc=f"Training {model_name}")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Get predictions
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].detach().cpu().numpy())  # fake class probability

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    epoch_loss = running_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0

    return epoch_loss, accuracy, precision, recall, f1, auc


def validate_epoch(model, loader, criterion, device, model_name):
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Validating {model_name}"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].detach().cpu().numpy())

    epoch_loss = running_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0

    cm = confusion_matrix(all_labels, all_preds)

    return epoch_loss, accuracy, precision, recall, f1, auc, cm


def export_to_onnx(model, model_name, input_size, device):
    """Export model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)

    output_path = MODEL_DIR / f"{model_name}_deepfake.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['pixel_values'],
        output_names=['logits'],
        dynamic_axes={
            'pixel_values': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    )
    print(f"✓ Exported {model_name} to {output_path}")
    return output_path


def train_model(model_name, args):
    """Train a specific model (legacy single-dataset mode)."""
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")

    # Model-specific settings
    if model_name == 'efficientnet':
        input_size = 224
        model = create_efficientnet_model(pretrained=not args.no_pretrained)
    elif model_name == 'mesonet':
        input_size = 256
        model = create_mesonet_model()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)

    # Data transforms
    train_transform = get_transforms(model_name, is_train=True, input_size=input_size)
    val_transform = get_transforms(model_name, is_train=False, input_size=input_size)

    # Datasets
    train_dataset = DeepfakeDataset(DATA_ROOT, 'Train', train_transform, input_size)
    val_dataset = DeepfakeDataset(DATA_ROOT, 'Validation', val_transform, input_size)

    if len(train_dataset) == 0:
        print(f"ERROR: No training data found in {DATA_ROOT}/Train")
        return None

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    if model_name == 'efficientnet':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_auc = 0.0
    best_model_path = MODEL_DIR / f"{model_name}_deepfake_best.pth"

    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [], 'train_auc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_auc': []
    }

    for epoch in range(args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{args.epochs} - {model_name.upper()}")
        print(f"{'='*50}")

        # Train
        train_loss, train_acc, train_prec, train_rec, train_f1, train_auc = train_epoch(
            model, train_loader, criterion, optimizer, device, model_name
        )

        # Validate
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, cm = validate_epoch(
            model, val_loader, criterion, device, model_name
        )

        scheduler.step()

        # Log metrics
        print(f"\nTrain - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
        print(f"Val Confusion Matrix:\n{cm}")

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['train_auc'].append(train_auc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'val_acc': val_acc,
            }, best_model_path)
            print(f"✓ Saved best model (AUC: {val_auc:.4f})")

    # Load best model for export
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Export to ONNX
    export_to_onnx(model, model_name, input_size, device)

    # Save training history
    history_path = MODEL_DIR / f"{model_name}_training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n✓ {model_name.upper()} training complete!")
    print(f"  Best Val AUC: {best_val_auc:.4f}")
    print(f"  Model saved to: {best_model_path}")
    print(f"  History saved to: {history_path}")

    return model, history


def train_model_with_config(model_name, args):
    """Train a specific model using multi-dataset config (Task 1 & 2)."""
    if not MULTI_DATASET_AVAILABLE:
        print("Multi-dataset modules not available, falling back to legacy training")
        return train_model(model_name, args)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")

    # Model-specific settings
    if model_name == 'efficientnet':
        input_size = 224
        model = create_efficientnet_model(pretrained=not args.no_pretrained)
    elif model_name == 'mesonet':
        input_size = 256
        model = create_mesonet_model()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)

    # Load config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent.parent / args.config

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get degradation config
    degradation_config = config.get('degradation', {})
    degradation_enabled = degradation_config.get('enabled', True) and not args.no_degradation
    degradation_prob = degradation_config.get('probability', 0.5)

    # Get holdout family
    holdout_family = config.get('holdout_generator_family')
    if not args.holdout_validation:
        holdout_family = None

    print(f"Multi-dataset training with config: {config_path}")
    print(f"  Degradation: {'enabled' if degradation_enabled else 'disabled'}")
    print(f"  Holdout family: {holdout_family or 'none'}")

    # Get transforms with degradation augmentations (Task 2)
    if degradation_enabled and MULTI_DATASET_AVAILABLE:
        try:
            train_transform = get_training_transforms(
                model_name, input_size, degradation_prob
            )
            val_transform = get_albumentations_transforms(model_name, input_size, is_train=False)
        except Exception as e:
            print(f"Warning: Could not use degradation transforms: {e}")
            train_transform = get_albumentations_transforms_legacy(model_name, True, input_size)
            val_transform = get_albumentations_transforms_legacy(model_name, False, input_size)
    else:
        train_transform = get_albumentations_transforms_legacy(model_name, True, input_size)
        val_transform = get_albumentations_transforms_legacy(model_name, False, input_size)

    # Create dataloaders using multi-dataset loader
    loaders = create_dataloaders(
        config_path=str(config_path),
        batch_size=args.batch_size,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=val_transform,
        input_size=input_size,
        holdout_family=holdout_family,
        sampling_strategy='balanced',
        num_workers=args.num_workers
    )

    train_loader = loaders['train']
    val_loader = loaders['val']

    if len(train_loader.dataset) == 0:
        print(f"ERROR: No training data found")
        return None

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    if model_name == 'efficientnet':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_auc = 0.0
    best_model_path = MODEL_DIR / f"{model_name}_deepfake_best.pth"

    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [], 'train_auc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_auc': []
    }

    for epoch in range(args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{args.epochs} - {model_name.upper()}")
        print(f"{'='*50}")

        # Train
        train_loss, train_acc, train_prec, train_rec, train_f1, train_auc = train_epoch(
            model, train_loader, criterion, optimizer, device, model_name
        )

        # Validate
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, cm = validate_epoch(
            model, val_loader, criterion, device, model_name
        )

        scheduler.step()

        # Log metrics
        print(f"\nTrain - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
        print(f"Val Confusion Matrix:\n{cm}")

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['train_auc'].append(train_auc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'val_acc': val_acc,
            }, best_model_path)
            print(f"✓ Saved best model (AUC: {val_auc:.4f})")

    # Load best model for export
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Export to ONNX
    export_to_onnx(model, model_name, input_size, device)

    # Save training history
    history_path = MODEL_DIR / f"{model_name}_training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Holdout validation (if enabled)
    if holdout_family and 'holdout' in loaders:
        print(f"\n{'='*50}")
        print(f"HOLDOUT VALIDATION - {holdout_family}")
        print(f"{'='*50}")
        holdout_loss, holdout_acc, holdout_prec, holdout_rec, holdout_f1, holdout_auc, holdout_cm = validate_epoch(
            model, loaders['holdout'], criterion, device, model_name
        )
        print(f"Holdout - Loss: {holdout_loss:.4f}, Acc: {holdout_acc:.4f}, F1: {holdout_f1:.4f}, AUC: {holdout_auc:.4f}")
        print(f"Holdout Confusion Matrix:\n{holdout_cm}")
        history['holdout_auc'] = holdout_auc
        history['holdout_acc'] = holdout_acc

    print(f"\n✓ {model_name.upper()} training complete!")
    print(f"  Best Val AUC: {best_val_auc:.4f}")
    if holdout_family:
        print(f"  Holdout AUC: {history.get('holdout_auc', 'N/A'):.4f}")
    print(f"  Model saved to: {best_model_path}")
    print(f"  History saved to: {history_path}")

    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train deepfake detection models')
    parser.add_argument('--model', type=str, choices=['efficientnet', 'mesonet', 'both'],
                        default='both', help='Model to train')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data workers')
    parser.add_argument('--cpu', action='store_true', help='Force CPU training')
    parser.add_argument('--no-pretrained', action='store_true', help='Don\'t use pretrained weights')

    # Multi-dataset config options (Task 1 & 2)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to dataset config YAML (enables multi-dataset training)')
    parser.add_argument('--no-degradation', action='store_true',
                        help='Disable degradation augmentations (Task 2)')
    parser.add_argument('--holdout-validation', action='store_true',
                        help='Enable generator-family holdout validation (Task 1)')
    parser.add_argument('--degradation-prob', type=float, default=0.5,
                        help='Probability of applying each degradation transform')

    args = parser.parse_args()

    print("="*60)
    print("DEEPFAKE DETECTOR TRAINING")
    print("="*60)
    print(f"Dataset: {DATA_ROOT}")
    print(f"Model output dir: {MODEL_DIR}")
    print(f"Model(s): {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {'CPU' if args.cpu else 'CUDA if available'}")
    if args.config:
        print(f"Config: {args.config}")
        print(f"Degradation: {'disabled' if args.no_degradation else 'enabled'}")
        print(f"Holdout validation: {'enabled' if args.holdout_validation else 'disabled'}")
    print("="*60)

    if args.config:
        # Multi-dataset training mode
        if args.model in ['efficientnet', 'both']:
            train_model_with_config('efficientnet', args)

        if args.model in ['mesonet', 'both']:
            train_model_with_config('mesonet', args)
    else:
        # Legacy single-dataset training mode
        if args.model in ['efficientnet', 'both']:
            train_model('efficientnet', args)

        if args.model in ['mesonet', 'both']:
            train_model('mesonet', args)

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    main()