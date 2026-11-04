#!/usr/bin/env python3
"""
Train Temporal Model for Video-Level Deepfake Detection (Task 3)

Usage:
    python scripts/train_temporal.py --video-dir dataset/videos --model-type transformer --epochs 20
    python scripts/train_temporal.py --video-dir dataset/videos --model-type bigru --epochs 20
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import timm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.frame_sampler import VideoFrameDataset, create_video_dataloaders, collate_video_batch
from scripts.temporal_model import (
    TemporalTransformer, BiGRUTemporal, TemporalDetector,
    create_temporal_model, export_temporal_to_onnx
)
from scripts.meta_classifier import MetaClassifier, train_meta_classifier

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def extract_efficientnet_features(model: nn.Module, frames: torch.Tensor) -> torch.Tensor:
    """
    Extract penultimate layer features from EfficientNet-B0.

    Args:
        model: EfficientNet-B0 model
        frames: (B, T, C, H, W) or (B*T, C, H, W)

    Returns:
        Features (B, T, D) where D=1280 for EfficientNet-B0
    """
    original_shape = frames.shape

    if frames.dim() == 5:
        B, T, C, H, W = frames.shape
        frames = frames.view(B * T, C, H, W)
    else:
        B = 1
        T = frames.size(0)

    with torch.no_grad():
        # Get features from penultimate layer
        features = model.forward_features(frames)

        # Global average pooling if needed
        if features.dim() == 4:
            features = features.mean(dim=[2, 3])  # (B*T, D)

    return features.view(B, T, -1)


def train_epoch(
    temporal_model: nn.Module,
    efficientnet: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    model_name: str
) -> Tuple[float, float, float, float, float]:
    """Train temporal model for one epoch."""
    temporal_model.train()
    efficientnet.eval()  # Freeze backbone

    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    pbar = tqdm(loader, desc=f"Training {model_name}")
    for frames, labels, metadata in pbar:
        frames = frames.to(device)  # (B, T, C, H, W)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Extract frame embeddings from EfficientNet
        embeddings = extract_efficientnet_features(efficientnet, frames)  # (B, T, 1280)

        # Temporal model forward
        logits = temporal_model(embeddings)  # (B, 2)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        probs = torch.softmax(logits, dim=1)
        _, preds = torch.max(logits, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    epoch_loss = running_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0

    return epoch_loss, accuracy, precision, recall, f1, auc


def validate_epoch(
    temporal_model: nn.Module,
    efficientnet: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    model_name: str
) -> Tuple[float, float, float, float, float, np.ndarray]:
    """Validate temporal model for one epoch."""
    temporal_model.eval()
    efficientnet.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for frames, labels, metadata in tqdm(loader, desc=f"Validating {model_name}"):
            frames = frames.to(device)
            labels = labels.to(device)

            embeddings = extract_efficientnet_features(efficientnet, frames)
            logits = temporal_model(embeddings)

            loss = criterion(logits, labels)
            running_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    epoch_loss = running_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0

    cm = confusion_matrix(all_labels, all_preds)

    return epoch_loss, accuracy, precision, recall, f1, auc, cm


def train_temporal_model(
    temporal_model: nn.Module,
    efficientnet: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    model_name: str
) -> Tuple[nn.Module, Dict]:
    """Train temporal model with frozen EfficientNet backbone."""

    criterion = nn.CrossEntropyLoss()

    if args.model_type == 'transformer':
        optimizer = optim.AdamW(
            temporal_model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = optim.Adam(
            temporal_model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_auc = 0.0
    best_model_path = MODEL_DIR / f"{model_name}_temporal_best.pth"

    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [], 'train_auc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_auc': []
    }

    for epoch in range(args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{args.epochs} - {model_name.upper()} (Temporal)")
        print(f"{'='*50}")

        # Train
        train_loss, train_acc, train_prec, train_rec, train_f1, train_auc = train_epoch(
            temporal_model, efficientnet, train_loader, criterion, optimizer, device, model_name
        )

        # Validate
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, cm = validate_epoch(
            temporal_model, efficientnet, val_loader, criterion, device, model_name
        )

        scheduler.step()

        print(f"\nTrain - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
        print(f"Val CM:\n{cm}")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['train_auc'].append(train_auc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                'epoch': epoch,
                'temporal_model_state_dict': temporal_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'val_acc': val_acc,
            }, best_model_path)
            print(f"✓ Saved best temporal model (AUC: {val_auc:.4f})")

    # Load best model
    checkpoint = torch.load(best_model_path, map_location=device)
    temporal_model.load_state_dict(checkpoint['temporal_model_state_dict'])

    # Save history
    history_path = MODEL_DIR / f"{model_name}_temporal_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n✓ {model_name.upper()} temporal training complete!")
    print(f"  Best Val AUC: {best_val_auc:.4f}")
    print(f"  Model saved to: {best_model_path}")
    print(f"  History saved to: {history_path}")

    return temporal_model, history


def train_meta_fusion(
    temporal_model: nn.Module,
    efficientnet: nn.Module,
    frame_ensemble_probs: np.ndarray,
    val_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace
) -> Any:
    """
    Train meta-classifier on validation set using per-frame ensemble + temporal outputs.

    Args:
        temporal_model: Trained temporal model
        efficientnet: EfficientNet backbone
        frame_ensemble_probs: Pre-computed per-frame ensemble probabilities (N, 2)
        val_loader: Validation dataloader
        device: Training device
        args: Training arguments

    Returns:
        Trained meta-classifier
    """
    print("\n" + "="*50)
    print("TRAINING META-CLASSIFIER FOR FUSION")
    print("="*50)

    temporal_model.eval()
    efficientnet.eval()

    # Collect temporal predictions on validation set
    all_temporal_probs = []
    all_labels = []

    with torch.no_grad():
        for frames, labels, metadata in tqdm(val_loader, desc="Collecting temporal probs"):
            frames = frames.to(device)
            embeddings = extract_efficientnet_features(efficientnet, frames)
            logits = temporal_model(embeddings)
            probs = torch.softmax(logits, dim=1)

            all_temporal_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    temporal_probs = np.array(all_temporal_probs)
    labels = np.array(all_labels)

    # Use provided frame ensemble probs or compute them
    if frame_ensemble_probs is not None:
        if len(frame_ensemble_probs) != len(labels):
            print(f"Warning: frame_ensemble_probs length ({len(frame_ensemble_probs)}) != labels ({len(labels)})")
            # Truncate or pad
            min_len = min(len(frame_ensemble_probs), len(labels))
            frame_ensemble_probs = frame_ensemble_probs[:min_len]
            temporal_probs = temporal_probs[:min_len]
            labels = labels[:min_len]
    else:
        # Generate dummy frame probs (in practice, run per-frame ensemble)
        print("Warning: No frame ensemble probs provided, using temporal probs as placeholder")
        frame_ensemble_probs = temporal_probs.copy()

    # Train meta-classifier
    meta_classifier = train_meta_classifier(
        frame_probs=frame_ensemble_probs,
        temporal_probs=temporal_probs,
        labels=labels,
        model_type=args.meta_type,
        device=str(device),
        epochs=args.meta_epochs,
        batch_size=args.meta_batch_size,
        lr=args.meta_lr
    )

    # Evaluate fusion
    from scripts.meta_classifier import evaluate_fusion, compare_fusion_strategies

    metrics = evaluate_fusion(
        meta_classifier,
        frame_ensemble_probs,
        temporal_probs,
        labels
    )

    print(f"\nFusion Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")

    # Compare strategies
    strategies = compare_fusion_strategies(
        frame_ensemble_probs,
        temporal_probs,
        labels
    )
    print("\nFusion Strategy Comparison:")
    for name, m in strategies.items():
        print(f"  {name}: Acc={m['accuracy']:.4f}, AUC={m.get('auc', 0):.4f}")

    return meta_classifier


def main():
    parser = argparse.ArgumentParser(description='Train temporal model for video deepfake detection')
    parser.add_argument('--video-dir', type=str, default='dataset/videos',
                        help='Directory with real/ and fake/ subdirectories')
    parser.add_argument('--model-type', type=str, choices=['transformer', 'bigru'],
                        default='transformer', help='Temporal model type')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size (small for video)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data workers')
    parser.add_argument('--cpu', action='store_true', help='Force CPU training')
    parser.add_argument('--num-frames', type=int, default=16, help='Frames per sequence')
    parser.add_argument('--stride', type=int, default=2, help='Frame stride')
    parser.add_argument('--target-size', type=int, nargs=2, default=[224, 224], help='Target frame size H W')
    parser.add_argument('--efficientnet-path', type=str,
                        default='models/efficientnet_deepfake_best.pth',
                        help='Path to pretrained EfficientNet checkpoint')
    parser.add_argument('--freeze-backbone', action='store_true', default=True,
                        help='Freeze EfficientNet backbone')

    # Meta-classifier options
    parser.add_argument('--train-meta', action='store_true',
                        help='Train meta-classifier after temporal training')
    parser.add_argument('--meta-type', type=str, choices=['mlp', 'attention', 'gbm'],
                        default='mlp', help='Meta-classifier type')
    parser.add_argument('--meta-epochs', type=int, default=50, help='Meta-classifier epochs')
    parser.add_argument('--meta-batch-size', type=int, default=64, help='Meta-classifier batch size')
    parser.add_argument('--meta-lr', type=float, default=1e-3, help='Meta-classifier learning rate')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")

    target_size = tuple(args.target_size)
    model_name = f"{args.model_type}_temporal_{args.num_frames}f"

    print("="*60)
    print("TEMPORAL MODEL TRAINING")
    print("="*60)
    print(f"Video dir: {args.video_dir}")
    print(f"Model type: {args.model_type}")
    print(f"Frames: {args.num_frames}, Stride: {args.stride}")
    print(f"Target size: {target_size}")
    print(f"EfficientNet: {args.efficientnet_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"LR: {args.lr}")
    print(f"Device: {device}")
    print("="*60)

    # Load EfficientNet backbone
    print("\nLoading EfficientNet backbone...")
    efficientnet = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
    efficientnet_path = Path(args.efficientnet_path)
    if not efficientnet_path.is_absolute():
        efficientnet_path = Path(__file__).parent.parent / args.efficientnet_path

    if efficientnet_path.exists():
        checkpoint = torch.load(efficientnet_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            efficientnet.load_state_dict(checkpoint['model_state_dict'])
        else:
            efficientnet.load_state_dict(checkpoint)
        print(f"✓ Loaded EfficientNet from {efficientnet_path}")
    else:
        print(f"Warning: EfficientNet checkpoint not found at {efficientnet_path}")
        print("Using ImageNet pretrained weights")
        efficientnet = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)

    efficientnet = efficientnet.to(device)
    efficientnet.eval()

    # Freeze backbone
    for param in efficientnet.parameters():
        param.requires_grad = False

    # Create temporal model
    print(f"\nCreating {args.model_type} temporal model...")
    temporal_model = create_temporal_model(
        args.model_type,
        embed_dim=1280,
        num_layers=4 if args.model_type == 'transformer' else 2,
        num_heads=8,
        hidden_dim=512,
        num_classes=2,
        dropout=0.1,
        max_frames=args.num_frames
    )
    temporal_model = temporal_model.to(device)

    # Create dataloaders
    print("\nCreating dataloaders...")
    try:
        loaders = create_video_dataloaders(
            video_dir=args.video_dir,
            batch_size=args.batch_size,
            num_frames=args.num_frames,
            target_size=target_size,
            num_workers=args.num_workers,
            train_split=0.7,
            val_split=0.15
        )
        print(f"  Train: {len(loaders['train'].dataset)} samples")
        print(f"  Val: {len(loaders['val'].dataset)} samples")
        print(f"  Test: {len(loaders['test'].dataset)} samples")
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        print("Make sure video directory has real/ and fake/ subdirectories with video files")
        return

    # Train temporal model
    temporal_model, history = train_temporal_model(
        temporal_model, efficientnet,
        loaders['train'], loaders['val'],
        device, args, model_name
    )

    # Train meta-classifier for fusion
    if args.train_meta:
        meta_classifier = train_meta_fusion(
            temporal_model, efficientnet,
            frame_ensemble_probs=None,  # Would need per-frame ensemble outputs
            val_loader=loaders['val'],
            device=device,
            args=args
        )

        # Save meta-classifier
        meta_path = MODEL_DIR / f"{model_name}_meta_best.pth"
        from scripts.meta_classifier import save_meta_classifier_pytorch
        save_meta_classifier_pytorch(meta_classifier, str(meta_path))

    # Export temporal model to ONNX
    print("\nExporting temporal model to ONNX...")
    try:
        export_temporal_to_onnx(
            TemporalDetector(efficientnet, temporal_model),
            str(MODEL_DIR / f"{model_name}.onnx"),
            num_frames=args.num_frames,
            input_size=target_size[0],
            device=str(device)
        )
    except Exception as e:
        print(f"ONNX export failed (expected for dynamic seq len): {e}")
        print("Consider using TorchScript for temporal branch")

    print("\n" + "="*60)
    print("TEMPORAL TRAINING COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    main()