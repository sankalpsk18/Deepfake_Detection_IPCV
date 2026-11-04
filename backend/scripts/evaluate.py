#!/usr/bin/env python3
"""
Evaluate trained deepfake detection models on test set.

Usage:
    # Legacy single-dataset evaluation
    python scripts/evaluate.py --model efficientnet
    python scripts/evaluate.py --model mesonet
    python scripts/evaluate.py --model both --ensemble

    # Multi-dataset evaluation with config (Task 1)
    python scripts/evaluate.py --config dataset_config.yaml --model both --per-dataset --per-generator
"""

import os
import sys
import argparse
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc
)
import timm

# Import multi-dataset modules
try:
    from scripts.multi_dataset_loader import MultiDatasetLoader, get_albumentations_transforms
    MULTI_DATASET_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Multi-dataset modules not available: {e}")
    MULTI_DATASET_AVAILABLE = False

# Paths - dataset is in project root
DATA_ROOT = Path(__file__).parent.parent.parent / "dataset"
MODEL_DIR = Path(__file__).parent.parent / "models"


class DeepfakeDataset(Dataset):
    """Dataset for deepfake detection evaluation (legacy single-dataset)."""

    def __init__(self, root_dir, split='test', transform=None, model_input_size=224):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.model_input_size = model_input_size
        self.samples = []

        for label, class_name in enumerate(['Real', 'Fake']):
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
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
            return image, label, str(img_path)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            dummy = Image.new('RGB', (self.model_input_size, self.model_input_size), (0, 0, 0))
            if self.transform:
                return self.transform(dummy), label, str(img_path)
            return torch.zeros(3, self.model_input_size, self.model_input_size), label, str(img_path)


def get_transforms(model_name, input_size=224):
    """Get transforms for evaluation (legacy torchvision)."""
    if model_name == 'efficientnet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif model_name == 'mesonet':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def create_efficientnet_model(num_classes=2):
    """Create EfficientNet-B0 model."""
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes)
    return model


def create_mesonet_model(num_classes=2):
    """Create MesoNet model."""
    class MesoNet(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
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
            self.pool4 = nn.MaxPool2d(4, 4)

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


def load_model(model_name, device):
    """Load trained model from checkpoint."""
    if model_name == 'efficientnet':
        model = create_efficientnet_model()
        input_size = 224
        checkpoint_path = MODEL_DIR / "efficientnet_deepfake_best.pth"
    elif model_name == 'mesonet':
        model = create_mesonet_model()
        input_size = 256
        checkpoint_path = MODEL_DIR / "mesonet_deepfake_best.pth"
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return None, None

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✓ Loaded {model_name} from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Val AUC: {checkpoint.get('val_auc', 'unknown'):.4f}")
    print(f"  Val Acc: {checkpoint.get('val_acc', 'unknown'):.4f}")

    return model, input_size


def evaluate_model(model, loader, device, model_name):
    """Evaluate model on test set."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_paths = []

    with torch.no_grad():
        for images, labels, paths in tqdm(loader, desc=f"Evaluating {model_name}"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].detach().cpu().numpy())  # fake class probability
            all_paths.extend(paths)

    return np.array(all_labels), np.array(all_preds), np.array(all_probs), all_paths


def compute_metrics(labels, preds, probs):
    """Compute all evaluation metrics."""
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    try:
        auc_score = roc_auc_score(labels, probs)
    except:
        auc_score = 0.0

    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # same as recall
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    # EER (Equal Error Rate) - approximate
    fpr_vals, tpr_vals, thresholds = roc_curve(labels, probs)
    eer_idx = np.argmin(np.abs(fpr_vals - (1 - tpr_vals)))
    eer = fpr_vals[eer_idx]

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_score,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'fpr': fpr,
        'fnr': fnr,
        'eer': eer,
        'confusion_matrix': cm.tolist(),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }


def evaluate_ensemble(efficientnet_model, mesonet_model, device, eff_loader, mes_loader):
    """Evaluate ensemble of both models."""
    print("\n" + "="*60)
    print("ENSEMBLE EVALUATION (70% EfficientNet + 30% MesoNet)")
    print("="*60)

    efficientnet_model.eval()
    mesonet_model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        # We need to iterate through both loaders in parallel
        # Since batch sizes might differ, let's re-create loaders with same batch size
        pass  # We'll do this differently - collect all probs first

    # Actually, let's just run both models and collect probs separately
    # Then combine them

    # This is handled in main() by collecting probs from each model

    return None


def print_results(model_name, metrics):
    """Print formatted evaluation results."""
    print(f"\n{'='*60}")
    print(f"RESULTS: {model_name.upper()}")
    print(f"{'='*60}")
    print(f"Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision:    {metrics['precision']:.4f}")
    print(f"Recall:       {metrics['recall']:.4f}")
    print(f"F1-Score:     {metrics['f1']:.4f}")
    print(f"AUC-ROC:      {metrics['auc']:.4f}")
    print(f"Specificity:  {metrics['specificity']:.4f}")
    print(f"Sensitivity:  {metrics['sensitivity']:.4f}")
    print(f"FPR:          {metrics['fpr']:.4f}")
    print(f"FNR:          {metrics['fnr']:.4f}")
    print(f"EER:          {metrics['eer']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {metrics['tn']:5d}  FP: {metrics['fp']:5d}")
    print(f"  FN: {metrics['fn']:5d}  TP: {metrics['tp']:5d}")


def compute_metrics_by_group(labels, preds, probs, groups, group_name):
    """Compute metrics broken down by a grouping (dataset, generator family, etc.)."""
    unique_groups = np.unique(groups)
    group_metrics = {}

    for group in unique_groups:
        mask = groups == group
        if mask.sum() < 2:
            continue
        group_labels = labels[mask]
        group_preds = preds[mask]
        group_probs = probs[mask]

        if len(np.unique(group_labels)) < 2:
            continue

        accuracy = accuracy_score(group_labels, group_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(group_labels, group_preds, average='binary')
        try:
            auc_score = roc_auc_score(group_labels, group_probs)
        except:
            auc_score = 0.0

        group_metrics[group] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc_score,
            'num_samples': len(group_labels),
            'real_count': int((group_labels == 0).sum()),
            'fake_count': int((group_labels == 1).sum()),
        }

    return group_metrics


def evaluate_model_with_metadata(model, loader, device, model_name):
    """Evaluate model and return predictions with metadata."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_paths = []
    all_metadata = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {model_name}"):
            if len(batch) == 3:
                images, labels, paths = batch
                metadata_list = [{}] * len(labels)
            else:
                images, labels, metadata_list = batch
                paths = [m.get('path', '') for m in metadata_list]

            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].detach().cpu().numpy())
            all_paths.extend(paths)
            all_metadata.extend(metadata_list)

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
        all_paths,
        all_metadata
    )


def print_group_metrics(group_metrics, group_name):
    """Print metrics broken down by group."""
    print(f"\n--- {group_name} Breakdown ---")
    for group, metrics in sorted(group_metrics.items()):
        print(f"  {group:20s}: Acc={metrics['accuracy']:.4f}, AUC={metrics['auc']:.4f}, "
              f"F1={metrics['f1']:.4f}, N={metrics['num_samples']} "
              f"(R:{metrics['real_count']}/F:{metrics['fake_count']})")


def evaluate_with_config(model, config_path, model_name, device, batch_size, num_workers,
                         per_dataset=False, per_generator=False):
    """Evaluate model using multi-dataset config with optional per-group breakdown."""
    if not MULTI_DATASET_AVAILABLE:
        print("Multi-dataset modules not available")
        return None

    # Get transform
    input_size = 224 if model_name == 'efficientnet' else 256
    transform = get_albumentations_transforms(model_name, input_size, is_train=False)

    # Load test dataset
    test_dataset = MultiDatasetLoader(
        config_path=config_path,
        split='test',
        transform=transform,
        input_size=input_size,
        holdout_generator_family=None,
        sampling_strategy='uniform'
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Evaluate
    labels, preds, probs, paths, metadata = evaluate_model_with_metadata(
        model, test_loader, device, model_name
    )

    # Overall metrics
    overall_metrics = compute_metrics(labels, preds, probs)

    results = {
        'overall': overall_metrics,
        'per_dataset': None,
        'per_generator': None,
        'per_manipulation': None
    }

    if per_dataset:
        dataset_names = [m.get('dataset', 'unknown') for m in metadata]
        results['per_dataset'] = compute_metrics_by_group(
            labels, preds, probs, np.array(dataset_names), 'Dataset'
        )
        print_group_metrics(results['per_dataset'], 'Dataset')

    if per_generator:
        generator_families = [m.get('generator_family', 'unknown') for m in metadata]
        results['per_generator'] = compute_metrics_by_group(
            labels, preds, probs, np.array(generator_families), 'Generator Family'
        )
        print_group_metrics(results['per_generator'], 'Generator Family')

    manipulation_types = [m.get('manipulation_type', 'unknown') for m in metadata]
    results['per_manipulation'] = compute_metrics_by_group(
        labels, preds, probs, np.array(manipulation_types), 'Manipulation Type'
    )
    print_group_metrics(results['per_manipulation'], 'Manipulation Type')

    return results


def save_results(model_name, metrics, output_path):
    """Save results to JSON."""
    results = {
        'model': model_name,
        'metrics': metrics
    }
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate deepfake detection models')
    parser.add_argument('--model', type=str, choices=['efficientnet', 'mesonet', 'both'],
                        default='both', help='Model to evaluate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data workers')
    parser.add_argument('--cpu', action='store_true', help='Force CPU evaluation')
    parser.add_argument('--ensemble-weights', type=str, default='0.7,0.3',
                        help='Ensemble weights (eff,mes)')

    # Multi-dataset evaluation options (Task 1)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to dataset config YAML (enables multi-dataset evaluation)')
    parser.add_argument('--per-dataset', action='store_true',
                        help='Report per-dataset metrics (requires --config)')
    parser.add_argument('--per-generator', action='store_true',
                        help='Report per-generator-family metrics (requires --config)')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")

    # Parse ensemble weights
    eff_weight, mes_weight = map(float, args.ensemble_weights.split(','))
    total = eff_weight + mes_weight
    eff_weight, mes_weight = eff_weight / total, mes_weight / total
    print(f"Ensemble weights: EfficientNet={eff_weight:.2f}, MesoNet={mes_weight:.2f}")

    results = {}

    # Multi-dataset evaluation mode
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = Path(__file__).parent.parent / args.config

        print(f"\nMulti-dataset evaluation using config: {config_path}")
        print(f"  Per-dataset breakdown: {'enabled' if args.per_dataset else 'disabled'}")
        print(f"  Per-generator breakdown: {'enabled' if args.per_generator else 'disabled'}")

        # Evaluate EfficientNet
        if args.model in ['efficientnet', 'both']:
            print("\n" + "="*60)
            print("EVALUATING EFFICIENTNET-B0")
            print("="*60)

            model, input_size = load_model('efficientnet', device)
            if model is not None:
                eval_results = evaluate_with_config(
                    model, str(config_path), 'efficientnet', device,
                    args.batch_size, args.num_workers,
                    per_dataset=args.per_dataset,
                    per_generator=args.per_generator
                )
                if eval_results:
                    print_results('EfficientNet-B0', eval_results['overall'])
                    save_results('efficientnet', eval_results, MODEL_DIR / 'efficientnet_test_results.json')
                    results['efficientnet'] = eval_results

        # Evaluate MesoNet
        if args.model in ['mesonet', 'both']:
            print("\n" + "="*60)
            print("EVALUATING MESONET")
            print("="*60)

            model, input_size = load_model('mesonet', device)
            if model is not None:
                eval_results = evaluate_with_config(
                    model, str(config_path), 'mesonet', device,
                    args.batch_size, args.num_workers,
                    per_dataset=args.per_dataset,
                    per_generator=args.per_generator
                )
                if eval_results:
                    print_results('MesoNet', eval_results['overall'])
                    save_results('mesonet', eval_results, MODEL_DIR / 'mesonet_test_results.json')
                    results['mesonet'] = eval_results

        # Evaluate Ensemble
        if args.model == 'both' and 'efficientnet' in results and 'mesonet' in results:
            print("\n" + "="*60)
            print("ENSEMBLE EVALUATION")
            print("="*60)

            eff_probs = results['efficientnet']['overall']['probs'] if 'probs' in results['efficientnet']['overall'] else None
            # For ensemble, we need to re-evaluate both models on same data
            # This is simplified - in practice, use the saved probs from individual evaluations
            print("Note: Ensemble evaluation requires both models evaluated on identical data.")
            print("Run with --config and both models will be evaluated on same multi-dataset test set.")

    else:
        # Legacy single-dataset evaluation mode
        # Evaluate EfficientNet
        if args.model in ['efficientnet', 'both']:
            print("\n" + "="*60)
            print("EVALUATING EFFICIENTNET-B0")
            print("="*60)

            model, input_size = load_model('efficientnet', device)
            if model is not None:
                transform = get_transforms('efficientnet', input_size)
                test_dataset = DeepfakeDataset(DATA_ROOT, 'Test', transform, input_size)
                test_loader = DataLoader(
                    test_dataset, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True
                )

                labels, preds, probs, paths = evaluate_model(model, test_loader, device, 'efficientnet')
                metrics = compute_metrics(labels, preds, probs)
                print_results('EfficientNet-B0', metrics)
                save_results('efficientnet', metrics, MODEL_DIR / 'efficientnet_test_results.json')
                results['efficientnet'] = {'labels': labels, 'preds': preds, 'probs': probs, 'metrics': metrics}

        # Evaluate MesoNet
        if args.model in ['mesonet', 'both']:
            print("\n" + "="*60)
            print("EVALUATING MESONET")
            print("="*60)

            model, input_size = load_model('mesonet', device)
            if model is not None:
                transform = get_transforms('mesonet', input_size)
                test_dataset = DeepfakeDataset(DATA_ROOT, 'Test', transform, input_size)
                test_loader = DataLoader(
                    test_dataset, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True
                )

                labels, preds, probs, paths = evaluate_model(model, test_loader, device, 'mesonet')
                metrics = compute_metrics(labels, preds, probs)
                print_results('MesoNet', metrics)
                save_results('mesonet', metrics, MODEL_DIR / 'mesonet_test_results.json')
                results['mesonet'] = {'labels': labels, 'preds': preds, 'probs': probs, 'metrics': metrics}

        # Evaluate Ensemble
        if args.model == 'both' and 'efficientnet' in results and 'mesonet' in results:
            print("\n" + "="*60)
            print("ENSEMBLE EVALUATION")
            print("="*60)

            # Both models evaluated on same test set (same order since no shuffle)
            eff_probs = results['efficientnet']['probs']
            mes_probs = results['mesonet']['probs']
            labels = results['efficientnet']['labels']

            # Weighted ensemble
            ensemble_probs = eff_weight * eff_probs + mes_weight * mes_probs
            ensemble_preds = (ensemble_probs > 0.5).astype(int)

            metrics = compute_metrics(labels, ensemble_preds, ensemble_probs)
            print_results(f'Ensemble ({eff_weight:.0%}/{mes_weight:.0%})', metrics)
            save_results('ensemble', metrics, MODEL_DIR / 'ensemble_test_results.json')

            # Also try different thresholds
            print("\n--- Threshold Analysis ---")
            for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
                thresh_preds = (ensemble_probs > thresh).astype(int)
                acc = accuracy_score(labels, thresh_preds)
                prec, rec, f1, _ = precision_recall_fscore_support(labels, thresh_preds, average='binary')
                print(f"  Threshold {thresh:.1f}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")

            # Per-class analysis
            print("\n--- Per-Class Analysis ---")
            real_mask = labels == 0
            fake_mask = labels == 1
            print(f"Real samples:  {real_mask.sum()} - Mean prob(fake): {ensemble_probs[real_mask].mean():.4f}")
            print(f"Fake samples:  {fake_mask.sum()} - Mean prob(fake): {ensemble_probs[fake_mask].mean():.4f}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for model_key in ['efficientnet', 'mesonet', 'ensemble']:
        if model_key in results:
            if 'overall' in results[model_key]:
                m = results[model_key]['overall']
            else:
                m = results[model_key]['metrics'] if model_key != 'ensemble' else metrics
            print(f"  {model_key:15s}: Acc={m['accuracy']:.4f}, F1={m['f1']:.4f}, AUC={m['auc']:.4f}")


if __name__ == '__main__':
    main()