#!/usr/bin/env python3
"""
Generator-Family Holdout Validation Protocol

Implements cross-generator generalization evaluation by holding out
one generator family from training and testing on it.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report
)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.multi_dataset_loader import MultiDatasetLoader, create_dataloaders, get_albumentations_transforms
from scripts.train import create_efficientnet_model, create_mesonet_model


def evaluate_on_family(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    family_name: str
) -> Dict[str, float]:
    """Evaluate model on a specific generator family."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels, metadata in tqdm(loader, desc=f"Evaluating {family_name}"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0

    cm = confusion_matrix(all_labels, all_preds)

    return {
        'family': family_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm.tolist(),
        'num_samples': len(all_labels)
    }


def run_holdout_validation(
    config_path: str,
    model_name: str = 'efficientnet',
    model_path: Optional[str] = None,
    holdout_family: Optional[str] = None,
    device: str = 'cpu',
    batch_size: int = 32
) -> Dict[str, Any]:
    """
    Run generator-family holdout validation.

    Args:
        config_path: Path to dataset_config.yaml
        model_name: 'efficientnet' or 'mesonet'
        model_path: Path to trained model checkpoint
        holdout_family: Generator family to hold out (from config if None)
        device: 'cpu' or 'cuda'
        batch_size: Batch size for evaluation

    Returns:
        Dictionary with holdout validation results
    """
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    # Load config to get holdout family
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    holdout_family = holdout_family or config.get('holdout_generator_family')
    if not holdout_family:
        raise ValueError("No holdout family specified in config or arguments")

    print(f"Holdout family: {holdout_family}")

    # Create model
    if model_name == 'efficientnet':
        model = create_efficientnet_model()
        input_size = 224
    elif model_name == 'mesonet':
        model = create_mesonet_model()
        input_size = 256
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)

    # Load checkpoint
    if model_path and Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded model from {model_path}")
    else:
        print("Warning: No model checkpoint provided, using random weights")

    model.eval()

    # Get transforms
    test_transform = get_albumentations_transforms(model_name, input_size, is_train=False)

    # Create test dataloader for all families
    test_dataset = MultiDatasetLoader(
        config_path=config_path,
        split='test',
        transform=test_transform,
        input_size=input_size,
        holdout_generator_family=None,
        sampling_strategy='uniform'
    )

    # Get all families
    families = test_dataset.get_generator_families()
    print(f"Generator families in test set: {families}")

    # Evaluate on each family separately
    results = {}
    for family in families:
        family_indices = [i for i, s in enumerate(test_dataset.samples)
                         if s.generator_family == family]
        family_subset = Subset(test_dataset, family_indices)
        family_loader = DataLoader(family_subset, batch_size=batch_size, shuffle=False, num_workers=4)

        print(f"\nEvaluating on {family} ({len(family_subset)} samples)...")
        metrics = evaluate_on_family(model, family_loader, device, family)
        results[family] = metrics
        print(f"  Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")

    # Special focus on holdout family
    holdout_metrics = results.get(holdout_family, {})
    in_domain_families = [f for f in families if f != holdout_family]

    if in_domain_families:
        in_domain_acc = np.mean([results[f]['accuracy'] for f in in_domain_families])
        in_domain_auc = np.mean([results[f]['auc'] for f in in_domain_families])
    else:
        in_domain_acc = 0
        in_domain_auc = 0

    summary = {
        'holdout_family': holdout_family,
        'model': model_name,
        'per_family_results': results,
        'holdout_metrics': holdout_metrics,
        'in_domain_avg_accuracy': in_domain_acc,
        'in_domain_avg_auc': in_domain_auc,
        'generalization_gap': in_domain_acc - holdout_metrics.get('accuracy', 0) if holdout_metrics else 0,
        'generalization_gap_auc': in_domain_auc - holdout_metrics.get('auc', 0) if holdout_metrics else 0
    }

    print("\n" + "="*60)
    print("HOLDOUT VALIDATION SUMMARY")
    print("="*60)
    print(f"Holdout family: {holdout_family}")
    print(f"  Accuracy: {holdout_metrics.get('accuracy', 0):.4f}")
    print(f"  AUC: {holdout_metrics.get('auc', 0):.4f}")
    print(f"  F1: {holdout_metrics.get('f1', 0):.4f}")
    print(f"In-domain average accuracy: {in_domain_acc:.4f}")
    print(f"In-domain average AUC: {in_domain_auc:.4f}")
    print(f"Generalization gap (acc): {summary['generalization_gap']:.4f}")
    print(f"Generalization gap (AUC): {summary['generalization_gap_auc']:.4f}")

    return summary


def compare_models_holdout(
    config_path: str,
    model_paths: Dict[str, str],  # {'efficientnet': path, 'mesonet': path}
    holdout_family: Optional[str] = None,
    device: str = 'cpu',
    batch_size: int = 32
) -> Dict[str, Any]:
    """Compare multiple models on holdout validation."""
    results = {}

    for model_name, model_path in model_paths.items():
        if Path(model_path).exists():
            print(f"\n{'='*60}")
            print(f"Evaluating {model_name}")
            print(f"{'='*60}")
            results[model_name] = run_holdout_validation(
                config_path=config_path,
                model_name=model_name,
                model_path=model_path,
                holdout_family=holdout_family,
                device=device,
                batch_size=batch_size
            )
        else:
            print(f"Warning: Model path not found for {model_name}: {model_path}")

    # Summary comparison
    print("\n" + "="*60)
    print("MODEL COMPARISON ON HOLDOUT FAMILY")
    print("="*60)
    for model_name, res in results.items():
        holdout = res.get('holdout_metrics', {})
        print(f"{model_name:15s}: Acc={holdout.get('accuracy', 0):.4f}, "
              f"AUC={holdout.get('auc', 0):.4f}, "
              f"F1={holdout.get('f1', 0):.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Generator-family holdout validation')
    parser.add_argument('--config', type=str, default='backend/scripts/dataset_config.yaml',
                        help='Path to dataset config')
    parser.add_argument('--model', type=str, choices=['efficientnet', 'mesonet', 'both'],
                        default='efficientnet', help='Model to evaluate')
    parser.add_argument('--model-path', type=str, help='Path to model checkpoint')
    parser.add_argument('--holdout-family', type=str, help='Generator family to hold out')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--output', type=str, help='Output JSON file for results')

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent.parent / args.config

    if args.model == 'both':
        model_paths = {
            'efficientnet': args.model_path or 'backend/models/efficientnet_deepfake_best.pth',
            'mesonet': args.model_path or 'backend/models/mesonet_deepfake_best.pth'
        }
        results = compare_models_holdout(
            config_path=str(config_path),
            model_paths=model_paths,
            holdout_family=args.holdout_family,
            device=args.device,
            batch_size=args.batch_size
        )
    else:
        results = run_holdout_validation(
            config_path=str(config_path),
            model_name=args.model,
            model_path=args.model_path,
            holdout_family=args.holdout_family,
            device=args.device,
            batch_size=args.batch_size
        )

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()