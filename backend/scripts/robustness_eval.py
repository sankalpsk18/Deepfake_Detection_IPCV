#!/usr/bin/env python3
"""
Robustness Evaluation Module (Task 2)

Evaluates model accuracy as a function of degradation severity.
Generates accuracy-vs-severity curves for production readiness assessment.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.multi_dataset_loader import MultiDatasetLoader, get_albumentations_transforms
from scripts.degradation_augmentations import (
    create_fixed_severity_augmentation,
    apply_degradation_to_image,
    get_default_degradation_config
)
from scripts.train import create_efficientnet_model, create_mesonet_model


class RobustnessEvaluator:
    """
    Evaluates model robustness across degradation severities.
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        device: torch.device,
        input_size: int
    ):
        self.model = model
        self.model_name = model_name
        self.device = device
        self.input_size = input_size
        self.model.eval()

    def evaluate_on_loader(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on a dataloader."""
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels, _ in tqdm(loader, desc="Evaluating", leave=False):
                images = images.to(self.device)
                outputs = self.model(images)
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
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm.tolist(),
            'num_samples': len(all_labels)
        }

    def evaluate_severity(
        self,
        dataset: MultiDatasetLoader,
        severity: str,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Evaluate model on dataset with fixed severity degradation.

        Args:
            dataset: Base dataset (test split)
            severity: 'mild', 'moderate', 'severe'
            batch_size: Batch size

        Returns:
            Metrics dictionary
        """
        # Create fixed severity transform
        transform = create_fixed_severity_augmentation(
            severity, self.model_name, self.input_size
        )

        # Create degraded dataset
        degraded_dataset = MultiDatasetLoader(
            config_path=dataset.config_path,
            split='test',
            transform=transform,
            input_size=self.input_size,
            holdout_generator_family=None,
            sampling_strategy='uniform'
        )

        loader = DataLoader(
            degraded_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        return self.evaluate_on_loader(loader)

    def run_full_robustness_eval(
        self,
        dataset: MultiDatasetLoader,
        severity_levels: List[str] = ['clean', 'mild', 'moderate', 'severe'],
        batch_size: int = 32
    ) -> Dict[str, Dict[str, float]]:
        """
        Run complete robustness evaluation across severity levels.

        Args:
            dataset: Base test dataset
            severity_levels: List of severity levels ('clean' = no degradation)
            batch_size: Batch size

        Returns:
            Dict mapping severity -> metrics
        """
        results = {}

        for severity in severity_levels:
            print(f"\nEvaluating {severity} severity...")

            if severity == 'clean':
                # Use original dataset without degradation
                transform = get_albumentations_transforms(
                    self.model_name, self.input_size, is_train=False
                )
                clean_dataset = MultiDatasetLoader(
                    config_path=dataset.config_path,
                    split='test',
                    transform=transform,
                    input_size=self.input_size,
                    holdout_generator_family=None,
                    sampling_strategy='uniform'
                )
                loader = DataLoader(clean_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            else:
                loader = DataLoader(
                    MultiDatasetLoader(
                        config_path=dataset.config_path,
                        split='test',
                        transform=create_fixed_severity_augmentation(severity, self.model_name, self.input_size),
                        input_size=self.input_size,
                        holdout_generator_family=None,
                        sampling_strategy='uniform'
                    ),
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=4
                )

            metrics = self.evaluate_on_loader(loader)
            results[severity] = metrics
            print(f"  {severity}: Acc={metrics['accuracy']:.4f}, AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}")

        return results


def run_robustness_evaluation(
    config_path: str,
    model_path: str,
    model_name: str = 'efficientnet',
    device: str = 'cpu',
    batch_size: int = 32,
    severity_levels: List[str] = None,
    output_path: Optional[str] = None,
    plot_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run complete robustness evaluation for a trained model.

    Args:
        config_path: Path to dataset_config.yaml
        model_path: Path to trained model checkpoint
        model_name: 'efficientnet' or 'mesonet'
        device: 'cpu' or 'cuda'
        batch_size: Batch size
        severity_levels: List of severities to test
        output_path: Optional JSON output path
        plot_path: Optional plot output path

    Returns:
        Evaluation results dictionary
    """
    if severity_levels is None:
        severity_levels = ['clean', 'mild', 'moderate', 'severe']

    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    # Load model
    if model_name == 'efficientnet':
        model = create_efficientnet_model()
        input_size = 224
    elif model_name == 'mesonet':
        model = create_mesonet_model()
        input_size = 256
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)

    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model not found at {model_path}, using random weights")

    model.eval()

    # Create base test dataset (clean)
    transform = get_albumentations_transforms(model_name, input_size, is_train=False)
    test_dataset = MultiDatasetLoader(
        config_path=config_path,
        split='test',
        transform=transform,
        input_size=input_size,
        holdout_generator_family=None,
        sampling_strategy='uniform'
    )

    # Create evaluator
    evaluator = RobustnessEvaluator(model, model_name, device, input_size)

    # Run evaluation
    results = evaluator.run_full_robustness_eval(
        test_dataset, severity_levels, batch_size
    )

    # Add metadata
    results['metadata'] = {
        'model': model_name,
        'model_path': model_path,
        'config': config_path,
        'severity_levels': severity_levels,
        'test_samples': len(test_dataset)
    }

    # Print summary
    print("\n" + "="*60)
    print("ROBUSTNESS EVALUATION SUMMARY")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Test samples: {len(test_dataset)}")
    print()
    for sev in severity_levels:
        m = results[sev]
        print(f"  {sev:10s}: Acc={m['accuracy']:.4f} ({m['accuracy']*100:.1f}%), "
              f"AUC={m['auc']:.4f}, F1={m['f1']:.4f}")

    # Calculate degradation drops
    if 'clean' in results:
        clean_acc = results['clean']['accuracy']
        clean_auc = results['clean']['auc']
        print(f"\nDegradation drops (relative to clean):")
        for sev in ['mild', 'moderate', 'severe']:
            if sev in results:
                acc_drop = clean_acc - results[sev]['accuracy']
                auc_drop = clean_auc - results[sev]['auc']
                print(f"  {sev:10s}: ΔAcc={acc_drop:.4f} ({acc_drop*100:.1f}%), "
                      f"ΔAUC={auc_drop:.4f}")

    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    # Generate plot
    if plot_path:
        plot_robustness_curve(results, severity_levels, plot_path)
        print(f"Plot saved to {plot_path}")

    return results


def plot_robustness_curve(
    results: Dict[str, Dict[str, float]],
    severity_levels: List[str],
    output_path: str
):
    """Generate accuracy vs severity plot."""
    # Filter to actual severity levels in results
    available_severities = [s for s in severity_levels if s in results]
    if not available_severities:
        return

    x_pos = list(range(len(available_severities)))
    accuracies = [results[s]['accuracy'] for s in available_severities]
    aucs = [results[s]['auc'] for s in available_severities]
    f1s = [results[s]['f1'] for s in available_severities]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy plot
    ax1.plot(x_pos, accuracies, 'o-', color='#2563eb', linewidth=2, markersize=8, label='Accuracy')
    ax1.set_xlabel('Degradation Severity')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy vs Degradation Severity')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(available_severities)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    ax1.legend()

    # Add value labels
    for i, acc in enumerate(accuracies):
        ax1.annotate(f'{acc:.3f}', (x_pos[i], acc), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9)

    # AUC plot
    ax2.plot(x_pos, aucs, 's-', color='#16a34a', linewidth=2, markersize=8, label='AUC-ROC')
    ax2.plot(x_pos, f1s, '^-', color='#f59e0b', linewidth=2, markersize=8, label='F1-Score')
    ax2.set_xlabel('Degradation Severity')
    ax2.set_ylabel('Score')
    ax2.set_title('AUC & F1 vs Degradation Severity')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(available_severities)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def compare_models_robustness(
    config_path: str,
    model_paths: Dict[str, str],  # {'efficientnet': path, 'mesonet': path}
    device: str = 'cpu',
    batch_size: int = 32,
    output_path: Optional[str] = None,
    plot_path: Optional[str] = None
) -> Dict[str, Any]:
    """Compare robustness of multiple models."""
    all_results = {}

    for model_name, model_path in model_paths.items():
        if Path(model_path).exists():
            print(f"\n{'='*60}")
            print(f"Evaluating {model_name} robustness")
            print(f"{'='*60}")
            results = run_robustness_evaluation(
                config_path=config_path,
                model_path=model_path,
                model_name=model_name,
                device=device,
                batch_size=batch_size
            )
            all_results[model_name] = results
        else:
            print(f"Warning: Model not found: {model_path}")

    # Comparison plot
    if plot_path and all_results:
        plot_model_comparison(all_results, plot_path)

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nComparison results saved to {output_path}")

    return all_results


def plot_model_comparison(
    all_results: Dict[str, Dict],
    output_path: str
):
    """Plot comparison of multiple models' robustness curves."""
    severity_levels = ['clean', 'mild', 'moderate', 'severe']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = {'efficientnet': '#2563eb', 'mesonet': '#16a34a'}

    for model_name, results in all_results.items():
        color = colors.get(model_name, '#6366f1')
        available = [s for s in severity_levels if s in results]
        x_pos = list(range(len(available)))
        accuracies = [results[s]['accuracy'] for s in available]
        aucs = [results[s]['auc'] for s in available]

        ax1.plot(x_pos, accuracies, 'o-', color=color, linewidth=2, markersize=8,
                 label=f'{model_name} (Acc)')
        ax2.plot(x_pos, aucs, 's-', color=color, linewidth=2, markersize=8,
                 label=f'{model_name} (AUC)')

    ax1.set_xlabel('Degradation Severity')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Comparison: Accuracy vs Severity')
    ax1.set_xticks(range(len(severity_levels)))
    ax1.set_xticklabels(severity_levels)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    ax1.legend()

    ax2.set_xlabel('Degradation Severity')
    ax2.set_ylabel('AUC-ROC')
    ax2.set_title('Model Comparison: AUC vs Severity')
    ax2.set_xticks(range(len(severity_levels)))
    ax2.set_xticklabels(severity_levels)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Robustness evaluation across degradation severities')
    parser.add_argument('--config', type=str, default='backend/scripts/dataset_config.yaml',
                        help='Path to dataset config')
    parser.add_argument('--model', type=str, choices=['efficientnet', 'mesonet', 'both'],
                        default='efficientnet', help='Model to evaluate')
    parser.add_argument('--model-path', type=str, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--severities', type=str, nargs='+',
                        default=['clean', 'mild', 'moderate', 'severe'],
                        help='Severity levels to test')
    parser.add_argument('--output', type=str, help='Output JSON file')
    parser.add_argument('--plot', type=str, help='Output plot file (PNG)')

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent.parent / args.config

    if args.model == 'both':
        model_paths = {
            'efficientnet': args.model_path or 'backend/models/efficientnet_deepfake_best.pth',
            'mesonet': args.model_path or 'backend/models/mesonet_deepfake_best.pth'
        }
        compare_models_robustness(
            config_path=str(config_path),
            model_paths=model_paths,
            device=args.device,
            batch_size=args.batch_size,
            output_path=args.output,
            plot_path=args.plot
        )
    else:
        run_robustness_evaluation(
            config_path=str(config_path),
            model_path=args.model_path or f'backend/models/{args.model}_deepfake_best.pth',
            model_name=args.model,
            device=args.device,
            batch_size=args.batch_size,
            severity_levels=args.severities,
            output_path=args.output,
            plot_path=args.plot
        )


if __name__ == '__main__':
    main()