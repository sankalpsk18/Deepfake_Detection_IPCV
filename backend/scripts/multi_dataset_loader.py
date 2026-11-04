#!/usr/bin/env python3
"""
Multi-Dataset Loader for Deepfake Detection Training

Unified dataset loader supporting multiple datasets with balanced sampling
across generator families and cross-generator holdout validation.
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, WeightedRandomSampler, Subset
from PIL import Image
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""
    name: str
    path: str
    manipulation_types: List[str]
    generator_family: str
    weight: float
    splits: Dict[str, str]


@dataclass
class Sample:
    """A single sample with metadata."""
    path: Path
    label: int  # 0 = real, 1 = fake
    dataset_name: str
    generator_family: str
    manipulation_type: str
    split: str


class MultiDatasetLoader(Dataset):
    """
    Unified dataset loader for multi-dataset deepfake training.

    Features:
    - Loads multiple datasets with a common schema
    - Balanced sampling across generator families
    - Cross-generator holdout validation
    - Per-sample metadata for analysis
    """

    def __init__(
        self,
        config_path: str,
        split: str = 'train',
        transform=None,
        input_size: int = 224,
        holdout_generator_family: Optional[str] = None,
        sampling_strategy: str = 'balanced',
        min_samples_per_generator: int = 100
    ):
        """
        Initialize the multi-dataset loader.

        Args:
            config_path: Path to dataset_config.yaml
            split: 'train', 'val', or 'test'
            transform: Image transforms
            input_size: Model input size
            holdout_generator_family: Generator family to exclude from training
            sampling_strategy: 'balanced', 'weighted', or 'uniform'
            min_samples_per_generator: Minimum samples per generator family
        """
        self.config_path = Path(config_path)
        self.split = split
        self.transform = transform
        self.input_size = input_size
        self.holdout_generator_family = holdout_generator_family
        self.sampling_strategy = sampling_strategy
        self.min_samples_per_generator = min_samples_per_generator

        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Parse dataset configurations
        self.datasets = [DatasetConfig(**d) for d in self.config['datasets']]
        self.holdout_family = holdout_generator_family or self.config.get('holdout_generator_family')
        self.sampling_config = self.config.get('sampling', {})

        # Discover all samples
        self.samples = self._discover_samples()

        # Apply holdout if specified
        if self.holdout_family and self.split == 'train':
            self.samples = [s for s in self.samples if s.generator_family != self.holdout_family]
            print(f"Holdout: Excluded generator family '{self.holdout_family}' from training")

        # Compute sampling weights
        self.weights = self._compute_weights()

        print(f"Loaded {len(self.samples)} samples from {split} split")
        self._print_statistics()

    def _discover_samples(self) -> List[Sample]:
        """Discover all samples across all datasets."""
        samples = []

        for ds_config in self.datasets:
            ds_path = Path(ds_config.path)
            if not ds_path.exists():
                print(f"Warning: Dataset path not found: {ds_path}")
                continue

            # Get split directory
            split_dir = ds_config.splits.get(self.split)
            if not split_dir:
                print(f"Warning: Split '{self.split}' not defined for dataset {ds_config.name}")
                continue

            split_path = ds_path / split_dir
            if not split_path.exists():
                print(f"Warning: Split directory not found: {split_path}")
                continue

            # Load real samples
            real_dir = split_path / 'Real'
            if not real_dir.exists():
                real_dir = split_path / 'real'
            if real_dir.exists():
                for img_path in real_dir.glob('*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                        # Determine manipulation type for real images
                        manip_type = 'real'
                        samples.append(Sample(
                            path=img_path,
                            label=0,
                            dataset_name=ds_config.name,
                            generator_family=ds_config.generator_family,
                            manipulation_type=manip_type,
                            split=self.split
                        ))

            # Load fake samples
            fake_dir = split_path / 'Fake'
            if not fake_dir.exists():
                fake_dir = split_path / 'fake'
            if fake_dir.exists():
                for img_path in fake_dir.glob('*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                        # Assign manipulation type round-robin or from filename
                        manip_type = self._get_manipulation_type(img_path, ds_config.manipulation_types)
                        samples.append(Sample(
                            path=img_path,
                            label=1,
                            dataset_name=ds_config.name,
                            generator_family=ds_config.generator_family,
                            manipulation_type=manip_type,
                            split=self.split
                        ))

        return samples

    def _get_manipulation_type(self, img_path: Path, manipulation_types: List[str]) -> str:
        """Determine manipulation type from filename or assign round-robin."""
        # Try to extract from filename (e.g., fake_Deepfakes_001.jpg)
        for mtype in manipulation_types:
            if mtype.lower() in img_path.name.lower():
                return mtype
        # Default to first type
        return manipulation_types[0] if manipulation_types else 'unknown'

    def _compute_weights(self) -> List[float]:
        """Compute sampling weights based on strategy."""
        if not self.samples:
            return []

        if self.sampling_strategy == 'uniform':
            return [1.0] * len(self.samples)

        # Count samples per generator family
        family_counts = {}
        family_weights = {}
        for s in self.samples:
            family_counts[s.generator_family] = family_counts.get(s.generator_family, 0) + 1

        # Get dataset weights from config
        dataset_weights = {d.generator_family: d.weight for d in self.datasets}

        if self.sampling_strategy == 'balanced':
            # Inverse frequency weighting per generator family
            total_samples = len(self.samples)
            for family, count in family_counts.items():
                # Weight = dataset_weight / count (normalized)
                base_weight = dataset_weights.get(family, 1.0)
                family_weights[family] = base_weight / max(count, self.min_samples_per_generator)

            # Normalize weights
            weight_sum = sum(family_weights.values())
            family_weights = {k: v / weight_sum for k, v in family_weights.items()}

        elif self.sampling_strategy == 'weighted':
            # Use dataset weights directly, normalize per sample
            for family in family_counts:
                family_weights[family] = dataset_weights.get(family, 1.0)

        # Assign weights to samples
        weights = [family_weights.get(s.generator_family, 1.0) for s in self.samples]
        return weights

    def get_weighted_sampler(self) -> WeightedRandomSampler:
        """Get WeightedRandomSampler for balanced sampling."""
        if not self.weights:
            return None
        return WeightedRandomSampler(
            weights=self.weights,
            num_samples=len(self.samples),
            replacement=True
        )

    def get_holdout_split(self, holdout_family: str) -> Tuple['MultiDatasetLoader', 'MultiDatasetLoader']:
        """
        Split dataset into training (excluding holdout_family) and holdout validation.

        Returns:
            (train_dataset, holdout_dataset)
        """
        train_indices = [i for i, s in enumerate(self.samples) if s.generator_family != holdout_family]
        holdout_indices = [i for i, s in enumerate(self.samples) if s.generator_family == holdout_family]

        train_subset = Subset(self, train_indices)
        holdout_subset = Subset(self, holdout_indices)

        return train_subset, holdout_subset

    def get_generator_families(self) -> List[str]:
        """Get unique generator families in this dataset."""
        return list(set(s.generator_family for s in self.samples))

    def get_dataset_names(self) -> List[str]:
        """Get unique dataset names in this dataset."""
        return list(set(s.dataset_name for s in self.samples))

    def get_manipulation_types(self) -> List[str]:
        """Get unique manipulation types in this dataset."""
        return list(set(s.manipulation_type for s in self.samples))

    def _print_statistics(self):
        """Print dataset statistics."""
        if not self.samples:
            return

        # Per dataset
        dataset_counts = {}
        family_counts = {}
        manip_counts = {}
        label_counts = {0: 0, 1: 0}

        for s in self.samples:
            dataset_counts[s.dataset_name] = dataset_counts.get(s.dataset_name, 0) + 1
            family_counts[s.generator_family] = family_counts.get(s.generator_family, 0) + 1
            manip_counts[s.manipulation_type] = manip_counts.get(s.manipulation_type, 0) + 1
            label_counts[s.label] += 1

        print(f"  By dataset: {dataset_counts}")
        print(f"  By generator family: {family_counts}")
        print(f"  By manipulation type: {manip_counts}")
        print(f"  By label - Real: {label_counts[0]}, Fake: {label_counts[1]}")

        if self.holdout_family:
            holdout_count = sum(1 for s in self.samples if s.generator_family == self.holdout_family)
            print(f"  Holdout family '{self.holdout_family}' samples: {holdout_count}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        """Get a single sample with metadata."""
        sample = self.samples[idx]

        try:
            image = Image.open(sample.path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading {sample.path}: {e}")
            # Return black image as fallback
            if self.transform:
                dummy = Image.new('RGB', (self.input_size, self.input_size), (0, 0, 0))
                image = self.transform(dummy)
            else:
                image = torch.zeros(3, self.input_size, self.input_size)

        metadata = {
            'path': str(sample.path),
            'dataset': sample.dataset_name,
            'generator_family': sample.generator_family,
            'manipulation_type': sample.manipulation_type,
            'split': sample.split
        }

        return image, sample.label, metadata


def create_dataloaders(
    config_path: str,
    batch_size: int = 32,
    train_transform=None,
    val_transform=None,
    test_transform=None,
    input_size: int = 224,
    holdout_family: Optional[str] = None,
    sampling_strategy: str = 'balanced',
    num_workers: int = 4
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create train/val/test dataloaders from config.

    Returns dict with keys: 'train', 'val', 'test', 'holdout' (if applicable)
    """
    loaders = {}

    # Training loader with weighted sampling
    train_dataset = MultiDatasetLoader(
        config_path=config_path,
        split='train',
        transform=train_transform,
        input_size=input_size,
        holdout_generator_family=holdout_family,
        sampling_strategy=sampling_strategy
    )

    sampler = train_dataset.get_weighted_sampler()
    shuffle = sampler is None

    loaders['train'] = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    # Validation loader
    val_dataset = MultiDatasetLoader(
        config_path=config_path,
        split='val',
        transform=val_transform,
        input_size=input_size,
        holdout_generator_family=None,  # Don't apply holdout to val
        sampling_strategy='uniform'
    )

    loaders['val'] = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Test loader
    test_dataset = MultiDatasetLoader(
        config_path=config_path,
        split='test',
        transform=test_transform,
        input_size=input_size,
        holdout_generator_family=None,
        sampling_strategy='uniform'
    )

    loaders['test'] = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Holdout loader (if specified)
    if holdout_family:
        holdout_dataset = MultiDatasetLoader(
            config_path=config_path,
            split='test',  # Use test split for holdout evaluation
            transform=test_transform,
            input_size=input_size,
            holdout_generator_family=None,
            sampling_strategy='uniform'
        )
        # Filter to only holdout family
        holdout_indices = [i for i, s in enumerate(holdout_dataset.samples)
                          if s.generator_family == holdout_family]
        holdout_subset = torch.utils.data.Subset(holdout_dataset, holdout_indices)

        loaders['holdout'] = torch.utils.data.DataLoader(
            holdout_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        print(f"Holdout validation loader created for family '{holdout_family}' with {len(holdout_subset)} samples")

    return loaders


def get_albumentations_transforms(model_name: str, input_size: int, is_train: bool = True):
    """
    Get Albumentations transforms for the specified model.

    Args:
        model_name: 'efficientnet' or 'mesonet'
        input_size: Model input size
        is_train: Whether training transforms
    """
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


if __name__ == '__main__':
    # Test the loader
    config_path = Path(__file__).parent / 'dataset_config.yaml'
    if config_path.exists():
        dataset = MultiDatasetLoader(config_path, split='train')
        print(f"Total samples: {len(dataset)}")
        print(f"Families: {dataset.get_generator_families()}")

        # Test weighted sampler
        sampler = dataset.get_weighted_sampler()
        if sampler:
            print("Weighted sampler created successfully")
    else:
        print(f"Config not found: {config_path}")