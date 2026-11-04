#!/usr/bin/env python3
"""
Meta-Classifier for Fusion of Temporal + Per-Frame Predictions (Task 3)

Implements learned fusion of per-frame ensemble predictions and temporal model
predictions using a trainable meta-classifier.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class MetaClassifier(nn.Module):
    """
    Meta-classifier for fusing per-frame and temporal predictions.

    Takes concatenated probabilities from:
    - Per-frame ensemble: [frame_real_prob, frame_fake_prob]
    - Temporal model: [temporal_real_prob, temporal_fake_prob]

    And outputs final video-level prediction.
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dims: List[int] = [32, 16],
        num_classes: int = 2,
        dropout: float = 0.2,
        activation: str = 'relu'
    ):
        """
        Initialize meta-classifier.

        Args:
            input_dim: Input dimension (4 = 2 probs from frame + 2 probs from temporal)
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of output classes
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu', 'tanh')
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Build activation
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'gelu':
            act_fn = nn.GELU
        elif activation == 'tanh':
            act_fn = nn.Tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build MLP
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, frame_probs: torch.Tensor, temporal_probs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            frame_probs: Per-frame ensemble probabilities (B, 2) or (B, T, 2)
            temporal_probs: Temporal model probabilities (B, 2)

        Returns:
            Fused logits (B, num_classes)
        """
        # Handle per-frame probs - average over time if needed
        if frame_probs.dim() == 3:
            # (B, T, 2) -> average over frames -> (B, 2)
            frame_probs = frame_probs.mean(dim=1)

        # Ensure temporal_probs is (B, 2)
        if temporal_probs.dim() == 3:
            temporal_probs = temporal_probs.mean(dim=1)

        # Concatenate: [frame_real, frame_fake, temporal_real, temporal_fake]
        fused_input = torch.cat([frame_probs, temporal_probs], dim=-1)

        # Forward through MLP
        logits = self.network(fused_input)
        return logits


class AttentionMetaClassifier(nn.Module):
    """
    Attention-based meta-classifier that learns to weight frame vs temporal
    predictions based on their confidence/reliability.
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()

        self.frame_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.temporal_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Cross-attention between frame and temporal features
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, frame_probs: torch.Tensor, temporal_probs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with cross-attention fusion.

        Args:
            frame_probs: (B, 2) or (B, T, 2)
            temporal_probs: (B, 2)

        Returns:
            Fused logits (B, num_classes)
        """
        B = frame_probs.size(0)

        # Handle per-frame probs
        if frame_probs.dim() == 3:
            # (B, T, 2) -> (B, T, hidden)
            frame_feats = self.frame_encoder(frame_probs)
            # Average over frames for sequence representation
            frame_feat = frame_feats.mean(dim=1)  # (B, hidden)
        else:
            frame_feat = self.frame_encoder(frame_probs)  # (B, hidden)

        temporal_feat = self.temporal_encoder(temporal_probs)  # (B, hidden)

        # Cross-attention: frame attends to temporal, temporal attends to frame
        frame_attended, _ = self.cross_attention(
            frame_feat.unsqueeze(1),  # (B, 1, hidden)
            temporal_feat.unsqueeze(1),  # (B, 1, hidden)
            temporal_feat.unsqueeze(1)
        )
        temporal_attended, _ = self.cross_attention(
            temporal_feat.unsqueeze(1),
            frame_feat.unsqueeze(1),
            frame_feat.unsqueeze(1)
        )

        frame_attended = frame_attended.squeeze(1)  # (B, hidden)
        temporal_attended = temporal_attended.squeeze(1)

        # Fuse
        fused = torch.cat([frame_attended, temporal_attended], dim=-1)
        logits = self.fusion(fused)

        return logits


class GradientBoostingMetaClassifier:
    """
    Scikit-learn Gradient Boosting meta-classifier for production deployment.

    Can be trained on validation set outputs and exported for fast inference
    without PyTorch dependency.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        random_state: int = 42
    ):
        from sklearn.ensemble import GradientBoostingClassifier

        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )
        self.is_fitted = False

    def train(
        self,
        frame_probs: np.ndarray,
        temporal_probs: np.ndarray,
        labels: np.ndarray
    ):
        """
        Train on validation set.

        Args:
            frame_probs: (N, 2) per-frame ensemble probabilities
            temporal_probs: (N, 2) temporal model probabilities
            labels: (N,) ground truth labels (0 or 1)
        """
        # Concatenate features
        X = np.concatenate([frame_probs, temporal_probs], axis=1)
        self.model.fit(X, labels)
        self.is_fitted = True
        logger.info(f"Meta-classifier trained on {len(labels)} samples")

    def predict_proba(
        self,
        frame_probs: np.ndarray,
        temporal_probs: np.ndarray
    ) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() first.")

        X = np.concatenate([frame_probs, temporal_probs], axis=1)
        return self.model.predict_proba(X)

    def predict(
        self,
        frame_probs: np.ndarray,
        temporal_probs: np.ndarray
    ) -> np.ndarray:
        """Predict class labels."""
        return self.predict_proba(frame_probs, temporal_probs).argmax(axis=1)

    def save(self, path: str):
        """Save model to disk."""
        import joblib
        joblib.dump(self.model, path)
        logger.info(f"Meta-classifier saved to {path}")

    @classmethod
    def load(cls, path: str):
        """Load model from disk."""
        import joblib
        model = joblib.load(path)
        instance = cls()
        instance.model = model
        instance.is_fitted = True
        return instance


def train_meta_classifier(
    frame_probs: np.ndarray,
    temporal_probs: np.ndarray,
    labels: np.ndarray,
    model_type: str = 'mlp',  # 'mlp', 'attention', 'gbm'
    device: str = 'cpu',
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_split: float = 0.2
) -> Any:
    """
    Train a meta-classifier on validation outputs.

    Args:
        frame_probs: (N, 2) per-frame ensemble probabilities
        temporal_probs: (N, 2) temporal model probabilities
        labels: (N,) ground truth labels
        model_type: 'mlp', 'attention', or 'gbm'
        device: Training device
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        val_split: Validation split fraction

    Returns:
        Trained meta-classifier
    """
    device = torch.device(device)

    # Prepare data
    X = np.concatenate([frame_probs, temporal_probs], axis=1)
    y = labels

    # Train/val split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=42, stratify=y
    )

    if model_type == 'gbm':
        # Gradient Boosting (scikit-learn)
        meta = GradientBoostingMetaClassifier()
        meta.train(frame_probs, temporal_probs, labels)
        return meta

    # PyTorch models
    if model_type == 'mlp':
        model = MetaClassifier(input_dim=4)
    elif model_type == 'attention':
        model = AttentionMetaClassifier(input_dim=4)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = model.to(device)

    # Convert to tensors
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.LongTensor(y_train).to(device)
    X_val = torch.FloatTensor(X_val).to(device)
    y_val = torch.LongTensor(y_val).to(device)

    # Split input features
    frame_train = X_train[:, :2]
    temporal_train = X_train[:, 2:]
    frame_val = X_val[:, :2]
    temporal_val = X_val[:, 2:]

    # Data loaders
    train_dataset = torch.utils.data.TensorDataset(frame_train, temporal_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(frame_val, temporal_val, y_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for frame_b, temporal_b, y_b in train_loader:
            optimizer.zero_grad()
            logits = model(frame_b, temporal_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (logits.argmax(dim=1) == y_b).sum().item()
            train_total += y_b.size(0)

        scheduler.step()

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for frame_b, temporal_b, y_b in val_loader:
                logits = model(frame_b, temporal_b)
                val_correct += (logits.argmax(dim=1) == y_b).sum().item()
                val_total += y_b.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

    # Load best model
    model.load_state_dict(best_state)
    logger.info(f"Training complete. Best Val Acc: {best_val_acc:.4f}")

    return model


def evaluate_fusion(
    meta_classifier: Any,
    frame_probs: np.ndarray,
    temporal_probs: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate fusion performance.

    Args:
        meta_classifier: Trained meta-classifier
        frame_probs: (N, 2) frame probabilities
        temporal_probs: (N, 2) temporal probabilities
        labels: (N,) ground truth labels

    Returns:
        Dictionary with metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        roc_auc_score, confusion_matrix
    )

    # Get predictions
    if isinstance(meta_classifier, GradientBoostingMetaClassifier):
        fused_probs = meta_classifier.predict_proba(frame_probs, temporal_probs)
        fused_preds = meta_classifier.predict(frame_probs, temporal_probs)
    else:
        # PyTorch model
        device = next(meta_classifier.parameters()).device
        meta_classifier.eval()
        with torch.no_grad():
            frame_t = torch.FloatTensor(frame_probs).to(device)
            temporal_t = torch.FloatTensor(temporal_probs).to(device)
            logits = meta_classifier(frame_t, temporal_t)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            fused_probs = probs
            fused_preds = probs.argmax(axis=1)

    # Compute metrics
    accuracy = accuracy_score(labels, fused_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, fused_preds, average='binary')
    try:
        auc = roc_auc_score(labels, fused_probs[:, 1])
    except:
        auc = 0.0

    cm = confusion_matrix(labels, fused_preds)
    tn, fp, fn, tp = cm.ravel()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'sensitivity': recall,
        'confusion_matrix': cm.tolist(),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }


def compare_fusion_strategies(
    frame_probs: np.ndarray,
    temporal_probs: np.ndarray,
    labels: np.ndarray,
    ensemble_weights: Tuple[float, float] = (0.7, 0.3)
) -> Dict[str, Dict[str, float]]:
    """
    Compare different fusion strategies.

    Args:
        frame_probs: (N, 2) frame ensemble probabilities
        temporal_probs: (N, 2) temporal probabilities
        labels: (N,) ground truth labels
        ensemble_weights: (frame_weight, temporal_weight) for weighted average

    Returns:
        Dictionary of strategy -> metrics
    """
    from sklearn.metrics import accuracy_score, roc_auc_score

    results = {}

    # Strategy 1: Frame only
    frame_preds = frame_probs.argmax(axis=1)
    results['frame_only'] = {
        'accuracy': accuracy_score(labels, frame_preds),
        'auc': roc_auc_score(labels, frame_probs[:, 1])
    }

    # Strategy 2: Temporal only
    temporal_preds = temporal_probs.argmax(axis=1)
    results['temporal_only'] = {
        'accuracy': accuracy_score(labels, temporal_preds),
        'auc': roc_auc_score(labels, temporal_probs[:, 1])
    }

    # Strategy 3: Weighted average (fixed)
    w_frame, w_temporal = ensemble_weights
    weighted_probs = w_frame * frame_probs + w_temporal * temporal_probs
    weighted_preds = weighted_probs.argmax(axis=1)
    results['weighted_avg'] = {
        'accuracy': accuracy_score(labels, weighted_preds),
        'auc': roc_auc_score(labels, weighted_probs[:, 1])
    }

    # Strategy 4: Max confidence
    max_conf = np.maximum(frame_probs.max(axis=1), temporal_probs.max(axis=1))
    # Choose prediction from model with higher confidence
    frame_conf = frame_probs.max(axis=1)
    temporal_conf = temporal_probs.max(axis=1)
    max_conf_preds = np.where(frame_conf > temporal_conf, frame_preds, temporal_preds)
    results['max_confidence'] = {
        'accuracy': accuracy_score(labels, max_conf_preds)
    }

    # Strategy 5: Product of probabilities (Bayesian fusion)
    product_probs = frame_probs * temporal_probs
    product_probs = product_probs / product_probs.sum(axis=1, keepdims=True)
    product_preds = product_probs.argmax(axis=1)
    results['product_fusion'] = {
        'accuracy': accuracy_score(labels, product_preds),
        'auc': roc_auc_score(labels, product_probs[:, 1])
    }

    return results


def save_meta_classifier_pytorch(model: nn.Module, path: str):
    """Save PyTorch meta-classifier."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'input_dim': model.input_dim if hasattr(model, 'input_dim') else 4
    }, path)
    logger.info(f"PyTorch meta-classifier saved to {path}")


def load_meta_classifier_pytorch(path: str, device: str = 'cpu') -> nn.Module:
    """Load PyTorch meta-classifier."""
    checkpoint = torch.load(path, map_location=device)
    model_class = checkpoint['model_class']

    if model_class == 'MetaClassifier':
        model = MetaClassifier(input_dim=checkpoint.get('input_dim', 4))
    elif model_class == 'AttentionMetaClassifier':
        model = AttentionMetaClassifier(input_dim=checkpoint.get('input_dim', 4))
    else:
        raise ValueError(f"Unknown model class: {model_class}")

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    logger.info(f"Loaded meta-classifier from {path}")
    return model


if __name__ == '__main__':
    # Test meta-classifiers
    print("Testing MetaClassifier...")
    model = MetaClassifier()
    frame_probs = torch.randn(4, 2)
    temporal_probs = torch.randn(4, 2)
    out = model(frame_probs, temporal_probs)
    print(f"  Input: frame={frame_probs.shape}, temporal={temporal_probs.shape}")
    print(f"  Output: {out.shape}")

    print("\nTesting AttentionMetaClassifier...")
    model = AttentionMetaClassifier()
    out = model(frame_probs, temporal_probs)
    print(f"  Output: {out.shape}")

    print("\nTesting GradientBoostingMetaClassifier...")
    gbm = GradientBoostingMetaClassifier()
    # Dummy data
    frame_probs = np.random.rand(100, 2)
    frame_probs = frame_probs / frame_probs.sum(axis=1, keepdims=True)
    temporal_probs = np.random.rand(100, 2)
    temporal_probs = temporal_probs / temporal_probs.sum(axis=1, keepdims=True)
    labels = np.random.randint(0, 2, 100)
    gbm.train(frame_probs, temporal_probs, labels)
    preds = gbm.predict(frame_probs, temporal_probs)
    print(f"  Predictions: {preds.shape}")

    print("\nAll tests passed!")