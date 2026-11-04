#!/usr/bin/env python3
"""
Temporal Consistency Modeling (Task 3)

Implements temporal models for video-level deepfake detection:
- Temporal Transformer (Option B, stronger)
- Bidirectional GRU with Attention (Option A, simpler)

Both models operate on per-frame embeddings from EfficientNet-B0 (1280-dim).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TemporalTransformer(nn.Module):
    """
    Temporal Transformer for video-level deepfake detection.

    Uses a learnable [CLS] token for sequence classification,
    similar to Vision Transformer but operating on frame embeddings.
    """

    def __init__(
        self,
        embed_dim: int = 1280,          # EfficientNet-B0 penultimate layer output
        num_layers: int = 4,
        num_heads: int = 8,
        hidden_dim: int = 512,
        num_classes: int = 2,
        dropout: float = 0.1,
        max_frames: int = 32,
        use_cls_token: bool = True,
        pool_type: str = 'cls'          # 'cls', 'mean', 'max', 'attention'
    ):
        """
        Initialize Temporal Transformer.

        Args:
            embed_dim: Input embedding dimension (1280 for EfficientNet-B0)
            num_layers: Number of Transformer encoder layers
            num_heads: Number of attention heads
            hidden_dim: Feedforward hidden dimension
            num_classes: Number of output classes (2 for real/fake)
            dropout: Dropout probability
            max_frames: Maximum sequence length for positional encoding
            use_cls_token: Whether to use learnable CLS token
            pool_type: Pooling strategy for sequence classification
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.use_cls_token = use_cls_token
        self.pool_type = pool_type
        self.max_frames = max_frames

        # Input projection (optional, for dimension matching)
        self.input_proj = nn.Linear(embed_dim, embed_dim)

        # Learnable CLS token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
            seq_len = max_frames + 1
        else:
            seq_len = max_frames

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN architecture
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Pooling layer (if not using CLS token)
        if not use_cls_token or pool_type == 'attention':
            self.attention_pool = nn.Sequential(
                nn.Linear(embed_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.use_cls_token:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module_weights)

    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Frame embeddings (B, T, D) where T <= max_frames
            mask: Optional padding mask (B, T) - True for valid frames

        Returns:
            Logits (B, num_classes)
        """
        B, T, D = x.shape
        assert D == self.embed_dim, f"Input dim {D} != embed_dim {self.embed_dim}"

        # Project input
        x = self.input_proj(x)

        # Add CLS token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
            x = torch.cat([cls_tokens, x], dim=1)  # (B, T+1, D)

        # Add positional encoding
        x = x + self.pos_embed[:, :x.size(1)]
        x = self.pos_drop(x)

        # Prepare attention mask for padding
        if mask is not None:
            if self.use_cls_token:
                # CLS token is always valid
                cls_mask = torch.ones(B, 1, dtype=torch.bool, device=x.device)
                mask = torch.cat([cls_mask, mask], dim=1)
            # Transformer expects False for valid positions
            src_key_padding_mask = ~mask
        else:
            src_key_padding_mask = None

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Pooling
        if self.use_cls_token and self.pool_type == 'cls':
            # Use CLS token
            pooled = x[:, 0]  # (B, D)
        elif self.pool_type == 'mean':
            # Mean pooling over valid frames
            if mask is not None:
                if self.use_cls_token:
                    mask = mask[:, 1:]  # Remove CLS from mask
                pooled = (x[:, 1:] * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                pooled = x.mean(dim=1)
        elif self.pool_type == 'max':
            if mask is not None:
                if self.use_cls_token:
                    x = x[:, 1:]
                    mask = mask[:, 1:]
                x = x.masked_fill(~mask.unsqueeze(-1), -1e9)
            pooled = x.max(dim=1)[0]
        elif self.pool_type == 'attention':
            # Attention pooling
            if self.use_cls_token:
                attn_x = x[:, 1:]  # Exclude CLS from attention pooling
            else:
                attn_x = x
            attn_weights = self.attention_pool(attn_x).squeeze(-1)  # (B, T)
            if mask is not None:
                if self.use_cls_token:
                    mask = mask[:, 1:]
                attn_weights = attn_weights.masked_fill(~mask, -1e9)
            attn_weights = F.softmax(attn_weights, dim=-1)
            pooled = (attn_x * attn_weights.unsqueeze(-1)).sum(dim=1)
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")

        # Classification
        pooled = self.norm(pooled)
        logits = self.head(pooled)

        return logits


class BiGRUTemporal(nn.Module):
    """
    Bidirectional GRU with Attention Pooling for video-level deepfake detection.

    Simpler alternative to Transformer, faster to train and export.
    """

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
        pool_type: str = 'attention'  # 'attention', 'mean', 'max', 'last'
    ):
        """
        Initialize BiGRU Temporal model.

        Args:
            input_dim: Input embedding dimension
            hidden_dim: GRU hidden dimension
            num_layers: Number of GRU layers
            num_classes: Number of output classes
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional GRU
            pool_type: Pooling strategy
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pool_type = pool_type

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

        gru_output_dim = hidden_dim * (2 if bidirectional else 1)

        # Attention pooling
        if pool_type == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(gru_output_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )

        # Classification head
        self.norm = nn.LayerNorm(gru_output_dim)
        self.head = nn.Sequential(
            nn.Linear(gru_output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Frame embeddings (B, T, D)
            mask: Optional padding mask (B, T) - True for valid frames

        Returns:
            Logits (B, num_classes)
        """
        B, T, D = x.shape

        # GRU forward
        if mask is not None:
            # Pack padded sequence for efficiency
            lengths = mask.sum(dim=1).cpu()
            x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        gru_out, _ = self.gru(x)

        if mask is not None:
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)
            # gru_out: (B, T, hidden_dim * 2)

        # Pooling
        if self.pool_type == 'attention':
            attn_weights = self.attention(gru_out).squeeze(-1)  # (B, T)
            if mask is not None:
                attn_weights = attn_weights.masked_fill(~mask, -1e9)
            attn_weights = F.softmax(attn_weights, dim=-1)
            pooled = (gru_out * attn_weights.unsqueeze(-1)).sum(dim=1)
        elif self.pool_type == 'mean':
            if mask is not None:
                pooled = (gru_out * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                pooled = gru_out.mean(dim=1)
        elif self.pool_type == 'max':
            if mask is not None:
                gru_out = gru_out.masked_fill(~mask.unsqueeze(-1), -1e9)
            pooled = gru_out.max(dim=1)[0]
        elif self.pool_type == 'last':
            # Use last valid frame
            if mask is not None:
                last_indices = mask.sum(dim=1) - 1  # (B,)
                pooled = torch.stack([gru_out[i, last_indices[i]] for i in range(B)])
            else:
                pooled = gru_out[:, -1]
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")

        # Classification
        pooled = self.norm(pooled)
        logits = self.head(pooled)

        return logits


def create_temporal_model(
    model_type: str = 'transformer',
    embed_dim: int = 1280,
    **kwargs
) -> nn.Module:
    """
    Factory function to create temporal model.

    Args:
        model_type: 'transformer' or 'bigru'
        embed_dim: Input embedding dimension
        **kwargs: Additional arguments for model constructor

    Returns:
        Temporal model instance
    """
    if model_type == 'transformer':
        return TemporalTransformer(embed_dim=embed_dim, **kwargs)
    elif model_type == 'bigru':
        return BiGRUTemporal(input_dim=embed_dim, **kwargs)
    else:
        raise ValueError(f"Unknown temporal model type: {model_type}")


class TemporalDetector(nn.Module):
    """
    Complete temporal detection pipeline:
    1. EfficientNet backbone (frozen) for frame embeddings
    2. Temporal model for sequence classification
    3. Optional meta-classifier for fusion with per-frame predictions
    """

    def __init__(
        self,
        efficientnet: nn.Module,
        temporal_model: nn.Module,
        num_classes: int = 2,
        freeze_backbone: bool = True
    ):
        """
        Initialize temporal detector.

        Args:
            efficientnet: EfficientNet-B0 model (should output embeddings, not logits)
            temporal_model: TemporalTransformer or BiGRUTemporal
            num_classes: Number of classes
            freeze_backbone: Whether to freeze EfficientNet weights
        """
        super().__init__()
        self.efficientnet = efficientnet
        self.temporal_model = temporal_model
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            for param in self.efficientnet.parameters():
                param.requires_grad = False

        # Meta-classifier for fusion (4 inputs: frame_real, frame_fake, temporal_real, temporal_fake)
        self.meta_classifier = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def extract_frame_embeddings(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings from EfficientNet backbone.

        Args:
            frames: (B, T, C, H, W) or (B*T, C, H, W)

        Returns:
            Embeddings (B, T, D)
        """
        # Handle both 5D and 4D inputs
        if frames.dim() == 5:
            B, T, C, H, W = frames.shape
            frames = frames.view(B * T, C, H, W)
        else:
            # Assume already (B*T, C, H, W)
            B = 1
            T = frames.size(0)

        with torch.set_grad_enabled(not self.freeze_backbone):
            # Extract features from penultimate layer
            features = self.efficientnet.forward_features(frames)

            # Global average pooling if needed
            if features.dim() == 4:
                features = features.mean(dim=[2, 3])  # (B*T, D)

        return features.view(B, T, -1)  # (B, T, D)

    def forward(
        self,
        frames: torch.Tensor,
        frame_logits: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            frames: Video frames (B, T, C, H, W)
            frame_logits: Optional per-frame logits from ensemble (B, T, 2)
            mask: Optional padding mask (B, T)

        Returns:
            Video-level logits (B, num_classes)
        """
        # Extract frame embeddings
        embeddings = self.extract_frame_embeddings(frames)  # (B, T, D)

        # Temporal modeling
        temporal_logits = self.temporal_model(embeddings, mask)  # (B, 2)

        # Fusion with per-frame predictions if available
        if frame_logits is not None:
            # frame_logits: (B, T, 2) -> average over frames -> (B, 2)
            frame_probs = F.softmax(frame_logits, dim=-1).mean(dim=1)  # (B, 2)
            temporal_probs = F.softmax(temporal_logits, dim=-1)  # (B, 2)

            # Concatenate: [frame_real, frame_fake, temporal_real, temporal_fake]
            fusion_input = torch.cat([frame_probs, temporal_probs], dim=-1)  # (B, 4)
            fused_logits = self.meta_classifier(fusion_input)
            return fused_logits

        return temporal_logits


def export_temporal_to_onnx(
    temporal_detector: TemporalDetector,
    output_path: str,
    num_frames: int = 16,
    input_size: int = 224,
    device: str = 'cpu'
):
    """
    Export temporal detector to ONNX.

    Note: Requires careful handling of dynamic sequence length.
    """
    temporal_detector.eval()

    # Create dummy input
    dummy_frames = torch.randn(1, num_frames, 3, input_size, input_size).to(device)

    torch.onnx.export(
        temporal_detector,
        dummy_frames,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['frames'],
        output_names=['logits'],
        dynamic_axes={
            'frames': {0: 'batch', 1: 'num_frames'},
            'logits': {0: 'batch'}
        }
    )
    print(f"✓ Exported temporal model to {output_path}")


if __name__ == '__main__':
    # Test models
    print("Testing TemporalTransformer...")
    model = TemporalTransformer(embed_dim=1280, num_layers=4, num_heads=8)
    x = torch.randn(2, 16, 1280)
    mask = torch.ones(2, 16, dtype=torch.bool)
    out = model(x, mask)
    print(f"  Input: {x.shape}, Output: {out.shape}")

    print("\nTesting BiGRUTemporal...")
    model = BiGRUTemporal(input_dim=1280, hidden_dim=256, num_layers=2)
    x = torch.randn(2, 16, 1280)
    out = model(x, mask)
    print(f"  Input: {x.shape}, Output: {out.shape}")

    print("\nTesting TemporalDetector...")
    # Create dummy EfficientNet that outputs embeddings
    import timm
    efficientnet = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
    efficientnet.forward_features = lambda x: x  # Dummy

    temporal = TemporalTransformer(embed_dim=1280)
    detector = TemporalDetector(efficientnet, temporal)

    frames = torch.randn(1, 8, 3, 224, 224)
    out = detector(frames)
    print(f"  Frames: {frames.shape}, Output: {out.shape}")

    print("\nAll tests passed!")