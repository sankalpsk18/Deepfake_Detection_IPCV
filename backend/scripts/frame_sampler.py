#!/usr/bin/env python3
"""
Video Frame Sequence Sampling (Task 3)

Extracts frame sequences from videos for temporal consistency modeling.
Uses existing MediaPipe face detection + crop pipeline per frame.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import logging

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.preprocessing import detect_and_crop_face, resize_with_padding

logger = logging.getLogger(__name__)


def sample_frame_sequence(
    video_path: str,
    num_frames: int = 16,
    stride: int = 2,
    target_size: Tuple[int, int] = (224, 224),
    face_confidence: float = 0.5,
    face_padding: float = 0.3,
    sampling_strategy: str = 'uniform',  # 'uniform', 'random', 'center'
    max_frames: Optional[int] = None
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """
    Extract a sequence of frames from a video with face detection and cropping.

    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample
        stride: Frame sampling stride (every Nth frame)
        target_size: Target size for face crops (H, W)
        face_confidence: MediaPipe face detection confidence threshold
        face_padding: Padding around detected face (ratio)
        sampling_strategy: 'uniform', 'random', or 'center'
        max_frames: Maximum frames to read from video (None = all)

    Returns:
        Tuple of (frame_tensor, metadata_list)
        frame_tensor: (N, C, H, W) normalized tensors ready for model
        metadata_list: List of dicts with frame info (frame_idx, face_detected, bbox, etc.)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if max_frames:
        total_frames = min(total_frames, max_frames)

    if total_frames == 0:
        cap.release()
        raise ValueError(f"Video has no frames: {video_path}")

    # Determine frame indices to sample
    if sampling_strategy == 'uniform':
        # Uniform sampling across video duration
        if total_frames <= num_frames * stride:
            # Video too short, use all available frames with stride 1
            frame_indices = list(range(0, total_frames, max(1, total_frames // num_frames)))
        else:
            # Start from middle of first segment for better coverage
            start_idx = (total_frames - num_frames * stride) // 2
            frame_indices = [start_idx + i * stride for i in range(num_frames)]
    elif sampling_strategy == 'center':
        # Sample around center of video
        center = total_frames // 2
        half_window = num_frames * stride // 2
        start_idx = max(0, center - half_window)
        frame_indices = [start_idx + i * stride for i in range(num_frames)]
        frame_indices = [idx for idx in frame_indices if idx < total_frames]
    elif sampling_strategy == 'random':
        # Random sampling
        if total_frames <= num_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = sorted(np.random.choice(total_frames, num_frames, replace=False))
    else:
        raise ValueError(f"Unknown sampling_strategy: {sampling_strategy}")

    # Ensure we have exactly num_frames (pad if needed)
    frame_indices = frame_indices[:num_frames]
    while len(frame_indices) < num_frames:
        # Pad with last valid index
        frame_indices.append(frame_indices[-1] if frame_indices else 0)

    frames = []
    metadata_list = []

    for idx, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            # Frame read failed, use black frame
            logger.warning(f"Failed to read frame {frame_idx} from {video_path}")
            frame = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
            face_detected = False
            bbox = None
        else:
            # Detect and crop face
            cropped, face_detected, bbox = detect_and_crop_face(
                frame,
                confidence_threshold=face_confidence,
                padding=face_padding
            )

            if face_detected and cropped.size > 0:
                frame = cropped
            else:
                # Use full frame if no face detected
                frame = frame

        # Resize with padding to maintain aspect ratio
        resized = resize_with_padding(frame, target_size)

        # Normalize for EfficientNet (ImageNet stats)
        normalized = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        normalized = (normalized - mean) / std

        # Convert to CHW tensor
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float()

        frames.append(tensor)

        metadata_list.append({
            'frame_index': frame_idx,
            'sequence_index': idx,
            'face_detected': face_detected,
            'bbox': bbox,
            'original_shape': frame.shape[:2] if 'frame' in locals() else None
        })

    cap.release()

    # Stack frames: (N, C, H, W)
    frame_tensor = torch.stack(frames, dim=0)

    return frame_tensor, metadata_list


def sample_frame_sequence_opencv(
    video_path: str,
    num_frames: int = 16,
    target_size: Tuple[int, int] = (224, 224),
    **kwargs
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """
    Simplified frame sampling using OpenCV only (no MediaPipe).
    Faster but less accurate face detection.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        raise ValueError(f"Video has no frames: {video_path}")

    # Uniform sampling
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()

    frames = []
    metadata_list = []

    # Use Haar cascade for faster face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    for idx, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            frame = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
            face_detected = False
            bbox = None
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) > 0:
                # Use largest face
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                face_detected = True
                bbox = (x, y, w, h)

                # Add padding
                pad_x = int(w * 0.3)
                pad_y = int(h * 0.3)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(frame.shape[1], x + w + pad_x)
                y2 = min(frame.shape[0], y + h + pad_y)
                frame = frame[y1:y2, x1:x2]
            else:
                face_detected = False
                bbox = None

        resized = resize_with_padding(frame, target_size)
        normalized = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        normalized = (normalized - mean) / std
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float()

        frames.append(tensor)
        metadata_list.append({
            'frame_index': frame_idx,
            'sequence_index': idx,
            'face_detected': face_detected,
            'bbox': bbox
        })

    cap.release()
    frame_tensor = torch.stack(frames, dim=0)
    return frame_tensor, metadata_list


def extract_frames_for_training(
    video_path: str,
    label: int,  # 0=real, 1=fake
    num_frames: int = 16,
    target_size: Tuple[int, int] = (224, 224),
    **kwargs
) -> Dict[str, Any]:
    """
    Extract frames from a video for training the temporal model.

    Returns dict with:
        - frames: (N, C, H, W) tensor
        - label: 0 or 1
        - metadata: list of frame metadata
    """
    frames, metadata = sample_frame_sequence(
        video_path,
        num_frames=num_frames,
        target_size=target_size,
        **kwargs
    )

    return {
        'frames': frames,
        'label': label,
        'metadata': metadata,
        'video_path': video_path
    }


class VideoFrameDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for video frame sequences.

    Each sample returns a frame sequence tensor and label.
    """

    def __init__(
        self,
        video_paths: List[str],
        labels: List[int],
        num_frames: int = 16,
        target_size: Tuple[int, int] = (224, 224),
        stride: int = 2,
        face_confidence: float = 0.5,
        face_padding: float = 0.3,
        sampling_strategy: str = 'uniform',
        transform=None
    ):
        """
        Args:
            video_paths: List of video file paths
            labels: List of labels (0=real, 1=fake)
            num_frames: Number of frames per sequence
            target_size: Target frame size (H, W)
            stride: Frame sampling stride
            face_confidence: Face detection confidence
            face_padding: Face crop padding
            sampling_strategy: 'uniform', 'random', 'center'
            transform: Optional transform to apply to each frame
        """
        assert len(video_paths) == len(labels), "Paths and labels must have same length"

        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.target_size = target_size
        self.stride = stride
        self.face_confidence = face_confidence
        self.face_padding = face_padding
        self.sampling_strategy = sampling_strategy
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        try:
            frames, metadata = sample_frame_sequence(
                video_path,
                num_frames=self.num_frames,
                stride=self.stride,
                target_size=self.target_size,
                face_confidence=self.face_confidence,
                face_padding=self.face_padding,
                sampling_strategy=self.sampling_strategy
            )

            if self.transform:
                # Apply transform to each frame
                frames = torch.stack([self.transform(f) for f in frames])

            return frames, label, metadata

        except Exception as e:
            logger.warning(f"Error processing {video_path}: {e}")
            # Return dummy data
            dummy_frames = torch.zeros(self.num_frames, 3, *self.target_size)
            dummy_metadata = [{'error': str(e)}] * self.num_frames
            return dummy_frames, label, dummy_metadata


def create_video_dataloaders(
    video_dir: str,
    batch_size: int = 4,
    num_frames: int = 16,
    target_size: Tuple[int, int] = (224, 224),
    num_workers: int = 4,
    train_split: float = 0.7,
    val_split: float = 0.15,
    seed: int = 42
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create train/val/test dataloaders from a directory of videos.

    Expected directory structure:
        video_dir/
            real/
                video1.mp4
                video2.mp4
                ...
            fake/
                video1.mp4
                video2.mp4
                ...
    """
    video_dir = Path(video_dir)
    real_dir = video_dir / 'real'
    fake_dir = video_dir / 'fake'

    video_paths = []
    labels = []

    # Collect real videos
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        for path in real_dir.glob(f'*{ext}'):
            video_paths.append(str(path))
            labels.append(0)

    # Collect fake videos
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        for path in fake_dir.glob(f'*{ext}'):
            video_paths.append(str(path))
            labels.append(1)

    if not video_paths:
        raise ValueError(f"No videos found in {video_dir}")

    # Shuffle and split
    np.random.seed(seed)
    indices = np.random.permutation(len(video_paths))

    train_end = int(len(indices) * train_split)
    val_end = train_end + int(len(indices) * val_split)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    def create_loader(indices):
        dataset = VideoFrameDataset(
            [video_paths[i] for i in indices],
            [labels[i] for i in indices],
            num_frames=num_frames,
            target_size=target_size
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_video_batch
        )

    return {
        'train': create_loader(train_idx),
        'val': create_loader(val_idx),
        'test': create_loader(test_idx)
    }


def collate_video_batch(batch):
    """
    Custom collate function for variable-length video sequences.
    Pads sequences to same length within batch.
    """
    frames_list = []
    labels = []
    metadata_list = []

    max_frames = max(f.shape[0] for f, _, _ in batch)

    for frames, label, metadata in batch:
        # Pad frames if needed
        if frames.shape[0] < max_frames:
            padding = torch.zeros(max_frames - frames.shape[0], *frames.shape[1:])
            frames = torch.cat([frames, padding], dim=0)
        frames_list.append(frames)
        labels.append(label)
        metadata_list.append(metadata)

    return torch.stack(frames_list), torch.tensor(labels), metadata_list


def get_temporal_transforms(model_name: str, input_size: int, is_train: bool = True):
    """
    Get transforms for temporal model frames.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

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
    # Test frame sampling
    import sys

    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # Create dummy video for testing
        print("Creating dummy video for testing...")
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            video_path = f.name

        # Create dummy video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
        for i in range(60):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            out.write(frame)
        out.release()

    print(f"Testing frame sampling on: {video_path}")
    frames, metadata = sample_frame_sequence(video_path, num_frames=16, stride=2)

    print(f"Frames shape: {frames.shape}")
    print(f"Metadata entries: {len(metadata)}")
    for m in metadata[:3]:
        print(f"  {m}")

    print("\nTest passed!")