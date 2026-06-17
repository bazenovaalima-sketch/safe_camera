"""
RWF-2000 Dataset: Real World Fights

Dataset structure:
RWF-2000/
├── train/
│   ├── Fight/
│   └── NonFight/
└── val/
    ├── Fight/
    └── NonFight/
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


class RWF2000Dataset(Dataset):
    """Real World Fights dataset (2000 videos)"""

    def __init__(self, root_dir, split='train', num_frames=16, augment=False):
        """
        Args:
            root_dir: Path to RWF-2000 directory
            split: 'train' or 'val'
            num_frames: Number of frames to extract
            augment: Apply augmentation
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_frames = num_frames
        self.augment = augment

        # Find fight and non-fight videos
        fight_dir = self.root_dir / split / 'Fight'
        nonf_dir = self.root_dir / split / 'NonFight'

        self.videos = []
        self.labels = []

        # Load fight videos (label=1)
        if fight_dir.exists():
            for video_file in sorted(list(fight_dir.glob('*.mp4')) + list(fight_dir.glob('*.avi'))):
                self.videos.append(str(video_file))
                self.labels.append(1)

        # Load non-fight videos (label=0)
        if nonf_dir.exists():
            for video_file in sorted(list(nonf_dir.glob('*.mp4')) + list(nonf_dir.glob('*.avi'))):
                self.videos.append(str(video_file))
                self.labels.append(0)

        num_fights = sum(self.labels)
        num_normal = len(self.labels) - num_fights

        print(f"{split.upper()} set:")
        print(f"  Total: {len(self.videos)} videos")
        print(f"  Fights: {num_fights}")
        print(f"  Normal: {num_normal}")

        if len(self.videos) == 0:
            raise ValueError(f"No videos found in {self.root_dir}")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path = self.videos[idx]
        label = self.labels[idx]

        frames = self._load_frames(video_path)

        if self.augment:
            frames = self._augment_frames(frames)

        return frames, label

    def _load_frames(self, video_path):
        """Load num_frames from video with uniform sampling"""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            # Return black frames if video can't be opened
            print(f"Warning: Could not open {video_path}")
            return torch.zeros((3, self.num_frames, 224, 224))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < 1:
            print(f"Warning: {video_path} has 0 frames")
            return torch.zeros((3, self.num_frames, 224, 224))

        # Uniform sampling of frame indices
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if ret:
                # Resize to 224x224
                frame = cv2.resize(frame, (224, 224))
                # BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Normalize
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
            else:
                # If frame can't be read, use black frame
                frames.append(np.zeros((224, 224, 3), dtype=np.float32))

        cap.release()

        # Stack frames: (T, H, W, C)
        frames = np.stack(frames, axis=0)

        # Convert to tensor and rearrange to (C, T, H, W)
        frames = torch.FloatTensor(frames).permute(3, 0, 1, 2)

        return frames

    def _augment_frames(self, frames):
        """Apply augmentation to frames"""
        frames_np = frames.permute(1, 2, 3, 0).numpy()  # (T, H, W, C)

        # Random horizontal flip
        if np.random.rand() > 0.5:
            frames_np = np.flip(frames_np, axis=2).copy()  # Copy to fix stride

        # Random brightness/contrast
        if np.random.rand() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            frames_np = np.clip(frames_np * factor, 0, 1)

        # Random Gaussian noise
        if np.random.rand() > 0.7:
            noise = np.random.normal(0, 0.02, frames_np.shape)
            frames_np = np.clip(frames_np + noise, 0, 1)

        # Convert back (ensure contiguous)
        frames_np = np.ascontiguousarray(frames_np)
        frames = torch.FloatTensor(frames_np).permute(3, 0, 1, 2)  # (C, T, H, W)

        return frames


if __name__ == '__main__':
    # Test dataset
    from torch.utils.data import DataLoader

    dataset = RWF2000Dataset('data/rwf2000/RWF-2000', split='train', num_frames=16)

    print(f"\nDataset size: {len(dataset)}")

    # Load one sample
    frames, label = dataset[0]
    print(f"Frames shape: {frames.shape}")  # (3, 16, 224, 224)
    print(f"Label: {label}")  # 0 or 1

    # Test dataloader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    for batch_frames, batch_labels in dataloader:
        print(f"\nBatch shapes:")
        print(f"  Frames: {batch_frames.shape}")  # (2, 3, 16, 224, 224)
        print(f"  Labels: {batch_labels.shape}")  # (2,)
        break
