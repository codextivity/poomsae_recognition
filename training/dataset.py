"""
PyTorch Dataset for Poomsae Recognition
Loads sliding window data for LSTM training

Features:
- 22-class support
- Advanced augmentation (time warp, flip, rotate, scale)
- Oversampling for short movements
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import numpy as np
from pathlib import Path

# Short movement classes that need oversampling
SHORT_MOVEMENT_CLASSES = [6, 12, 14, 17]  # 6_1, 12_1, 14_1, 16_1


class PoomsaeDataset(Dataset):
    """Dataset for loading windowed poomsae keypoint sequences"""

    def __init__(self, windows_files, normalize=True, augment=False):
        """
        Args:
            windows_files: List of paths to window .npz files
            normalize: Whether to normalize keypoints
            augment: Whether to apply data augmentation
        """
        self.normalize = normalize
        self.augment = augment

        # Load all windows from all files
        self.X = []
        self.y = []
        self.metadata = []

        for file_path in windows_files:
            data = np.load(file_path, allow_pickle=True)

            self.X.append(data['X'])
            self.y.append(data['y'])

            # Store metadata if available
            num_samples = len(data['y'])
            for i in range(num_samples):
                meta = {
                    'source_file': str(file_path)
                }
                if 'movement_names' in data:
                    meta['movement_name'] = data['movement_names'][i]
                if 'quality' in data:
                    meta['quality'] = data['quality'][i]

                self.metadata.append(meta)

        # Concatenate all data
        self.X = np.concatenate(self.X, axis=0)
        self.y = np.concatenate(self.y, axis=0)

        print(f"Loaded {len(self)} samples from {len(windows_files)} files")
        print(f"  X shape: {self.X.shape}")
        print(f"  y shape: {self.y.shape}")

        # Calculate normalization statistics
        if self.normalize:
            self._calculate_stats()

    def _calculate_stats(self):
        """Calculate mean and std for normalization"""
        self.mean = np.mean(self.X, axis=(0, 1))
        self.std = np.std(self.X, axis=(0, 1)) + 1e-8

        print(f"  Normalization stats calculated")
        print(f"    Mean shape: {self.mean.shape}")
        print(f"    Std shape: {self.std.shape}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        """Get one sample"""
        x = self.X[idx].copy()
        y = self.y[idx]

        # Normalize
        if self.normalize:
            x = (x - self.mean) / self.std

        # Augment (only during training)
        if self.augment:
            x = self._augment(x)

        # Convert to tensors
        x = torch.FloatTensor(x)
        y = torch.LongTensor([y]).squeeze()

        return x, y

    def _augment(self, x):
        """
        Apply random augmentation for skeleton sequences.

        x shape: (seq_len, 78) where 78 = 26 keypoints × 3 (x, y, conf)
        """
        seq_len, num_features = x.shape

        # 1. Random noise injection (30% chance)
        if np.random.random() < 0.3:
            noise_level = np.random.uniform(0.005, 0.02)
            noise = np.random.normal(0, noise_level, x.shape)
            x = x + noise

        # 2. Time warping / speed perturbation (40% chance)
        # Crucial for short movements - creates speed variations
        if np.random.random() < 0.4:
            speed_factor = np.random.uniform(0.8, 1.2)  # 80% to 120% speed
            new_len = int(seq_len * speed_factor)
            if new_len > 2:
                # Interpolate to new length
                old_indices = np.linspace(0, seq_len - 1, seq_len)
                new_indices = np.linspace(0, seq_len - 1, new_len)
                x_warped = np.zeros((new_len, num_features))
                for f in range(num_features):
                    x_warped[:, f] = np.interp(new_indices, old_indices, x[:, f])

                # Resample back to original length
                if new_len != seq_len:
                    final_indices = np.linspace(0, new_len - 1, seq_len)
                    x_final = np.zeros((seq_len, num_features))
                    for f in range(num_features):
                        x_final[:, f] = np.interp(final_indices, np.arange(new_len), x_warped[:, f])
                    x = x_final

        # 3. Random temporal shift (20% chance)
        if np.random.random() < 0.2:
            shift = np.random.randint(-3, 4)
            x = np.roll(x, shift, axis=0)

        # 4. Horizontal flip / mirror (30% chance)
        # Swap left and right keypoints
        if np.random.random() < 0.3:
            x = self._horizontal_flip(x)

        # 5. Scale augmentation (25% chance)
        # Random scaling of spatial coordinates
        if np.random.random() < 0.25:
            scale = np.random.uniform(0.9, 1.1)
            # Only scale x,y coordinates (not confidence)
            x_reshaped = x.reshape(seq_len, 26, 3)
            x_reshaped[:, :, :2] *= scale
            x = x_reshaped.reshape(seq_len, num_features)

        # 6. Random rotation (20% chance)
        if np.random.random() < 0.2:
            angle = np.random.uniform(-15, 15)  # degrees
            x = self._rotate_skeleton(x, angle)

        # 7. Keypoint dropout (15% chance)
        # Randomly zero out some keypoints
        if np.random.random() < 0.15:
            x_reshaped = x.reshape(seq_len, 26, 3)
            num_drop = np.random.randint(1, 4)  # Drop 1-3 keypoints
            drop_indices = np.random.choice(26, num_drop, replace=False)
            x_reshaped[:, drop_indices, :] = 0
            x = x_reshaped.reshape(seq_len, num_features)

        return x

    def _horizontal_flip(self, x):
        """
        Flip skeleton horizontally and swap left/right keypoints.

        Halpe26 keypoint pairs (left/right):
        - 1,2 (eyes), 3,4 (ears), 5,6 (shoulders)
        - 7,8 (elbows), 9,10 (wrists), 11,12 (hips)
        - 13,14 (knees), 15,16 (ankles), 20,21 (big toes)
        - 22,23 (small toes), 24,25 (heels)
        """
        seq_len = x.shape[0]
        x_reshaped = x.reshape(seq_len, 26, 3)

        # Flip x coordinates (negate)
        x_reshaped[:, :, 0] = -x_reshaped[:, :, 0]

        # Swap left/right keypoint pairs
        swap_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10),
                      (11, 12), (13, 14), (15, 16), (20, 21), (22, 23), (24, 25)]

        for left, right in swap_pairs:
            x_reshaped[:, [left, right], :] = x_reshaped[:, [right, left], :]

        return x_reshaped.reshape(seq_len, -1)

    def _rotate_skeleton(self, x, angle_degrees):
        """Rotate skeleton by angle around center"""
        seq_len = x.shape[0]
        x_reshaped = x.reshape(seq_len, 26, 3)

        angle_rad = np.radians(angle_degrees)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        # Rotation matrix
        x_coords = x_reshaped[:, :, 0].copy()
        y_coords = x_reshaped[:, :, 1].copy()

        x_reshaped[:, :, 0] = x_coords * cos_a - y_coords * sin_a
        x_reshaped[:, :, 1] = x_coords * sin_a + y_coords * cos_a

        return x_reshaped.reshape(seq_len, -1)

    def get_class_distribution(self):
        """Get distribution of movement classes"""
        unique, counts = np.unique(self.y, return_counts=True)
        return dict(zip(unique, counts))


def create_dataloaders(data_dir, batch_size=32, train_split=0.7, val_split=0.15,
                       num_workers=4, seed=42, oversample_short=True):
    """
    Create train, validation, and test dataloaders

    Args:
        data_dir: Directory containing window .npz files
        batch_size: Batch size for training
        train_split: Fraction of data for training (0.0-1.0)
        val_split: Fraction of data for validation (0.0-1.0)
        num_workers: Number of data loading workers
        seed: Random seed for reproducibility
        oversample_short: Whether to oversample short movement classes

    Returns:
        train_loader, val_loader, test_loader
    """
    data_dir = Path(data_dir)

    # Find all window files
    window_files = list(data_dir.glob('*_windows.npz'))

    if not window_files:
        raise ValueError(f"No window files found in {data_dir}")

    print(f"\nFound {len(window_files)} window files:")
    for f in window_files:
        print(f"  - {f.name}")

    # Create full dataset
    full_dataset = PoomsaeDataset(window_files, normalize=True, augment=False)

    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # Enable augmentation only for training
    train_dataset.dataset.augment = True

    # Create sampler for oversampling short movements
    sampler = None
    shuffle = True

    if oversample_short:
        # Get labels for training samples
        train_indices = train_dataset.indices
        train_labels = full_dataset.y[train_indices]

        # Calculate sample weights (higher for short movements)
        sample_weights = np.ones(len(train_labels))
        for i, label in enumerate(train_labels):
            if label in SHORT_MOVEMENT_CLASSES:
                sample_weights[i] = 3.0  # 3x more likely to be sampled

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_labels),
            replacement=True
        )
        shuffle = False  # Can't shuffle with sampler

        short_count = sum(1 for l in train_labels if l in SHORT_MOVEMENT_CLASSES)
        print(f"\n[OK] Oversampling enabled for short movements")
        print(f"  Short movement samples: {short_count} ({100*short_count/len(train_labels):.1f}%)")
        print(f"  Short classes: {SHORT_MOVEMENT_CLASSES}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"\nDataset splits:")
    print(f"  Train: {len(train_dataset)} samples ({train_split*100:.0f}%)")
    print(f"  Val:   {len(val_dataset)} samples ({val_split*100:.0f}%)")
    print(f"  Test:  {len(test_dataset)} samples ({(1-train_split-val_split)*100:.0f}%)")

    # Show class distribution
    print(f"\nClass distribution:")
    class_dist = full_dataset.get_class_distribution()
    for class_id, count in sorted(class_dist.items()):
        print(f"  Movement {class_id+1}: {count} samples")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    import sys

    # You need to update this path to your actual windows directory
    windows_dir = "data/processed/windows"

    if not Path(windows_dir).exists():
        print(f"Error: Directory {windows_dir} does not exist")
        print("Please create windows first using preprocessing scripts")
        sys.exit(1)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=windows_dir,
        batch_size=16,
        train_split=0.7,
        val_split=0.15
    )

    # Test loading one batch
    print("\nTesting batch loading...")
    for x_batch, y_batch in train_loader:
        print(f"Batch shapes:")
        print(f"  X: {x_batch.shape}")
        print(f"  y: {y_batch.shape}")
        print(f"  y values: {y_batch[:5]}")  # First 5 labels
        break

    print("\n✓ Dataset test successful!")