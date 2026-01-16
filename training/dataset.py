"""
PyTorch Dataset for Poomsae Recognition
Loads sliding window data for LSTM training
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from pathlib import Path


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
        """Apply random augmentation"""
        # Random noise injection
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.01, x.shape)
            x = x + noise

        # Random temporal shift
        if np.random.random() < 0.2:
            shift = np.random.randint(-5, 6)
            x = np.roll(x, shift, axis=0)

        return x

    def get_class_distribution(self):
        """Get distribution of movement classes"""
        unique, counts = np.unique(self.y, return_counts=True)
        return dict(zip(unique, counts))


def create_dataloaders(data_dir, batch_size=32, train_split=0.7, val_split=0.15,
                       num_workers=4, seed=42):
    """
    Create train, validation, and test dataloaders

    Args:
        data_dir: Directory containing window .npz files
        batch_size: Batch size for training
        train_split: Fraction of data for training (0.0-1.0)
        val_split: Fraction of data for validation (0.0-1.0)
        num_workers: Number of data loading workers
        seed: Random seed for reproducibility

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

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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