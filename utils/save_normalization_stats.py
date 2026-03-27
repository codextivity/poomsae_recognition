"""Save normalization stats from training data for inference."""

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from configs.paths import Paths


def save_normalization_stats(windows_dir=None, checkpoint_dir=None, output_name='normalization_stats.pkl'):
    """Calculate and save normalization stats from all training windows."""
    windows_dir = Path(windows_dir) if windows_dir is not None else Paths.WINDOWS_DIR
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else Paths.CHECKPOINTS_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load all window files
    window_files = list(windows_dir.glob('*_windows.npz'))

    if not window_files:
        print(f"No window files found in {windows_dir}")
        return

    print(f"Loading {len(window_files)} window files...")

    all_X = []
    for f in window_files:
        data = np.load(f)
        all_X.append(data['X'])

    X = np.concatenate(all_X, axis=0)
    print(f"Total samples: {X.shape[0]}")
    print(f"Shape: {X.shape}")  # (N, seq_len, 78)

    # Calculate mean and std across samples and time steps
    mean = np.mean(X, axis=(0, 1))  # (78,)
    std = np.std(X, axis=(0, 1)) + 1e-8  # (78,)

    print(f"Mean shape: {mean.shape}")
    print(f"Std shape: {std.shape}")

    # Save
    stats = {
        'mean': mean,
        'std': std,
        'num_samples': X.shape[0],
        'sequence_length': X.shape[1],
        'input_size': X.shape[2],
    }

    output_path = checkpoint_dir / output_name
    with open(output_path, 'wb') as f:
        pickle.dump(stats, f)

    print(f"\nSaved to: {output_path}")
    print(f"  Mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"  Std range: [{std.min():.4f}, {std.max():.4f}]")
    return output_path


if __name__ == "__main__":
    save_normalization_stats()
