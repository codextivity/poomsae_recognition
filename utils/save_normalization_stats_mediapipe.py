"""
Save normalization stats from MediaPipe windows data.

This keeps MediaPipe stats separate from RTMPose stats.
"""

import argparse
import pickle
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from configs.paths import Paths


def save_normalization_stats_mediapipe(windows_dir: Path, checkpoint_dir: Path, output_name: str):
    """Calculate and save normalization stats from MediaPipe window files."""
    windows_dir = Path(windows_dir)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    window_files = sorted(windows_dir.glob("*_windows.npz"))
    if not window_files:
        print(f"No window files found in {windows_dir}")
        return

    print(f"Loading {len(window_files)} MediaPipe window files...")

    all_x = []
    for file_path in window_files:
        data = np.load(file_path)
        all_x.append(data["X"])

    x = np.concatenate(all_x, axis=0)
    print(f"Total samples: {x.shape[0]}")
    print(f"Shape: {x.shape}")  # (N, seq_len, 78)

    mean = np.mean(x, axis=(0, 1))
    std = np.std(x, axis=(0, 1)) + 1e-8

    print(f"Mean shape: {mean.shape}")
    print(f"Std shape: {std.shape}")

    stats = {
        "mean": mean,
        "std": std,
        "num_samples": int(x.shape[0]),
        "sequence_length": int(x.shape[1]),
        "input_size": int(x.shape[2]),
        "source": "mediapipe_windows",
    }

    output_path = checkpoint_dir / output_name
    with open(output_path, "wb") as f:
        pickle.dump(stats, f)

    print(f"\nSaved to: {output_path}")
    print(f"  Mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"  Std range: [{std.min():.4f}, {std.max():.4f}]")


def main():
    parser = argparse.ArgumentParser(description="Save normalization stats for MediaPipe windows")
    parser.add_argument(
        "--windows-dir",
        type=str,
        default=str(Paths.PROCESSED_ROOT / "windows_mediapipe"),
        help="Directory containing MediaPipe *_windows.npz files",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(Paths.ROOT / "checkpoints" / "22_classes_model_mediapipe"),
        help="Directory to save normalization_stats.pkl",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="normalization_stats.pkl",
        help="Output pickle filename",
    )
    args = parser.parse_args()

    save_normalization_stats_mediapipe(
        windows_dir=Path(args.windows_dir),
        checkpoint_dir=Path(args.checkpoint_dir),
        output_name=args.output_name,
    )


if __name__ == "__main__":
    main()
