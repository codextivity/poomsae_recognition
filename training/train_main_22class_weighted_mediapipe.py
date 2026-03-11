"""
MediaPipe-specific entrypoint for 22-class weighted LSTM training.

It reuses the existing training logic but switches data/checkpoint paths
to MediaPipe-specific directories.
"""

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from configs.paths import Paths
from configs.policy_config import PolicyConfig


def main():
    parser = argparse.ArgumentParser(description="Train 22-class weighted model using MediaPipe windows")
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
        help="Directory to save model checkpoints and training history",
    )
    parser.add_argument(
        "--policy-profile",
        type=str,
        default="",
        choices=["baseline", "short_aware", "custom", ""],
        help="Optional override for PolicyConfig.PROFILE",
    )
    args = parser.parse_args()

    windows_dir = Path(args.windows_dir)
    checkpoint_dir = Path(args.checkpoint_dir)

    if args.policy_profile:
        PolicyConfig.PROFILE = args.policy_profile

    # Redirect shared Paths used by the original training script.
    Paths.WINDOWS_DIR = windows_dir
    Paths.CHECKPOINTS_DIR = checkpoint_dir

    print("\n" + "=" * 70)
    print("MEDIAPIPE TRAINING ENTRYPOINT")
    print("=" * 70)
    print(f"Windows dir: {Paths.WINDOWS_DIR}")
    print(f"Checkpoint dir: {Paths.CHECKPOINTS_DIR}")
    print(f"Policy profile: {PolicyConfig.PROFILE}")
    print("=" * 70 + "\n")

    # Lazy import so --help works even if torch is not installed in this interpreter.
    from training.train_main_22class_weighted import main as train_main_main
    train_main_main()


if __name__ == "__main__":
    main()
