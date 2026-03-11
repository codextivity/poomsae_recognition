"""
Create sliding windows specifically for MediaPipe keypoints.

This script reuses the core windowing logic from create_windows.py but keeps
MediaPipe input/output paths separate, so you can tune behavior later without
touching the RTMPose window generation flow.
"""

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from configs.paths import Paths
from configs.policy_config import PolicyConfig
from preprocessing.create_windows import SlidingWindowCreator


class MediaPipeWindowCreator(SlidingWindowCreator):
    """Dedicated entrypoint for MediaPipe window creation."""

    def __init__(self):
        super().__init__()

    def process_single(self, keypoints_dir, annotations_dir, output_dir, video_name):
        """Process one video by base name (e.g., P002)."""
        keypoints_dir = Path(keypoints_dir)
        annotations_dir = Path(annotations_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        base = str(video_name).replace("_keypoints.pkl", "").replace("_annotations.json", "")

        keypoints_file = keypoints_dir / f"{base}_keypoints.pkl"
        ann_file = annotations_dir / f"{base}_annotations.json"
        output_file = output_dir / f"{base}_windows.npz"

        if not keypoints_file.exists():
            raise FileNotFoundError(f"Keypoints file not found: {keypoints_file}")
        if not ann_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")

        print(f"\n{'='*70}")
        print(f"CREATING WINDOWS (MEDIAPIPE): {base}")
        print(f"Keypoints: {keypoints_file}")
        print(f"Annotations: {ann_file}")
        print(f"Output: {output_file}")
        print(f"Policy profile: {PolicyConfig.PROFILE}")
        print(f"{'='*70}")

        count = self.process_video(keypoints_file, ann_file, output_file)
        print(f"\n[OK] {base}: saved {count} windows")


def main():
    parser = argparse.ArgumentParser(description="Create windows from MediaPipe keypoints")
    parser.add_argument(
        "--keypoints-dir",
        type=str,
        default=str(Paths.PROCESSED_ROOT / "keypoints_mediapipe"),
        help="Directory containing *_keypoints.pkl from MediaPipe extractor",
    )
    parser.add_argument(
        "--annotations-dir",
        type=str,
        default=str(Paths.RAW_ANNOTATIONS),
        help="Directory containing *_annotations.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Paths.PROCESSED_ROOT / "windows_mediapipe"),
        help="Output directory for *_windows.npz",
    )
    parser.add_argument(
        "--video-name",
        type=str,
        default="",
        help="Optional single video base name (e.g., P002). If omitted, process all.",
    )
    parser.add_argument(
        "--policy-profile",
        type=str,
        default="",
        choices=["baseline", "short_aware", "custom", ""],
        help="Optional override for PolicyConfig.PROFILE",
    )
    args = parser.parse_args()

    if args.policy_profile:
        PolicyConfig.PROFILE = args.policy_profile

    creator = MediaPipeWindowCreator()

    if args.video_name:
        creator.process_single(
            keypoints_dir=args.keypoints_dir,
            annotations_dir=args.annotations_dir,
            output_dir=args.output_dir,
            video_name=args.video_name,
        )
    else:
        creator.process_all(
            keypoints_dir=args.keypoints_dir,
            annotations_dir=args.annotations_dir,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
