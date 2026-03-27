"""Restructure reference keypoints into view/jang/movement/reference_name/keypoints.npz."""

import argparse
import shutil
from pathlib import Path


def restructure_reference_keypoints(source_root: Path, destination_root: Path):
    source_root = Path(source_root)
    destination_root = Path(destination_root)

    if not source_root.exists():
        raise FileNotFoundError(f"Source root not found: {source_root}")

    copied = 0
    skipped = 0

    for npz_path in source_root.glob('*/*/*/frames/*/keypoints.npz'):
        relative = npz_path.relative_to(source_root)
        # Expected: view/jang/reference_name/frames/movement_id/keypoints.npz
        parts = relative.parts
        if len(parts) != 6:
            skipped += 1
            continue

        view_name, jang_name, reference_name, frames_dir, movement_id, filename = parts
        if frames_dir != 'frames' or filename != 'keypoints.npz':
            skipped += 1
            continue

        destination_dir = destination_root / view_name / jang_name / movement_id / reference_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_file = destination_dir / 'keypoints.npz'

        shutil.copy2(npz_path, destination_file)
        copied += 1

    return copied, skipped


def main():
    parser = argparse.ArgumentParser(
        description='Restructure references into view/jang/movement/reference_name/keypoints.npz'
    )
    parser.add_argument('source_root', type=Path, help='Source root, e.g. compare/references_batch/Trim_0')
    parser.add_argument('destination_root', type=Path, help='Destination root, e.g. compare/references_by_movement')
    args = parser.parse_args()

    copied, skipped = restructure_reference_keypoints(args.source_root, args.destination_root)
    print(f'Copied: {copied}')
    print(f'Skipped: {skipped}')


if __name__ == '__main__':
    main()
