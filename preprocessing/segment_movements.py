"""Segment keypoints by movement annotations"""

import json
import numpy as np
import pickle
from pathlib import Path

from configs.paths import Paths


class MovementSegmenter:
    def __init__(self):
        pass

    def segment_video(self, keypoints_path, annotations_path, output_dir):
        """
        Segment keypoints into individual movements

        Args:
            keypoints_path: Path to keypoints pickle file
            annotations_path: Path to annotations JSON
            output_dir: Where to save segmented movements
        """
        # Load keypoints
        with open(keypoints_path, 'rb') as f:
            data = pickle.load(f)

        keypoints = data['keypoints']  # (num_frames, 26, 3)
        fps = data['fps']

        # Load annotations
        with open(annotations_path) as f:
            ann_data = json.load(f)

        annotations = ann_data['annotations']
        video_name = Path(keypoints_path).stem.replace('_keypoints', '')

        # Add end times
        for i in range(len(annotations)):
            annotations[i]['startTime'] = float(annotations[i]['startTime'])
            if i < len(annotations) - 1:
                annotations[i]['endTime'] = float(annotations[i + 1]['startTime'])
            else:
                annotations[i]['endTime'] = len(keypoints) / fps

        # Segment each movement
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for idx, ann in enumerate(annotations):
            start_frame = int(ann['startTime'] * fps)
            end_frame = int(ann['endTime'] * fps)

            # Extract keypoints for this movement
            movement_keypoints = keypoints[start_frame:end_frame]

            # Save
            output_file = output_dir / f"{video_name}_movement_{idx + 1}.npz"

            np.savez_compressed(
                output_file,
                keypoints=movement_keypoints,
                movement_name=ann['movement'],
                movement_number=idx + 1,
                start_frame=start_frame,
                end_frame=end_frame,
                duration=ann['endTime'] - ann['startTime'],
                fps=fps
            )

            print(f"✓ Saved movement {idx + 1}: {output_file.name}")

    def process_all(self, keypoints_dir, annotations_dir, output_dir):
        """Process all videos"""
        keypoints_dir = Path(keypoints_dir)
        annotations_dir = Path(annotations_dir)

        for keypoints_file in keypoints_dir.glob('*_keypoints.pkl'):
            base_name = keypoints_file.stem.replace('_keypoints', '')
            ann_file = annotations_dir / f"{base_name}_annotations.json"

            if not ann_file.exists():
                print(f"⚠ No annotations for {base_name}")
                continue

            print(f"\nSegmenting {base_name}...")
            self.segment_video(keypoints_file, ann_file, output_dir)


if __name__ == "__main__":
    segmenter = MovementSegmenter()

    segmenter.process_all(
        keypoints_dir=Paths.KEYPOINTS_DIR,
        annotations_dir=Paths.RAW_ANNOTATIONS,
        output_dir=Paths.MOVEMENTS_DIR
    )