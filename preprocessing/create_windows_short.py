"""
Create sliding windows for SHORT MOVEMENT LSTM training

Specialized window creator for fast movements:
- 6_1 (right punch)
- 12_1 (left punch)
- 14_1 (right front kick)
- 16_1 (left front kick)

Uses 16-frame windows with stride 2 for dense sampling.
"""
import numpy as np
import pickle
from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).parent.parent))
from configs.lstm_config_short import LSTMConfigShort
from configs.paths import Paths


class ShortMovementWindowCreator:
    """Creates windows optimized for short/fast movement detection"""

    # Target short movements (movement IDs from annotations)
    # Format in annotations: "X_Y_description"
    # These are the fast movements that need specialized detection:
    #   6_1: 오른 지르기 (right punch) - 19 frames mean, 9 frames min
    #   12_1: 왼 지르기 (left punch) - 20 frames mean, 12 frames min
    #   14_1: 오른발 앞차기 (right front kick) - 25 frames mean, 5 frames min!
    #   16_1: 왼발 앞차기 (left front kick) - 24 frames mean, 8 frames min
    SHORT_MOVEMENTS = {
        '6_1': 0,   # Right punch -> class 0
        '12_1': 1,  # Left punch -> class 1
        '14_1': 2,  # Right front kick -> class 2
        '16_1': 3,  # Left front kick -> class 3
    }
    OTHER_CLASS = 4  # Non-short movements

    # Map short model class back to 22-class main model indices
    SHORT_TO_MAIN_CLASS = {
        0: 6,   # 6_1
        1: 12,  # 12_1
        2: 14,  # 14_1
        3: 17,  # 16_1 (note: class 17 in 22-class system)
    }

    def __init__(self):
        self.config = LSTMConfigShort()
        self.sequence_length = self.config.SEQUENCE_LENGTH  # 16 frames
        self.stride = self.config.STRIDE  # 2 frames

    def parse_movement_id(self, movement_str):
        """
        Parse movement ID from format "X_Y_description"

        Args:
            movement_str: String like "6_1_오른 지르기"

        Returns:
            str: Movement ID like "6_1"
        """
        parts = str(movement_str).strip().split('_')

        # Format: "X_Y_description" -> "X_Y"
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            return f"{int(parts[0])}_{int(parts[1])}"

        # Format: "X_description" -> "X"
        if len(parts) >= 1 and parts[0].isdigit():
            return str(int(parts[0]))

        return movement_str

    def is_short_movement(self, movement_id):
        """Check if movement is one of the target short movements"""
        return movement_id in self.SHORT_MOVEMENTS

    def get_class_label(self, movement_id):
        """
        Get class label for movement

        Returns:
            int: 0-3 for short movements, 4 for others
        """
        if movement_id in self.SHORT_MOVEMENTS:
            return self.SHORT_MOVEMENTS[movement_id]
        return self.OTHER_CLASS

    def create_windows_from_keypoints(self, keypoints_path, annotations_path):
        """
        Create sliding windows from full video keypoints

        Args:
            keypoints_path: Path to keypoints pickle
            annotations_path: Path to annotations JSON

        Returns:
            windows: List of window dictionaries
        """
        # Load keypoints
        with open(keypoints_path, 'rb') as f:
            data = pickle.load(f)

        keypoints = data['keypoints']  # (num_frames, 26, 3)
        fps = data['fps']

        # Load annotations
        with open(annotations_path, encoding='utf-8') as f:
            ann_data = json.load(f)

        annotations = ann_data['annotations']
        video_name = Path(keypoints_path).stem.replace('_keypoints', '')

        # Prepare annotations with parsed movement IDs
        annotations_processed = []

        for ann in annotations:
            movement_id = self.parse_movement_id(ann['movement'])

            annotations_processed.append({
                'movement': ann['movement'],
                'movement_id': movement_id,
                'startTime': float(ann['startTime']),
                'is_short': self.is_short_movement(movement_id),
                'class_label': self.get_class_label(movement_id)
            })

        # Sort by start time
        annotations_processed.sort(key=lambda x: x['startTime'])

        # Add end times
        for i in range(len(annotations_processed)):
            if i < len(annotations_processed) - 1:
                annotations_processed[i]['endTime'] = annotations_processed[i + 1]['startTime']
            else:
                annotations_processed[i]['endTime'] = len(keypoints) / fps

        # Create sliding windows with dense sampling
        windows = []
        short_count = 0
        other_count = 0

        for start_frame in range(0, len(keypoints) - self.sequence_length, self.stride):
            end_frame = start_frame + self.sequence_length

            # Extract window
            window_keypoints = keypoints[start_frame:end_frame]

            # Calculate window time
            window_start_time = start_frame / fps
            window_end_time = end_frame / fps
            window_duration = self.sequence_length / fps

            # Find label using majority vote
            label_info = self._majority_vote_label(
                window_start_time,
                window_end_time,
                window_duration,
                annotations_processed
            )

            if label_info['class_label'] is not None and label_info['percentage'] >= 50:
                windows.append({
                    'keypoints': window_keypoints,
                    'label': label_info['class_label'],
                    'movement_id': label_info['movement_id'],
                    'movement_name': label_info['movement'],
                    'is_short': label_info['is_short'],
                    'percentage': label_info['percentage'],
                    'quality': label_info['quality'],
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                })

                if label_info['is_short']:
                    short_count += 1
                else:
                    other_count += 1

        print(f"  {video_name}: {short_count} short movement windows, {other_count} other windows")
        return windows

    def _majority_vote_label(self, window_start, window_end, window_duration, annotations):
        """Calculate label using majority vote algorithm"""
        overlaps = {}

        for ann in annotations:
            overlap_start = max(window_start, ann['startTime'])
            overlap_end = min(window_end, ann['endTime'])

            if overlap_start < overlap_end:
                overlap_duration = overlap_end - overlap_start
                percentage = (overlap_duration / window_duration) * 100

                overlaps[ann['movement_id']] = {
                    'movement': ann['movement'],
                    'movement_id': ann['movement_id'],
                    'is_short': ann['is_short'],
                    'class_label': ann['class_label'],
                    'percentage': percentage
                }

        if not overlaps:
            return {'class_label': None, 'movement_id': None, 'percentage': 0, 'quality': 'none'}

        # Find dominant movement
        dominant = max(overlaps.items(), key=lambda x: x[1]['percentage'])
        percentage = dominant[1]['percentage']

        # Determine quality
        if percentage >= 70:
            quality = 'high'
        elif percentage >= 50:
            quality = 'medium'
        else:
            quality = 'low'

        return {
            'movement': dominant[1]['movement'],
            'movement_id': dominant[1]['movement_id'],
            'is_short': dominant[1]['is_short'],
            'class_label': dominant[1]['class_label'],
            'percentage': percentage,
            'quality': quality
        }

    def process_all(self, keypoints_dir, annotations_dir, output_dir):
        """Process all videos and create short movement dataset"""
        keypoints_dir = Path(keypoints_dir)
        annotations_dir = Path(annotations_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        keypoint_files = sorted(keypoints_dir.glob('*_keypoints.pkl'))

        if not keypoint_files:
            print(f"[!] No keypoint files found in {keypoints_dir}")
            return

        print(f"\n{'='*70}")
        print("CREATING SHORT MOVEMENT WINDOWS")
        print(f"Window: {self.sequence_length} frames, Stride: {self.stride} frames")
        print(f"Target movements: {list(self.SHORT_MOVEMENTS.keys())}")
        print(f"{'='*70}\n")

        all_windows = []

        for keypoints_file in keypoint_files:
            base_name = keypoints_file.stem.replace('_keypoints', '')
            ann_file = annotations_dir / f"{base_name}_annotations.json"

            if not ann_file.exists():
                print(f"  [!] No annotations for {base_name}")
                continue

            try:
                windows = self.create_windows_from_keypoints(keypoints_file, ann_file)
                all_windows.extend(windows)
            except Exception as e:
                print(f"  [!] ERROR processing {base_name}: {e}")
                import traceback
                traceback.print_exc()

        if not all_windows:
            print("[!] No windows created!")
            return

        # Separate short and other windows
        short_windows = [w for w in all_windows if w['is_short']]
        other_windows = [w for w in all_windows if not w['is_short']]

        print(f"\n{'='*70}")
        print("DATASET SUMMARY")
        print(f"{'='*70}")
        print(f"Total short movement windows: {len(short_windows)}")
        print(f"Total other windows: {len(other_windows)}")

        # Count per class
        class_counts = {}
        for w in all_windows:
            label = w['label']
            class_counts[label] = class_counts.get(label, 0) + 1

        class_names = ['6_1 (R punch)', '12_1 (L punch)', '14_1 (R kick)', '16_1 (L kick)', 'other']
        print("\nPer-class distribution:")
        for label in sorted(class_counts.keys()):
            print(f"  Class {label} ({class_names[label]}): {class_counts[label]} windows")

        # Balance the dataset - undersample "other" class
        # Keep ratio of about 1:1 (short:other)
        if len(other_windows) > len(short_windows) * 2:
            np.random.seed(42)
            other_indices = np.random.choice(
                len(other_windows),
                size=len(short_windows) * 2,
                replace=False
            )
            other_windows_balanced = [other_windows[i] for i in other_indices]
            print(f"\nBalanced 'other' class: {len(other_windows)} -> {len(other_windows_balanced)}")
            training_windows = short_windows + other_windows_balanced
        else:
            training_windows = all_windows

        # Shuffle
        np.random.seed(42)
        indices = np.random.permutation(len(training_windows))
        training_windows = [training_windows[i] for i in indices]

        # Save
        X = np.array([w['keypoints'] for w in training_windows])
        y = np.array([w['label'] for w in training_windows])

        # Reshape: (num_windows, 16, 26, 3) -> (num_windows, 16, 78)
        X_flat = X.reshape(X.shape[0], X.shape[1], -1)

        output_path = output_dir / "short_movements_windows.npz"
        np.savez_compressed(
            output_path,
            X=X_flat,
            y=y,
            movement_ids=[w['movement_id'] for w in training_windows],
            movement_names=[w['movement_name'] for w in training_windows],
            quality=[w['quality'] for w in training_windows],
        )

        print(f"\n[OK] Saved {len(training_windows)} windows to {output_path}")
        print(f"  Shape X: {X_flat.shape}")
        print(f"  Shape y: {y.shape}")

        # Final class distribution
        final_counts = {}
        for label in y:
            final_counts[label] = final_counts.get(label, 0) + 1
        print("\nFinal class distribution:")
        for label in sorted(final_counts.keys()):
            print(f"  Class {label} ({class_names[label]}): {final_counts[label]} windows")


def main():
    creator = ShortMovementWindowCreator()

    creator.process_all(
        keypoints_dir=Paths.KEYPOINTS_DIR,
        annotations_dir=Paths.RAW_ANNOTATIONS,
        output_dir=Paths.WINDOWS_DIR
    )


if __name__ == "__main__":
    main()
