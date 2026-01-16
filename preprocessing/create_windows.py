"""
Create sliding windows for LSTM training - FIXED VERSION

This version properly handles annotation format: "X. Movement Name"
and includes validation to ensure all 20 movements are present.
"""
import numpy as np
import pickle
from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).parent.parent))
from configs.lstm_config import LSTMConfig
from configs.paths import Paths


class SlidingWindowCreator:
    def __init__(self):
        self.config = LSTMConfig()
        self.sequence_length = self.config.SEQUENCE_LENGTH
        self.stride = self.config.STRIDE
        self.expected_movements = 20  # Taegeuk 1 has 20 movements

    def parse_movement_number(self, movement_str):
        """
        Parse movement number from format "X. Name" or just "X"

        Args:
            movement_str: String like "2. 아래막기" or "2"

        Returns:
            int: Movement number (e.g., 2)
        """
        movement_str = str(movement_str).strip()

        # Format: "2. 아래막기" -> extract "2"
        if '.' in movement_str:
            number_str = movement_str.split('.')[0].strip()
            return int(number_str)
        else:
            # Just a number
            return int(movement_str)

    def validate_annotations(self, annotations, video_name):
        """
        Validate that annotations have all expected movements

        Args:
            annotations: List of annotation dictionaries
            video_name: Name of video for error reporting

        Returns:
            bool: True if valid, False otherwise
        """
        movement_numbers = []

        for ann in annotations:
            try:
                mov_num = self.parse_movement_number(ann['movement'])
                movement_numbers.append(mov_num)
            except Exception as e:
                print(f"❌ ERROR parsing movement in {video_name}: {ann.get('movement')} - {e}")
                return False

        # Check for expected movements (1-20)
        expected = set(range(1, self.expected_movements + 1))
        found = set(movement_numbers)

        missing = expected - found
        extra = found - expected
        duplicates = [num for num in found if movement_numbers.count(num) > 1]

        if missing:
            print(f"❌ {video_name}: Missing movements: {sorted(missing)}")
            return False

        if extra:
            print(f"❌ {video_name}: Extra/invalid movements: {sorted(extra)}")
            return False

        if duplicates:
            print(f"❌ {video_name}: Duplicate movements: {sorted(set(duplicates))}")
            return False

        if len(movement_numbers) != self.expected_movements:
            print(f"❌ {video_name}: Expected {self.expected_movements} movements, found {len(movement_numbers)}")
            return False

        return True

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

        # Load annotations with UTF-8 encoding
        with open(annotations_path, encoding='utf-8') as f:
            ann_data = json.load(f)

        annotations = ann_data['annotations']
        video_name = Path(keypoints_path).stem.replace('_keypoints', '')

        # Validate annotations
        if not self.validate_annotations(annotations, video_name):
            print(f"⚠️  Skipping {video_name} due to annotation errors")
            return []

        # Prepare annotations with end times and PARSED movement numbers
        annotations_with_end = []

        for ann in annotations:
            # Parse movement number from format "X. Name"
            try:
                movement_num = self.parse_movement_number(ann['movement'])
            except Exception as e:
                print(f"❌ Failed to parse movement: {ann.get('movement')} - {e}")
                continue

            ann_processed = {
                'movement': ann['movement'],  # Keep original for reference
                'startTime': float(ann['startTime']),
                'movement_number': movement_num  # Use PARSED number, not index!
            }

            annotations_with_end.append(ann_processed)

        # Sort by movement number to ensure correct order
        annotations_with_end.sort(key=lambda x: x['movement_number'])

        # Add end times
        for i in range(len(annotations_with_end)):
            if i < len(annotations_with_end) - 1:
                annotations_with_end[i]['endTime'] = annotations_with_end[i + 1]['startTime']
            else:
                annotations_with_end[i]['endTime'] = len(keypoints) / fps

        # Create sliding windows
        windows = []

        for start_frame in range(0, len(keypoints) - self.sequence_length, self.stride):
            end_frame = start_frame + self.sequence_length

            # Extract window
            window_keypoints = keypoints[start_frame:end_frame]  # (90, 26, 3)

            # Calculate window time
            window_start_time = start_frame / fps
            window_end_time = end_frame / fps
            window_duration = self.sequence_length / fps

            # Find label using majority vote
            label_info = self._majority_vote_label(
                window_start_time,
                window_end_time,
                window_duration,
                annotations_with_end
            )

            if label_info['label'] is not None:
                windows.append({
                    'keypoints': window_keypoints,
                    'label': label_info['movement_number'] - 1,  # 0-indexed (1 -> 0, 2 -> 1, ..., 20 -> 19)
                    'movement_name': label_info['label'],
                    'percentage': label_info['percentage'],
                    'quality': label_info['quality'],
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'start_time': window_start_time,
                    'end_time': window_end_time
                })

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

                overlaps[ann['movement_number']] = {
                    'movement': ann['movement'],
                    'percentage': percentage
                }

        if not overlaps:
            return {'label': None, 'movement_number': None, 'percentage': 0, 'quality': 'none'}

        # Find dominant movement
        dominant = max(overlaps.items(), key=lambda x: x[1]['percentage'])
        movement_number = dominant[0]
        percentage = dominant[1]['percentage']

        # Determine quality
        if percentage >= 70:
            quality = 'high'
        elif percentage >= 50:
            quality = 'medium'
        else:
            quality = 'low'

        return {
            'label': dominant[1]['movement'],
            'movement_number': movement_number,
            'percentage': percentage,
            'quality': quality
        }

    def process_video(self, keypoints_path, annotations_path, output_path):
        """Process one video and save windows"""
        print(f"\nCreating windows for {keypoints_path.name}...")

        windows = self.create_windows_from_keypoints(keypoints_path, annotations_path)

        if not windows:
            print(f"❌ No windows created for {keypoints_path.name}")
            return

        # Separate by quality
        high_quality = [w for w in windows if w['quality'] == 'high']
        medium_quality = [w for w in windows if w['quality'] == 'medium']
        low_quality = [w for w in windows if w['quality'] == 'low']

        print(f"  Total windows: {len(windows)}")
        print(f"  High quality (≥70%): {len(high_quality)}")
        print(f"  Medium quality (50-70%): {len(medium_quality)}")
        print(f"  Low quality (<50%): {len(low_quality)}")

        # Show label distribution for this video
        label_counts = {}
        for w in windows:
            label = w['label']
            label_counts[label] = label_counts.get(label, 0) + 1

        print(f"  Label distribution:")
        for label in sorted(label_counts.keys()):
            print(f"    Movement {label}: {label_counts[label]} windows")

        # Save windows (use high + medium quality only)
        training_windows = high_quality + medium_quality

        if training_windows:
            # Stack all keypoints
            X = np.array([w['keypoints'] for w in training_windows])  # (num_windows, 90, 26, 3)
            y = np.array([w['label'] for w in training_windows])  # (num_windows,)

            # Reshape keypoints: (num_windows, 90, 78) - flatten x,y,conf for each frame
            X_flat = X.reshape(X.shape[0], X.shape[1], -1)

            np.savez_compressed(
                output_path,
                X=X_flat,
                y=y,
                movement_names=[w['movement_name'] for w in training_windows],
                quality=[w['quality'] for w in training_windows],
                percentage=[w['percentage'] for w in training_windows]
            )

            print(f"✓ Saved {len(training_windows)} windows to {output_path}")
        else:
            print("⚠️  No valid windows created")

    def process_all(self, keypoints_dir, annotations_dir, output_dir):
        """Process all videos"""
        keypoints_dir = Path(keypoints_dir)
        annotations_dir = Path(annotations_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        keypoint_files = sorted(keypoints_dir.glob('*_keypoints.pkl'))

        if not keypoint_files:
            print(f"❌ No keypoint files found in {keypoints_dir}")
            return

        print(f"\n{'='*70}")
        print(f"PROCESSING {len(keypoint_files)} VIDEOS")
        print(f"{'='*70}")

        successful = 0
        failed = 0
        total_windows = 0

        for keypoints_file in keypoint_files:
            base_name = keypoints_file.stem.replace('_keypoints', '')
            ann_file = annotations_dir / f"{base_name}_annotations.json"
            output_file = output_dir / f"{base_name}_windows.npz"

            if not ann_file.exists():
                print(f"\n⚠️  No annotations for {base_name}")
                failed += 1
                continue

            try:
                self.process_video(keypoints_file, ann_file, output_file)

                # Count windows created
                if output_file.exists():
                    data = np.load(output_file)
                    total_windows += len(data['y'])
                    successful += 1
                else:
                    failed += 1

            except Exception as e:
                print(f"❌ ERROR processing {base_name}: {e}")
                import traceback
                traceback.print_exc()
                failed += 1

        # Final summary
        print(f"\n{'='*70}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"✓ Successful: {successful}/{len(keypoint_files)}")
        print(f"❌ Failed: {failed}/{len(keypoint_files)}")
        print(f"Total windows created: {total_windows}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    creator = SlidingWindowCreator()

    creator.process_all(
        keypoints_dir=Paths.KEYPOINTS_DIR,
        annotations_dir=Paths.RAW_ANNOTATIONS,
        output_dir=Paths.WINDOWS_DIR
    )