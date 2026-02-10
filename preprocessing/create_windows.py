"""
Create sliding windows for LSTM training - 22 CLASS VERSION

Handles annotation format: "X_Y_description" (e.g., "6_1_오른 지르기")
Supports 22 classes including sub-movements (14_1, 14_2, 16_1, 16_2)
"""
import numpy as np
import pickle
from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).parent.parent))
from configs.lstm_config import LSTMConfig
from configs.paths import Paths


# 22-class mapping: movement_id -> class_index
CLASS_MAPPING = {
    '0_1': 0,   # 기본준비
    '1_1': 1,   # 왼 앞서기 왼 아래막기
    '2_1': 2,   # 오른 앞서기 오른 지르기
    '3_1': 3,   # 오른 앞서기 오른 아래막기
    '4_1': 4,   # 왼 앞서기 왼 지르기
    '5_1': 5,   # 왼 앞굽이 왼 아래막기
    '6_1': 6,   # 오른 지르기 (SHORT)
    '7_1': 7,   # 오른 앞서기 왼 안막기
    '8_1': 8,   # 왼 앞서기 오른 지르기
    '9_1': 9,   # 왼 앞서기 오른 안막기
    '10_1': 10, # 오른 앞서기 왼 지르기
    '11_1': 11, # 오른 앞굽이 오른 아래막기
    '12_1': 12, # 왼 지르기 (SHORT)
    '13_1': 13, # 왼 앞서기 왼 얼굴막기
    '14_1': 14, # 오른발 앞차기 (SHORT)
    '14_2': 15, # 오른 앞서기 오른 지르기
    '15_1': 16, # 오른 앞서기 오른 얼굴막기
    '16_1': 17, # 왼발 앞차기 (SHORT)
    '16_2': 18, # 왼 앞서기 왼 지르기
    '17_1': 19, # 왼 앞굽이 왼 아래막기
    '18_1': 20, # 오른 앞굽이 오른 지르기(기합)
    '19_1': 21, # 기본바로
}

CLASS_NAMES = [
    '0_1_기본준비',
    '1_1_왼 앞서기 왼 아래막기',
    '2_1_오른 앞서기 오른 지르기',
    '3_1_오른 앞서기 오른 아래막기',
    '4_1_왼 앞서기 왼 지르기',
    '5_1_왼 앞굽이 왼 아래막기',
    '6_1_오른 지르기',
    '7_1_오른 앞서기 왼 안막기',
    '8_1_왼 앞서기 오른 지르기',
    '9_1_왼 앞서기 오른 안막기',
    '10_1_오른 앞서기 왼 지르기',
    '11_1_오른 앞굽이 오른 아래막기',
    '12_1_왼 지르기',
    '13_1_왼 앞서기 왼 얼굴막기',
    '14_1_오른발 앞차기',
    '14_2_오른 앞서기 오른 지르기',
    '15_1_오른 앞서기 오른 얼굴막기',
    '16_1_왼발 앞차기',
    '16_2_왼 앞서기 왼 지르기',
    '17_1_왼 앞굽이 왼 아래막기',
    '18_1_오른 앞굽이 오른 지르기(기합)',
    '19_1_기본바로',
]


def parse_movement_id(movement_str: str) -> str:
    """
    Parse movement ID from format "X_Y_description"

    Examples:
        "6_1_오른 지르기" -> "6_1"
        "14_2_오른 앞서기 오른 지르기" -> "14_2"
    """
    parts = str(movement_str).strip().split('_')

    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return f"{int(parts[0])}_{int(parts[1])}"

    if len(parts) >= 1 and parts[0].isdigit():
        return f"{int(parts[0])}_1"

    return movement_str


class SlidingWindowCreator:
    def __init__(self):
        self.config = LSTMConfig()
        self.sequence_length = self.config.SEQUENCE_LENGTH
        self.stride = self.config.STRIDE
        self.num_classes = self.config.NUM_CLASSES  # 22

    def validate_annotations(self, annotations, video_name):
        """Validate annotations have expected movements"""
        movement_ids = set()

        for ann in annotations:
            mov_id = parse_movement_id(ann.get('movement', ''))
            movement_ids.add(mov_id)

        # Check all 22 classes are present
        expected = set(CLASS_MAPPING.keys())
        missing = expected - movement_ids
        extra = movement_ids - expected

        if missing:
            print(f"[!] {video_name}: Missing movements: {sorted(missing)}")
            # Don't fail - some videos might have slight variations

        if extra:
            print(f"[!] {video_name}: Unknown movements: {sorted(extra)}")

        return True  # Continue processing even with warnings

    def create_windows_from_keypoints(self, keypoints_path, annotations_path):
        """Create sliding windows from full video keypoints"""
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

        # Validate
        self.validate_annotations(annotations, video_name)

        # Process annotations
        annotations_processed = []
        for ann in annotations:
            mov_id = parse_movement_id(ann.get('movement', ''))

            # Get class index from mapping
            if mov_id not in CLASS_MAPPING:
                print(f"[!] Unknown movement ID: {mov_id} in {video_name}")
                continue

            class_idx = CLASS_MAPPING[mov_id]

            annotations_processed.append({
                'movement': ann['movement'],
                'movement_id': mov_id,
                'class_idx': class_idx,
                'startTime': float(ann['startTime']),
            })

        # Sort by start time
        annotations_processed.sort(key=lambda x: x['startTime'])

        # Add end times
        for i in range(len(annotations_processed)):
            if i < len(annotations_processed) - 1:
                annotations_processed[i]['endTime'] = annotations_processed[i + 1]['startTime']
            else:
                annotations_processed[i]['endTime'] = len(keypoints) / fps

        # Create sliding windows
        windows = []

        for start_frame in range(0, len(keypoints) - self.sequence_length, self.stride):
            end_frame = start_frame + self.sequence_length

            window_keypoints = keypoints[start_frame:end_frame]

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

            if label_info['class_idx'] is not None:
                windows.append({
                    'keypoints': window_keypoints,
                    'label': label_info['class_idx'],
                    'movement_id': label_info['movement_id'],
                    'movement_name': label_info['movement'],
                    'percentage': label_info['percentage'],
                    'quality': label_info['quality'],
                    'start_frame': start_frame,
                    'end_frame': end_frame,
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

                overlaps[ann['movement_id']] = {
                    'movement': ann['movement'],
                    'movement_id': ann['movement_id'],
                    'class_idx': ann['class_idx'],
                    'percentage': percentage
                }

        if not overlaps:
            return {'class_idx': None, 'movement_id': None, 'percentage': 0, 'quality': 'none'}

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
            'class_idx': dominant[1]['class_idx'],
            'percentage': percentage,
            'quality': quality
        }

    def process_video(self, keypoints_path, annotations_path, output_path):
        """Process one video and save windows"""
        print(f"\nProcessing {keypoints_path.name}...")

        windows = self.create_windows_from_keypoints(keypoints_path, annotations_path)

        if not windows:
            print(f"[!] No windows created for {keypoints_path.name}")
            return 0

        # Separate by quality
        high_quality = [w for w in windows if w['quality'] == 'high']
        medium_quality = [w for w in windows if w['quality'] == 'medium']
        low_quality = [w for w in windows if w['quality'] == 'low']

        print(f"  Total: {len(windows)} | High: {len(high_quality)} | Med: {len(medium_quality)} | Low: {len(low_quality)}")

        # Use high + medium quality for training
        training_windows = high_quality + medium_quality

        if training_windows:
            X = np.array([w['keypoints'] for w in training_windows])
            y = np.array([w['label'] for w in training_windows])

            # Reshape: (N, seq_len, 26, 3) -> (N, seq_len, 78)
            X_flat = X.reshape(X.shape[0], X.shape[1], -1)

            np.savez_compressed(
                output_path,
                X=X_flat,
                y=y,
                movement_ids=[w['movement_id'] for w in training_windows],
                movement_names=[w['movement_name'] for w in training_windows],
                quality=[w['quality'] for w in training_windows],
                percentage=[w['percentage'] for w in training_windows]
            )

            print(f"  [OK] Saved {len(training_windows)} windows")
            return len(training_windows)

        return 0

    def process_all(self, keypoints_dir, annotations_dir, output_dir):
        """Process all videos"""
        keypoints_dir = Path(keypoints_dir)
        annotations_dir = Path(annotations_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        keypoint_files = sorted(keypoints_dir.glob('*_keypoints.pkl'))

        if not keypoint_files:
            print(f"[!] No keypoint files found in {keypoints_dir}")
            return

        print(f"\n{'='*70}")
        print(f"CREATING WINDOWS FOR {len(keypoint_files)} VIDEOS")
        print(f"Sequence length: {self.sequence_length} frames")
        print(f"Stride: {self.stride} frames")
        print(f"Number of classes: {self.num_classes}")
        print(f"{'='*70}")

        total_windows = 0
        successful = 0

        for keypoints_file in keypoint_files:
            base_name = keypoints_file.stem.replace('_keypoints', '')
            ann_file = annotations_dir / f"{base_name}_annotations.json"
            output_file = output_dir / f"{base_name}_windows.npz"

            if not ann_file.exists():
                print(f"\n[!] No annotations for {base_name}")
                continue

            try:
                count = self.process_video(keypoints_file, ann_file, output_file)
                if count > 0:
                    total_windows += count
                    successful += 1
            except Exception as e:
                print(f"[!] ERROR processing {base_name}: {e}")
                import traceback
                traceback.print_exc()

        # Summary
        print(f"\n{'='*70}")
        print("COMPLETE")
        print(f"{'='*70}")
        print(f"Videos processed: {successful}/{len(keypoint_files)}")
        print(f"Total windows: {total_windows}")
        print(f"Output: {output_dir}")

        # Show class distribution
        self._print_class_distribution(output_dir)

    def _print_class_distribution(self, output_dir):
        """Print distribution of classes across all windows"""
        output_dir = Path(output_dir)
        all_labels = []

        for npz_file in output_dir.glob('*.npz'):
            data = np.load(npz_file)
            all_labels.extend(data['y'].tolist())

        if not all_labels:
            return

        print(f"\n{'='*70}")
        print("CLASS DISTRIBUTION")
        print(f"{'='*70}")

        class_counts = {}
        for label in all_labels:
            class_counts[label] = class_counts.get(label, 0) + 1

        print(f"\n{'Class':<8} {'ID':<8} {'Count':<10} {'Percent':<10}")
        print(f"{'-'*40}")

        total = len(all_labels)
        for class_idx in sorted(class_counts.keys()):
            count = class_counts[class_idx]
            pct = 100.0 * count / total
            # Get movement ID from class index
            mov_id = list(CLASS_MAPPING.keys())[list(CLASS_MAPPING.values()).index(class_idx)]
            print(f"{class_idx:<8} {mov_id:<8} {count:<10} {pct:.1f}%")

        print(f"\nTotal samples: {total}")
        print(f"{'='*70}\n")


def main():
    creator = SlidingWindowCreator()

    creator.process_all(
        keypoints_dir=Paths.KEYPOINTS_DIR,
        annotations_dir=Paths.RAW_ANNOTATIONS,
        output_dir=Paths.WINDOWS_DIR
    )


if __name__ == "__main__":
    main()
