"""Create sliding windows for LSTM training using annotation-derived class metadata."""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from configs.class_metadata import (
    get_short_class_indices,
    metadata_payload,
    parse_movement_id,
    resolve_annotation_file,
    save_class_metadata_json,
    build_class_metadata_from_annotation_files,
)
from configs.lstm_config import LSTMConfig
from configs.paths import Paths
from configs.policy_config import PolicyConfig


class SlidingWindowCreator:
    def __init__(self):
        self.config = LSTMConfig()
        self.sequence_length = self.config.SEQUENCE_LENGTH
        self.stride = self.config.STRIDE
        PolicyConfig.apply_profile()
        self.policy = PolicyConfig

        self.class_mapping = {}
        self.class_names = []
        self.num_classes = 0
        self.short_class_indices = set()

    def _set_class_metadata(self, class_mapping, class_names):
        self.class_mapping = dict(class_mapping)
        self.class_names = list(class_names)
        self.num_classes = len(self.class_mapping)
        self.short_class_indices = get_short_class_indices(self.class_mapping, self.policy.SHORT_MOVEMENT_IDS)
        self.policy.SHORT_CLASS_INDICES = set(self.short_class_indices)

    def _is_short_class(self, class_idx):
        return int(class_idx) in self.short_class_indices

    def _keep_window_for_training(self, window):
        quality = window.get('quality', 'none')
        class_idx = int(window.get('label', -1))
        overlap_pct = float(window.get('percentage', 0.0))

        if quality in ('high', 'medium'):
            return True

        if quality != 'low':
            return False

        if not self.policy.KEEP_LOW_QUALITY_WINDOWS:
            return False

        if self.policy.KEEP_LOW_FOR_SHORT_CLASSES_ONLY and not self._is_short_class(class_idx):
            return False

        return overlap_pct >= self.policy.LOW_QUALITY_MIN_OVERLAP_PCT

    def validate_annotations(self, annotations, video_name):
        movement_ids = []
        for ann in annotations:
            mov_id = parse_movement_id(ann.get('movement', ''))
            if mov_id:
                movement_ids.append(mov_id)

        if not movement_ids:
            print(f"[!] {video_name}: no valid movement IDs found")
            return False

        unknown = sorted({mov_id for mov_id in movement_ids if mov_id not in self.class_mapping})
        if unknown:
            print(f"[!] {video_name}: Unknown movements: {unknown}")

        return True

    def _annotation_start_time(self, ann):
        value = ann.get('startTime')
        return float(value) if value is not None else None

    def _annotation_end_time(self, ann):
        value = ann.get('endTime')
        if value is None or value == '':
            return None
        return float(value)

    def _build_processed_annotations(self, annotations, keypoints, fps, video_name):
        processed = []

        for ann in annotations:
            mov_id = parse_movement_id(ann.get('movement', ''))
            if mov_id not in self.class_mapping:
                print(f"[!] Unknown movement ID: {mov_id} in {video_name}")
                continue

            start_time = self._annotation_start_time(ann)
            if start_time is None:
                print(f"[!] Missing startTime for {ann.get('movement', mov_id)} in {video_name}")
                continue

            processed.append({
                'movement': ann.get('movement', mov_id),
                'movement_id': mov_id,
                'class_idx': self.class_mapping[mov_id],
                'startTime': start_time,
                'endTime': self._annotation_end_time(ann),
            })

        processed.sort(key=lambda x: x['startTime'])

        video_end = len(keypoints) / fps if fps > 0 else 0.0
        for i in range(len(processed)):
            if processed[i]['endTime'] is not None:
                continue
            if i < len(processed) - 1:
                processed[i]['endTime'] = processed[i + 1]['startTime']
            else:
                processed[i]['endTime'] = video_end

        return processed

    def create_windows_from_keypoints(self, keypoints_path, annotations_path):
        with open(keypoints_path, 'rb') as f:
            data = pickle.load(f)

        keypoints = data['keypoints']
        fps = data['fps']

        with open(annotations_path, 'r', encoding='utf-8') as f:
            ann_data = json.load(f)

        annotations = ann_data['annotations']
        video_name = Path(keypoints_path).stem.replace('_keypoints', '')
        self.validate_annotations(annotations, video_name)

        annotations_processed = self._build_processed_annotations(annotations, keypoints, fps, video_name)

        windows = []
        for start_frame in range(0, len(keypoints) - self.sequence_length, self.stride):
            end_frame = start_frame + self.sequence_length
            window_keypoints = keypoints[start_frame:end_frame]

            window_start_time = start_frame / fps
            window_end_time = end_frame / fps
            window_duration = self.sequence_length / fps

            label_info = self._majority_vote_label(
                window_start_time,
                window_end_time,
                window_duration,
                annotations_processed,
            )

            if label_info['class_idx'] is None:
                continue

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
                    'percentage': percentage,
                }

        if not overlaps:
            return {'class_idx': None, 'movement_id': None, 'percentage': 0.0, 'quality': 'none'}

        dominant = max(overlaps.items(), key=lambda x: x[1]['percentage'])
        percentage = dominant[1]['percentage']

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
            'quality': quality,
        }

    def process_video(self, keypoints_path, annotations_path, output_path):
        print(f"\nProcessing {Path(keypoints_path).name}...")
        windows = self.create_windows_from_keypoints(keypoints_path, annotations_path)

        if not windows:
            print(f"[!] No windows created for {Path(keypoints_path).name}")
            return 0

        high_quality = [w for w in windows if w['quality'] == 'high']
        medium_quality = [w for w in windows if w['quality'] == 'medium']
        low_quality = [w for w in windows if w['quality'] == 'low']

        print(f"  Total: {len(windows)} | High: {len(high_quality)} | Med: {len(medium_quality)} | Low: {len(low_quality)}")

        training_windows = [w for w in windows if self._keep_window_for_training(w)]
        dropped_windows = len(windows) - len(training_windows)
        kept_low = sum(1 for w in training_windows if w['quality'] == 'low')
        kept_low_short = sum(1 for w in training_windows if w['quality'] == 'low' and self._is_short_class(w['label']))

        print(
            '  Policy:'
            f' keep_low={self.policy.KEEP_LOW_QUALITY_WINDOWS},'
            f' short_only={self.policy.KEEP_LOW_FOR_SHORT_CLASSES_ONLY},'
            f' low_min_overlap={self.policy.LOW_QUALITY_MIN_OVERLAP_PCT:.1f}%'
        )
        print(f"  Selected: {len(training_windows)} (dropped: {dropped_windows}, kept_low: {kept_low}, kept_low_short: {kept_low_short})")

        if not training_windows:
            return 0

        X = np.array([w['keypoints'] for w in training_windows])
        y = np.array([w['label'] for w in training_windows], dtype=np.int64)
        X_flat = X.reshape(X.shape[0], X.shape[1], -1)
        class_meta = metadata_payload(self.class_mapping, self.class_names, self.policy.SHORT_MOVEMENT_IDS)

        np.savez_compressed(
            output_path,
            X=X_flat,
            y=y,
            movement_ids=np.array([w['movement_id'] for w in training_windows], dtype=object),
            movement_names=np.array([w['movement_name'] for w in training_windows], dtype=object),
            quality=np.array([w['quality'] for w in training_windows], dtype=object),
            percentage=np.array([w['percentage'] for w in training_windows], dtype=np.float32),
            class_names=np.array(class_meta['class_names'], dtype=object),
            class_mapping_json=json.dumps(class_meta['class_mapping'], ensure_ascii=False),
            num_classes=np.int32(class_meta['num_classes']),
            short_class_indices=np.array(class_meta['short_class_indices'], dtype=np.int32),
            short_movement_ids=np.array(class_meta['short_movement_ids'], dtype=object),
        )

        print(f"  [OK] Saved {len(training_windows)} windows")
        return len(training_windows)

    def process_all(self, keypoints_dir, annotations_dir, output_dir):
        keypoints_dir = Path(keypoints_dir)
        annotations_dir = Path(annotations_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        keypoint_files = sorted(keypoints_dir.glob('*_keypoints.pkl'))
        if not keypoint_files:
            print(f"[!] No keypoint files found in {keypoints_dir}")
            return

        matched = []
        for keypoints_file in keypoint_files:
            base_name = keypoints_file.stem.replace('_keypoints', '')
            ann_file = resolve_annotation_file(annotations_dir, base_name)
            if ann_file is None:
                print(f"\n[!] No annotations for {base_name}")
                continue
            matched.append((keypoints_file, ann_file, output_dir / f'{base_name}_windows.npz'))

        if not matched:
            print(f"[!] No matched keypoint/annotation pairs found")
            return

        class_mapping, class_names = build_class_metadata_from_annotation_files([ann for _, ann, _ in matched])
        self._set_class_metadata(class_mapping, class_names)
        metadata_path = save_class_metadata_json(output_dir, self.class_mapping, self.class_names, self.policy.SHORT_MOVEMENT_IDS)

        print(f"\n{'=' * 70}")
        print(f"CREATING WINDOWS FOR {len(matched)} VIDEOS")
        print(f"Sequence length: {self.sequence_length} frames")
        print(f"Stride: {self.stride} frames")
        print(f"Number of classes: {self.num_classes}")
        print(f"Short class indices: {sorted(self.short_class_indices)}")
        print(f"Policy profile: {self.policy.PROFILE}")
        print(f"Class metadata: {metadata_path}")
        print(f"{'=' * 70}")

        total_windows = 0
        successful = 0

        for keypoints_file, ann_file, output_file in matched:
            try:
                count = self.process_video(keypoints_file, ann_file, output_file)
                if count > 0:
                    total_windows += count
                    successful += 1
            except Exception as exc:
                print(f"[!] ERROR processing {keypoints_file.stem}: {exc}")
                import traceback
                traceback.print_exc()

        print(f"\n{'=' * 70}")
        print('COMPLETE')
        print(f"{'=' * 70}")
        print(f"Videos processed: {successful}/{len(matched)}")
        print(f"Total windows: {total_windows}")
        print(f"Output: {output_dir}")
        self._print_class_distribution(output_dir)

    def _print_class_distribution(self, output_dir):
        output_dir = Path(output_dir)
        all_labels = []
        for npz_file in output_dir.glob('*_windows.npz'):
            data = np.load(npz_file, allow_pickle=True)
            all_labels.extend(data['y'].tolist())

        if not all_labels:
            return

        inverse = {idx: mov_id for mov_id, idx in self.class_mapping.items()}
        print(f"\n{'=' * 70}")
        print('CLASS DISTRIBUTION')
        print(f"{'=' * 70}")
        print(f"\n{'Class':<8} {'ID':<8} {'Count':<10} {'Percent':<10}")
        print('-' * 40)

        total = len(all_labels)
        class_counts = {}
        for label in all_labels:
            label = int(label)
            class_counts[label] = class_counts.get(label, 0) + 1

        for class_idx in sorted(class_counts):
            count = class_counts[class_idx]
            pct = 100.0 * count / total
            mov_id = inverse.get(class_idx, f'class_{class_idx}')
            print(f"{class_idx:<8} {mov_id:<8} {count:<10} {pct:.1f}%")

        print(f"\nTotal samples: {total}")
        print(f"{'=' * 70}\n")


def main():
    creator = SlidingWindowCreator()
    creator.process_all(
        keypoints_dir=Paths.KEYPOINTS_DIR,
        annotations_dir=Paths.RAW_ANNOTATIONS,
        output_dir=Paths.WINDOWS_DIR,
    )


if __name__ == '__main__':
    main()
