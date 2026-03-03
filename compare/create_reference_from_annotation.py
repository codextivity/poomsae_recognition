"""
Create Reference Database from Annotation JSON File

Uses manually labeled annotations for accurate movement boundaries.
Much more reliable than automatic detection for transition frames.

Usage:
    python create_reference_from_annotation.py \
        --video data/raw/videos/P011.mp4 \
        --annotation data/raw/annotations/P011_annotations.json \
        --output compare/references/master_22class.pkl
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import numpy as np
from pathlib import Path
import pickle
import argparse
import json
import sys

sys.path.append(str(Path(__file__).parent.parent))
from preprocessing.create_windows import CLASS_MAPPING, CLASS_NAMES


class ReferenceCreatorFromAnnotation:
    """Create reference database from annotation JSON file"""

    def __init__(self, device='cuda'):
        # Initialize RTMPose
        try:
            from rtmlib import BodyWithFeet
            self.pose_estimator = BodyWithFeet(
                to_openpose=False,
                mode='balanced',
                backend='onnxruntime',
                device=device
            )
            print("RTMPose initialized")
        except ImportError:
            raise ImportError("rtmlib not installed. Run: pip install rtmlib")

    def normalize_keypoints(self, keypoints):
        """Normalize keypoints: hip-centered + height-scaled"""
        coords = keypoints[:, :2].copy()
        conf = keypoints[:, 2:3].copy()

        if np.all(coords == 0):
            return keypoints

        # Hip-centering (keypoint 19 = pelvis in Halpe26)
        hip = coords[19:20, :]
        centered = coords - hip

        # Height normalization
        head = coords[0, :]
        feet = (coords[15, :] + coords[16, :]) / 2
        height = np.linalg.norm(head - feet)

        if height < 1e-3:
            shoulder_width = np.linalg.norm(coords[5, :] - coords[6, :])
            height = shoulder_width * 3 if shoulder_width > 1e-3 else 1.0

        normalized = centered / height
        return np.concatenate([normalized, conf], axis=1)

    def extract_keypoints_from_frame(self, frame):
        """Extract and normalize keypoints from frame"""
        keypoints, scores = self.pose_estimator(frame)

        if len(keypoints) > 0:
            kp = np.concatenate([keypoints[0], scores[0].reshape(-1, 1)], axis=1)
            kp_normalized = self.normalize_keypoints(kp)
            return kp_normalized, kp.copy()
        return np.zeros((26, 3)), np.zeros((26, 3))

    def extract_key_poses(self, movement_keypoints):
        """Extract key poses from movement sequence"""
        n_frames = len(movement_keypoints)

        if n_frames == 0:
            return {}

        key_poses = {
            'start': movement_keypoints[0],
            'end': movement_keypoints[-1],
        }

        if n_frames >= 3:
            mid_idx = n_frames // 2
            key_poses['middle'] = movement_keypoints[mid_idx]

        if n_frames >= 5:
            # Peak pose (maximum movement frame)
            diffs = np.linalg.norm(
                np.diff(movement_keypoints[:, :, :2], axis=0),
                axis=(1, 2)
            )
            peak_idx = np.argmax(diffs)
            key_poses['peak'] = movement_keypoints[peak_idx]

        return key_poses

    def parse_movement_id(self, movement_str):
        """Parse movement ID from format 'X_Y_description'"""
        parts = str(movement_str).strip().split('_')
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            return f"{parts[0]}_{parts[1]}"
        return None

    def create_reference(self, video_path, annotation_path, output_path):
        """Create reference database from annotation file"""
        video_path = Path(video_path)
        annotation_path = Path(annotation_path)

        if not video_path.exists():
            print(f"Video not found: {video_path}")
            return None

        if not annotation_path.exists():
            print(f"Annotation not found: {annotation_path}")
            return None

        # Load annotation
        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotation = json.load(f)

        annotations = annotation['annotations']
        total_movements = annotation.get('totalMovements', len(annotations))

        print(f"\n{'='*70}")
        print("CREATING REFERENCE FROM ANNOTATION")
        print(f"{'='*70}")
        print(f"Video: {video_path.name}")
        print(f"Annotation: {annotation_path.name}")
        print(f"Total movements: {total_movements}")
        print(f"{'='*70}\n")

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video info: {width}x{height}, {fps:.1f} FPS, {total_frames} frames")
        print(f"Duration: {total_frames/fps:.1f}s\n")

        # Build movement segments with start/end frames
        segments = []
        for i, ann in enumerate(annotations):
            movement_name = ann['movement']
            start_frame = ann['frame']
            start_time = float(ann['startTime'])

            # End frame is start of next movement (or end of video)
            if i + 1 < len(annotations):
                end_frame = annotations[i + 1]['frame'] - 1
            else:
                # Last movement: extend to end of video or reasonable duration
                end_frame = min(start_frame + int(fps * 3), total_frames - 1)

            # Parse movement ID
            movement_id = self.parse_movement_id(movement_name)
            if movement_id and movement_id in CLASS_MAPPING:
                class_idx = CLASS_MAPPING[movement_id]
            else:
                # Fallback
                class_idx = i

            segments.append({
                'movement_name': movement_name,
                'movement_id': movement_id,
                'class_idx': class_idx,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'start_time': start_time,
                'duration': (end_frame - start_frame) / fps
            })

        # Reference data structure
        reference_data = {
            'video_name': video_path.name,
            'annotation_file': annotation_path.name,
            'fps': fps,
            'total_frames': total_frames,
            'width': width,
            'height': height,
            'num_classes': 22,
            'class_names': CLASS_NAMES,
            'class_mapping': CLASS_MAPPING,
            'movements': []
        }

        # Extract keypoints for each segment
        for seg in segments:
            start_frame = seg['start_frame']
            end_frame = seg['end_frame']
            movement_name = seg['movement_name']

            print(f"Processing: {movement_name}")
            print(f"  Frames: {start_frame} - {end_frame} ({end_frame - start_frame + 1} frames)")

            movement_kps = []
            movement_kps_raw = []
            frame_nums = []

            # Seek to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            for frame_num in range(start_frame, end_frame + 1):
                ret, frame = cap.read()
                if not ret:
                    break

                # Extract keypoints
                kp_normalized, kp_raw = self.extract_keypoints_from_frame(frame)
                movement_kps.append(kp_normalized)
                movement_kps_raw.append(kp_raw)
                frame_nums.append(frame_num)

            movement_kps = np.array(movement_kps)
            movement_kps_raw = np.array(movement_kps_raw)

            # Extract key poses
            key_poses = self.extract_key_poses(movement_kps)

            movement_data = {
                'movement_number': seg['class_idx'],
                'movement_id': seg['movement_id'],
                'movement_name': movement_name,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'start_time': seg['start_time'],
                'duration': seg['duration'],
                'num_frames': len(movement_kps),
                'all_keypoints': movement_kps,           # Normalized (N, 26, 3)
                'all_keypoints_raw': movement_kps_raw,   # Raw pixel coords (N, 26, 3)
                'frame_numbers': frame_nums,
                'key_poses': key_poses,
                'avg_keypoint': movement_kps.mean(axis=0) if len(movement_kps) > 0 else np.zeros((26, 3))
            }

            reference_data['movements'].append(movement_data)
            print(f"  -> Extracted {len(movement_kps)} frames, key poses: {list(key_poses.keys())}")

        cap.release()

        # Save reference
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump(reference_data, f)

        # Print summary
        print(f"\n{'='*70}")
        print("REFERENCE CREATED SUCCESSFULLY")
        print(f"{'='*70}")
        print(f"Output: {output_path}")
        print(f"Movements: {len(reference_data['movements'])}")
        print(f"\n{'No':<4} {'ID':<6} {'Name':<35} {'Frames':<15} {'Duration':<10}")
        print("-" * 75)

        for i, mov in enumerate(reference_data['movements']):
            mov_id = mov.get('movement_id', '?')
            name = mov['movement_name'][:35]
            frames = f"{mov['start_frame']}-{mov['end_frame']}"
            duration = f"{mov['duration']:.2f}s"
            print(f"{i+1:<4} {mov_id:<6} {name:<35} {frames:<15} {duration:<10}")

        print(f"{'='*70}\n")

        return reference_data


def main():
    parser = argparse.ArgumentParser(description='Create reference from annotation')
    parser.add_argument('--video', required=True, help='Video file path')
    parser.add_argument('--annotation', required=True, help='Annotation JSON file path')
    parser.add_argument('--output', default='compare/references/master_22class.pkl',
                        help='Output reference file')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')

    args = parser.parse_args()

    creator = ReferenceCreatorFromAnnotation(args.device)
    creator.create_reference(args.video, args.annotation, args.output)


if __name__ == "__main__":
    main()
