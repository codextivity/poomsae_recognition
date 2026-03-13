"""
Create Complete Reference Database with Images and Keypoints

Features:
- Reads annotation JSON for accurate movement boundaries
- Trims first N frames from movements 1-21 (transition frames)
- Extracts keypoints with RTMPose or MediaPipe
- Saves images in movement folders
- Saves keypoints.npz in each folder
- Creates master_22class.pkl reference database
- Creates index.json

Usage:
    python create_reference_complete.py \
        --video data/reference/annotations/P001.mp4 \
        --annotation data/reference/annotations/P001_annotations.json \
        --output compare/references
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


class ReferenceCreator:
    """Create complete reference database with images and keypoints"""

    # Normalization reference joints (hip joints in Halpe26)
    NORM_CENTER_JOINT = 19  # Pelvis/hip center
    NORM_REF_JOINTS = [11, 12]  # Left hip, Right hip

    def __init__(self, device='cuda', trim_start_frames=3, pose_backend='rtmpose'):
        """
        Args:
            device: 'cuda' or 'cpu'
            trim_start_frames: Number of frames to trim from start of movements 1-21
            pose_backend: 'rtmpose' or 'mediapipe'
        """
        self.device = device
        self.trim_start_frames = trim_start_frames
        self.pose_backend = str(pose_backend).strip().lower()
        self.pose_estimator = None
        self.mp_extractor = None

        self._init_pose_backend()

    def _init_pose_backend(self):
        """Initialize selected pose backend."""
        if self.pose_backend == 'mediapipe':
            try:
                from preprocessing.extract_keypoints_mediapipe import MediaPipeExtractor
                self.mp_extractor = MediaPipeExtractor(
                    normalize=False,
                    model_complexity=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    static_image_mode=False,
                )
                print("MediaPipe initialized")
                return
            except ImportError:
                raise ImportError("mediapipe not installed. Run: pip install mediapipe")

        if self.pose_backend != 'rtmpose':
            raise ValueError(f"Unsupported pose backend: {self.pose_backend}")

        try:
            from rtmlib import BodyWithFeet
            self.pose_estimator = BodyWithFeet(
                to_openpose=False,
                mode='balanced',
                backend='onnxruntime',
                device=self.device
            )
            print(f"RTMPose initialized (device: {self.device})")
        except ImportError:
            raise ImportError("rtmlib not installed. Run: pip install rtmlib")

    def normalize_keypoints(self, keypoints):
        """
        Normalize keypoints: hip-centered + height-scaled

        Args:
            keypoints: (26, 3) array [x, y, conf]

        Returns:
            normalized: (26, 3) array [x_norm, y_norm, conf]
        """
        coords = keypoints[:, :2].copy()
        conf = keypoints[:, 2:3].copy()

        if np.all(coords == 0):
            return keypoints

        # Hip-centering (keypoint 19 = pelvis in Halpe26)
        hip = coords[self.NORM_CENTER_JOINT:self.NORM_CENTER_JOINT + 1, :]
        centered = coords - hip

        # Height normalization (head to feet)
        head = coords[0, :]
        feet = (coords[15, :] + coords[16, :]) / 2
        height = np.linalg.norm(head - feet)

        if height < 1e-3:
            shoulder_width = np.linalg.norm(coords[5, :] - coords[6, :])
            height = shoulder_width * 3 if shoulder_width > 1e-3 else 1.0

        normalized = centered / height
        return np.concatenate([normalized, conf], axis=1)

    def extract_keypoints_from_frame(self, frame):
        """Extract keypoints from a single frame"""
        if self.pose_backend == 'mediapipe':
            kp_raw = self.mp_extractor._extract_halpe26_from_frame(frame)
            kp_norm = self.normalize_keypoints(kp_raw)
            return kp_norm.astype(np.float32), kp_raw.astype(np.float32)

        keypoints, scores = self.pose_estimator(frame)

        if len(keypoints) > 0:
            kp_raw = np.concatenate([keypoints[0], scores[0].reshape(-1, 1)], axis=1)
            kp_norm = self.normalize_keypoints(kp_raw)
            return kp_norm.astype(np.float32), kp_raw.astype(np.float32)

        return np.zeros((26, 3), dtype=np.float32), np.zeros((26, 3), dtype=np.float32)

    def extract_key_poses(self, keypoints_norm):
        """Extract key poses from movement sequence"""
        n_frames = len(keypoints_norm)

        if n_frames == 0:
            return {}

        key_poses = {
            'start': keypoints_norm[0],
            'end': keypoints_norm[-1],
        }

        if n_frames >= 3:
            mid_idx = n_frames // 2
            key_poses['middle'] = keypoints_norm[mid_idx]

        if n_frames >= 5:
            # Peak pose (maximum movement frame)
            diffs = np.linalg.norm(
                np.diff(keypoints_norm[:, :, :2], axis=0),
                axis=(1, 2)
            )
            peak_idx = np.argmax(diffs)
            key_poses['peak'] = keypoints_norm[peak_idx]

        return key_poses

    def get_movement_id(self, movement_name):
        """Extract movement ID like '0_1', '14_2' from movement name"""
        parts = str(movement_name).strip().split('_')
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            return f"{parts[0]}_{parts[1]}"
        return None

    def create_meta(self, movement_id, movement_name, fps):
        """Create metadata JSON string for .npz file"""
        meta = {
            "version": "kp_v1",
            "movement_id": movement_id,
            "movement_name": movement_name,
            "pose_backend": self.pose_backend,
            "fps": int(fps),
            "norm": {
                "center": "hip",
                "scale": "height",
                "ref_joints": self.NORM_REF_JOINTS,
                "notes": "x,y normalized by torso height, centered at hip (joint 19)"
            }
        }
        return json.dumps(meta, ensure_ascii=False)

    def create_reference(self, video_path, annotation_path, output_dir):
        """
        Create complete reference database

        Args:
            video_path: Path to video file
            annotation_path: Path to annotation JSON
            output_dir: Output directory for all files
        """
        video_path = Path(video_path)
        annotation_path = Path(annotation_path)
        output_dir = Path(output_dir)

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

        print(f"\n{'='*70}")
        print("CREATING COMPLETE REFERENCE DATABASE")
        print(f"{'='*70}")
        print(f"Video: {video_path.name}")
        print(f"Annotation: {annotation_path.name}")
        print(f"Output: {output_dir}")
        print(f"Pose backend: {self.pose_backend}")
        print(f"Trim start frames: {self.trim_start_frames} (for movements 1-21)")
        print(f"{'='*70}\n")

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video: {width}x{height}, {fps:.1f} FPS, {total_frames} frames")
        print(f"Duration: {total_frames/fps:.1f}s\n")

        # Create output directories
        frames_dir = output_dir / 'frames'
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Build movement segments
        segments = []
        for i, ann in enumerate(annotations):
            movement_name = ann['movement']
            start_frame = ann['frame']
            movement_id = self.get_movement_id(movement_name)

            # End frame is start of next movement - 1
            if i + 1 < len(annotations):
                end_frame = annotations[i + 1]['frame'] - 1
            else:
                end_frame = min(start_frame + int(fps * 3), total_frames - 1)

            # Apply trimming for movements 1-21 (not the first movement 0_1)
            trim_frames = 0
            if i > 0:  # Not the first movement
                trim_frames = self.trim_start_frames
                start_frame = start_frame + trim_frames

            # Get class index
            if movement_id and movement_id in CLASS_MAPPING:
                class_idx = CLASS_MAPPING[movement_id]
            else:
                class_idx = i

            segments.append({
                'index': i,
                'movement_id': movement_id,
                'movement_name': movement_name,
                'class_idx': class_idx,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'original_start': ann['frame'],
                'trimmed_frames': trim_frames,
                'duration': (end_frame - start_frame + 1) / fps
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
            'trim_start_frames': self.trim_start_frames,
            'class_names': CLASS_NAMES,
            'class_mapping': CLASS_MAPPING,
            'pose_backend': self.pose_backend,
            'movements': []
        }

        # Index data
        index_data = {
            'video': video_path.name,
            'annotation': annotation_path.name,
            'fps': fps,
            'width': width,
            'height': height,
            'trim_start_frames': self.trim_start_frames,
            'pose_backend': self.pose_backend,
            'movements': []
        }

        # Process each movement
        for seg in segments:
            movement_id = seg['movement_id']
            movement_name = seg['movement_name']
            start_frame = seg['start_frame']
            end_frame = seg['end_frame']
            num_frames = end_frame - start_frame + 1

            print(f"[{seg['index']+1:02d}/22] {movement_id} - {movement_name}")
            if seg['trimmed_frames'] > 0:
                print(f"        Original start: {seg['original_start']}, Trimmed: {seg['trimmed_frames']} frames")
            print(f"        Frames: {start_frame} - {end_frame} ({num_frames} frames)")

            # Create movement folder
            movement_dir = frames_dir / movement_id
            movement_dir.mkdir(parents=True, exist_ok=True)

            # Storage for this movement
            keypoints_norm_list = []
            keypoints_raw_list = []
            frame_indices = []

            # Seek to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            for frame_num in range(start_frame, end_frame + 1):
                ret, frame = cap.read()
                if not ret:
                    break

                # Extract keypoints
                kp_norm, kp_raw = self.extract_keypoints_from_frame(frame)
                keypoints_norm_list.append(kp_norm)
                keypoints_raw_list.append(kp_raw)
                frame_indices.append(frame_num)

                # Save frame as image
                img_filename = f"frame_{frame_num:05d}.jpg"
                cv2.imwrite(str(movement_dir / img_filename), frame)

            # Convert to numpy arrays
            keypoints_norm = np.array(keypoints_norm_list, dtype=np.float32)  # (N, 26, 3)
            keypoints_raw = np.array(keypoints_raw_list, dtype=np.float32)    # (N, 26, 3)
            frames_array = np.array(frame_indices, dtype=np.int32)            # (N,)
            img_wh = np.array([width, height], dtype=np.int32)                # (2,)

            # Create meta JSON
            meta_str = self.create_meta(movement_id, movement_name, fps)

            # Save keypoints.npz
            npz_path = movement_dir / 'keypoints.npz'
            np.savez(
                npz_path,
                keypoints_norm=keypoints_norm,
                frames=frames_array,
                keypoints_raw=keypoints_raw,
                img_wh=img_wh,
                meta=meta_str
            )

            # Extract key poses
            key_poses = self.extract_key_poses(keypoints_norm)

            # Add to reference data
            movement_data = {
                'movement_number': seg['class_idx'],
                'movement_id': movement_id,
                'movement_name': movement_name,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'original_start': seg['original_start'],
                'trimmed_frames': seg['trimmed_frames'],
                'duration': seg['duration'],
                'num_frames': len(keypoints_norm),
                'all_keypoints': keypoints_norm,
                'all_keypoints_raw': keypoints_raw,
                'frame_numbers': frame_indices,
                'key_poses': key_poses,
                'avg_keypoint': keypoints_norm.mean(axis=0) if len(keypoints_norm) > 0 else np.zeros((26, 3))
            }
            reference_data['movements'].append(movement_data)

            # Add to index
            index_data['movements'].append({
                'index': seg['index'],
                'movement_id': movement_id,
                'name': movement_name,
                'folder': movement_id,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'original_start': seg['original_start'],
                'trimmed_frames': seg['trimmed_frames'],
                'num_frames': len(keypoints_norm),
                'duration': seg['duration']
            })

            print(f"        Saved: {len(keypoints_norm)} images + keypoints.npz")

        cap.release()

        # Save master reference PKL
        pkl_path = output_dir / 'master_22class.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(reference_data, f)
        print(f"\n[OK] Saved: {pkl_path}")

        # Save index JSON
        index_path = frames_dir / 'index.json'
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
        print(f"[OK] Saved: {index_path}")

        # Print summary
        print(f"\n{'='*70}")
        print("REFERENCE CREATION COMPLETE")
        print(f"{'='*70}")
        print(f"\nOutput structure:")
        print(f"  {output_dir}/")
        print(f"  ├── master_22class.pkl")
        print(f"  └── frames/")
        print(f"      ├── index.json")
        for seg in segments[:3]:
            print(f"      ├── {seg['movement_id']}/")
            print(f"      │   ├── frame_XXXXX.jpg")
            print(f"      │   └── keypoints.npz")
        print(f"      ├── ...")
        print(f"      └── 19_1/")
        print(f"          ├── frame_XXXXX.jpg")
        print(f"          └── keypoints.npz")

        print(f"\n{'No':<4} {'ID':<6} {'Frames':<15} {'Trimmed':<8} {'Images':<8}")
        print("-" * 50)
        total_images = 0
        for seg in segments:
            mov_id = seg['movement_id']
            frames_str = f"{seg['start_frame']}-{seg['end_frame']}"
            trim_str = f"-{seg['trimmed_frames']}" if seg['trimmed_frames'] > 0 else "0"
            num_imgs = seg['end_frame'] - seg['start_frame'] + 1
            total_images += num_imgs
            print(f"{seg['index']+1:<4} {mov_id:<6} {frames_str:<15} {trim_str:<8} {num_imgs:<8}")

        print("-" * 50)
        print(f"Total images: {total_images}")
        print(f"{'='*70}\n")

        return reference_data


def main():
    parser = argparse.ArgumentParser(description='Create complete reference database')
    parser.add_argument('--video', required=True, help='Video file path')
    parser.add_argument('--annotation', required=True, help='Annotation JSON file path')
    parser.add_argument('--output', default='compare/references', help='Output directory')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    parser.add_argument(
        '--pose-backend',
        type=str,
        default='rtmpose',
        choices=['rtmpose', 'mediapipe'],
        help='Pose backend for extracting reference keypoints'
    )
    parser.add_argument('--trim', type=int, default=3,
                        help='Frames to trim from start of movements 1-21 (default: 3)')

    args = parser.parse_args()

    creator = ReferenceCreator(
        device=args.device,
        trim_start_frames=args.trim,
        pose_backend=args.pose_backend,
    )
    creator.create_reference(args.video, args.annotation, args.output)


if __name__ == "__main__":
    main()


"""
# 1) Reference (RTMPose)
python compare/create_reference_complete.py --video "data/reference/annotations/P001.mp4" --annotation "data/reference/annotations/P001_annotations.json" --output "compare/references_rtmpose" --pose-backend rtmpose

# 2) Reference (MediaPipe)
python compare/create_reference_complete.py --video "data/reference/annotations/P001.mp4" --annotation "data/reference/annotations/P001_annotations.json" --output "compare/references_mediapipe" --pose-backend mediapipe

# 3) Student process (RTMPose)
python compare/process_student.py --video "data/reference/videos/P011.mp4" --output "compare/students_rtmpose/P011" --pose-backend rtmpose

# 4) Student process (MediaPipe)
python compare/process_student.py --video "data/reference/videos/P011.mp4" --output "compare/students_mediapipe/P011" --pose-backend mediapipe

# 5) Compare (must match backend pairs)
python compare/compare_with_reference.py --student "compare/students_rtmpose/P011" --reference "compare/references_rtmpose" --output "compare/students_rtmpose/P011/comparison.json"
python compare/compare_with_reference.py --student "compare/students_mediapipe/P011" --reference "compare/references_mediapipe" --output "compare/students_mediapipe/P011/comparison.json"

"""