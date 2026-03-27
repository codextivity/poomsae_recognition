"""
Extract RTMPose Keypoints with Normalization

FINAL VERSION - Use this to replace preprocessing/extract_keypoints.py

Features:
- Hip-centered normalization (position invariant)
- Height-normalized scaling (scale invariant)
- Works with any video resolution
- Halpe26 keypoint format (26 keypoints including feet)
"""

import argparse
import cv2
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))
from configs.paths import Paths


class RTMLibExtractor:
    """Extract and normalize keypoints using RTMLib"""

    HALPE26_CONNECTIONS = [
        (0, 18), (17, 18),
        (0, 1), (0, 2), (1, 3), (2, 4),
        (18, 5), (18, 6),
        (5, 6), (5, 11), (6, 12), (11, 12), (18, 19),
        (5, 7), (7, 9),
        (6, 8), (8, 10),
        (11, 13), (13, 15),
        (12, 14), (14, 16),
        (15, 20), (15, 22), (15, 24), (20, 22),
        (16, 21), (16, 23), (16, 25), (21, 23),
    ]

    def __init__(
        self,
        normalize=True,
        track_single_person=True,
        save_previews=False,
        preview_dir=None,
        preview_every=30,
        save_suspicious=True,
    ):
        print("Initializing RTMPose via RTMLib...")

        try:
            from rtmlib import BodyWithFeet
            self.BodyWithFeet = BodyWithFeet
        except ImportError:
            print("ERROR: rtmlib not installed")
            print("Run: pip install rtmlib")
            raise

        self.pose_estimator = self.BodyWithFeet(
            to_openpose=False,  # Use Halpe26 format
            mode='balanced',
            backend='onnxruntime',
            device='cuda'
        )

        self.num_keypoints = 26
        self.normalize = normalize
        self.track_single_person = track_single_person
        self.min_valid_joint_score = 0.20
        self.max_center_jump_factor = 1.75
        self.max_pose_jump_factor = 2.50
        self.max_track_lost_frames = 10
        self.low_confidence_threshold = 0.30
        self.save_previews = save_previews
        self.preview_dir = Path(preview_dir) if preview_dir else (Paths.PROCESSED_ROOT / 'keypoints_preview')
        self.preview_every = max(1, int(preview_every))
        self.save_suspicious = save_suspicious
        self._reset_tracking_state()

        print(f"[OK] RTMPose initialized")
        print(f"  Keypoints: {self.num_keypoints} (Halpe26 - Body + Feet)")
        print(f"  Normalization: {'ENABLED (hip-centered + height-scaled)' if normalize else 'DISABLED'}")
        print(f"  Single-person tracking: {'ENABLED' if track_single_person else 'DISABLED'}")
        if self.save_previews:
            print(f"  Preview overlays: ENABLED ({self.preview_every}-frame sampling)")
            print(f"  Preview dir: {self.preview_dir}")

    def _reset_tracking_state(self):
        """Reset per-video subject tracking state."""
        self.prev_selected_pose = None
        self.track_lost_frames = 0

    def _pack_person(self, person_keypoints, person_scores):
        return np.concatenate([person_keypoints, person_scores.reshape(-1, 1)], axis=1)

    def _valid_joint_mask(self, pose):
        coords = pose[:, :2]
        scores = pose[:, 2]
        return (scores >= self.min_valid_joint_score) & np.any(coords != 0, axis=1)

    def _estimate_center(self, pose):
        coords = pose[:, :2]
        valid = self._valid_joint_mask(pose)

        anchor_indices = [19, 11, 12]
        anchor_points = []
        for idx in anchor_indices:
            if idx < len(pose) and valid[idx]:
                anchor_points.append(coords[idx])

        if anchor_points:
            return np.mean(anchor_points, axis=0)

        if np.any(valid):
            return np.mean(coords[valid], axis=0)

        return None

    def _estimate_scale(self, pose):
        coords = pose[:, :2]
        valid = self._valid_joint_mask(pose)

        if np.sum(valid) < 2:
            return 1.0

        valid_coords = coords[valid]
        bbox_size = valid_coords.max(axis=0) - valid_coords.min(axis=0)
        bbox_diag = float(np.linalg.norm(bbox_size))
        return max(bbox_diag, 1.0)

    def _pose_distance(self, candidate, reference):
        candidate_coords = candidate[:, :2]
        reference_coords = reference[:, :2]
        valid = self._valid_joint_mask(candidate) & self._valid_joint_mask(reference)

        if np.sum(valid) < 4:
            return float("inf")

        joint_distances = np.linalg.norm(candidate_coords[valid] - reference_coords[valid], axis=1)
        return float(np.mean(joint_distances))

    def _mean_confidence(self, pose):
        valid = self._valid_joint_mask(pose)
        if not np.any(valid):
            return 0.0
        return float(np.mean(pose[valid, 2]))

    def _draw_pose_overlay(self, frame, pose, labels=None):
        overlay = frame.copy()

        for start, end in self.HALPE26_CONNECTIONS:
            if start >= len(pose) or end >= len(pose):
                continue
            if pose[start, 2] < self.low_confidence_threshold or pose[end, 2] < self.low_confidence_threshold:
                continue
            if pose[start, 0] <= 0 or pose[end, 0] <= 0:
                continue

            pt1 = (int(pose[start, 0]), int(pose[start, 1]))
            pt2 = (int(pose[end, 0]), int(pose[end, 1]))
            cv2.line(overlay, pt1, pt2, (0, 255, 0), 2)

        for kp in pose:
            if kp[2] < self.low_confidence_threshold or kp[0] <= 0:
                continue
            cv2.circle(overlay, (int(kp[0]), int(kp[1])), 4, (0, 0, 255), -1)

        if labels:
            y = 30
            for label in labels:
                cv2.putText(
                    overlay,
                    str(label),
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                y += 28

        return overlay

    def _save_preview(self, preview_dir, frame_idx, frame, pose, note):
        if preview_dir is None:
            return
        labels = [
            f"frame={frame_idx:05d}",
            f"mean_conf={self._mean_confidence(pose):.3f}",
            note,
        ]
        preview = self._draw_pose_overlay(frame, pose, labels=labels)
        safe_note = note.replace(' ', '_').replace('/', '_')
        output_path = preview_dir / f"{frame_idx:05d}_{safe_note}.jpg"
        cv2.imwrite(str(output_path), preview)

    def _select_initial_person(self, packed_people, frame_shape):
        """Pick the main subject for the first confident frame."""
        frame_h, frame_w = frame_shape[:2]
        frame_center = np.array([frame_w / 2.0, frame_h / 2.0], dtype=np.float32)

        best_idx = None
        best_score = -float("inf")

        for idx, pose in enumerate(packed_people):
            center = self._estimate_center(pose)
            if center is None:
                continue

            mean_score = float(np.mean(pose[:, 2]))
            scale = self._estimate_scale(pose)
            center_distance = float(np.linalg.norm(center - frame_center))
            center_penalty = center_distance / max(frame_w, frame_h, 1)
            score = (mean_score * 2.0) + (scale / max(frame_w, frame_h, 1)) - center_penalty

            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx

    def _select_tracked_person(self, packed_people):
        """Choose the detection that best matches the previously selected subject."""
        if self.prev_selected_pose is None:
            return None

        prev_center = self._estimate_center(self.prev_selected_pose)
        prev_scale = self._estimate_scale(self.prev_selected_pose)
        if prev_center is None:
            return None

        best_idx = None
        best_score = float("inf")

        for idx, pose in enumerate(packed_people):
            center = self._estimate_center(pose)
            if center is None:
                continue

            center_distance = float(np.linalg.norm(center - prev_center))
            pose_distance = self._pose_distance(pose, self.prev_selected_pose)
            scale = max(self._estimate_scale(pose), prev_scale, 1.0)

            normalized_center_distance = center_distance / scale
            normalized_pose_distance = pose_distance / scale if np.isfinite(pose_distance) else float("inf")

            score = normalized_center_distance + (0.5 * normalized_pose_distance)
            if score < best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            return None

        best_pose = packed_people[best_idx]
        best_center = self._estimate_center(best_pose)
        center_distance = float(np.linalg.norm(best_center - prev_center))
        pose_distance = self._pose_distance(best_pose, self.prev_selected_pose)
        scale = max(self._estimate_scale(best_pose), prev_scale, 1.0)

        if (center_distance / scale) > self.max_center_jump_factor:
            return None

        if np.isfinite(pose_distance) and (pose_distance / scale) > self.max_pose_jump_factor:
            return None

        return best_idx

    def normalize_keypoints(self, keypoints):
        """
        Normalize keypoints: hip-centered + height-scaled

        Args:
            keypoints: (num_frames, 26, 3) - [x, y, confidence]
        Returns:
            normalized: (num_frames, 26, 3)
        """
        if not self.normalize or len(keypoints) == 0:
            return keypoints

        normalized_keypoints = []

        for frame_kp in keypoints:
            coords = frame_kp[:, :2].copy()
            confidence = frame_kp[:, 2:3].copy()

            # Skip if no detection
            if np.all(coords == 0):
                normalized_keypoints.append(frame_kp)
                continue

            # Center on hip (keypoint 19)
            hip = coords[19:20, :]
            centered = coords - hip

            # Calculate height (head to feet)
            head = coords[0, :]
            left_ankle = coords[15, :]
            right_ankle = coords[16, :]
            feet = (left_ankle + right_ankle) / 2
            height = np.linalg.norm(head - feet)

            # Fallback if height invalid
            if height < 1e-3:
                shoulder_width = np.linalg.norm(coords[5, :] - coords[6, :])
                height = shoulder_width * 3 if shoulder_width > 1e-3 else 1.0

            # Normalize by height
            normalized = centered / height

            # Recombine
            frame_normalized = np.concatenate([normalized, confidence], axis=1)
            normalized_keypoints.append(frame_normalized)

        return np.array(normalized_keypoints)

    def extract_from_video(self, video_path, save_path=None):
        """Extract keypoints from video"""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        all_keypoints = []
        self._reset_tracking_state()
        preview_dir = None
        if self.save_previews:
            preview_dir = self.preview_dir / video_path.stem
            preview_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing {video_path.name}...")
        print(f"  Resolution: {width}x{height}, FPS: {fps:.2f}, Frames: {total_frames}")

        with tqdm(total=total_frames, desc="Extracting") as pbar:
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                keypoints, info = self.extract_from_frame(frame)
                all_keypoints.append(keypoints)

                should_save_preview = self.save_previews and (
                    frame_idx % self.preview_every == 0
                    or (self.save_suspicious and info['is_suspicious'])
                )
                if should_save_preview:
                    note = info['reason']
                    if info['is_suspicious']:
                        note = f"suspicious_{note}"
                    else:
                        note = "sample"
                    self._save_preview(preview_dir, frame_idx, frame, keypoints, note)

                frame_idx += 1
                pbar.update(1)

        cap.release()

        keypoints_array = np.array(all_keypoints)
        print(f"  Extracted: {keypoints_array.shape}")

        if self.normalize:
            print(f"  Normalizing...")
            keypoints_array = self.normalize_keypoints(keypoints_array)
            print(f"  [OK] Normalized")

        if save_path:
            data = {
                'keypoints': keypoints_array,
                'fps': fps,
                'num_frames': len(keypoints_array),
                'video_resolution': (width, height),
                'normalized': self.normalize
            }

            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)

            print(f"  [OK] Saved: {save_path.name}")

        return keypoints_array

    def extract_from_frame(self, frame):
        """Extract keypoints from single frame"""
        keypoints, scores = self.pose_estimator(frame)
        zero_pose = np.zeros((26, 3))

        if len(keypoints) > 0:
            packed_people = [
                self._pack_person(person_keypoints, person_scores)
                for person_keypoints, person_scores in zip(keypoints, scores)
            ]
            reason = 'tracked_subject'

            if not self.track_single_person:
                selected_pose = packed_people[0]
                reason = 'first_detection'
            else:
                selected_idx = self._select_tracked_person(packed_people)

                if selected_idx is None:
                    if self.prev_selected_pose is None or self.track_lost_frames >= self.max_track_lost_frames:
                        selected_idx = self._select_initial_person(packed_people, frame.shape)
                        reason = 'initial_reacquire'
                    else:
                        self.track_lost_frames += 1
                        return zero_pose, {'reason': 'track_lost', 'is_suspicious': True}

                if selected_idx is None:
                    self.track_lost_frames += 1
                    return zero_pose, {'reason': 'no_subject_match', 'is_suspicious': True}

                selected_pose = packed_people[selected_idx]
                self.prev_selected_pose = selected_pose.copy()
                self.track_lost_frames = 0

            mean_conf = self._mean_confidence(selected_pose)
            is_suspicious = mean_conf < self.low_confidence_threshold
            if is_suspicious:
                reason = f'low_conf_{mean_conf:.2f}'

            return selected_pose, {'reason': reason, 'is_suspicious': is_suspicious}

        self.track_lost_frames += 1
        return zero_pose, {'reason': 'no_detection', 'is_suspicious': True}

    def process_all_videos(self, video_dir, output_dir):
        """Process all videos"""
        video_dir = Path(video_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        video_extensions = ['.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV']
        video_files = [f for f in video_dir.iterdir() if f.suffix in video_extensions]

        if not video_files:
            print(f"No videos found in {video_dir}")
            return

        print(f"\n{'='*70}")
        print(f"PROCESSING {len(video_files)} VIDEOS")
        print(f"{'='*70}")

        for video_file in video_files:
            output_file = output_dir / f"{video_file.stem}_keypoints.pkl"

            if output_file.exists():
                print(f"\n[SKIP] {video_file.name} (already exists)")
                continue

            self.extract_from_video(video_file, output_file)

        print(f"\n{'='*70}")
        print(f"[OK] COMPLETE")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract RTMPose keypoints from videos.")
    parser.add_argument('--video-dir', type=Path, default=Paths.RAW_VIDEOS, help='Input video directory')
    parser.add_argument('--output-dir', type=Path, default=Paths.KEYPOINTS_DIR, help='Output keypoint directory')
    parser.add_argument('--disable-normalize', action='store_true', help='Disable hip/height normalization before saving')
    parser.add_argument('--disable-tracking', action='store_true', help='Disable single-person tracking and use first detection')
    parser.add_argument('--save-previews', action='store_true', help='Save sampled overlay previews as images')
    parser.add_argument(
        '--preview-dir',
        type=Path,
        default=Paths.PROCESSED_ROOT / 'keypoints_preview',
        help='Directory to save overlay previews',
    )
    parser.add_argument('--preview-every', type=int, default=30, help='Save one sampled overlay every N frames')
    parser.add_argument('--no-suspicious-previews', action='store_true', help='Do not save extra suspicious-frame overlays')
    args = parser.parse_args()

    Paths.create_directories()
    extractor = RTMLibExtractor(
        normalize=not args.disable_normalize,
        track_single_person=not args.disable_tracking,
        save_previews=args.save_previews,
        preview_dir=args.preview_dir,
        preview_every=args.preview_every,
        save_suspicious=not args.no_suspicious_previews,
    )
    extractor.process_all_videos(
        video_dir=args.video_dir,
        output_dir=args.output_dir
    )



'''
python preprocessing/extract_keypoints.py --save-previews --preview-dir data/processed/keypoints_preview

'''