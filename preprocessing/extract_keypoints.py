"""
Extract RTMPose Keypoints with Normalization

FINAL VERSION - Use this to replace preprocessing/extract_keypoints.py

Features:
- Hip-centered normalization (position invariant)
- Height-normalized scaling (scale invariant)
- Works with any video resolution
- Halpe26 keypoint format (26 keypoints including feet)
"""

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

    def __init__(self, normalize=True):
        print("Initializing RTMPose via RTMLib...")

        try:
            from rtmlib import BodyWithFeet
            self.BodyWithFeet = BodyWithFeet
        except ImportError:
            print("ERROR: rtmlib not installed")
            print("Run: pip install rtmlib")
            raise

        self.pose_estimator = self.BodyWithFeet(
            mode='balanced',
            backend='onnxruntime',
            device='cuda'
        )

        self.num_keypoints = 26
        self.normalize = normalize

        print(f"✓ RTMPose initialized")
        print(f"  Keypoints: {self.num_keypoints} (Body + Feet)")
        print(f"  Normalization: {'ENABLED (hip-centered + height-scaled)' if normalize else 'DISABLED'}")

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

        print(f"\nProcessing {video_path.name}...")
        print(f"  Resolution: {width}x{height}, FPS: {fps:.2f}, Frames: {total_frames}")

        with tqdm(total=total_frames, desc="Extracting") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                keypoints = self.extract_from_frame(frame)
                all_keypoints.append(keypoints)
                pbar.update(1)

        cap.release()

        keypoints_array = np.array(all_keypoints)
        print(f"  Extracted: {keypoints_array.shape}")

        if self.normalize:
            print(f"  Normalizing...")
            keypoints_array = self.normalize_keypoints(keypoints_array)
            print(f"  ✓ Normalized")

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

            print(f"  ✓ Saved: {save_path.name}")

        return keypoints_array

    def extract_from_frame(self, frame):
        """Extract keypoints from single frame"""
        keypoints, scores = self.pose_estimator(frame)

        if len(keypoints) > 0:
            person_keypoints = keypoints[0]
            person_scores = scores[0]
            return np.concatenate([person_keypoints, person_scores.reshape(-1, 1)], axis=1)
        else:
            return np.zeros((26, 3))

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
                print(f"\n⊙ Skipping {video_file.name} (already exists)")
                continue

            self.extract_from_video(video_file, output_file)

        print(f"\n{'='*70}")
        print(f"✓ COMPLETE")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    Paths.create_directories()
    extractor = RTMLibExtractor(normalize=True)
    extractor.process_all_videos(
        video_dir=Paths.RAW_VIDEOS,
        output_dir=Paths.KEYPOINTS_DIR
    )