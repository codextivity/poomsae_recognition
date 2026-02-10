"""
Create Reference Database from Master Video

Usage:
    python create_reference.py --video master.mp4 --segments master_segments.csv
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from rtmlib import BodyWithFeet


class ReferenceCreator:
    """Extract and save reference data from master video"""

    def __init__(self):
        self.pose_estimator = BodyWithFeet(
            to_openpose=False,
            mode='balanced',
            backend='onnxruntime',
            device='cpu'
        )

    def normalize_keypoints(self, keypoints):
        """Hip-centered + height-normalized (same as training)"""
        coords = keypoints[:, :2].copy()
        confidence = keypoints[:, 2:3].copy()

        if np.all(coords == 0):
            return keypoints

        # Hip-centering (keypoint 25 = pelvis in Halpe26)
        hip = coords[25:26, :] if coords.shape[0] > 25 else coords[11:12, :]
        centered = coords - hip

        # Height normalization
        head = coords[0, :]
        feet = (coords[15, :] + coords[16, :]) / 2
        height = np.linalg.norm(head - feet)

        if height < 1e-3:
            height = np.linalg.norm(coords[5, :] - coords[6, :]) * 3
        if height < 1e-3:
            height = 1.0

        normalized = centered / height
        return np.concatenate([normalized, confidence], axis=1)

    def extract_reference(self, video_path, segments_csv, output_path):
        """Extract reference keypoints for each movement"""

        cap = cv2.VideoCapture(str(video_path))
        segments = pd.read_csv(segments_csv)
        fps = cap.get(cv2.CAP_PROP_FPS)

        reference_data = {
            'video_name': Path(video_path).name,
            'fps': fps,
            'movements': []
        }

        print(f"\n{'=' * 70}")
        print(f"EXTRACTING REFERENCE DATA")
        print(f"{'=' * 70}\n")

        for idx, seg in segments.iterrows():
            movement_num = seg['movement_number']
            start_frame = int(seg['start_frame'])
            end_frame = int(seg['end_frame'])

            print(f"Processing Movement {movement_num}: {seg['movement_name']}")
            print(f"  Frames: {start_frame}-{end_frame} ({end_frame - start_frame + 1} frames)")

            # Extract keypoints for this movement
            movement_keypoints = []
            frame_nums = []

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            for frame_num in range(start_frame, end_frame + 1):
                ret, frame = cap.read()
                if not ret:
                    break

                # Extract keypoints
                keypoints, scores = self.pose_estimator(frame)

                if len(keypoints) > 0:
                    person_kp = keypoints[0]
                    person_scores = scores[0]
                    kp_with_conf = np.concatenate([
                        person_kp,
                        person_scores.reshape(-1, 1)
                    ], axis=1)

                    # Normalize
                    kp_normalized = self.normalize_keypoints(kp_with_conf)
                    movement_keypoints.append(kp_normalized)
                    frame_nums.append(frame_num)
                else:
                    # No detection - use zeros
                    movement_keypoints.append(np.zeros((26, 3)))
                    frame_nums.append(frame_num)

            movement_keypoints = np.array(movement_keypoints)

            # Extract key poses (start, middle, end)
            key_poses = self.extract_key_poses(movement_keypoints)

            movement_data = {
                'movement_number': movement_num,
                'movement_name': seg['movement_name'],
                'start_frame': start_frame,
                'end_frame': end_frame,
                'duration': seg['duration'],
                'all_keypoints': movement_keypoints,  # (N, 26, 3)
                'frame_numbers': frame_nums,
                'key_poses': key_poses,
                'avg_keypoint': movement_keypoints.mean(axis=0)  # (26, 3)
            }

            reference_data['movements'].append(movement_data)
            print(f"  ✓ Extracted {len(movement_keypoints)} frames")

        cap.release()

        # Save reference
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump(reference_data, f)

        print(f"\n{'=' * 70}")
        print(f"✅ Reference data saved: {output_path}")
        print(f"{'=' * 70}\n")

        return reference_data

    def extract_key_poses(self, movement_keypoints):
        """Extract key poses from movement sequence"""
        n_frames = len(movement_keypoints)

        if n_frames == 0:
            return {}

        key_poses = {
            'start': movement_keypoints[0],  # First frame
            'end': movement_keypoints[-1],  # Last frame
        }

        # Middle pose(s)
        if n_frames >= 3:
            mid_idx = n_frames // 2
            key_poses['middle'] = movement_keypoints[mid_idx]

        # Peak pose (frame with maximum movement)
        if n_frames >= 5:
            # Calculate frame-to-frame differences
            diffs = np.linalg.norm(
                np.diff(movement_keypoints[:, :, :2], axis=0),
                axis=(1, 2)
            )
            peak_idx = np.argmax(diffs)
            key_poses['peak'] = movement_keypoints[peak_idx]

        return key_poses


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Create reference database')
    parser.add_argument('--video', required=True, help='Master video')
    parser.add_argument('--segments', required=True, help='Master segments CSV')
    parser.add_argument('--output', default='students/keypoints/P001_keypoints.pkl', help='Output path')

    args = parser.parse_args()

    creator = ReferenceCreator()
    creator.extract_reference(args.video, args.segments, args.output)


if __name__ == "__main__":
    main()

    # python create_reference.py --video "D:\All Docs\All Projects\Pycharm\poomsae_recognition\data\reference\videos\P001.mp4" --segments "D:\All Docs\All Projects\Pycharm\poomsae_recognition\results\movement_segments\P001_segments_20260121_113138.csv"