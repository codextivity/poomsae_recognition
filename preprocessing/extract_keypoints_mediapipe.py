"""
Extract MediaPipe Pose keypoints and convert to HALPE26-compatible format.

Purpose:
- Generate keypoint .pkl files with the same schema as RTMPose extractor:
  data['keypoints']: (num_frames, 26, 3) [x, y, confidence]
  data['fps'], data['num_frames'], data['video_resolution'], data['normalized']
- Keep downstream window/training scripts unchanged.

Recommended for backend comparison:
- Save to a separate directory (default: data/processed/keypoints_mediapipe)
- Run create_windows + train on this directory as a separate experiment.
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


class MediaPipeExtractor:
    """Extract and normalize keypoints using MediaPipe Pose."""

    # MediaPipe PoseLandmark indices we use
    MP = {
        "nose": 0,
        "left_eye": 2,
        "right_eye": 5,
        "left_ear": 7,
        "right_ear": 8,
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_ankle": 27,
        "right_ankle": 28,
        "left_heel": 29,
        "right_heel": 30,
        "left_foot_index": 31,   # big toe proxy
        "right_foot_index": 32,  # big toe proxy
    }

    def __init__(
        self,
        normalize=True,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=False,
    ):
        try:
            import mediapipe as mp
        except ImportError:
            print("ERROR: mediapipe not installed")
            print("Run: pip install mediapipe")
            raise

        self.num_keypoints = 26
        self.normalize = normalize

        self.pose = mp.solutions.pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        print("[OK] MediaPipe Pose initialized")
        print(f"  Keypoints: {self.num_keypoints} (HALPE26-compatible mapping)")
        print(f"  Normalize: {'ENABLED' if normalize else 'DISABLED'}")
        print(
            "  Pose config:"
            f" model_complexity={model_complexity},"
            f" min_det={min_detection_confidence},"
            f" min_track={min_tracking_confidence}"
        )

    def normalize_keypoints(self, keypoints):
        """
        Normalize keypoints: hip-centered + height-scaled.

        Args:
            keypoints: (num_frames, 26, 3) [x, y, confidence]
        Returns:
            normalized: (num_frames, 26, 3)
        """
        if not self.normalize or len(keypoints) == 0:
            return keypoints

        normalized_keypoints = []
        for frame_kp in keypoints:
            coords = frame_kp[:, :2].copy()
            confidence = frame_kp[:, 2:3].copy()

            if np.all(coords == 0):
                normalized_keypoints.append(frame_kp)
                continue

            # Center on hip center (index 19)
            hip = coords[19:20, :]
            centered = coords - hip

            # Height proxy: head (0) to mean ankles (15,16)
            head = coords[0, :]
            feet = (coords[15, :] + coords[16, :]) / 2
            height = np.linalg.norm(head - feet)

            if height < 1e-3:
                shoulder_width = np.linalg.norm(coords[5, :] - coords[6, :])
                height = shoulder_width * 3 if shoulder_width > 1e-3 else 1.0

            normalized = centered / height
            frame_normalized = np.concatenate([normalized, confidence], axis=1)
            normalized_keypoints.append(frame_normalized)

        return np.array(normalized_keypoints)

    def _set_from_mp(self, out, dst_idx, mp_landmarks, src_idx, width, height):
        """Copy one MediaPipe landmark into HALPE26 slot."""
        lm = mp_landmarks[src_idx]
        out[dst_idx, 0] = lm.x * width
        out[dst_idx, 1] = lm.y * height
        # Pose landmarks expose visibility; use it as confidence proxy.
        vis = float(getattr(lm, "visibility", 1.0))
        out[dst_idx, 2] = float(np.clip(vis, 0.0, 1.0))

    @staticmethod
    def _safe_unit(vec):
        """Return unit vector and norm; robust to near-zero vectors."""
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            return np.array([0.0, -1.0], dtype=np.float32), norm
        return (vec / norm).astype(np.float32), norm

    def _synthesize_head_top(self, out):
        """
        Synthesize HALPE head-top (idx 17) from face center + neck direction.

        This is more stable than extending only nose->neck, which can jitter or overlap.
        """
        neck_xy = out[18, :2]
        neck_conf = float(out[18, 2])

        face_idxs = [0, 1, 2, 3, 4]  # nose, eyes, ears
        valid_face = [i for i in face_idxs if out[i, 2] >= 0.2]
        if valid_face:
            face_xy = np.mean(out[valid_face, :2], axis=0)
            face_conf = float(np.mean(out[valid_face, 2]))
        else:
            face_xy = out[0, :2].copy()
            face_conf = float(out[0, 2])

        # Use neck->face axis. Fallback to neck->nose.
        direction, dir_norm = self._safe_unit(face_xy - neck_xy)
        if dir_norm < 1e-6:
            direction, _ = self._safe_unit(out[0, :2] - neck_xy)

        shoulder_width = float(np.linalg.norm(out[5, :2] - out[6, :2]))
        if shoulder_width < 1e-6:
            hip_width = float(np.linalg.norm(out[11, :2] - out[12, :2]))
            shoulder_width = hip_width * 0.9 if hip_width > 1e-6 else 50.0

        # Head top roughly ~35% shoulder width above face center.
        head_offset = 0.35 * shoulder_width
        out[17, :2] = face_xy + direction * head_offset
        out[17, 2] = max(0.0, min(1.0, min(neck_conf, face_conf) * 0.9))

    def _synthesize_small_toe(self, out, side):
        """
        Synthesize HALPE small toe from heel->toe foot axis + perpendicular offset.

        side: 'left' or 'right'
        """
        if side == "left":
            ankle_idx, big_idx, heel_idx, small_idx = 15, 20, 24, 22
            lateral_sign = 1.0
        else:
            ankle_idx, big_idx, heel_idx, small_idx = 16, 21, 25, 23
            lateral_sign = -1.0

        ankle_xy = out[ankle_idx, :2]
        big_xy = out[big_idx, :2]
        heel_xy = out[heel_idx, :2]

        ankle_conf = float(out[ankle_idx, 2])
        big_conf = float(out[big_idx, 2])
        heel_conf = float(out[heel_idx, 2])

        # Foot axis prefers heel->big-toe; fallback to ankle->big-toe.
        foot_axis = big_xy - heel_xy
        axis_unit, axis_len = self._safe_unit(foot_axis)
        if axis_len < 1e-6:
            axis_unit, axis_len = self._safe_unit(big_xy - ankle_xy)

        if axis_len < 1e-6:
            # No reliable geometry: keep near big toe with reduced confidence.
            out[small_idx, :2] = big_xy
            out[small_idx, 2] = max(0.0, min(1.0, big_conf * 0.6))
            return

        perp = np.array([-axis_unit[1], axis_unit[0]], dtype=np.float32)
        toe_forward = 0.12 * axis_len
        toe_lateral = 0.22 * axis_len

        out[small_idx, :2] = big_xy + axis_unit * toe_forward + lateral_sign * perp * toe_lateral
        out[small_idx, 2] = max(0.0, min(1.0, min(ankle_conf, big_conf, heel_conf) * 0.85))

    def _extract_halpe26_from_frame(self, frame_bgr):
        """Run MediaPipe Pose and convert 33 landmarks to HALPE26-compatible 26 points."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.pose.process(frame_rgb)

        if result.pose_landmarks is None:
            return np.zeros((26, 3), dtype=np.float32)

        mp_landmarks = result.pose_landmarks.landmark
        h, w = frame_bgr.shape[:2]

        out = np.zeros((26, 3), dtype=np.float32)

        # Direct mappings
        direct_map = {
            0: self.MP["nose"],
            1: self.MP["left_eye"],
            2: self.MP["right_eye"],
            3: self.MP["left_ear"],
            4: self.MP["right_ear"],
            5: self.MP["left_shoulder"],
            6: self.MP["right_shoulder"],
            7: self.MP["left_elbow"],
            8: self.MP["right_elbow"],
            9: self.MP["left_wrist"],
            10: self.MP["right_wrist"],
            11: self.MP["left_hip"],
            12: self.MP["right_hip"],
            13: self.MP["left_knee"],
            14: self.MP["right_knee"],
            15: self.MP["left_ankle"],
            16: self.MP["right_ankle"],
            20: self.MP["left_foot_index"],   # big toe
            21: self.MP["right_foot_index"],  # big toe
            24: self.MP["left_heel"],
            25: self.MP["right_heel"],
        }
        for dst_idx, src_idx in direct_map.items():
            self._set_from_mp(out, dst_idx, mp_landmarks, src_idx, w, h)

        # Synthesized keypoints to match expected HALPE26 slots
        # 18: neck (mid-shoulder)
        out[18, :2] = (out[5, :2] + out[6, :2]) * 0.5
        out[18, 2] = min(out[5, 2], out[6, 2])

        # 19: hip center (mid-hip)
        out[19, :2] = (out[11, :2] + out[12, :2]) * 0.5
        out[19, 2] = min(out[11, 2], out[12, 2])

        # 17: head top approximation from face center + neck direction
        self._synthesize_head_top(out)

        # 22/23: small toe approximation from foot orientation
        self._synthesize_small_toe(out, side="left")
        self._synthesize_small_toe(out, side="right")

        # Keep synthesized points inside image bounds.
        out[:, 0] = np.clip(out[:, 0], 0.0, float(w - 1))
        out[:, 1] = np.clip(out[:, 1], 0.0, float(h - 1))

        return out

    def extract_from_video(self, video_path, save_path=None):
        """Extract keypoints from one video."""
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
                keypoints = self._extract_halpe26_from_frame(frame)
                all_keypoints.append(keypoints)
                pbar.update(1)

        cap.release()

        keypoints_array = np.array(all_keypoints, dtype=np.float32)
        print(f"  Extracted: {keypoints_array.shape}")

        if self.normalize:
            print("  Normalizing...")
            keypoints_array = self.normalize_keypoints(keypoints_array)
            print("  [OK] Normalized")

        if save_path:
            data = {
                "keypoints": keypoints_array,
                "fps": fps,
                "num_frames": len(keypoints_array),
                "video_resolution": (width, height),
                "normalized": self.normalize,
                "source": "mediapipe_pose",
                "format": "halpe26_compatible",
            }
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump(data, f)
            print(f"  [OK] Saved: {save_path.name}")

        return keypoints_array

    def process_all_videos(self, video_dir, output_dir, overwrite=False):
        """Process all videos in a directory."""
        video_dir = Path(video_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        video_extensions = {".mp4", ".avi", ".mov", ".MP4", ".AVI", ".MOV"}
        video_files = sorted([f for f in video_dir.iterdir() if f.suffix in video_extensions])

        if not video_files:
            print(f"No videos found in {video_dir}")
            return

        print(f"\n{'='*70}")
        print(f"PROCESSING {len(video_files)} VIDEOS (MediaPipe)")
        print(f"Input: {video_dir}")
        print(f"Output: {output_dir}")
        print(f"{'='*70}")

        for video_file in video_files:
            output_file = output_dir / f"{video_file.stem}_keypoints.pkl"
            if output_file.exists() and not overwrite:
                print(f"\n[SKIP] {video_file.name} (already exists)")
                continue
            self.extract_from_video(video_file, output_file)

        print(f"\n{'='*70}")
        print("[OK] COMPLETE")
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Extract MediaPipe keypoints in HALPE26-compatible format")
    parser.add_argument("--video-dir", type=str, default=str(Paths.RAW_VIDEOS), help="Input video directory")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Paths.PROCESSED_ROOT / "keypoints_mediapipe"),
        help="Output keypoints directory",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing *_keypoints.pkl files")
    parser.add_argument("--no-normalize", action="store_true", help="Disable hip/height normalization")
    parser.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2], help="MediaPipe model complexity")
    parser.add_argument("--min-det-conf", type=float, default=0.5, help="Min detection confidence")
    parser.add_argument("--min-track-conf", type=float, default=0.5, help="Min tracking confidence")
    args = parser.parse_args()

    extractor = MediaPipeExtractor(
        normalize=not args.no_normalize,
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_det_conf,
        min_tracking_confidence=args.min_track_conf,
        static_image_mode=False,
    )
    extractor.process_all_videos(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
