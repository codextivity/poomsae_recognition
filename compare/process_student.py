"""
Process Student Video for Comparison

Processes a student video using the trained model with sequential validation,
extracts keypoints for each detected movement, and saves in the same format
as the reference for comparison.

Usage:
    python process_student.py --video student.mp4 --output compare/students/student1
    python process_student.py --video student.mp4 --output compare/students/student1 --save-video
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import torch
import numpy as np
from pathlib import Path
import pickle
import argparse
import json
import sys
from collections import deque

sys.path.append(str(Path(__file__).parent.parent))
from models.lstm_classifier import PoomsaeLSTM
from configs.lstm_config import LSTMConfig
from configs.paths import Paths
from preprocessing.create_windows import CLASS_NAMES, CLASS_MAPPING


class StudentProcessor:
    """Process student video and extract keypoints for comparison"""

    # Short movement classes (faster detection needed)
    SHORT_MOVEMENT_CLASSES = [6, 12, 14, 17]  # 6_1, 12_1, 14_1, 16_1

    # Normalization settings (must match reference)
    NORM_CENTER_JOINT = 19  # Pelvis/hip center
    NORM_REF_JOINTS = [11, 12]  # Left hip, Right hip

    def __init__(
        self,
        model_path,
        device='cuda',
        trim_start_frames=3,
        pose_backend='rtmpose',
        stats_path=None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.trim_start_frames = trim_start_frames
        self.pose_backend = str(pose_backend).strip().lower()
        self.stats_path = Path(stats_path) if stats_path else None
        print(f"Using device: {self.device}")

        # Load config
        self.config = LSTMConfig()
        self.num_classes = self.config.NUM_CLASSES  # 22
        self.window_size = self.config.SEQUENCE_LENGTH  # 24

        # Load model
        self.model = PoomsaeLSTM(self.config)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded (epoch {checkpoint.get('epoch', '?')}, val_acc={checkpoint.get('best_val_acc', 0):.2f}%)")

        # Initialize pose backend
        self.pose_estimator = None
        self.mp_extractor = None
        self._init_pose_backend(device)

        # Load normalization stats
        self.load_normalization_stats()

        # Keypoint buffer
        self.keypoint_buffer = deque(maxlen=self.window_size)

        # Class names
        self.class_names = CLASS_NAMES

        # Sequential validation parameters
        self.confirm_frames_expected = 5
        self.confirm_frames_short = 5
        self.confirm_frames_normal = 10
        self.confirm_frames_future = 15

        self.conf_threshold_expected = 0.35
        self.conf_threshold_short = 0.80
        self.conf_threshold_normal = 0.60
        self.conf_threshold_skip = 0.85

        self.skip_wait_seconds = 2.0

        # Tracking state
        self.reset_state()

    def _init_pose_backend(self, device):
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
                device='cuda' if device == 'cuda' else 'cpu'
            )
            print("RTMPose initialized")
        except ImportError:
            raise ImportError("rtmlib not installed. Run: pip install rtmlib")

    def reset_state(self):
        """Reset all tracking state"""
        self.expected_next = 0
        self.current_movement = None
        self.movement_start_frame = None
        self.detected_movements = []
        self.skipped_movements = []
        self.sequence_complete = False
        self.max_movements = 22

        self.candidate_movement = None
        self.candidate_frames = 0
        self.candidate_confidence_sum = 0
        self.last_detection_frame = 0

        self.fps = 30.0
        self.keypoint_buffer.clear()

        # Storage for keypoints during processing
        self.frame_keypoints = []  # [(frame_num, kp_norm, kp_raw), ...]

    def load_normalization_stats(self):
        """Load normalization stats from training"""
        stats_file = self.stats_path if self.stats_path else (Paths.CHECKPOINTS_DIR / 'normalization_stats.pkl')

        if stats_file.exists():
            with open(stats_file, 'rb') as f:
                stats = pickle.load(f)
            self.mean = stats['mean']
            self.std = stats['std']
            print(f"Loaded normalization stats from {stats_file}")
        else:
            print(f"WARNING: normalization stats not found: {stats_file}")
            self.mean = np.zeros(78)
            self.std = np.ones(78)

    def normalize_keypoints(self, keypoints):
        """Normalize keypoints: hip-centered + height-scaled"""
        coords = keypoints[:, :2].copy()
        conf = keypoints[:, 2:3].copy()

        if np.all(coords == 0):
            return keypoints

        hip = coords[self.NORM_CENTER_JOINT:self.NORM_CENTER_JOINT + 1, :]
        centered = coords - hip

        head = coords[0, :]
        feet = (coords[15, :] + coords[16, :]) / 2
        height = np.linalg.norm(head - feet)

        if height < 1e-3:
            shoulder_width = np.linalg.norm(coords[5, :] - coords[6, :])
            height = shoulder_width * 3 if shoulder_width > 1e-3 else 1.0

        normalized = centered / height
        return np.concatenate([normalized, conf], axis=1)

    def extract_keypoints(self, frame):
        """Extract and normalize keypoints from frame"""
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

    def predict(self):
        """Make prediction from current keypoint buffer"""
        if len(self.keypoint_buffer) < self.window_size:
            return None, 0.0, np.zeros(self.num_classes)

        window = np.array(list(self.keypoint_buffer))
        window_flat = window.reshape(window.shape[0], -1)
        window_norm = (window_flat - self.mean) / self.std

        x = torch.FloatTensor(window_norm).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(x)
            probs = torch.softmax(output, dim=1)
            confidence, prediction = torch.max(probs, 1)

        return prediction.item(), confidence.item(), probs.cpu().numpy()[0]

    def get_confirmation_threshold(self, movement, is_expected):
        if is_expected:
            return self.confirm_frames_expected
        elif movement in self.SHORT_MOVEMENT_CLASSES:
            return self.confirm_frames_short
        elif movement > self.expected_next:
            return self.confirm_frames_future
        else:
            return self.confirm_frames_normal

    def get_confidence_threshold(self, movement, is_expected):
        if movement in [0, 21]:
            return self.conf_threshold_expected
        if is_expected:
            return self.conf_threshold_expected
        elif movement in self.SHORT_MOVEMENT_CLASSES:
            return self.conf_threshold_short
        elif movement > self.expected_next:
            return self.conf_threshold_skip
        else:
            return self.conf_threshold_normal

    def validate_movement(self, predicted, confidence, frame_num):
        """Validate prediction with sequential constraints"""
        current_time = frame_num / self.fps

        if self.sequence_complete:
            return None, "Sequence complete"

        if len(self.detected_movements) >= self.max_movements:
            self.sequence_complete = True
            return None, "Max movements reached"

        # First movement must be 0
        if self.expected_next == 0:
            if predicted == 21:
                return None, "Rejected 19_1 at start"
            if predicted == 0:
                return self._try_confirm(predicted, confidence, frame_num, is_expected=True)
            elif predicted > 0 and confidence > self.conf_threshold_skip:
                confirmed = self._try_confirm(predicted, confidence, frame_num, is_expected=False)
                if confirmed[0] is not None:
                    self._add_skipped(0, current_time, "Not detected at start")
                return confirmed
            else:
                return None, f"Waiting for 0_1"

        # Last movement only at end
        if predicted == 21:
            if self.expected_next == 21:
                confirmed = self._try_confirm(predicted, confidence, frame_num, is_expected=True)
                if confirmed[0] is not None:
                    self.sequence_complete = True
                return confirmed
            else:
                return None, f"Rejected 19_1"

        # Reject past movements
        if predicted < self.expected_next:
            return None, f"Rejected past movement"

        # Expected movement
        if predicted == self.expected_next:
            return self._try_confirm(predicted, confidence, frame_num, is_expected=True)

        # Future movement
        if predicted > self.expected_next:
            time_since_last = (frame_num - self.last_detection_frame) / self.fps
            if time_since_last < self.skip_wait_seconds:
                return None, f"Wait before skip"
            confirmed = self._try_confirm(predicted, confidence, frame_num, is_expected=False)
            if confirmed[0] is not None:
                for skip_idx in range(self.expected_next, predicted):
                    self._add_skipped(skip_idx, current_time, "Not detected")
            return confirmed

        return None, "Unknown state"

    def _try_confirm(self, movement, confidence, frame_num, is_expected):
        conf_threshold = self.get_confidence_threshold(movement, is_expected)
        confirm_frames = self.get_confirmation_threshold(movement, is_expected)

        if confidence < conf_threshold:
            self.candidate_movement = None
            self.candidate_frames = 0
            return None, f"Low confidence"

        if movement == self.candidate_movement:
            self.candidate_frames += 1
            self.candidate_confidence_sum += confidence

            if self.candidate_frames >= confirm_frames:
                avg_confidence = self.candidate_confidence_sum / self.candidate_frames
                self._reset_candidate()
                self.expected_next = movement + 1
                self.last_detection_frame = frame_num
                return movement, f"Confirmed"
            else:
                return None, f"Confirming"
        else:
            self.candidate_movement = movement
            self.candidate_frames = 1
            self.candidate_confidence_sum = confidence
            return None, f"New candidate"

    def _reset_candidate(self):
        self.candidate_movement = None
        self.candidate_frames = 0
        self.candidate_confidence_sum = 0

    def _add_skipped(self, movement_idx, time, reason):
        self.skipped_movements.append({
            'movement': movement_idx,
            'name': self.class_names[movement_idx],
            'time': time,
            'reason': reason
        })
        print(f"  [SKIP] {self.class_names[movement_idx]} - {reason}")

    def get_movement_id(self, class_idx):
        """Get movement ID from class index"""
        for mov_id, idx in CLASS_MAPPING.items():
            if idx == class_idx:
                return mov_id
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
            key_poses['middle'] = keypoints_norm[n_frames // 2]

        if n_frames >= 5:
            diffs = np.linalg.norm(
                np.diff(keypoints_norm[:, :, :2], axis=0),
                axis=(1, 2)
            )
            key_poses['peak'] = keypoints_norm[np.argmax(diffs)]

        return key_poses

    def process_video(self, video_path, output_dir, save_video=False, save_images=False):
        """
        Process student video and extract keypoints for each movement

        Args:
            video_path: Path to student video
            output_dir: Output directory for results
            save_video: Whether to save annotated video
            save_images: Whether to save frame images
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)

        if not video_path.exists():
            print(f"Video not found: {video_path}")
            return None

        self.reset_state()

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"\n{'='*70}")
        print("PROCESSING STUDENT VIDEO")
        print(f"{'='*70}")
        print(f"Video: {video_path.name}")
        print(f"Resolution: {width}x{height}, FPS: {self.fps:.1f}, Frames: {total_frames}")
        print(f"Duration: {total_frames/self.fps:.1f}s")
        print(f"Output: {output_dir}")
        print(f"Pose backend: {self.pose_backend}")
        print(f"{'='*70}\n")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = output_dir / 'frames'
        frames_dir.mkdir(exist_ok=True)

        # Video writer
        writer = None
        if save_video:
            video_output = output_dir / f"{video_path.stem}_processed.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(video_output), fourcc, self.fps, (width, height))
            print(f"Saving video to: {video_output}")

        # Process frames
        frame_num = 0
        self.frame_keypoints = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1

            # Extract keypoints
            kp_norm, kp_raw = self.extract_keypoints(frame)
            self.keypoint_buffer.append(kp_norm)
            self.frame_keypoints.append((frame_num, kp_norm, kp_raw, frame if save_images else None))

            # Predict
            pred, conf, probs = self.predict()

            if pred is not None:
                valid_movement, status = self.validate_movement(pred, conf, frame_num)

                if valid_movement is not None and valid_movement != self.current_movement:
                    # Save previous movement
                    if self.current_movement is not None and self.movement_start_frame is not None:
                        duration = (frame_num - self.movement_start_frame) / self.fps
                        self.detected_movements.append({
                            'movement': self.current_movement,
                            'movement_id': self.get_movement_id(self.current_movement),
                            'name': self.class_names[self.current_movement],
                            'start_frame': self.movement_start_frame,
                            'end_frame': frame_num - 1,
                            'duration': duration,
                            'confidence': conf
                        })

                    self.current_movement = valid_movement
                    self.movement_start_frame = frame_num

                    count = len(self.detected_movements) + 1
                    print(f"[{frame_num/self.fps:.1f}s] {self.class_names[valid_movement]} ({conf*100:.1f}%) [{count}/22]")

            if writer:
                # Simple overlay for video
                if self.current_movement is not None:
                    cv2.putText(frame, f"{self.class_names[self.current_movement][:30]}",
                               (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Detected: {len(self.detected_movements)+1}/22",
                               (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                writer.write(frame)

        # Handle last movement
        if self.current_movement is not None and self.movement_start_frame is not None:
            duration = (frame_num - self.movement_start_frame) / self.fps
            self.detected_movements.append({
                'movement': self.current_movement,
                'movement_id': self.get_movement_id(self.current_movement),
                'name': self.class_names[self.current_movement],
                'start_frame': self.movement_start_frame,
                'end_frame': frame_num,
                'duration': duration,
                'confidence': 0
            })

        cap.release()
        if writer:
            writer.release()

        # Save keypoints for each detected movement
        print(f"\n{'='*70}")
        print("SAVING MOVEMENT KEYPOINTS")
        print(f"{'='*70}\n")

        results = {
            'video_name': video_path.name,
            'fps': self.fps,
            'total_frames': total_frames,
            'width': width,
            'height': height,
            'pose_backend': self.pose_backend,
            'num_detected': len(self.detected_movements),
            'num_skipped': len(self.skipped_movements),
            'movements': [],
            'skipped': self.skipped_movements
        }

        for i, mov in enumerate(self.detected_movements):
            movement_id = mov['movement_id']
            movement_name = mov['name']
            start_frame = mov['start_frame']
            end_frame = mov['end_frame']

            # Apply trimming for movements after the first
            trim_frames = 0
            if i > 0:
                trim_frames = self.trim_start_frames
                start_frame = start_frame + trim_frames

            print(f"[{i+1:02d}/22] {movement_id} - {movement_name}")
            if trim_frames > 0:
                print(f"        Trimmed: {trim_frames} frames")
            print(f"        Frames: {start_frame} - {end_frame}")

            # Create movement folder
            movement_dir = frames_dir / movement_id
            movement_dir.mkdir(parents=True, exist_ok=True)

            # Get keypoints for this movement
            keypoints_norm_list = []
            keypoints_raw_list = []
            frame_indices = []

            for fn, kp_norm, kp_raw, frame_img in self.frame_keypoints:
                if start_frame <= fn <= end_frame:
                    keypoints_norm_list.append(kp_norm)
                    keypoints_raw_list.append(kp_raw)
                    frame_indices.append(fn)

                    # Save frame image if requested
                    if save_images and frame_img is not None:
                        img_path = movement_dir / f"frame_{fn:05d}.jpg"
                        cv2.imwrite(str(img_path), frame_img)

            if len(keypoints_norm_list) == 0:
                print(f"        WARNING: No keypoints found!")
                continue

            # Convert to numpy
            keypoints_norm = np.array(keypoints_norm_list, dtype=np.float32)
            keypoints_raw = np.array(keypoints_raw_list, dtype=np.float32)
            frames_array = np.array(frame_indices, dtype=np.int32)
            img_wh = np.array([width, height], dtype=np.int32)

            # Create meta
            meta_str = self.create_meta(movement_id, movement_name, self.fps)

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

            # Add to results
            mov_result = {
                'index': i,
                'movement': mov['movement'],
                'movement_id': movement_id,
                'name': movement_name,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'original_start': mov['start_frame'],
                'trimmed_frames': trim_frames,
                'duration': mov['duration'],
                'confidence': mov['confidence'],
                'num_frames': len(keypoints_norm),
                'keypoints_file': str(npz_path.relative_to(output_dir))
            }
            results['movements'].append(mov_result)

            print(f"        Saved: {len(keypoints_norm)} frames -> keypoints.npz")

        # Save results JSON
        results_path = output_dir / 'results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Print summary
        print(f"\n{'='*70}")
        print("PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Detected: {len(self.detected_movements)}/22 movements")
        if self.skipped_movements:
            print(f"Skipped: {len(self.skipped_movements)} movements")
        print(f"\nOutput:")
        print(f"  {output_dir}/")
        print(f"  ├── results.json")
        print(f"  └── frames/")
        for mov in results['movements'][:3]:
            print(f"      ├── {mov['movement_id']}/")
            print(f"      │   └── keypoints.npz")
        if len(results['movements']) > 3:
            print(f"      └── ...")
        print(f"{'='*70}\n")

        return results


def main():
    parser = argparse.ArgumentParser(description='Process student video for comparison')
    parser.add_argument('--video', required=True, help='Student video path')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    parser.add_argument(
        '--pose-backend',
        type=str,
        default='rtmpose',
        choices=['rtmpose', 'mediapipe'],
        help='Pose backend for student keypoint extraction'
    )
    parser.add_argument('--model-path', type=str, default='', help='Optional explicit model checkpoint path')
    parser.add_argument('--stats-path', type=str, default='', help='Optional explicit normalization stats path')
    parser.add_argument('--trim', type=int, default=3, help='Frames to trim from start')
    parser.add_argument('--save-video', action='store_true', help='Save annotated video')
    parser.add_argument('--save-images', action='store_true', help='Save frame images')

    args = parser.parse_args()

    if args.model_path:
        model_path = Path(args.model_path)
    elif args.pose_backend == 'mediapipe':
        model_path = Paths.CHECKPOINTS_DIR / '22_classes_model_mediapipe' / 'lstm_taegeuk1_best.pth'
    else:
        model_path = Paths.CHECKPOINTS_DIR / 'lstm_taegeuk1_best.pth'

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    if args.stats_path:
        stats_path = Path(args.stats_path)
    elif args.pose_backend == 'mediapipe':
        stats_path = Paths.CHECKPOINTS_DIR / '22_classes_model_mediapipe' / 'normalization_stats.pkl'
    else:
        stats_path = Paths.CHECKPOINTS_DIR / 'normalization_stats.pkl'

    processor = StudentProcessor(
        model_path,
        args.device,
        args.trim,
        pose_backend=args.pose_backend,
        stats_path=stats_path,
    )
    processor.process_video(
        args.video,
        args.output,
        save_video=args.save_video,
        save_images=args.save_images
    )


if __name__ == "__main__":
    main()
