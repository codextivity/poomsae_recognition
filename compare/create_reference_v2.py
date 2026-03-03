"""
Create Reference Database from Master Video using 22-Class Model

Automatically detects movements using the trained LSTM model and extracts
keypoints for each movement to create a reference for comparison.

Usage:
    python create_reference_v2.py --video P001.mp4 --output references/master_22class.pkl
    python create_reference_v2.py --video P001.mp4 --save-video  # Also save annotated video
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import torch
import numpy as np
from pathlib import Path
import pickle
import argparse
import sys
from collections import deque
from PIL import Image, ImageDraw, ImageFont

sys.path.append(str(Path(__file__).parent.parent))
from models.lstm_classifier import PoomsaeLSTM
from configs.lstm_config import LSTMConfig
from configs.paths import Paths
from preprocessing.create_windows import CLASS_NAMES, CLASS_MAPPING


class ReferenceCreator:
    """Create reference database using trained 22-class model"""

    # Short movement classes
    SHORT_MOVEMENT_CLASSES = [6, 12, 14, 17]  # 6_1, 12_1, 14_1, 16_1

    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
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

        # Initialize RTMPose
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

        # Load normalization stats
        self.load_normalization_stats()

        # Keypoint buffer
        self.keypoint_buffer = deque(maxlen=self.window_size)

        # Class names
        self.class_names = CLASS_NAMES

        # ========================================
        # SEQUENTIAL VALIDATION PARAMETERS
        # ========================================
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

    def reset_state(self):
        """Reset tracking state"""
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

        # Storage for movement keypoints
        self.movement_keypoints = {}  # movement_idx -> list of keypoints
        self.movement_keypoints_raw = {}  # movement_idx -> list of raw keypoints

    def load_normalization_stats(self):
        """Load normalization stats from training"""
        stats_file = Paths.CHECKPOINTS_DIR / 'normalization_stats.pkl'

        if stats_file.exists():
            with open(stats_file, 'rb') as f:
                stats = pickle.load(f)
            self.mean = stats['mean']
            self.std = stats['std']
            print(f"Loaded normalization stats")
        else:
            print("WARNING: normalization_stats.pkl not found!")
            self.mean = np.zeros(78)
            self.std = np.ones(78)

    def normalize_keypoints(self, keypoints):
        """Normalize keypoints: hip-centered + height-scaled"""
        coords = keypoints[:, :2].copy()
        conf = keypoints[:, 2:3].copy()

        if np.all(coords == 0):
            return keypoints

        hip = coords[19:20, :]
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
        keypoints, scores = self.pose_estimator(frame)

        if len(keypoints) > 0:
            kp = np.concatenate([keypoints[0], scores[0].reshape(-1, 1)], axis=1)
            kp_normalized = self.normalize_keypoints(kp)
            return kp_normalized, kp.copy()
        return np.zeros((26, 3)), np.zeros((26, 3))

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
        """Get number of frames needed to confirm a movement"""
        if is_expected:
            return self.confirm_frames_expected
        elif movement in self.SHORT_MOVEMENT_CLASSES:
            return self.confirm_frames_short
        elif movement > self.expected_next:
            return self.confirm_frames_future
        else:
            return self.confirm_frames_normal

    def get_confidence_threshold(self, movement, is_expected):
        """Get confidence threshold for a movement"""
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
        """Try to confirm a movement detection"""
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
        """Reset candidate tracking"""
        self.candidate_movement = None
        self.candidate_frames = 0
        self.candidate_confidence_sum = 0

    def _add_skipped(self, movement_idx, time, reason):
        """Add a movement to skipped list"""
        self.skipped_movements.append({
            'movement': movement_idx,
            'name': self.class_names[movement_idx],
            'time': time,
            'reason': reason
        })
        print(f"  [SKIP] {self.class_names[movement_idx]} - {reason}")

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

    def create_reference(self, video_path, output_path, save_video=False, video_output_path=None):
        """Create reference database from video"""
        video_path = Path(video_path)
        if not video_path.exists():
            print(f"Video not found: {video_path}")
            return None

        self.reset_state()

        cap = cv2.VideoCapture(str(video_path))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"\n{'='*70}")
        print("CREATING REFERENCE DATABASE (22-Class)")
        print(f"{'='*70}")
        print(f"Video: {video_path.name}")
        print(f"Resolution: {width}x{height}, FPS: {self.fps:.1f}, Frames: {total_frames}")
        print(f"Duration: {total_frames/self.fps:.1f}s")
        print(f"{'='*70}\n")

        writer = None
        if save_video:
            if video_output_path is None:
                video_output_path = video_path.stem + '_reference.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(video_output_path, fourcc, self.fps, (width, height))
            print(f"Saving video to: {video_output_path}")

        # Storage for all keypoints per frame
        all_frame_keypoints = []  # List of (frame_num, kp_normalized, kp_raw)

        frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1

            # Extract keypoints
            kp_normalized, kp_raw = self.extract_keypoints(frame)
            self.keypoint_buffer.append(kp_normalized)

            # Store for later segmentation
            all_frame_keypoints.append((frame_num, kp_normalized.copy(), kp_raw.copy()))

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
                            'name': self.class_names[self.current_movement],
                            'start_frame': self.movement_start_frame,
                            'end_frame': frame_num,
                            'duration': duration,
                            'confidence': conf
                        })

                    self.current_movement = valid_movement
                    self.movement_start_frame = frame_num

                    count = len(self.detected_movements) + 1
                    print(f"[{frame_num/self.fps:.1f}s] {self.class_names[valid_movement]} ({conf*100:.1f}%) [{count}/22]")

            if writer:
                # Draw info on frame for video output
                display_frame = self.draw_info(frame.copy(), self.current_movement,
                                               conf if pred else 0, frame_num, total_frames)
                writer.write(display_frame)

        # Handle last movement
        if self.current_movement is not None and self.movement_start_frame is not None:
            duration = (frame_num - self.movement_start_frame) / self.fps
            self.detected_movements.append({
                'movement': self.current_movement,
                'name': self.class_names[self.current_movement],
                'start_frame': self.movement_start_frame,
                'end_frame': frame_num,
                'duration': duration,
                'confidence': 0
            })

        cap.release()
        if writer:
            writer.release()

        # Build reference data
        print(f"\n{'='*70}")
        print("BUILDING REFERENCE DATA")
        print(f"{'='*70}\n")

        reference_data = {
            'video_name': video_path.name,
            'fps': self.fps,
            'total_frames': total_frames,
            'num_classes': 22,
            'class_names': CLASS_NAMES,
            'class_mapping': CLASS_MAPPING,
            'movements': []
        }

        # Extract keypoints for each detected movement
        for mov in self.detected_movements:
            movement_idx = mov['movement']
            start_frame = mov['start_frame']
            end_frame = mov['end_frame']

            print(f"Processing: {mov['name']}")
            print(f"  Frames: {start_frame} - {end_frame} ({end_frame - start_frame} frames)")

            # Get keypoints for this movement
            movement_kps = []
            movement_kps_raw = []
            frame_nums = []

            for fn, kp_norm, kp_raw in all_frame_keypoints:
                if start_frame <= fn <= end_frame:
                    movement_kps.append(kp_norm)
                    movement_kps_raw.append(kp_raw)
                    frame_nums.append(fn)

            movement_kps = np.array(movement_kps)  # (N, 26, 3)
            movement_kps_raw = np.array(movement_kps_raw)

            # Extract key poses
            key_poses = self.extract_key_poses(movement_kps)

            movement_data = {
                'movement_number': movement_idx,
                'movement_id': list(CLASS_MAPPING.keys())[list(CLASS_MAPPING.values()).index(movement_idx)],
                'movement_name': mov['name'],
                'start_frame': start_frame,
                'end_frame': end_frame,
                'duration': mov['duration'],
                'confidence': mov['confidence'],
                'all_keypoints': movement_kps,  # Normalized (N, 26, 3)
                'all_keypoints_raw': movement_kps_raw,  # Raw pixel coords (N, 26, 3)
                'frame_numbers': frame_nums,
                'key_poses': key_poses,
                'avg_keypoint': movement_kps.mean(axis=0) if len(movement_kps) > 0 else np.zeros((26, 3))
            }

            reference_data['movements'].append(movement_data)
            print(f"  -> Extracted {len(movement_kps)} frames, key poses: {list(key_poses.keys())}")

        # Add skipped movements info
        reference_data['skipped_movements'] = self.skipped_movements

        # Save reference
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump(reference_data, f)

        print(f"\n{'='*70}")
        print("REFERENCE CREATED SUCCESSFULLY")
        print(f"{'='*70}")
        print(f"Output: {output_path}")
        print(f"Movements: {len(reference_data['movements'])}/22")
        if self.skipped_movements:
            print(f"Skipped: {len(self.skipped_movements)}")
        print(f"{'='*70}\n")

        return reference_data

    def draw_info(self, frame, movement_idx, confidence, frame_num, total_frames):
        """Draw simple info on frame"""
        h, w = frame.shape[:2]

        # Overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Text
        if movement_idx is not None:
            name = self.class_names[movement_idx]
            # Use PIL for Korean
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 28)
            except:
                font = ImageFont.load_default()
            draw.text((10, 10), f"REF: {name}", font=font, fill=(0, 255, 0))
            draw.text((10, 45), f"[{len(self.detected_movements)+1}/22] Conf: {confidence*100:.1f}%",
                     font=font, fill=(255, 255, 255))
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Progress bar
        progress = frame_num / total_frames if total_frames > 0 else 0
        cv2.rectangle(frame, (10, h - 20), (w - 10, h - 5), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, h - 20), (10 + int((w - 20) * progress), h - 5), (0, 200, 0), -1)

        return frame


def main():
    parser = argparse.ArgumentParser(description='Create 22-class reference database')
    parser.add_argument('--video', required=True, help='Master video path')
    parser.add_argument('--output', default='compare/references/master_22class.pkl',
                        help='Output reference file')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--save-video', action='store_true', help='Save annotated video')
    parser.add_argument('--video-output', default=None, help='Output video path')

    args = parser.parse_args()

    model_path = Paths.CHECKPOINTS_DIR / 'lstm_taegeuk1_best.pth'
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    creator = ReferenceCreator(model_path, args.device)
    creator.create_reference(
        args.video,
        args.output,
        save_video=args.save_video,
        video_output_path=args.video_output
    )


if __name__ == "__main__":
    main()
