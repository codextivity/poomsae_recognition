"""
Video Testing for 22-Class Poomsae Model with Sequential Validation

Features:
- 22 classes (with sub-movements 14_1/14_2, 16_1/16_2)
- Sequential movement validation (movements must follow order)
- Handles skipped movements
- Special handling for similar poses (0_1 vs 19_1)
- Fast acceptance for short movements

Usage:
    python test_on_video_v2.py --video path/to/video.mp4
    python test_on_video_v2.py --video path/to/video.mp4 --save-video --output output.mp4
    python test_on_video_v2.py --webcam --camera 0
    python test_on_video_v2.py --video path/to/video.mp4 --raw-mode --raw-conf-threshold 0.4
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import torch
import numpy as np
from pathlib import Path
import argparse
import sys
from collections import deque
import pickle
from PIL import Image, ImageDraw, ImageFont

sys.path.append(str(Path(__file__).parent.parent))
from models.lstm_classifier import PoomsaeLSTM
from configs.lstm_config import LSTMConfig
from configs.paths import Paths
from configs.policy_config import PolicyConfig
from preprocessing.create_windows import CLASS_NAMES, CLASS_MAPPING


class VideoTester:
    """Test trained model on video with 22-class support and sequential validation"""

    # Short movement classes (faster detection needed)
    SHORT_MOVEMENT_CLASSES = [6, 12, 14, 17]  # 6_1, 12_1, 14_1, 16_1

    def __init__(self, model_path, device='cuda', raw_mode=False, raw_conf_threshold=0.0, raw_smoothing=3):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        PolicyConfig.apply_profile()
        self.policy = PolicyConfig
        self.raw_mode = bool(raw_mode)
        self.raw_conf_threshold = float(raw_conf_threshold)
        self.raw_smoothing = max(1, int(raw_smoothing))
        self.raw_history = deque(maxlen=self.raw_smoothing)

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

        # Print model info
        if 'model_config' in checkpoint:
            mc = checkpoint['model_config']
            print(f"Model config: seq_len={mc.get('sequence_length')}, classes={mc.get('num_classes')}")
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

        # Keypoint buffer for sliding window
        self.keypoint_buffer = deque(maxlen=self.window_size)

        # Class names (22 classes)
        self.class_names = CLASS_NAMES

        # ========================================
        # SEQUENTIAL VALIDATION PARAMETERS
        # ========================================

        # Confirmation thresholds (frames needed to confirm detection)
        self.confirm_frames_expected = 7       # Expected movement: conservative accept
        self.confirm_frames_short = 5          # Short movements: fast accept
        self.confirm_frames_normal = 10        # Normal movements: standard
        self.confirm_frames_future = 15        # Future movement (skip): slow accept

        # Confidence thresholds
        self.conf_threshold_expected = 0.50    # Stricter expected threshold for cleaner transitions
        self.conf_threshold_boundary = 0.35    # For start/end pose confusion (0_1 vs 19_1)
        self.conf_threshold_short = 0.80       # Higher for short movements
        self.conf_threshold_normal = 0.60      # Normal threshold
        self.conf_threshold_skip = 0.85        # High threshold before skipping

        # Transition timing guard to prevent early switching
        self.min_hold_seconds_normal = 0.45
        self.min_hold_seconds_short = 0.30

        # Webcam-only startup fallback (keeps video behavior unchanged)
        self.webcam_force_start_enabled = True
        self.webcam_force_start_after_seconds = 5.0

        # Skip timing
        self.skip_wait_seconds = 2.0           # Wait time before allowing skip
        if not self.policy.ALLOW_FUTURE_SKIP:
            print("Policy: future skip disabled")

        # ========================================
        # TRACKING STATE
        # ========================================

        self.expected_next = 0                 # Next expected movement (0-21)
        self.current_movement = None           # Currently confirmed movement
        self.movement_start_frame = None       # Frame when current movement started

        self.detected_movements = []           # List of confirmed detections
        self.skipped_movements = []            # List of skipped movements

        self.sequence_complete = False         # True when 19_1 is detected
        self.max_movements = 22                # Maximum detections allowed

        # Candidate tracking (for confirmation)
        self.candidate_movement = None         # Movement being evaluated
        self.candidate_frames = 0              # Frames candidate has been detected
        self.candidate_confidence_sum = 0      # Sum of confidences for averaging
        self.last_detection_frame = 0          # Frame of last confirmed detection

        # FPS (set during processing)
        self.fps = 30.0
        self.runtime_mode = 'video'
        if self.raw_mode:
            print(
                "Raw mode enabled: sequence validation bypassed "
                f"(conf>={self.raw_conf_threshold:.2f}, smoothing={self.raw_smoothing})"
            )

    def load_normalization_stats(self):
        """Load normalization stats from training"""
        stats_file = Paths.CHECKPOINTS_DIR / 'normalization_stats.pkl'

        if stats_file.exists():
            with open(stats_file, 'rb') as f:
                stats = pickle.load(f)
            self.mean = stats['mean']
            self.std = stats['std']
            print(f"Loaded normalization stats (samples={stats.get('num_samples', '?')})")
        else:
            print("WARNING: normalization_stats.pkl not found!")
            print("Run: python utils/save_normalization_stats.py")
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
        # Special case: 0_1 and 19_1 have low confidence due to similar poses
        if movement in [0, 21]:
            return self.conf_threshold_boundary

        if is_expected:
            return self.conf_threshold_expected
        elif movement in self.SHORT_MOVEMENT_CLASSES:
            return self.conf_threshold_short
        elif movement > self.expected_next:
            return self.conf_threshold_skip
        else:
            return self.conf_threshold_normal

    def _get_min_hold_seconds_for_current(self):
        """Minimum time the current movement should hold before switching."""
        if self.current_movement in self.SHORT_MOVEMENT_CLASSES:
            return self.min_hold_seconds_short
        return self.min_hold_seconds_normal

    def validate_movement(self, predicted, confidence, frame_num):
        """
        Validate prediction with sequential constraints and confirmation.

        Returns:
            (confirmed_movement, status_message) or (None, reason)
        """
        current_time = frame_num / self.fps

        # ========================================
        # RULE 1: Sequence complete - reject all
        # ========================================
        if self.sequence_complete:
            return None, "Sequence complete"

        # ========================================
        # RULE 2: Max movements reached
        # ========================================
        if len(self.detected_movements) >= self.max_movements:
            self.sequence_complete = True
            return None, "Max movements reached"

        # ========================================
        # RULE 3: First movement must be 0 (0_1)
        # ========================================
        if self.expected_next == 0:
            if predicted == 21:  # 19_1 at start - REJECT (pose confusion)
                return None, "Rejected 19_1 at start"

            if predicted == 0:
                # Expected first movement
                return self._try_confirm(predicted, confidence, frame_num, is_expected=True)
            elif (
                self.runtime_mode == 'webcam'
                and self.webcam_force_start_enabled
                and current_time >= self.webcam_force_start_after_seconds
            ):
                # Webcam can be noisy at startup; force-confirm 0_1 after warmup.
                forced_conf = max(confidence, self.conf_threshold_boundary + 0.01)
                return self._try_confirm(0, forced_conf, frame_num, is_expected=True)
            elif predicted > 0 and confidence > self.conf_threshold_skip:
                if not self.policy.ALLOW_FUTURE_SKIP:
                    return None, "Skip disabled by policy (waiting for 0_1)"
                # High confidence future movement - might have missed 0_1
                confirmed = self._try_confirm(predicted, confidence, frame_num, is_expected=False)
                if confirmed[0] is not None:
                    # Skip 0_1
                    self._add_skipped(0, current_time, "Not detected at start")
                return confirmed
            else:
                return None, f"Waiting for 0_1 (got {predicted})"

        # ========================================
        # RULE 4: Last movement (21/19_1) only at end
        # ========================================
        if predicted == 21:
            if self.expected_next == 21:
                # Expected final movement
                confirmed = self._try_confirm(predicted, confidence, frame_num, is_expected=True)
                if confirmed[0] is not None:
                    self.sequence_complete = True
                return confirmed
            else:
                # 19_1 detected too early - likely confusion with 0_1
                return None, f"Rejected 19_1 (expected {self.expected_next})"

        # ========================================
        # RULE 5: Reject past movements
        # ========================================
        if predicted < self.expected_next:
            return None, f"Rejected past movement {predicted}"

        # ========================================
        # RULE 6: Expected movement - fast accept
        # ========================================
        if predicted == self.expected_next:
            # Guard: don't switch too early while current movement is still unfolding.
            if self.current_movement is not None and self.movement_start_frame is not None:
                elapsed = (frame_num - self.movement_start_frame) / self.fps
                required = self._get_min_hold_seconds_for_current()
                if elapsed < required:
                    # Reset candidate so transition requires fresh consecutive evidence
                    # after minimum hold time is reached.
                    self._reset_candidate()
                    return None, f"Hold current movement ({elapsed:.2f}s / {required:.2f}s)"
            return self._try_confirm(predicted, confidence, frame_num, is_expected=True)

        # ========================================
        # RULE 7: Future movement - slow accept with skip
        # ========================================
        if predicted > self.expected_next:
            if not self.policy.ALLOW_FUTURE_SKIP:
                return None, f"Skip disabled (expected {self.expected_next})"
            time_since_last = (frame_num - self.last_detection_frame) / self.fps

            # Must wait before allowing skip
            if time_since_last < self.skip_wait_seconds:
                return None, f"Wait before skip ({time_since_last:.1f}s / {self.skip_wait_seconds}s)"

            # Try to confirm future movement
            confirmed = self._try_confirm(predicted, confidence, frame_num, is_expected=False)

            if confirmed[0] is not None:
                # Skip intermediate movements
                for skip_idx in range(self.expected_next, predicted):
                    self._add_skipped(skip_idx, current_time, "Not detected")

            return confirmed

        return None, "Unknown state"

    def select_movement(self, predicted, confidence, probs, frame_num):
        """Select movement based on runtime mode (validated or raw)."""
        if not self.raw_mode:
            return self.validate_movement(predicted, confidence, frame_num)

        if confidence < self.raw_conf_threshold:
            return None, f"Raw low confidence {confidence:.1%} < {self.raw_conf_threshold:.1%}"

        if self.raw_smoothing <= 1:
            return predicted, "Raw direct"

        self.raw_history.append(predicted)
        if len(self.raw_history) < self.raw_smoothing:
            return None, f"Raw buffering ({len(self.raw_history)}/{self.raw_smoothing})"

        counts = {}
        for p in self.raw_history:
            counts[p] = counts.get(p, 0) + 1

        max_count = max(counts.values())
        # Tie-break with recency.
        candidate = None
        for p in reversed(self.raw_history):
            if counts[p] == max_count:
                candidate = p
                break

        return candidate, f"Raw smoothed ({max_count}/{self.raw_smoothing})"

    def _try_confirm(self, movement, confidence, frame_num, is_expected):
        """Try to confirm a movement detection"""
        conf_threshold = self.get_confidence_threshold(movement, is_expected)
        confirm_frames = self.get_confirmation_threshold(movement, is_expected)

        # Check confidence threshold
        if confidence < conf_threshold:
            self.candidate_movement = None
            self.candidate_frames = 0
            return None, f"Low confidence {confidence:.1%} < {conf_threshold:.1%}"

        # Same candidate - accumulate
        if movement == self.candidate_movement:
            self.candidate_frames += 1
            self.candidate_confidence_sum += confidence

            if self.candidate_frames >= confirm_frames:
                # CONFIRMED!
                avg_confidence = self.candidate_confidence_sum / self.candidate_frames
                self._reset_candidate()
                self.expected_next = movement + 1
                self.last_detection_frame = frame_num
                return movement, f"Confirmed ({self.candidate_frames} frames, {avg_confidence:.1%})"
            else:
                return None, f"Confirming {movement} ({self.candidate_frames}/{confirm_frames})"

        # New candidate
        else:
            self.candidate_movement = movement
            self.candidate_frames = 1
            self.candidate_confidence_sum = confidence
            return None, f"New candidate {movement}"

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

    def draw_skeleton(self, frame, keypoints_raw):
        """Draw skeleton on frame"""
        connections = [
            (0, 18), (17, 18), (0, 1), (0, 2), (1, 3), (2, 4),
            (18, 5), (18, 6), (5, 6), (5, 11), (6, 12), (11, 12),
            (18, 19), (5, 7), (7, 9), (6, 8), (8, 10),
            (11, 13), (13, 15), (12, 14), (14, 16)
        ]

        for start, end in connections:
            if (start < len(keypoints_raw) and end < len(keypoints_raw) and
                keypoints_raw[start][2] > 0.3 and keypoints_raw[end][2] > 0.3):
                pt1 = (int(keypoints_raw[start][0]), int(keypoints_raw[start][1]))
                pt2 = (int(keypoints_raw[end][0]), int(keypoints_raw[end][1]))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        for kp in keypoints_raw:
            if kp[2] > 0.3 and kp[0] > 0:
                cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, (0, 0, 255), -1)

        return frame

    def draw_info(self, frame, movement_idx, confidence, probs, frame_num, total_frames):
        """Draw prediction info on frame using PIL for Korean text support"""
        h, w = frame.shape[:2]

        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 200), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Convert to PIL for Korean text
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        # Load Korean font
        try:
            font_large = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 32)
            font_medium = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 24)
            font_small = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 20)
        except:
            try:
                font_large = ImageFont.truetype("C:/Windows/Fonts/gulim.ttc", 32)
                font_medium = ImageFont.truetype("C:/Windows/Fonts/gulim.ttc", 24)
                font_small = ImageFont.truetype("C:/Windows/Fonts/gulim.ttc", 20)
            except:
                font_large = font_medium = font_small = ImageFont.load_default()

        # Current movement
        if movement_idx is not None:
            name = self.class_names[movement_idx] if movement_idx < len(self.class_names) else f"Class {movement_idx}"
            color = (0, 255, 0) if confidence > 0.7 else (255, 255, 0) if confidence > 0.5 else (255, 165, 0)
            draw.text((10, 10), f"{name}", font=font_large, fill=color)
            draw.text((10, 50), f"Confidence: {confidence*100:.1f}%", font=font_medium, fill=(255, 255, 255))

        # Detection count
        draw.text((10, 85), f"Detected: {len(self.detected_movements)}/22", font=font_medium, fill=(200, 200, 200))

        # Expected next (or raw mode info)
        if self.raw_mode:
            draw.text((10, 115), "Raw mode: sequence validation OFF", font=font_small, fill=(255, 200, 120))
        elif self.expected_next < 22:
            exp_name = self.class_names[self.expected_next]
            draw.text((10, 115), f"Expected: {exp_name}", font=font_small, fill=(150, 150, 255))

        # Top 3 predictions
        top3 = np.argsort(probs)[-3:][::-1]
        y = 145
        for idx in top3:
            name = self.class_names[idx] if idx < len(self.class_names) else f"Class {idx}"
            draw.text((10, y), f"{name}: {probs[idx]*100:.1f}%", font=font_small, fill=(150, 150, 150))
            y += 22

        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Progress bar
        progress = frame_num / total_frames if total_frames > 0 else 0
        cv2.rectangle(frame, (10, h - 25), (w - 10, h - 5), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, h - 25), (10 + int((w - 20) * progress), h - 5), (0, 255, 0), -1)

        # Time info
        current_time = frame_num / self.fps
        total_time = total_frames / self.fps
        cv2.putText(frame, f"{current_time:.1f}s / {total_time:.1f}s", (w - 150, h - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return frame

    def process_video(self, video_path, save_video=False, output_path='output.mp4', show_window=True):
        """Process video with sequential validation"""
        self.runtime_mode = 'video'
        self.raw_history.clear()
        video_path = Path(video_path)
        if not video_path.exists():
            print(f"Video not found: {video_path}")
            return

        cap = cv2.VideoCapture(str(video_path))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"\nProcessing: {video_path.name}")
        print(f"Resolution: {width}x{height}, FPS: {self.fps:.1f}, Frames: {total_frames}")
        print(f"Duration: {total_frames/self.fps:.1f}s")
        print(f"\n{'='*60}")
        if self.raw_mode:
            print("RAW MODE ENABLED (NO SEQUENCE VALIDATION)")
        else:
            print("SEQUENTIAL VALIDATION ENABLED")
        print(f"{'='*60}\n")

        writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
            print(f"Saving to: {output_path}")

        if show_window:
            cv2.namedWindow('Poomsae Recognition', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Poomsae Recognition', 1280, 720)

        frame_num = 0
        last_conf = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1

            # Extract keypoints
            kp_normalized, kp_raw = self.extract_keypoints(frame)
            self.keypoint_buffer.append(kp_normalized)

            # Draw skeleton
            frame = self.draw_skeleton(frame, kp_raw)

            # Predict
            pred, conf, probs = self.predict()

            if pred is not None:
                last_conf = conf

                # Choose movement (validated mode or raw mode)
                valid_movement, status = self.select_movement(pred, conf, probs, frame_num)

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

                    # Update current
                    self.current_movement = valid_movement
                    self.movement_start_frame = frame_num

                    count = len(self.detected_movements) + 1
                    print(f"[{frame_num/self.fps:.1f}s] {self.class_names[valid_movement]} ({conf*100:.1f}%) [{count}/22]")

            # Draw info
            display_movement = self.current_movement
            frame = self.draw_info(frame, display_movement, last_conf,
                                   probs if pred is not None else np.zeros(self.num_classes),
                                   frame_num, total_frames)

            if writer:
                writer.write(frame)

            if show_window:
                cv2.imshow('Poomsae Recognition', frame)
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), 27]:
                    break
                elif key == ord(' '):
                    cv2.waitKey(0)

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
        if show_window:
            cv2.destroyAllWindows()

        self.print_summary()

    def process_webcam(
        self,
        camera_index=0,
        save_video=False,
        output_path='webcam_output.mp4',
        show_window=True,
        allow_future_skip=False
    ):
        """Process live webcam stream with the same 22-class validation."""
        self.runtime_mode = 'webcam'
        self.raw_history.clear()

        # Webcam-only threshold tuning for more stable startup.
        original_confirm_frames_expected = self.confirm_frames_expected
        original_conf_threshold_expected = self.conf_threshold_expected
        original_conf_threshold_boundary = self.conf_threshold_boundary
        self.confirm_frames_expected = 5
        self.conf_threshold_expected = 0.45
        self.conf_threshold_boundary = 0.30

        # Webcam is noisier than offline video; strict sequence is safer by default.
        original_allow_future_skip = self.policy.ALLOW_FUTURE_SKIP
        if not allow_future_skip and original_allow_future_skip:
            self.policy.ALLOW_FUTURE_SKIP = False
            print("Webcam strict mode: future skip disabled")
        elif allow_future_skip:
            self.policy.ALLOW_FUTURE_SKIP = True
            print("Webcam mode: future skip enabled")

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Cannot open camera index {camera_index}")
            print("Try another camera index, for example: --camera 1")
            self.policy.ALLOW_FUTURE_SKIP = original_allow_future_skip
            return

        # Camera FPS is often unreliable on webcams; use a safe fallback.
        cam_fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = cam_fps if cam_fps and cam_fps > 1.0 else 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"\nWebcam mode (camera={camera_index})")
        print(f"Resolution: {width}x{height}, FPS: {self.fps:.1f}")
        print(f"\n{'='*60}")
        if self.raw_mode:
            print("RAW MODE ENABLED (WEBCAM, NO SEQUENCE VALIDATION)")
        else:
            print("SEQUENTIAL VALIDATION ENABLED (WEBCAM)")
        print("Controls: q/ESC quit, space pause/resume")
        print(f"{'='*60}\n")

        writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
            print(f"Saving to: {output_path}")

        if show_window:
            cv2.namedWindow('Poomsae Recognition (Webcam)', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Poomsae Recognition (Webcam)', 1280, 720)

        frame_num = 0
        last_conf = 0.0
        paused = False

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("Webcam frame read failed, stopping.")
                    break

                frame_num += 1

                # Extract keypoints
                kp_normalized, kp_raw = self.extract_keypoints(frame)
                self.keypoint_buffer.append(kp_normalized)

                # Draw skeleton
                frame = self.draw_skeleton(frame, kp_raw)

                # Predict
                pred, conf, probs = self.predict()

                if pred is not None:
                    last_conf = conf

                    # Choose movement (validated mode or raw mode)
                    valid_movement, status = self.select_movement(pred, conf, probs, frame_num)

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

                        # Update current
                        self.current_movement = valid_movement
                        self.movement_start_frame = frame_num

                        count = len(self.detected_movements) + 1
                        print(f"[{frame_num/self.fps:.1f}s] {self.class_names[valid_movement]} ({conf*100:.1f}%) [{count}/22]")

                # Draw info (no fixed total frames in webcam mode)
                display_movement = self.current_movement
                frame = self.draw_info(
                    frame,
                    display_movement,
                    last_conf,
                    probs if pred is not None else np.zeros(self.num_classes),
                    frame_num,
                    0
                )

                if writer:
                    writer.write(frame)
            else:
                # Keep showing last frame while paused.
                if frame is not None:
                    cv2.putText(frame, "PAUSED", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            if show_window:
                cv2.imshow('Poomsae Recognition (Webcam)', frame)
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), 27]:
                    break
                if key == ord(' '):
                    paused = not paused
            else:
                # Headless mode still needs an escape route when not displaying.
                if self.sequence_complete:
                    break

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
        if show_window:
            cv2.destroyAllWindows()

        # Restore policy state for any subsequent run in the same process.
        self.policy.ALLOW_FUTURE_SKIP = original_allow_future_skip
        self.confirm_frames_expected = original_confirm_frames_expected
        self.conf_threshold_expected = original_conf_threshold_expected
        self.conf_threshold_boundary = original_conf_threshold_boundary
        self.runtime_mode = 'video'

        self.print_summary()

    def print_summary(self):
        """Print detection summary"""
        print(f"\n{'='*60}")
        print("DETECTION SUMMARY")
        print(f"{'='*60}")
        print(f"Detected: {len(self.detected_movements)}/22 movements")

        if self.skipped_movements:
            print(f"Skipped: {len(self.skipped_movements)} movements")

        print(f"\n{'No.':<5} {'Movement':<40} {'Duration':<10} {'Conf':<8}")
        print(f"{'-'*65}")

        for i, mov in enumerate(self.detected_movements):
            conf_str = f"{mov.get('confidence', 0)*100:.1f}%" if mov.get('confidence') else "-"
            print(f"{i+1:<5} {mov['name']:<40} {mov['duration']:.2f}s     {conf_str}")

        if self.skipped_movements:
            print(f"\n{'='*60}")
            print("SKIPPED MOVEMENTS")
            print(f"{'='*60}")
            for skip in self.skipped_movements:
                print(f"  - {skip['name']} at {skip['time']:.1f}s ({skip['reason']})")

        print(f"\n{'='*60}")
        if self.raw_mode:
            unique_movs = sorted({mov['movement'] for mov in self.detected_movements})
            print("STATUS: RAW MODE (sequence validation disabled)")
            print(f"Unique classes seen: {len(unique_movs)}")
        elif len(self.detected_movements) == 22:
            print("STATUS: All 22 movements detected!")
        elif self.sequence_complete:
            print(f"STATUS: Sequence complete with {len(self.detected_movements)} detections")
            if self.skipped_movements:
                print(f"        ({len(self.skipped_movements)} movements were skipped)")
        else:
            missing = 22 - len(self.detected_movements)
            print(f"STATUS: Incomplete ({missing} movements not detected)")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Test Poomsae model on video')
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--video', help='Path to video file')
    source_group.add_argument('--webcam', action='store_true', help='Use webcam stream as input')
    parser.add_argument('--camera', type=int, default=0, help='Camera index for webcam mode (default: 0)')
    parser.add_argument(
        '--allow-future-skip-webcam',
        action='store_true',
        help='Enable future-skip jumps in webcam mode (default: disabled for strict sequence)'
    )
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--raw-mode', action='store_true', help='Bypass sequence validation and use raw model predictions')
    parser.add_argument('--raw-conf-threshold', type=float, default=0.0, help='Min confidence for raw mode (default: 0.0)')
    parser.add_argument('--raw-smoothing', type=int, default=3, help='Majority-vote window size in raw mode (default: 3)')
    parser.add_argument('--save-video', action='store_true', help='Save output video')
    parser.add_argument('--output', default='output_v2.mp4', help='Output video path')
    parser.add_argument('--no-display', action='store_true', help='Disable display window')
    args = parser.parse_args()

    model_path = Paths.CHECKPOINTS_DIR / 'lstm_taegeuk1_best.pth'
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    tester = VideoTester(
        model_path,
        args.device,
        raw_mode=args.raw_mode,
        raw_conf_threshold=args.raw_conf_threshold,
        raw_smoothing=args.raw_smoothing
    )
    if args.webcam:
        tester.process_webcam(
            camera_index=args.camera,
            save_video=args.save_video,
            output_path=args.output,
            show_window=not args.no_display,
            allow_future_skip=args.allow_future_skip_webcam
        )
    else:
        tester.process_video(
            args.video,
            save_video=args.save_video,
            output_path=args.output,
            show_window=not args.no_display
        )


if __name__ == "__main__":
    main()
