"""
Real-Time Webcam Poomsae Recognition (v6)

Works with live webcam feed - no need to know total duration upfront.

Key changes from video version:
1. Handles unknown total duration
2. Real-time processing optimizations
3. Visual feedback for live performance
4. Audio cues (optional) for movement transitions

Usage:
    python realtime_webcam.py
    python realtime_webcam.py --camera 1  # Use camera index 1
    python realtime_webcam.py --reference path/to/reference.pkl
"""
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import cv2
import torch
import numpy as np
from pathlib import Path
import argparse
import sys
import time
from collections import deque
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))
from models.lstm_classifier import PoomsaeLSTM
from configs.lstm_config import LSTMConfig
from configs.paths import Paths

import pickle


class RealtimeWebcamTester:
    """Real-time webcam poomsae recognition"""

    def __init__(self, model_path, reference_path=None, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load model
        self.config = LSTMConfig()
        self.model = PoomsaeLSTM(self.config)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f"✓ Model loaded")

        # Initialize RTMPose
        try:
            from rtmlib import BodyWithFeet
            self.pose_estimator = BodyWithFeet(
                to_openpose=False,
                mode='lightweight',  # Use lightweight for faster processing
                backend='onnxruntime',
                device='cuda' if device == 'cuda' else 'cpu'
            )
            print("✓ RTMPose initialized (lightweight mode for real-time)")
        except ImportError:
            raise ImportError("rtmlib not installed")

        self.movement_names = [
            "01. 준비자세", "02. 아래막기", "03. 몸통반대지르기", "04. 아래막기",
            "05. 몸통반대지르기", "06. 아래막기", "07. 몸통바로지르기", "08. 몸통안막기",
            "09. 몸통바로지르기", "10. 몸통안막기", "11. 몸통바로지르기", "12. 아래막기",
            "13. 몸통바로지르기", "14. 올려막기", "15. 앞차고몸통반대지르기", "16. 뒤로돌아올려막기",
            "17. 앞차고몸통반대지르기", "18. 아래막기", "19. 몸통지르기", "20. 준비자세",
        ]

        # ============================================================
        # TIMING PARAMETERS (data-driven, works without total_duration)
        # ============================================================

        # Expected durations based on typical performance
        self.movement_duration_stats = {
            0: (3.0, 1.0, 10.0),  # M1
            1: (1.0, 0.5, 2.0),  # M2
            2: (1.2, 0.5, 2.0),  # M3
            3: (1.0, 0.5, 2.0),  # M4
            4: (1.2, 0.5, 2.0),  # M5
            5: (1.5, 0.5, 3.0),  # M6
            6: (0.8, 0.3, 1.5),  # M7
            7: (1.2, 0.5, 2.0),  # M8
            8: (0.8, 0.3, 1.5),  # M9
            9: (1.2, 0.5, 2.0),  # M10
            10: (0.8, 0.3, 1.5),  # M11
            11: (2.0, 0.5, 4.0),  # M12
            12: (0.5, 0.2, 1.0),  # M13
            13: (1.0, 0.5, 2.0),  # M14
            14: (2.5, 1.0, 4.0),  # M15
            15: (1.5, 0.5, 2.5),  # M16
            16: (2.5, 1.0, 4.0),  # M17
            17: (0.8, 0.3, 1.5),  # M18
            18: (1.0, 0.4, 2.0),  # M19
            19: (3.0, 1.0, 10.0),  # M20
        }

        self.min_durations = {i: stats[1] for i, stats in self.movement_duration_stats.items()}
        self.wait_time_multiplier = 2.5
        self.max_absolute_wait = 5.0

        self.seen_confidence_threshold = 0.30
        self.high_confidence_threshold = 0.70
        self.consecutive_frames_for_skip = 15

        # ============================================================
        # TRACKING STATE
        # ============================================================

        self.movement_max_confidence = np.zeros(20)
        self.waiting_since_time = None
        self.consecutive_same_prediction_count = 0
        self.consecutive_same_prediction_movement = None
        self.skipped_movements = []

        self.window_size = self.config.SEQUENCE_LENGTH
        self.keypoint_buffer = deque(maxlen=self.window_size)
        self.prediction_history = deque(maxlen=5)

        self.previous_movement = None
        self.movement_segments = []
        self.current_segment_start = None
        self.current_segment_start_time = None
        self.frame_confidences = []

        self.total_movements_detected = 0
        self.sequence_complete = False
        self.expected_next_movement = 0

        # Session tracking
        self.session_start_time = None
        self.is_session_active = False

        self.load_normalization_stats()

        # Reference (optional)
        self.reference = None
        self.reference_durations = {}

        if reference_path and Path(reference_path).exists():
            with open(reference_path, 'rb') as f:
                self.reference = pickle.load(f)
            print(f"✓ Loaded reference")
            for mov in self.reference.get('movements', []):
                self.reference_durations[mov['movement_number'] - 1] = mov['duration']

        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = None

    def get_max_wait_time(self, movement_index):
        """
        Get max wait time - NO total_duration needed for webcam!
        Uses only movement-specific data.
        """
        # Priority 1: Reference duration
        if movement_index in self.reference_durations:
            ref_dur = self.reference_durations[movement_index]
            return min(ref_dur * self.wait_time_multiplier, self.max_absolute_wait)

        # Priority 2: Statistics
        if movement_index in self.movement_duration_stats:
            typical, min_d, max_d = self.movement_duration_stats[movement_index]
            wait = max(typical * self.wait_time_multiplier, max_d * 1.5)
            return min(wait, self.max_absolute_wait)

        # Fallback
        return 3.0

    def normalize_keypoints(self, keypoints):
        coords = keypoints[:, :2].copy()
        conf = keypoints[:, 2:3].copy()
        if np.all(coords == 0):
            return keypoints
        hip = coords[19:20, :]
        centered = coords - hip
        head, feet = coords[0, :], (coords[15, :] + coords[16, :]) / 2
        height = np.linalg.norm(head - feet)
        if height < 1e-3:
            height = np.linalg.norm(coords[5, :] - coords[6, :]) * 3
        if height < 1e-3:
            height = 1.0
        return np.concatenate([centered / height, conf], axis=1)

    def load_normalization_stats(self):
        stats_file = Paths.CHECKPOINTS_DIR / 'normalization_stats.pkl'
        if stats_file.exists():
            with open(stats_file, 'rb') as f:
                stats = pickle.load(f)
            self.mean, self.std = stats['mean'], stats['std']
            print(f"✓ Loaded normalization stats")
        else:
            self.mean, self.std = np.zeros(78), np.ones(78)

    def extract_keypoints(self, frame):
        keypoints, scores = self.pose_estimator(frame)
        if len(keypoints) > 0:
            kp = np.concatenate([keypoints[0], scores[0].reshape(-1, 1)], axis=1)
            return self.normalize_keypoints(kp), kp.copy()
        return np.zeros((26, 3)), np.zeros((26, 3))

    def predict(self):
        if len(self.keypoint_buffer) < self.window_size:
            return None, 0.0, np.zeros(20)
        window = np.array(list(self.keypoint_buffer))
        window_flat = window.reshape(window.shape[0], -1)
        window_norm = (window_flat - self.mean) / self.std
        x = torch.FloatTensor(window_norm).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(x)
            probs = torch.softmax(out, dim=1)
            conf, pred = torch.max(probs, 1)
        return pred.item(), conf.item(), probs.cpu().numpy()[0]

    def get_smoothed_prediction(self, pred):
        self.prediction_history.append(pred)
        if len(self.prediction_history) >= 3:
            counts = {}
            for p in self.prediction_history:
                counts[p] = counts.get(p, 0) + 1
            return max(counts, key=counts.get)
        return pred

    def reset_waiting_state(self):
        self.waiting_since_time = None
        self.consecutive_same_prediction_count = 0
        self.consecutive_same_prediction_movement = None

    def reset_session(self):
        """Reset everything for a new session"""
        self.movement_max_confidence = np.zeros(20)
        self.waiting_since_time = None
        self.consecutive_same_prediction_count = 0
        self.consecutive_same_prediction_movement = None
        self.skipped_movements = []
        self.keypoint_buffer.clear()
        self.prediction_history.clear()
        self.previous_movement = None
        self.movement_segments = []
        self.current_segment_start = None
        self.current_segment_start_time = None
        self.frame_confidences = []
        self.total_movements_detected = 0
        self.sequence_complete = False
        self.expected_next_movement = 0
        self.session_start_time = None
        self.is_session_active = False
        print("\n🔄 Session reset!")

    def update_confidence_tracking(self, probs):
        for i in range(20):
            self.movement_max_confidence[i] = max(self.movement_max_confidence[i], probs[i])

    def check_skip_conditions(self, expected, predicted, current_time, conf):
        if predicted <= expected:
            return False, "Can't skip backward"
        if self.waiting_since_time is None:
            return False, "Just started waiting"

        wait_time = current_time - self.waiting_since_time
        max_wait = self.get_max_wait_time(expected)

        if wait_time < max_wait:
            return False, f"Wait {wait_time:.1f}s < {max_wait:.1f}s"

        for mov in range(expected, predicted):
            if self.movement_max_confidence[mov] > self.seen_confidence_threshold:
                return False, f"M{mov + 1} seen ({self.movement_max_confidence[mov] * 100:.0f}%)"

        if self.consecutive_same_prediction_count < self.consecutive_frames_for_skip:
            return False, f"Need {self.consecutive_frames_for_skip} frames"

        if conf < self.high_confidence_threshold:
            return False, f"Conf {conf * 100:.0f}% < {self.high_confidence_threshold * 100:.0f}%"

        return True, f"Skip after {wait_time:.1f}s"

    def validate_movement_sequence(self, predicted, current_time, conf, probs):
        """Validate - NO total_duration needed!"""

        if self.sequence_complete:
            return None, "Complete"

        self.update_confidence_tracking(probs)

        # First movement detection - starts the session
        if self.previous_movement is None:
            # Look for M1 (ready stance) or M20 (same pose)
            if predicted == 0 or predicted == 19:
                self.session_start_time = current_time
                self.is_session_active = True
                return 0, "Session started - M1"

            # If we've been waiting too long without M1, accept any high-confidence prediction
            if current_time > 5 and conf > 0.8:
                self.session_start_time = current_time
                self.is_session_active = True
                return 0, "Forced M1 start"

            return None, "Waiting for ready stance (M1)"

        # Minimum duration check
        if self.current_segment_start_time:
            dur = current_time - self.current_segment_start_time
            min_d = self.min_durations.get(self.previous_movement, 0.3)
            if dur < min_d and not (predicted == self.previous_movement + 1 and conf > 0.85):
                return None, f"Min dur {dur:.2f}s"

        # Same movement - reset timer!
        if predicted == self.previous_movement:
            self.consecutive_same_prediction_count = 0
            self.waiting_since_time = None  # Reset timer when back to current
            return None, "Same"

        expected = self.previous_movement + 1
        if expected >= 20:
            self.sequence_complete = True
            return None, "Complete"

        # Accept expected movement
        if predicted == expected:
            self.reset_waiting_state()
            return predicted, f"M{expected + 1} accepted"

        # Start/continue waiting
        if self.waiting_since_time is None:
            self.waiting_since_time = current_time

        if predicted == self.consecutive_same_prediction_movement:
            self.consecutive_same_prediction_count += 1
        else:
            self.consecutive_same_prediction_movement = predicted
            self.consecutive_same_prediction_count = 1

        # Check skip
        skip, reason = self.check_skip_conditions(expected, predicted, current_time, conf)

        if skip:
            for mov in range(expected, predicted):
                self.skipped_movements.append({
                    'movement': mov + 1,
                    'max_conf': self.movement_max_confidence[mov],
                    'time': current_time
                })
                print(f"\n  ⚠️  M{mov + 1} SKIPPED")
            self.reset_waiting_state()
            return predicted, reason

        # Restart after M20
        if predicted == 0 and self.previous_movement == 19:
            self.reset_waiting_state()
            return 0, "Restart"

        wait = current_time - self.waiting_since_time
        max_w = self.get_max_wait_time(expected)
        return None, f"Wait M{expected + 1} ({wait:.1f}s/{max_w:.1f}s)"

    def draw_skeleton(self, frame, kp):
        conns = [(0, 18), (17, 18), (0, 1), (0, 2), (1, 3), (2, 4), (18, 5), (18, 6), (5, 6), (5, 11), (6, 12),
                 (11, 12), (18, 19), (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)]
        for s, e in conns:
            if s < len(kp) and e < len(kp) and kp[s][2] > 0.3 and kp[e][2] > 0.3:
                cv2.line(frame, (int(kp[s][0]), int(kp[s][1])), (int(kp[e][0]), int(kp[e][1])), (0, 255, 0), 2)
        for p in kp:
            if p[2] > 0.3 and p[0] > 0:
                cv2.circle(frame, (int(p[0]), int(p[1])), 4, (0, 0, 255), -1)
        return frame

    def draw_info(self, frame, mov_id, conf, probs, current_time):
        """Draw info overlay for webcam"""
        h, w = frame.shape[:2]

        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 220), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Calculate FPS
        if self.last_frame_time is not None:
            fps = 1.0 / (time.time() - self.last_frame_time + 0.001)
            self.fps_history.append(fps)
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0

        try:
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 32)
                font_sm = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 24)
                font_lg = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 40)
            except:
                font = font_sm = font_lg = ImageFont.load_default()

            # Status bar
            status_color = (0, 255, 0) if self.is_session_active else (255, 255, 0)
            status_text = f"🎯 {self.total_movements_detected}/20" if self.is_session_active else "⏳ Waiting for start..."
            draw.text((10, 5), status_text, font=font_lg, fill=status_color)

            # FPS and time
            draw.text((w - 150, 5), f"FPS: {avg_fps:.0f}", font=font_sm, fill=(255, 255, 255))
            if self.session_start_time:
                elapsed = current_time - self.session_start_time
                draw.text((w - 150, 30), f"Time: {elapsed:.1f}s", font=font_sm, fill=(255, 255, 255))

            # Current movement
            if mov_id is not None:
                mov_text = f"{self.movement_names[mov_id]}"
                draw.text((10, 55), mov_text, font=font, fill=(0, 255, 0))
                draw.text((10, 90), f"Confidence: {conf * 100:.0f}%", font=font_sm, fill=(200, 200, 200))
            else:
                draw.text((10, 55), "Buffering...", font=font, fill=(255, 255, 0))

            # Expected next
            if self.previous_movement is not None and self.previous_movement < 19:
                exp = self.previous_movement + 1
                exp_conf = probs[exp] if exp < 20 else 0
                draw.text((10, 120), f"Next: M{exp + 1} ({exp_conf * 100:.0f}%)", font=font_sm, fill=(255, 255, 0))

            # Top 3 predictions
            if np.any(probs > 0):
                top3 = np.argsort(probs)[-3:][::-1]
                y = 150
                for idx in top3:
                    color = (0, 255, 0) if idx == mov_id else (150, 150, 150)
                    draw.text((10, y), f"{self.movement_names[idx]}: {probs[idx] * 100:.0f}%", font=font_sm, fill=color)
                    y += 22

            frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        except Exception as e:
            # Fallback to OpenCV text
            cv2.putText(frame, f"Movement: {mov_id + 1 if mov_id else '?'}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Progress bar (movements completed)
        bar_y = h - 40
        bar_h = 30
        cv2.rectangle(frame, (10, bar_y), (w - 10, bar_y + bar_h), (50, 50, 50), -1)

        if self.total_movements_detected > 0:
            progress_w = int((w - 20) * self.total_movements_detected / 20)
            color = (0, 255, 0) if self.sequence_complete else (0, 200, 255)
            cv2.rectangle(frame, (10, bar_y), (10 + progress_w, bar_y + bar_h), color, -1)

        # Movement markers
        for i in range(20):
            x = 10 + int((w - 20) * (i + 0.5) / 20)
            detected = any(s['movement_number'] == i + 1 for s in self.movement_segments) or \
                       (self.previous_movement is not None and i <= self.previous_movement)
            color = (0, 255, 0) if detected else (100, 100, 100)
            cv2.circle(frame, (x, bar_y + bar_h // 2), 5, color, -1)

        # Instructions
        cv2.putText(frame, "R: Reset | Q: Quit | SPACE: Pause", (10, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return frame

    def run_webcam(self, camera_index=0, save_video=False, output_path='webcam_output.mp4'):
        """Main webcam loop"""

        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print(f"❌ Cannot open camera {camera_index}")
            print("   Try: --camera 1 or --camera 2")
            return

        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"\n{'=' * 60}")
        print(f"🎥 WEBCAM REAL-TIME RECOGNITION")
        print(f"{'=' * 60}")
        print(f"Resolution: {actual_w}x{actual_h}")
        print(f"Camera FPS: {actual_fps}")
        print(f"\nControls:")
        print(f"  R     - Reset session")
        print(f"  SPACE - Pause/Resume")
        print(f"  Q/ESC - Quit")
        print(f"{'=' * 60}")
        print(f"\n👤 Stand in ready stance (준비자세) to begin...\n")

        writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, 30, (actual_w, actual_h))

        cv2.namedWindow('Poomsae Recognition', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Poomsae Recognition', 1280, 720)

        paused = False
        frame_count = 0
        start_time = time.time()

        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("❌ Failed to read frame")
                        break

                    frame_count += 1
                    current_time = time.time() - start_time

                    # Flip horizontally for mirror effect (more intuitive)
                    frame = cv2.flip(frame, 1)

                    # Extract keypoints and predict
                    kp_norm, kp_raw = self.extract_keypoints(frame)
                    self.keypoint_buffer.append(kp_norm)
                    frame = self.draw_skeleton(frame, kp_raw)

                    raw, conf, probs = self.predict()

                    if raw is not None:
                        smooth = self.get_smoothed_prediction(raw)
                        valid, reason = self.validate_movement_sequence(smooth, current_time, conf, probs)

                        if valid is not None and valid != self.previous_movement and not self.sequence_complete:
                            # End previous segment
                            if self.previous_movement is not None and self.current_segment_start_time:
                                seg = {
                                    'movement_number': self.previous_movement + 1,
                                    'movement_name': self.movement_names[self.previous_movement],
                                    'duration': current_time - self.current_segment_start_time,
                                }
                                self.movement_segments.append(seg)
                                print(f"  ✓ {seg['movement_name']} ({seg['duration']:.2f}s)")

                            # Start new segment
                            self.current_segment_start_time = current_time
                            self.previous_movement = valid
                            self.frame_confidences = [conf]
                            self.total_movements_detected += 1
                            self.reset_waiting_state()

                            print(
                                f"\n▶ M{valid + 1}: {self.movement_names[valid]} [{self.total_movements_detected}/20]")

                            if valid == 19:
                                self.sequence_complete = True
                                elapsed = current_time - self.session_start_time
                                print(f"\n🎉 COMPLETE! Total time: {elapsed:.1f}s")
                                print(f"   Movements detected: {self.total_movements_detected}/20")
                                if self.skipped_movements:
                                    print(f"   Skipped: {[s['movement'] for s in self.skipped_movements]}")

                        elif self.frame_confidences is not None:
                            self.frame_confidences.append(conf)

                        mov_id = valid if valid is not None else self.previous_movement
                    else:
                        mov_id = None

                    frame = self.draw_info(frame, mov_id, conf if raw else 0,
                                           probs if raw is not None else np.zeros(20), current_time)

                    if writer:
                        writer.write(frame)

                    self.last_frame_time = time.time()
                    current_frame = frame.copy()

                else:
                    frame = current_frame.copy()
                    cv2.putText(frame, "PAUSED", (actual_w // 2 - 80, actual_h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)

                cv2.imshow('Poomsae Recognition', frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:  # Q or ESC
                    break
                elif key == ord('r'):  # Reset
                    self.reset_session()
                    start_time = time.time()
                    frame_count = 0
                elif key == ord(' '):  # Pause
                    paused = not paused
                    print("PAUSED" if paused else "RESUMED")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            if writer:
                writer.release()

            self._print_summary()

    def _print_summary(self):
        """Print session summary"""
        print(f"\n{'=' * 60}")
        print(f"SESSION SUMMARY")
        print(f"{'=' * 60}")
        print(f"Movements detected: {len(self.movement_segments)}/20")

        if self.skipped_movements:
            print(f"\n⚠️ Skipped movements:")
            for s in self.skipped_movements:
                print(f"   M{s['movement']}: max conf {s['max_conf'] * 100:.0f}%")

        if self.movement_segments:
            print(f"\nDetected movements:")
            total_time = sum(s.get('duration', 0) for s in self.movement_segments)
            for s in self.movement_segments:
                dur = s.get('duration', 0)
                print(f"   M{s['movement_number']:2d}: {s['movement_name']:<25} {dur:.2f}s")
            print(f"\nTotal active time: {total_time:.1f}s")

        print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description='Real-time webcam poomsae recognition')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--reference', type=str, default=None, help='Reference file path')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--save-video', action='store_true', help='Save output video')
    parser.add_argument('--output', type=str, default='webcam_output.mp4', help='Output path')

    args = parser.parse_args()

    model_path = Paths.CHECKPOINTS_DIR / 'lstm_taegeuk1_best.pth'
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return

    tester = RealtimeWebcamTester(
        model_path=model_path,
        reference_path=args.reference,
        device=args.device
    )

    tester.run_webcam(
        camera_index=args.camera,
        save_video=args.save_video,
        output_path=args.output
    )


if __name__ == "__main__":
    main()