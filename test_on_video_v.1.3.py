"""
Interactive Video Testing - DATA-DRIVEN VERSION (v5)

Uses actual movement duration data for wait times instead of arbitrary values.

Wait time logic:
1. If reference available: use reference duration × 2.5
2. If no reference: use training data statistics
3. Fallback: adaptive based on remaining time

Usage:
    python test_on_video_v5.py --video path/to/video.mp4 --reference path/to/ref.pkl
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
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import csv

sys.path.append(str(Path(__file__).parent))
from models.lstm_classifier import PoomsaeLSTM
from configs.lstm_config import LSTMConfig
from configs.paths import Paths

import pickle
from scipy.spatial import procrustes


class InteractiveVideoTester:
    """Interactive video testing with DATA-DRIVEN wait times"""

    def __init__(self, model_path, reference_path=None, device='cuda', debug_log=True):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.debug_log = debug_log
        self.prediction_log = []

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
                to_openpose=False, mode='balanced', backend='onnxruntime',
                device='cuda' if device == 'cuda' else 'cpu'
            )
            print("✓ RTMPose initialized")
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
        # DATA-DRIVEN PARAMETERS
        # Based on typical Taegeuk 1 performance data analysis
        # ============================================================

        # Expected durations from typical performances (seconds)
        # Format: (typical, min, max) based on video analysis
        self.movement_duration_stats = {
            0: (3.0, 1.0, 10.0),  # M1: Ready (variable start)
            1: (1.0, 0.5, 2.0),  # M2: 아래막기
            2: (1.2, 0.5, 2.0),  # M3: 몸통반대지르기
            3: (1.0, 0.5, 2.0),  # M4: 아래막기
            4: (1.2, 0.5, 2.0),  # M5: 몸통반대지르기
            5: (1.5, 0.5, 3.0),  # M6: 아래막기
            6: (0.8, 0.3, 1.5),  # M7: 몸통바로지르기 (fast)
            7: (1.2, 0.5, 2.0),  # M8: 몸통안막기
            8: (0.8, 0.3, 1.5),  # M9: 몸통바로지르기
            9: (1.2, 0.5, 2.0),  # M10: 몸통안막기
            10: (0.8, 0.3, 1.5),  # M11: 몸통바로지르기
            11: (2.0, 0.5, 4.0),  # M12: 아래막기 (turn)
            12: (0.5, 0.2, 1.0),  # M13: 몸통바로지르기 (SHORTEST!)
            13: (1.0, 0.5, 2.0),  # M14: 올려막기
            14: (2.5, 1.0, 4.0),  # M15: 앞차고 (kick combo)
            15: (1.5, 0.5, 2.5),  # M16: 뒤로돌아올려막기
            16: (2.5, 1.0, 4.0),  # M17: 앞차고
            17: (0.8, 0.3, 1.5),  # M18: 아래막기 (short!)
            18: (1.0, 0.4, 2.0),  # M19: 몸통지르기
            19: (3.0, 1.0, 10.0),  # M20: Ready (variable end)
        }

        self.min_durations = {i: stats[1] for i, stats in self.movement_duration_stats.items()}

        # Wait time = duration × multiplier
        self.wait_time_multiplier = 2.5
        self.max_absolute_wait = 5.0

        self.seen_confidence_threshold = 0.30
        self.high_confidence_threshold = 0.70
        self.consecutive_frames_for_skip = 15

        # ============================================================
        # TRACKING
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

        self.load_normalization_stats()

        # Reference
        self.reference = None
        self.reference_durations = {}
        self.comparison_scores = {}

        if reference_path and Path(reference_path).exists():
            with open(reference_path, 'rb') as f:
                self.reference = pickle.load(f)
            print(f"✓ Loaded reference: {self.reference['video_name']}")
            self._extract_reference_durations()
        else:
            print(f"  Using default duration statistics")

        self._print_wait_times()
        self.movement_keypoints_buffer = []

    def _extract_reference_durations(self):
        """Extract durations from reference"""
        if self.reference is None:
            return

        print(f"\n📊 Reference Durations (will use these for wait times):")
        for mov in self.reference.get('movements', []):
            mov_idx = mov['movement_number'] - 1
            self.reference_durations[mov_idx] = mov['duration']
            print(f"   M{mov['movement_number']}: {mov['duration']:.2f}s")
        print()

    def _print_wait_times(self):
        """Show calculated wait times"""
        print(f"\n⏱️  WAIT TIMES (data-driven):")
        print(f"{'Mov':<5} {'Typical':<10} {'Max Wait':<10} {'Source':<10}")
        print(f"{'-' * 35}")
        for i in range(20):
            max_wait = self.get_max_wait_time(i)
            if i in self.reference_durations:
                typical = self.reference_durations[i]
                source = "Reference"
            else:
                typical = self.movement_duration_stats[i][0]
                source = "Default"
            print(f"M{i + 1:<3} {typical:<10.2f} {max_wait:<10.2f} {source:<10}")
        print()

    def get_max_wait_time(self, movement_index, current_time=None, total_duration=None):
        """Get wait time based on actual data"""

        # Priority 1: Reference duration
        if movement_index in self.reference_durations:
            ref_dur = self.reference_durations[movement_index]
            return min(ref_dur * self.wait_time_multiplier, self.max_absolute_wait)

        # Priority 2: Statistics
        if movement_index in self.movement_duration_stats:
            typical, min_d, max_d = self.movement_duration_stats[movement_index]
            wait = max(typical * self.wait_time_multiplier, max_d * 1.5)
            return min(wait, self.max_absolute_wait)

        # Priority 3: Adaptive
        if current_time and total_duration:
            remaining = 20 - movement_index
            time_left = total_duration - current_time
            if remaining > 0 and time_left > 0:
                return min(time_left / remaining * 2, self.max_absolute_wait)

        return 2.5

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

    def update_confidence_tracking(self, probs):
        for i in range(20):
            self.movement_max_confidence[i] = max(self.movement_max_confidence[i], probs[i])

    def check_skip_conditions(self, expected, predicted, current_time, conf, total_dur):
        if predicted <= expected:
            return False, "Can't skip backward"
        if self.waiting_since_time is None:
            return False, "Just started waiting"

        wait_time = current_time - self.waiting_since_time
        max_wait = self.get_max_wait_time(expected, current_time, total_dur)

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

    def validate_movement_sequence(self, predicted, current_time, total_dur, conf, probs, frame):
        if self.sequence_complete:
            return None, "Complete"

        self.update_confidence_tracking(probs)

        if self.previous_movement is None:
            if current_time < 15:
                if predicted == 0:
                    return 0, "M1 accepted"
                if predicted == 19:
                    return 0, "M20→M1"
                if current_time > 3:
                    return 0, "Force M1"
                return None, "Buffering"
            return 0, "Force M1"

        if self.current_segment_start_time:
            dur = current_time - self.current_segment_start_time
            min_d = self.min_durations.get(self.previous_movement, 0.3)
            if dur < min_d and not (predicted == self.previous_movement + 1 and conf > 0.85):
                return None, f"Min dur {dur:.2f}s < {min_d:.2f}s"

        if predicted == self.previous_movement:
            self.consecutive_same_prediction_count = 0
            return None, "Same"

        expected = self.previous_movement + 1
        if expected >= 20:
            self.sequence_complete = True
            return None, "Complete"

        if predicted == expected:
            self.reset_waiting_state()
            return predicted, f"M{expected + 1} accepted"

        if self.waiting_since_time is None:
            self.waiting_since_time = current_time

        if predicted == self.consecutive_same_prediction_movement:
            self.consecutive_same_prediction_count += 1
        else:
            self.consecutive_same_prediction_movement = predicted
            self.consecutive_same_prediction_count = 1

        skip, reason = self.check_skip_conditions(expected, predicted, current_time, conf, total_dur)

        if skip:
            for mov in range(expected, predicted):
                self.skipped_movements.append({
                    'movement': mov + 1,
                    'max_conf': self.movement_max_confidence[mov],
                    'time': current_time
                })
                print(f"\n  ⚠️  M{mov + 1} SKIPPED (max conf: {self.movement_max_confidence[mov] * 100:.1f}%)")
            self.reset_waiting_state()
            return predicted, reason

        if predicted == 0 and self.previous_movement == 19:
            self.reset_waiting_state()
            return 0, "Restart"

        wait = current_time - self.waiting_since_time
        max_w = self.get_max_wait_time(expected, current_time, total_dur)
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

    def draw_info(self, frame, mov_id, conf, probs, frame_num, total_frames, fps):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 200), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        if mov_id is not None:
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 32)
                font_sm = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 24)
            except:
                font = font_sm = ImageFont.load_default()

            draw.text((10, 10), f"{self.movement_names[mov_id]} {conf * 100:.0f}%", font=font, fill=(0, 255, 0))

            if self.previous_movement is not None and self.previous_movement < 19:
                exp = self.previous_movement + 1
                exp_conf = probs[exp] if exp < 20 else 0
                max_w = self.get_max_wait_time(exp)
                draw.text((10, 50), f"Expected M{exp + 1}: {exp_conf * 100:.0f}% (max wait: {max_w:.1f}s)",
                          font=font_sm, fill=(255, 255, 0))

            top3 = np.argsort(probs)[-3:][::-1]
            y = 90
            for idx in top3:
                draw.text((10, y), f"{self.movement_names[idx]}: {probs[idx] * 100:.0f}%", font=font_sm,
                          fill=(255, 255, 255))
                y += 28
            frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        cv2.rectangle(frame, (10, h - 25), (w - 10, h - 5), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, h - 25), (10 + int((w - 20) * frame_num / total_frames), h - 5), (0, 255, 0), -1)
        return frame

    def test_video_interactive(self, video_path, save_video=False, output_path='output.mp4'):
        video_path = Path(video_path)
        if not video_path.exists():
            print(f"❌ Not found: {video_path}")
            return

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)) if save_video else None

        print(f"\n{'=' * 60}")
        print(f"DATA-DRIVEN TEST (v5): {video_path.name}")
        print(f"Duration: {total_frames / fps:.1f}s, FPS: {fps:.0f}")
        print(f"{'=' * 60}\n")

        cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Video', 1280, 720)

        frame_num = 0
        paused = False

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_num += 1

                kp_norm, kp_raw = self.extract_keypoints(frame)
                self.keypoint_buffer.append(kp_norm)
                frame = self.draw_skeleton(frame, kp_raw)

                raw, conf, probs = self.predict()

                if raw is not None:
                    smooth = self.get_smoothed_prediction(raw)
                    cur_t, tot_t = frame_num / fps, total_frames / fps

                    valid, reason = self.validate_movement_sequence(smooth, cur_t, tot_t, conf, probs, frame_num)

                    if valid is not None and valid != self.previous_movement and not self.sequence_complete:
                        if self.previous_movement is not None and self.current_segment_start:
                            seg = {
                                'movement_number': self.previous_movement + 1,
                                'movement_name': self.movement_names[self.previous_movement],
                                'duration': (frame_num - 1 - self.current_segment_start) / fps,
                                'avg_confidence': np.mean(self.frame_confidences) if self.frame_confidences else 0
                            }
                            self.movement_segments.append(seg)
                            print(f"  → {seg['movement_name']} ended ({seg['duration']:.2f}s)")

                        self.current_segment_start = frame_num
                        self.current_segment_start_time = cur_t
                        self.previous_movement = valid
                        self.frame_confidences = [conf]
                        self.total_movements_detected += 1
                        self.expected_next_movement = valid + 1
                        self.reset_waiting_state()
                        self.movement_keypoints_buffer = [kp_norm]

                        print(
                            f"\n▶ M{valid + 1}: {self.movement_names[valid]} at {cur_t:.1f}s [{self.total_movements_detected}/20]")
                        if valid == 19:
                            self.sequence_complete = True
                            print("✓ COMPLETE!")
                    elif self.frame_confidences is not None:
                        self.frame_confidences.append(conf)

                    mov_id = valid if valid is not None else self.previous_movement
                else:
                    mov_id = None

                frame = self.draw_info(frame, mov_id, conf if raw else 0, probs if raw else np.zeros(20), frame_num,
                                       total_frames, fps)
                if writer:
                    writer.write(frame)
                current_frame = frame.copy()
            else:
                frame = current_frame.copy()
                cv2.putText(frame, "PAUSED", (w // 2 - 80, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)

            cv2.imshow('Video', frame)
            key = cv2.waitKey(int(1000 / fps) if not paused else 1) & 0xFF
            if key in [ord('q'), 27]:
                break
            elif key == ord(' '):
                paused = not paused

        cap.release()
        cv2.destroyAllWindows()
        if writer:
            writer.release()

        if self.previous_movement is not None and self.current_segment_start:
            self.movement_segments.append({
                'movement_number': self.previous_movement + 1,
                'movement_name': self.movement_names[self.previous_movement],
                'duration': (frame_num - self.current_segment_start) / fps,
                'avg_confidence': np.mean(self.frame_confidences) if self.frame_confidences else 0
            })

        self._print_summary()

    def _print_summary(self):
        print(f"\n{'=' * 60}")
        print(f"RESULTS: {len(self.movement_segments)}/20")

        detected = [s['movement_number'] for s in self.movement_segments]
        missing = set(range(1, 21)) - set(detected)

        if not missing:
            print("✅ All 20 movements detected!")
        else:
            print(f"⚠️  Missing: {sorted(missing)}")

        if self.skipped_movements:
            print(f"\n⚠️  Skipped (low confidence):")
            for s in self.skipped_movements:
                print(f"   M{s['movement']}: max {s['max_conf'] * 100:.0f}%")

        print(f"\n{'No.':<5} {'Movement':<30} {'Duration':<10}")
        print(f"{'-' * 45}")
        for s in self.movement_segments:
            print(f"{s['movement_number']:<5} {s['movement_name']:<30} {s['duration']:.2f}s")
        print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--reference', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save-video', action='store_true')
    parser.add_argument('--output', default='output_v5.mp4')
    parser.add_argument('--wait-mult', type=float, default=2.5, help='Wait multiplier (default: 2.5)')
    parser.add_argument('--max-wait', type=float, default=5.0, help='Max wait cap (default: 5.0)')
    args = parser.parse_args()

    model_path = Paths.CHECKPOINTS_DIR / 'lstm_taegeuk1_best.pth'
    if not model_path.exists():
        print("❌ Model not found")
        return

    tester = InteractiveVideoTester(model_path, args.reference, args.device)
    tester.wait_time_multiplier = args.wait_mult
    tester.max_absolute_wait = args.max_wait
    tester.test_video_interactive(args.video, args.save_video, args.output)


if __name__ == "__main__":
    main()

# python test_on_video_v.1.3.py --video "D:\All Docs\All Projects\Pycharm\poomsae_recognition\data\reference\videos\P011.mp4" --save-video --reference "D:\All Docs\All Projects\Pycharm\poomsae_recognition\compare\references\master_reference.pkl" --output P011_with_scores.mp4