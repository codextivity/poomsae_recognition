"""
Interactive Video Testing - With LSTM Prediction Logging

Shows video playing with real-time predictions overlaid.
Logs LSTM predictions for each frame to help debug issues.

Usage:
    python test_on_video_debug.py --video path/to/video.mp4

Controls:
    SPACE: Pause/Resume
    Q or ESC: Quit
    S: Save current frame as screenshot
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

sys.path.append(str(Path(__file__).parent.parent))
from models.lstm_classifier import PoomsaeLSTM
from configs.lstm_config import LSTMConfig
from configs.paths import Paths

import pickle
from scipy.spatial import procrustes


class InteractiveVideoTester:
    """Interactive video testing with live display and LSTM logging"""

    def __init__(self, model_path, reference_path=None, device='cuda', debug_log=True):
        """Initialize tester"""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Debug logging flag
        self.debug_log = debug_log
        self.prediction_log = []  # Store all predictions for CSV export

        # Load model
        self.config = LSTMConfig()
        self.model = PoomsaeLSTM(self.config)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f"✓ Model loaded")

        # Initialize RTMPose
        print("Initializing RTMPose...")
        try:
            from rtmlib import BodyWithFeet
            self.pose_estimator = BodyWithFeet(
                to_openpose=False,
                mode='balanced',
                backend='onnxruntime',
                device='cuda' if device == 'cuda' else 'cpu'
            )
            print("✓ RTMPose initialized")
        except ImportError:
            print("ERROR: rtmlib not installed")
            raise

        # Movement names
        self.movement_names = [
            "01. 준비자세", "02. 아래막기", "03. 몸통반대지르기", "04. 아래막기",
            "05. 몸통반대지르기", "06. 아래막기", "07. 몸통바로지르기", "08. 몸통안막기",
            "09. 몸통바로지르기", "10. 몸통안막기", "11. 몸통바로지르기", "12. 아래막기",
            "13. 몸통바로지르기", "14. 올려막기", "15. 앞차고몸통반대지르기", "16. 뒤로돌아올려막기",
            "17. 앞차고몸통반대지르기", "18. 아래막기", "19. 몸통지르기", "20. 준비자세",
        ]

        # Sliding window buffer
        self.window_size = self.config.SEQUENCE_LENGTH
        self.keypoint_buffer = deque(maxlen=self.window_size)

        # Prediction smoothing
        self.prediction_history = deque(maxlen=5)

        # Movement tracking for start/end frames
        self.previous_movement = None
        self.movement_segments = []
        self.current_segment_start = None
        self.frame_confidences = []

        self.total_movements_detected = 0
        self.sequence_complete = False
        self.expected_next_movement = 0

        # Normalization
        self.load_normalization_stats()

        # Load reference for comparison (optional)
        self.reference = None
        self.comparison_scores = {}
        if reference_path and Path(reference_path).exists():
            with open(reference_path, 'rb') as f:
                self.reference = pickle.load(f)
            print(f"✓ Loaded reference: {self.reference['video_name']}")
        else:
            if reference_path:
                print(f"⚠️  Reference not found: {reference_path}")
            print(f"  Running without comparison")

        self.movement_keypoints_buffer = []

    def calculate_pose_similarity(self, pose1, pose2):
        """Calculate pose similarity using Procrustes distance"""
        coords1 = pose1[:, :2]
        coords2 = pose2[:, :2]

        valid_mask = (np.linalg.norm(coords1, axis=1) > 1e-6) & \
                     (np.linalg.norm(coords2, axis=1) > 1e-6)

        if valid_mask.sum() < 5:
            return 0.0

        coords1_valid = coords1[valid_mask]
        coords2_valid = coords2[valid_mask]

        try:
            _, _, disparity = procrustes(coords1_valid, coords2_valid)
            similarity = max(0, 100 * (1 - disparity))
            return similarity
        except:
            return 0.

    def compare_movement_with_reference(self, movement_id, movement_keypoints, duration):
        """Compare completed movement with reference"""
        if self.reference is None:
            return None

        ref_movement = None
        for mov in self.reference['movements']:
            if mov['movement_number'] == movement_id + 1:
                ref_movement = mov
                break

        if ref_movement is None:
            return None

        movement_keypoints = np.array(movement_keypoints)
        ref_keypoints = ref_movement['all_keypoints']

        ref_duration = ref_movement['duration']
        duration_ratio = min(duration, ref_duration) / max(duration, ref_duration)
        temporal_score = duration_ratio * 100

        pose_scores = []
        n_frames = len(movement_keypoints)
        if n_frames > 0:
            start_sim = self.calculate_pose_similarity(movement_keypoints[0], ref_keypoints[0])
            pose_scores.append(start_sim)

            end_sim = self.calculate_pose_similarity(movement_keypoints[-1], ref_keypoints[-1])
            pose_scores.append(end_sim)

            if n_frames >= 3:
                mid_idx = n_frames // 2
                ref_mid_idx = len(ref_keypoints) // 2
                middle_sim = self.calculate_pose_similarity(
                    movement_keypoints[mid_idx], ref_keypoints[ref_mid_idx]
                )
                pose_scores.append(middle_sim)

        avg_pose_score = np.mean(pose_scores) if pose_scores else 0.0
        overall_score = temporal_score * 0.3 + avg_pose_score * 0.7

        return {
            'overall_score': overall_score,
            'temporal_score': temporal_score,
            'pose_score': avg_pose_score,
            'duration_diff': duration - ref_duration,
            'duration_ratio': duration_ratio
        }

    def normalize_keypoints(self, keypoints):
        """Hip-centered + height-normalized"""
        coords = keypoints[:, :2].copy()
        confidence = keypoints[:, 2:3].copy()

        if np.all(coords == 0):
            return keypoints

        hip = coords[19:20, :]
        centered = coords - hip

        head = coords[0, :]
        feet = (coords[15, :] + coords[16, :]) / 2
        height = np.linalg.norm(head - feet)

        if height < 1e-3:
            height = np.linalg.norm(coords[5, :] - coords[6, :]) * 3
        if height < 1e-3:
            height = 1.0

        normalized = centered / height
        return np.concatenate([normalized, confidence], axis=1)

    def load_normalization_stats(self):
        """Load mean/std from training data"""
        import pickle
        stats_file = Paths.CHECKPOINTS_DIR / 'normalization_stats.pkl'

        if stats_file.exists():
            with open(stats_file, 'rb') as f:
                stats = pickle.load(f)
                self.mean = stats['mean']
                self.std = stats['std']
            print(f"✓ Loaded normalization stats")
        else:
            print(f"⚠️  Stats file not found, computing from training data...")
            self.compute_normalization_stats()

    def compute_normalization_stats(self):
        """Compute stats from training data"""
        import pickle

        project_root = Path(__file__).parent.parent
        data_path = project_root / 'data' / 'processed' / 'windows'

        if not data_path.exists():
            print(f"❌ Training data not found at {data_path}")
            self.mean = np.zeros(78)
            self.std = np.ones(78)
            return

        files = list(data_path.glob('*.npz'))

        if not files:
            print("❌ No .npz files found!")
            self.mean = np.zeros(78)
            self.std = np.ones(78)
            return

        print(f"Loading {len(files)} training files...")
        all_data = []
        for f in files:
            data = np.load(f)
            X = data['X']
            all_data.append(X.reshape(-1, 78))

        all_data = np.concatenate(all_data, axis=0)
        self.mean = all_data.mean(axis=0)
        self.std = all_data.std(axis=0)
        self.std[self.std < 1e-6] = 1.0

        stats = {'mean': self.mean, 'std': self.std}
        stats_file = Paths.CHECKPOINTS_DIR / 'normalization_stats.pkl'
        stats_file.parent.mkdir(parents=True, exist_ok=True)

        with open(stats_file, 'wb') as f:
            pickle.dump(stats, f)

        print(f"✓ Computed and saved stats to {stats_file}")

    def extract_keypoints(self, frame):
        """Extract keypoints from frame"""
        keypoints, scores = self.pose_estimator(frame)

        if len(keypoints) > 0:
            person_kp = keypoints[0]
            person_scores = scores[0]
            kp_with_conf = np.concatenate([
                person_kp,
                person_scores.reshape(-1, 1)
            ], axis=1)

            raw_keypoints = kp_with_conf.copy()
            kp_normalized = self.normalize_keypoints(kp_with_conf)

            return kp_normalized, raw_keypoints
        else:
            return np.zeros((26, 3)), np.zeros((26, 3))

    def predict(self):
        """Predict from current buffer - returns all probabilities"""
        if len(self.keypoint_buffer) < self.window_size:
            return None, 0.0, np.zeros(20)

        window = np.array(list(self.keypoint_buffer))
        window_flat = window.reshape(window.shape[0], -1)
        window_normalized = (window_flat - self.mean) / self.std

        x = torch.FloatTensor(window_normalized).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(x)
            probabilities = torch.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, 1)

        return prediction.item(), confidence.item(), probabilities.cpu().numpy()[0]

    def get_smoothed_prediction(self, prediction):
        """Smooth predictions"""
        self.prediction_history.append(prediction)

        if len(self.prediction_history) >= 3:
            counts = {}
            for p in self.prediction_history:
                counts[p] = counts.get(p, 0) + 1
            return max(counts, key=counts.get)
        return prediction

    def log_prediction(self, frame_num, fps, raw_prediction, smoothed_prediction,
                       confidence, probabilities, validated_movement, reason=""):
        """Log LSTM prediction for debugging"""

        # Get top 5 predictions
        top5_indices = np.argsort(probabilities)[-5:][::-1]

        log_entry = {
            'frame': frame_num,
            'time': frame_num / fps,
            'raw_prediction': raw_prediction + 1 if raw_prediction is not None else None,
            'smoothed_prediction': smoothed_prediction + 1 if smoothed_prediction is not None else None,
            'validated_movement': validated_movement + 1 if validated_movement is not None else None,
            'top1_conf': probabilities[top5_indices[0]] * 100 if len(probabilities) > 0 else 0,
            'expected_next': self.expected_next_movement + 1,
            'current_movement': self.previous_movement + 1 if self.previous_movement is not None else None,
            'reason': reason,
            'top5': [(idx + 1, probabilities[idx] * 100) for idx in top5_indices],
            # Specific movements of interest
            'mov18_conf': probabilities[17] * 100,  # Movement 18
            'mov19_conf': probabilities[18] * 100,  # Movement 19
            'mov20_conf': probabilities[19] * 100,  # Movement 20
        }

        self.prediction_log.append(log_entry)

        # Print debug info for frames in focus region (around movement 18-20 transition)
        # You can adjust these frame numbers based on your video
        if self.debug_log:
            # Always print when movement changes
            is_transition = (validated_movement is not None and
                             validated_movement != self.previous_movement)

            # Print every 30 frames or during transitions or in focus region
            in_focus = (self.previous_movement is not None and
                        self.previous_movement >= 16)  # After movement 17

            if is_transition or (in_focus and frame_num % 10 == 0):
                top5_str = " | ".join([f"M{idx + 1}:{prob:.1f}%"
                                       for idx, prob in zip(top5_indices,
                                                            [probabilities[i] * 100 for i in top5_indices])])

                print(f"[Frame {frame_num:5d} | {frame_num / fps:6.2f}s] "
                      f"Raw: M{raw_prediction + 1 if raw_prediction else '?':2} → "
                      f"Smooth: M{smoothed_prediction + 1 if smoothed_prediction else '?':2} → "
                      f"Valid: M{validated_movement + 1 if validated_movement is not None else '?':2} | "
                      f"Expected: M{self.expected_next_movement + 1} | "
                      f"M18:{probabilities[17] * 100:5.1f}% M19:{probabilities[18] * 100:5.1f}% M20:{probabilities[19] * 100:5.1f}%")

                if reason:
                    print(f"         ↳ {reason}")

    def validate_movement_sequence(self, predicted_movement, current_time, total_duration):
        """Validate movement with temporal context"""

        if self.sequence_complete:
            return None, "Sequence complete"

        # === FIRST MOVEMENT (Movement 1) ===
        if self.previous_movement is None:
            if current_time < 15:
                if predicted_movement == 0:
                    return 0, "First movement accepted"
                elif predicted_movement == 19:
                    return 0, "M20 forced to M1 (same pose at start)"
                elif current_time > 3:
                    return 0, "Forced to M1 after 3s"
                else:
                    return None, "Still buffering for M1"
            else:
                return 0, "Forced to M1 after 15s"

        # Allow repeating current movement
        if predicted_movement == self.previous_movement:
            return None, "Same movement continuing"

        # Expected next movement
        expected_next = self.previous_movement + 1

        # Accept expected next movement
        if predicted_movement == expected_next:
            return predicted_movement, f"Expected M{expected_next + 1} accepted"

        # Special case: Movement 20 confusion
        if predicted_movement == 19:  # Movement 20
            if self.previous_movement >= 18 or current_time > (total_duration - 20):
                return predicted_movement, "M20 accepted (near end)"
            else:
                return None, f"M20 rejected (expected M{expected_next + 1}, current M{self.previous_movement + 1})"

        # Special case: Movement 1 after start
        if predicted_movement == 0:
            if self.previous_movement == 19:
                return predicted_movement, "M1 accepted (restart after M20)"
            else:
                return None, f"M1 rejected (not after M20)"

        # Allow skip by 1
        if predicted_movement == expected_next + 1 and predicted_movement < 19:
            return predicted_movement, f"Skip allowed: M{expected_next + 1} → M{predicted_movement + 1}"

        # Reject everything else
        return None, f"Out of sequence: predicted M{predicted_movement + 1}, expected M{expected_next + 1}"

    def draw_skeleton(self, frame, keypoints):
        """Draw skeleton with color-coded body parts"""
        connection_groups = {
            "head": {"connections": [(0, 18), (17, 18), (0, 1), (0, 2), (1, 3), (2, 4)], "color": (0, 255, 255)},
            "torso": {"connections": [(18, 5), (18, 6), (5, 6), (5, 11), (6, 12), (11, 12), (18, 19)],
                      "color": (0, 255, 0)},
            "left_arm": {"connections": [(5, 7), (7, 9)], "color": (255, 0, 0)},
            "right_arm": {"connections": [(6, 8), (8, 10)], "color": (0, 0, 255)},

            "left_leg": {"connections": [(11, 13), (13, 15), (15, 20), (15, 22), (15, 24), (20, 22)],
                         "color": (255, 255, 0)},
            "right_leg": {"connections": [(12, 14), (14, 16), (16, 21), (16, 23), (16, 25), (21, 23)],
                          "color": (255, 0, 255)},
        }

        for group_name, group_data in connection_groups.items():
            color = group_data["color"]
            for start, end in group_data["connections"]:
                if start >= len(keypoints) or end >= len(keypoints):
                    continue
                if keypoints[start][2] < 0.3 or keypoints[end][2] < 0.3:
                    continue
                if keypoints[start][0] <= 0 or keypoints[end][0] <= 0:
                    continue

                pt1 = (int(keypoints[start][0]), int(keypoints[start][1]))
                pt2 = (int(keypoints[end][0]), int(keypoints[end][1]))
                cv2.line(frame, pt1, pt2, color, 2)

        for kp in keypoints:
            if kp[2] < 0.3 or kp[0] <= 0:
                continue
            pt = (int(kp[0]), int(kp[1]))
            cv2.circle(frame, pt, 4, (0, 0, 255), -1)

        return frame

    def draw_info(self, frame, movement_id, confidence, probabilities, frame_num, total_frames, fps):
        """Draw prediction info with Korean support"""
        height, width = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 260), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        if movement_id is not None:
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame_pil)

            try:
                font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 36)
                font_small = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 26)
            except:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 36)
                    font_small = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 26)
                except:
                    font = ImageFont.load_default()
                    font_small = ImageFont.load_default()

            movement_name = self.movement_names[movement_id]
            text = f"Movement: {movement_name} {confidence * 100:.1f}%"
            draw.text((10, 10), text, font=font, fill=(0, 255, 0))

            if movement_id in self.comparison_scores:
                score = self.comparison_scores[movement_id]
                score_text = f"Score: {score['overall_score']:.1f}/100"
                if score['overall_score'] >= 80:
                    score_color = (0, 255, 0)
                elif score['overall_score'] >= 60:
                    score_color = (255, 255, 0)
                else:
                    score_color = (255, 0, 0)
                draw.text((10, 55), score_text, font=font_small, fill=score_color)

            # Show M18, M19, M20 confidence (debug info)
            debug_text = f"M18:{probabilities[17] * 100:.1f}% M19:{probabilities[18] * 100:.1f}% M20:{probabilities[19] * 100:.1f}%"
            draw.text((10, 90), debug_text, font=font_small, fill=(255, 255, 0))

            top3_indices = np.argsort(probabilities)[-3:][::-1]
            y = 130
            for idx in top3_indices:
                prob = probabilities[idx]
                name = self.movement_names[idx]
                text = f"{name}: {prob * 100:.1f}%"
                draw.text((10, y), text, font=font_small, fill=(255, 255, 255))
                y += 30

            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        else:
            cv2.putText(frame, "Buffering...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        buffer_text = f"Buffer: {len(self.keypoint_buffer)}/{self.window_size}"
        cv2.putText(frame, buffer_text, (10, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        progress = frame_num / total_frames
        bar_width = width - 20
        bar_x, bar_y = 10, height - 25
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (50, 50, 50), -1)
        progress_width = int(bar_width * progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + 20), (0, 255, 0), -1)

        time_text = f"{frame_num / fps:.1f}s / {total_frames / fps:.1f}s"
        cv2.putText(frame, time_text, (bar_x + bar_width - 100, bar_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return frame

    def save_prediction_log(self, video_path):
        """Save prediction log to CSV"""
        if not self.prediction_log:
            print("No predictions to save")
            return

        output_dir = Path('debug_logs')
        output_dir.mkdir(parents=True, exist_ok=True)

        video_name = Path(video_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = output_dir / f"{video_name}_lstm_predictions_{timestamp}.csv"

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'Frame', 'Time(s)',
                'Raw_Pred', 'Smoothed_Pred', 'Validated',
                'Expected_Next', 'Current_Mov',
                'Top1_Conf%', 'M18_Conf%', 'M19_Conf%', 'M20_Conf%',
                'Top1', 'Top1%', 'Top2', 'Top2%', 'Top3', 'Top3%',
                'Top4', 'Top4%', 'Top5', 'Top5%',
                'Reason'
            ])

            for entry in self.prediction_log:
                top5 = entry['top5']
                writer.writerow([
                    entry['frame'],
                    f"{entry['time']:.2f}",
                    entry['raw_prediction'],
                    entry['smoothed_prediction'],
                    entry['validated_movement'],
                    entry['expected_next'],
                    entry['current_movement'],
                    f"{entry['top1_conf']:.1f}",
                    f"{entry['mov18_conf']:.1f}",
                    f"{entry['mov19_conf']:.1f}",
                    f"{entry['mov20_conf']:.1f}",
                    top5[0][0], f"{top5[0][1]:.1f}",
                    top5[1][0], f"{top5[1][1]:.1f}",
                    top5[2][0], f"{top5[2][1]:.1f}",
                    top5[3][0], f"{top5[3][1]:.1f}",
                    top5[4][0], f"{top5[4][1]:.1f}",
                    entry['reason']
                ])

        print(f"\n📊 Prediction log saved: {csv_path}")
        print(f"   Total entries: {len(self.prediction_log)}")

        # Print summary for Movement 19
        mov19_predictions = [e for e in self.prediction_log
                             if e['raw_prediction'] == 19 or e['smoothed_prediction'] == 19]
        print(f"\n📌 Movement 19 Analysis:")
        print(f"   Times M19 was raw prediction: {sum(1 for e in self.prediction_log if e['raw_prediction'] == 19)}")
        print(
            f"   Times M19 was smoothed prediction: {sum(1 for e in self.prediction_log if e['smoothed_prediction'] == 19)}")
        print(f"   Times M19 was validated: {sum(1 for e in self.prediction_log if e['validated_movement'] == 19)}")

        # Find max M19 confidence
        max_m19_conf = max(e['mov19_conf'] for e in self.prediction_log)
        max_m19_frame = next(e['frame'] for e in self.prediction_log if e['mov19_conf'] == max_m19_conf)
        print(f"   Max M19 confidence: {max_m19_conf:.1f}% at frame {max_m19_frame}")

        return csv_path

    def test_video_interactive(self, video_path, save_video=False, output_path='output_video.mp4'):
        """Test video with live display"""
        video_path = Path(video_path)

        if not video_path.exists():
            print(f"❌ Video not found: {video_path}")
            return

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"❌ Cannot open video")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"💾 Will save video to: {output_path}")

        print(f"\n{'=' * 70}")
        print(f"INTERACTIVE VIDEO TEST (DEBUG MODE): {video_path.name}")
        print(f"{'=' * 70}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps:.2f}")
        print(f"Duration: {total_frames / fps:.2f}s")
        print(f"\nControls:")
        print(f"  SPACE: Pause/Resume")
        print(f"  Q or ESC: Quit")
        print(f"  S: Screenshot")
        print(f"{'=' * 70}")
        print(f"\n🔍 DEBUG: Logging all LSTM predictions...")
        print(f"{'=' * 70}\n")

        frame_num = 0
        paused = False
        screenshot_count = 0

        cv2.namedWindow('Video Test', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Video Test', 1280, 720)

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_num += 1

                kp_normalized, kp_raw = self.extract_keypoints(frame)
                self.keypoint_buffer.append(kp_normalized)
                frame = self.draw_skeleton(frame, kp_raw)

                raw_prediction, confidence, probabilities = self.predict()

                if raw_prediction is not None:
                    smoothed_prediction = self.get_smoothed_prediction(raw_prediction)

                    current_time = frame_num / fps
                    total_duration = total_frames / fps
                    validated_movement, reason = self.validate_movement_sequence(
                        smoothed_prediction, current_time, total_duration
                    )

                    # Log prediction
                    self.log_prediction(
                        frame_num, fps, raw_prediction, smoothed_prediction,
                        confidence, probabilities, validated_movement, reason
                    )

                    if validated_movement is not None:
                        movement_id = validated_movement
                        self.movement_keypoints_buffer.append(kp_normalized)

                        if movement_id != self.previous_movement and not self.sequence_complete:
                            # End previous segment
                            if self.previous_movement is not None and self.current_segment_start is not None:
                                avg_confidence = np.mean(self.frame_confidences) if self.frame_confidences else 0.0

                                segment = {
                                    'movement_number': self.previous_movement + 1,
                                    'movement_name': self.movement_names[self.previous_movement],
                                    'start_frame': self.current_segment_start,
                                    'end_frame': frame_num - 1,
                                    'start_time': self.current_segment_start / fps,
                                    'end_time': (frame_num - 1) / fps,
                                    'duration': ((frame_num - 1) - self.current_segment_start) / fps,
                                    'avg_confidence': avg_confidence
                                }
                                self.movement_segments.append(segment)

                                comparison = self.compare_movement_with_reference(
                                    self.previous_movement,
                                    self.movement_keypoints_buffer,
                                    segment['duration']
                                )

                                if comparison:
                                    self.comparison_scores[self.previous_movement] = comparison
                                    print(f"  → Ended: {segment['movement_name']} "
                                          f"(frames {segment['start_frame']}-{segment['end_frame']}, "
                                          f"duration: {segment['duration']:.2f}s, conf: {avg_confidence * 100:.1f}%)")
                                    print(f"  📊 Score: {comparison['overall_score']:.1f}/100 "
                                          f"(Temporal: {comparison['temporal_score']:.1f}, "
                                          f"Pose: {comparison['pose_score']:.1f})")

                                self.movement_keypoints_buffer = []

                            # Start new segment
                            self.current_segment_start = frame_num
                            self.previous_movement = movement_id
                            self.frame_confidences = [confidence]
                            self.total_movements_detected += 1
                            self.expected_next_movement = movement_id + 1

                            self.movement_keypoints_buffer = [kp_normalized]

                            print(f"\n▶ Movement {movement_id + 1}: {self.movement_names[movement_id]} "
                                  f"started at frame {frame_num} ({frame_num / fps:.2f}s) "
                                  f"[{self.total_movements_detected}/20]")

                            if movement_id == 19:
                                self.sequence_complete = True
                                print(f"✓ SEQUENCE COMPLETE! All 20 movements detected.")

                        elif self.frame_confidences is not None:
                            self.frame_confidences.append(confidence)
                    else:
                        movement_id = self.previous_movement if self.previous_movement is not None else 0
                else:
                    movement_id = None
                    # Log even when no prediction
                    self.log_prediction(frame_num, fps, None, None, 0.0, np.zeros(20), None, "Buffering")

                frame = self.draw_info(frame, movement_id, confidence if raw_prediction else 0.0,
                                       probabilities if raw_prediction is not None else np.zeros(20),
                                       frame_num, total_frames, fps)

                if video_writer is not None:
                    video_writer.write(frame)

                current_frame = frame.copy()
            else:
                frame = current_frame.copy()
                cv2.putText(frame, "PAUSED", (width // 2 - 100, height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)

            cv2.imshow('Video Test', frame)

            key = cv2.waitKey(int(1000 / fps) if not paused else 1) & 0xFF

            if key == ord('q') or key == 27:
                print("\nStopped by user")
                break
            elif key == ord(' '):
                paused = not paused
                if paused:
                    print(f"Paused at {frame_num / fps:.2f}s")
                else:
                    print("Resumed")
            elif key == ord('s'):
                screenshot_count += 1
                screenshot_path = f"screenshot_{screenshot_count:03d}.png"
                cv2.imwrite(screenshot_path, frame)
                print(f"Screenshot saved: {screenshot_path}")

        cap.release()
        cv2.destroyAllWindows()

        # Save last segment
        if self.previous_movement is not None and self.current_segment_start is not None:
            avg_confidence = np.mean(self.frame_confidences) if self.frame_confidences else 0.0

            segment = {
                'movement_number': self.previous_movement + 1,
                'movement_name': self.movement_names[self.previous_movement],
                'start_frame': self.current_segment_start,
                'end_frame': frame_num,
                'start_time': self.current_segment_start / fps,
                'end_time': frame_num / fps,
                'duration': (frame_num - self.current_segment_start) / fps,
                'avg_confidence': avg_confidence
            }
            self.movement_segments.append(segment)
            print(f"  → Detected: {segment['movement_name']} "
                  f"(frames {segment['start_frame']}-{segment['end_frame']}, "
                  f"{segment['duration']:.2f}s, conf: {avg_confidence * 100:.1f}%)")

        # Save movement segments
        self.save_movement_segments(video_path, fps)

        # Save prediction log
        self.save_prediction_log(video_path)

        if video_writer is not None:
            video_writer.release()
            print(f"✅ Video saved: {output_path}")

        print(f"\n{'=' * 70}")
        print(f"VIDEO TEST COMPLETE")
        print(f"{'=' * 70}")
        print(f"Processed {frame_num}/{total_frames} frames")
        print(f"Duration: {frame_num / fps:.2f}s")
        print(f"{'=' * 70}\n")

    def save_movement_segments(self, video_path, fps):
        """Save detected movement segments to CSV"""
        import pandas as pd

        if not self.movement_segments:
            print("\n⚠️  No movement segments detected!")
            return

        output_dir = Path('results/movement_segments')
        output_dir.mkdir(parents=True, exist_ok=True)

        video_name = Path(video_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = output_dir / f"{video_name}_segments_{timestamp}.csv"

        for seg in self.movement_segments:
            mov_id = seg['movement_number'] - 1
            if mov_id in self.comparison_scores:
                score = self.comparison_scores[mov_id]
                seg['comparison_score'] = score['overall_score']
                seg['temporal_score'] = score['temporal_score']
                seg['pose_score'] = score['pose_score']
                seg['duration_diff'] = score['duration_diff']
            else:
                seg['comparison_score'] = None
                seg['temporal_score'] = None
                seg['pose_score'] = None
                seg['duration_diff'] = None

        df = pd.DataFrame(self.movement_segments)
        df = df[['movement_number', 'movement_name', 'start_frame', 'end_frame',
                 'start_time', 'end_time', 'duration', 'avg_confidence']]
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        print(f"\n{'=' * 70}")
        print(f"MOVEMENT SEGMENTS SAVED")
        print(f"{'=' * 70}")
        print(f"File: {csv_path}")
        print(f"Total movements detected: {len(self.movement_segments)}/20")

        movement_numbers = [seg['movement_number'] for seg in self.movement_segments]
        expected_sequence = list(range(1, 21))

        if movement_numbers == expected_sequence:
            print(f"✅ PERFECT SEQUENCE: All 20 movements in correct order!")
        else:
            print(f"⚠️  INCOMPLETE/INCORRECT SEQUENCE:")
            missing = set(expected_sequence) - set(movement_numbers)
            if missing:
                print(f"   Missing movements: {sorted(missing)}")
            duplicates = [m for m in movement_numbers if movement_numbers.count(m) > 1]
            if duplicates:
                print(f"   Duplicate movements: {sorted(set(duplicates))}")

        print(f"\nMovement Summary:")
        print(f"{'No.':<5} {'Movement':<30} {'Start':<10} {'End':<10} {'Duration':<10} {'Conf%':<7}")
        print(f"{'-' * 75}")
        for seg in self.movement_segments:
            print(f"{seg['movement_number']:<5} {seg['movement_name']:<30} "
                  f"{seg['start_frame']:<10} {seg['end_frame']:<10} "
                  f"{seg['duration']:<10.2f} {seg['avg_confidence'] * 100:<7.1f}")
        print(f"{'=' * 70}\n")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Interactive video testing with LSTM debug logging')
    parser.add_argument('--video', type=str, required=True, help='Path to video')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device')
    parser.add_argument('--save-video', action='store_true', help='Save annotated video')
    parser.add_argument('--output', type=str, default='output_debug.mp4', help='Output video path')
    parser.add_argument('--reference', type=str, default=None, help='Path to reference file')
    parser.add_argument('--no-debug', action='store_true', help='Disable console debug output')

    args = parser.parse_args()

    model_path = Paths.CHECKPOINTS_DIR / 'lstm_taegeuk1_best.pth'

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return

    tester = InteractiveVideoTester(
        model_path=model_path,
        reference_path=args.reference,
        device=args.device,
        debug_log=not args.no_debug
    )

    tester.test_video_interactive(
        video_path=args.video,
        save_video=args.save_video,
        output_path=args.output
    )


if __name__ == "__main__":
    main()

# Usage:
# python test_on_video_debug.py --video "path/to/video.mp4" --save-video --output debug_output.mp4
