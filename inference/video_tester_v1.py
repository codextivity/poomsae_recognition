"""
Interactive Video Testing - With Live Display

Shows video playing with real-time predictions overlaid.

Usage:
    python test_on_video_interactive.py --video path/to/video.mp4

Controls:
    SPACE: Pause/Resume
    Q or ESC: Quit
    S: Save current frame as screenshot
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import sys
from collections import deque
from PIL import Image, ImageDraw, ImageFont
from visualization.skeleton import draw_complete_skeleton
import pandas as pd  # ADD THIS
from datetime import datetime  # ADD THIS

sys.path.append(str(Path(__file__).parent.parent))
from models.lstm_classifier import PoomsaeLSTM
from configs.lstm_config import LSTMConfig
from configs.paths import Paths


class InteractiveVideoTester:
    """Interactive video testing with live display"""

    def __init__(self, model_path, device='cuda'):
        """Initialize tester"""
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
        self.movement_segments = []  # List of all detected movement segments
        self.current_segment_start = None
        self.frame_confidences = []  # Track confidences for current segment

        # ADD THESE NEW LINES ↓↓↓
        self.total_movements_detected = 0
        self.sequence_complete = False  # Flag when 20 movements detected
        self.expected_next_movement = 0  # Start with Movement 1 (index 0)


        # Normalization - load from training data
        self.load_normalization_stats()

    def normalize_keypoints(self, keypoints):
        """Hip-centered + height-normalized"""
        coords = keypoints[:, :2].copy()
        confidence = keypoints[:, 2:3].copy()

        if np.all(coords == 0):
            return keypoints

        # Hip-centering
        hip = coords[19:20, :]
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

        # Fix: Use relative path from project root
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
            X = data['X']  # Shape: (N, 60, 78)
            all_data.append(X.reshape(-1, 78))

        all_data = np.concatenate(all_data, axis=0)
        self.mean = all_data.mean(axis=0)
        self.std = all_data.std(axis=0)
        self.std[self.std < 1e-6] = 1.0

        # Save
        stats = {'mean': self.mean, 'std': self.std}
        stats_file = Paths.CHECKPOINTS_DIR / 'normalization_stats.pkl'
        stats_file.parent.mkdir(parents=True, exist_ok=True)

        with open(stats_file, 'wb') as f:
            pickle.dump(stats, f)

        print(f"✓ Computed and saved stats to {stats_file}")
        print(f"  Mean range: [{self.mean.min():.3f}, {self.mean.max():.3f}]")
        print(f"  Std range: [{self.std.min():.3f}, {self.std.max():.3f}]")

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

            # Store raw keypoints for drawing
            raw_keypoints = kp_with_conf.copy()

            # Normalize for prediction
            kp_normalized = self.normalize_keypoints(kp_with_conf)

            return kp_normalized, raw_keypoints
        else:
            return np.zeros((26, 3)), np.zeros((26, 3))

    def predict(self):
        """Predict from current buffer"""
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

    def validate_movement_sequence(self, predicted_movement, current_time, total_duration):
        """
        Validate movement with temporal context - SIMPLIFIED
        """

        # If sequence complete, ignore all predictions
        if self.sequence_complete:
            return None

        # === FIRST MOVEMENT (Movement 1) ===
        if self.previous_movement is None:
            # Very first movement - waiting for Movement 1
            if current_time < 15:
                # Model predicts Movement 1 → Accept
                if predicted_movement == 0:
                    return 0
                # Model predicts Movement 20 → Force to Movement 1 (same pose!)
                elif predicted_movement == 19:
                    return 0  # Don't print every frame, just force
                # Other movement → Force to Movement 1 after 3 seconds
                elif current_time > 3:
                    return 0  # Don't print every frame, just force
                else:
                    return None  # Still buffering
            else:
                # After 15 seconds, force to Movement 1
                return 0

        # === AFTER MOVEMENT 1 STARTED ===
        # Now previous_movement is not None, Movement 1 has started

        # Allow repeating current movement
        if predicted_movement == self.previous_movement:
            return None  # Don't create new segment, just continue

        # Expected next movement (sequential)
        expected_next = self.previous_movement + 1

        # Accept expected next movement
        if predicted_movement == expected_next:
            return predicted_movement

        # Special case: Movement 20 can be confused with Movement 1
        if predicted_movement == 19:  # Movement 20
            # Only accept if we've seen Movement 19 OR we're near end of video
            if self.previous_movement >= 18 or current_time > (total_duration - 20):
                return predicted_movement
            else:
                return None  # Reject, probably confusion

        # Special case: Movement 1 predicted after start
        if predicted_movement == 0:  # Movement 1
            # Only accept if we just finished Movement 20 (restart)
            if self.previous_movement == 19:
                return predicted_movement
            else:
                return None  # Reject

        # Allow skip by 1 (e.g., 2→4 if 3 missed)
        if predicted_movement == expected_next + 1 and predicted_movement < 19:
            print(f"  ⚠️  Skipped Movement {expected_next + 1}, accepting Movement {predicted_movement + 1}")
            return predicted_movement

        # Reject everything else (out of sequence)
        return None

    def draw_skeleton(self, frame, keypoints):
        """Draw skeleton with CORRECT connections"""

        # Correct connections
        connections = [
            (5, 6), (5, 11), (6, 12), (11, 12),  # Torso
            (5, 7), (7, 9),  # Left arm
            (6, 8), (8, 10),  # Right arm
            (11, 13), (13, 15),  # Left leg
            (12, 14), (14, 16),  # Right leg
        ]

        # Draw lines
        for start, end in connections:
            if start >= len(keypoints) or end >= len(keypoints):
                continue
            if keypoints[start][2] < 0.3 or keypoints[end][2] < 0.3:
                continue
            if keypoints[start][0] <= 0 or keypoints[end][0] <= 0:
                continue

            pt1 = (int(keypoints[start][0]), int(keypoints[start][1]))
            pt2 = (int(keypoints[end][0]), int(keypoints[end][1]))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        # Draw points
        for kp in keypoints:
            if kp[2] < 0.3 or kp[0] <= 0:
                continue
            pt = (int(kp[0]), int(kp[1]))
            cv2.circle(frame, pt, 4, (0, 0, 255), -1)

        return frame

    def draw_info(self, frame, movement_id, confidence, probabilities, frame_num, total_frames, fps):
        """Draw prediction info with Korean support"""
        height, width = frame.shape[:2]

        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 260), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        if movement_id is not None:
            # Convert to PIL for Korean text
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame_pil)

            # Load Korean font (use system font)
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 36)  # Windows
                font_small = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 26)
            except:
                try:
                    font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 36)  # Linux
                    font_small = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 26)
                except:
                    font = ImageFont.load_default()
                    font_small = ImageFont.load_default()

            # Draw Korean text
            movement_name = self.movement_names[movement_id]
            text = f"Movement: {movement_name} {confidence * 100:.1f}%"
            draw.text((10, 10), text, font=font, fill=(0, 255, 0))

            # Confidence
            # conf_text = f"Confidence: {confidence * 100:.1f}%"
            # color = (0, 255, 0) if confidence > 0.7 else (255, 255, 0) if confidence > 0.5 else (255, 0, 0)
            # draw.text((10, 40), conf_text, font=font_small, fill=color)
            # print(f"Movement: {movement_name}, confidence: {confidence * 100:.1f}%")

            top_3 = "---Top 3---"
            draw.text((10, 60), top_3, font=font_small, fill=(255, 255, 255))
            # Top 3
            top3_indices = np.argsort(probabilities)[-3:][::-1]
            y = 90
            for idx in top3_indices:
                prob = probabilities[idx]
                name = self.movement_names[idx]
                text = f"{name}: {prob * 100:.1f}%"
                draw.text((10, y), text, font=font_small, fill=(255, 255, 255))
                y += 30

            # Convert back
            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        else:
            # Buffering (English OK)
            cv2.putText(frame, "Buffering...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Buffer and progress (English OK, keep as is)
        buffer_text = f"Buffer: {len(self.keypoint_buffer)}/{self.window_size}"
        cv2.putText(frame, buffer_text, (10, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Progress bar
        progress = frame_num / total_frames
        bar_width = width - 20
        bar_x, bar_y = 10, height - 25
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (50, 50, 50), -1)
        progress_width = int(bar_width * progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + 20), (0, 255, 0), -1)

        time_text = f"{frame_num / fps:.1f}s / {total_frames / fps:.1f}s"
        cv2.putText(frame, time_text, (bar_x + bar_width - 100, bar_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1)

        return frame

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

        # Initialize video writer if saving
        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"💾 Will save video to: {output_path}")

        print(f"\n{'=' * 70}")

        print(f"\n{'='*70}")
        print(f"INTERACTIVE VIDEO TEST: {video_path.name}")
        print(f"{'='*70}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps:.2f}")
        print(f"Duration: {total_frames/fps:.2f}s")
        print(f"\nControls:")
        print(f"  SPACE: Pause/Resume")
        print(f"  Q or ESC: Quit")
        print(f"  S: Screenshot")
        print(f"{'='*70}\n")

        frame_num = 0
        paused = False
        screenshot_count = 0

        # Create window
        cv2.namedWindow('Video Test', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Video Test', 1280, 720)

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_num += 1

                # Extract keypoints
                kp_normalized, kp_raw = self.extract_keypoints(frame)

                # Add to buffer
                self.keypoint_buffer.append(kp_normalized)

                # Draw skeleton
                frame = self.draw_skeleton(frame, kp_raw)

                # # Predict
                # movement_id, confidence, probabilities = self.predict()
                # if movement_id is not None:
                #     movement_id = self.get_smoothed_prediction(movement_id)
                #
                # # Draw info overlay
                # frame = self.draw_info(frame, movement_id, confidence, probabilities,
                #                        frame_num, total_frames, fps)

                # Predict

                movement_id, confidence, probabilities = self.predict()

                if movement_id is not None:
                    movement_id = self.get_smoothed_prediction(movement_id)

                    # Validate movement follows correct sequence
                    current_time = frame_num / fps
                    total_duration = total_frames / fps
                    validated_movement = self.validate_movement_sequence(
                        movement_id,
                        current_time,
                        total_duration
                    )

                    # Handle validation result
                    if validated_movement is not None:
                        # Valid movement detected
                        movement_id = validated_movement

                        # Check if this is a new movement (not just continuing current)
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
                                print(f"  → Ended: {segment['movement_name']} "
                                      f"(frames {segment['start_frame']}-{segment['end_frame']}, "
                                      f"duration: {segment['duration']:.2f}s, conf: {avg_confidence * 100:.1f}%)")

                            # Start new segment
                            self.current_segment_start = frame_num
                            self.previous_movement = movement_id
                            self.frame_confidences = [confidence]
                            self.total_movements_detected += 1
                            self.expected_next_movement = movement_id + 1

                            print(f"\n▶ Movement {movement_id + 1}: {self.movement_names[movement_id]} "
                                  f"started at frame {frame_num} ({frame_num / fps:.2f}s) "
                                  f"[{self.total_movements_detected}/20]")

                            # Check if sequence complete
                            if movement_id == 19:
                                self.sequence_complete = True
                                print(f"✓ SEQUENCE COMPLETE! All 20 movements detected.")

                        elif self.frame_confidences is not None:
                            # Same movement continuing, update confidence
                            self.frame_confidences.append(confidence)
                    else:
                        # Invalid movement - just display previous
                        movement_id = self.previous_movement if self.previous_movement is not None else 0
                        # Don't track, don't create segment

                # Draw info overlay
                frame = self.draw_info(frame, movement_id, confidence, probabilities,
                                       frame_num, total_frames, fps)

                # Save frame to video if enabled
                if video_writer is not None:
                    video_writer.write(frame)

                # Store frame for pause
                current_frame = frame.copy()
            else:
                # Show paused frame
                frame = current_frame.copy()
                cv2.putText(frame, "PAUSED", (width//2 - 100, height//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)

            # Display
            cv2.imshow('Video Test', frame)

            # Handle keys
            key = cv2.waitKey(int(1000/fps) if not paused else 1) & 0xFF

            if key == ord('q') or key == 27:  # Q or ESC
                print("\nStopped by user")
                break
            elif key == ord(' '):  # SPACE
                paused = not paused
                if paused:
                    print(f"Paused at {frame_num/fps:.2f}s")
                else:
                    print("Resumed")
            elif key == ord('s'):  # S
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

        # Save movement segments to CSV
        self.save_movement_segments(video_path, fps)

        # Close video writer
        if video_writer is not None:
            video_writer.release()
            print(f"✅ Video saved: {output_path}")

        print(f"\n{'='*70}")
        print(f"VIDEO TEST COMPLETE")
        print(f"{'='*70}")
        print(f"Processed {frame_num}/{total_frames} frames")
        print(f"Duration: {frame_num/fps:.2f}s")

        if screenshot_count > 0:
            print(f"Screenshots saved: {screenshot_count}")

        print(f"{'='*70}\n")

    def save_movement_segments(self, video_path, fps):
        """Save detected movement segments to CSV"""
        import pandas as pd
        from datetime import datetime

        if not self.movement_segments:
            print("\n⚠️  No movement segments detected!")
            return

        # Create output directory
        output_dir = Path('results/movement_segments')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create filename
        video_name = Path(video_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = output_dir / f"{video_name}_segments_{timestamp}.csv"

        # Convert to DataFrame
        df = pd.DataFrame(self.movement_segments)

        # Reorder columns
        df = df[['movement_number', 'movement_name', 'start_frame', 'end_frame',
                 'start_time', 'end_time', 'duration', 'avg_confidence']]

        # Save to CSV
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        print(f"\n{'=' * 70}")
        print(f"MOVEMENT SEGMENTS SAVED")
        print(f"{'=' * 70}")
        print(f"File: {csv_path}")
        print(f"Total movements detected: {len(self.movement_segments)}/20")

        # ADD VALIDATION CHECK ↓↓↓
        # Validate sequence completeness
        movement_numbers = [seg['movement_number'] for seg in self.movement_segments]
        expected_sequence = list(range(1, 21))  # [1, 2, 3, ..., 20]

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
            out_of_order = movement_numbers != sorted(movement_numbers)
            if out_of_order:
                print(f"   Out of order: {movement_numbers}")

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
    parser = argparse.ArgumentParser(description='Interactive video testing')
    parser.add_argument('--video', type=str, required=True, help='Path to video')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device')
    parser.add_argument('--save-video', action='store_true',
                        help='Save annotated video')
    parser.add_argument('--output', type=str, default='P001_output_video_32.mp4',
                        help='Output video path')

    args = parser.parse_args()

    # Model path
    model_path = Paths.CHECKPOINTS_DIR / 'lstm_taegeuk1_best.pth'

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return

    # Create tester
    tester = InteractiveVideoTester(model_path=model_path, device=args.device)

    # Test
    tester.test_video_interactive(
    video_path=args.video,
    save_video=args.save_video,
    output_path=args.output)


if __name__ == "__main__":
    main()