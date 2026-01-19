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
from complete_skeleton import draw_complete_skeleton

sys.path.append(str(Path(__file__).parent))
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
        self.window_size = self.config.SEQUENCE_LENGTH  # 60
        self.keypoint_buffer = deque(maxlen=self.window_size)

        # Prediction smoothing
        self.prediction_history = deque(maxlen=5)

        # Normalization
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
        project_root = Path(__file__).parent
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
                font = ImageFont.truetype("malgun.ttf", 36)  # Windows
                font_small = ImageFont.truetype("malgun.ttf", 26)
            except:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 36)  # Linux
                    font_small = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 26)
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

                # Predict
                movement_id, confidence, probabilities = self.predict()
                if movement_id is not None:
                    movement_id = self.get_smoothed_prediction(movement_id)

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