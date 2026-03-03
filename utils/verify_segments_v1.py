"""
Visual Segment Verification Tool

Allows you to verify detected movement segments by:
1. Jumping to start/end frames
2. Seeing segment boundaries marked on video
3. Manual review with keyboard controls

Usage:
    python verify_segments_v2.py --video P001.mp4 --segments P001_segments.csv
"""

import cv2
import pandas as pd
import argparse
from pathlib import Path


class SegmentVerifier:
    """Verify movement segments visually"""

    def __init__(self, video_path, segments_csv):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(str(video_path))

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Load segments
        self.segments = pd.read_csv(segments_csv)
        self.current_segment_idx = 0

        print(f"\n{'=' * 70}")
        print(f"SEGMENT VERIFICATION TOOL")
        print(f"{'=' * 70}")
        print(f"Video: {Path(video_path).name}")
        print(f"Segments: {len(self.segments)}")
        print(f"FPS: {self.fps:.2f}")
        print(f"Duration: {self.total_frames / self.fps:.2f}s")
        print(f"{'=' * 70}\n")

    def jump_to_frame(self, frame_num):
        """Jump to specific frame"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    def draw_segment_info(self, frame, frame_num):
        """Draw current segment information on frame"""

        # Find which segment current frame belongs to
        current_segment = None
        for idx, seg in self.segments.iterrows():
            if seg['start_frame'] <= frame_num <= seg['end_frame']:
                current_segment = seg
                break

        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, 200), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        if current_segment is not None:
            # Movement info
            text1 = f"Movement {current_segment['movement_number']}: {current_segment['movement_name']}"
            text2 = f"Segment: Frame {current_segment['start_frame']}-{current_segment['end_frame']}"
            text3 = f"Duration: {current_segment['duration']:.2f}s | Confidence: {current_segment['avg_confidence'] * 100:.1f}%"
            text4 = f"Current: Frame {frame_num} ({frame_num / self.fps:.2f}s)"

            # Color based on position in segment
            progress = (frame_num - current_segment['start_frame']) / (
                        current_segment['end_frame'] - current_segment['start_frame'])
            if progress < 0.2:
                color = (0, 255, 255)  # Yellow = start
            elif progress > 0.8:
                color = (0, 165, 255)  # Orange = end
            else:
                color = (0, 255, 0)  # Green = middle

            cv2.putText(frame, text1, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, text2, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, text3, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, text4, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Progress bar for current segment
            bar_width = self.width - 40
            bar_x, bar_y = 20, 160
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), (50, 50, 50), -1)

            progress_width = int(bar_width * progress)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + 20), color, -1)

            # Mark start and end
            cv2.line(frame, (bar_x, bar_y), (bar_x, bar_y + 20), (0, 255, 255), 3)  # Start
            cv2.line(frame, (bar_x + bar_width, bar_y), (bar_x + bar_width, bar_y + 20), (0, 165, 255), 3)  # End

        else:
            text = f"Frame {frame_num} ({frame_num / self.fps:.2f}s) - NO SEGMENT"
            cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Controls help
        help_y = self.height - 120
        cv2.putText(frame, "Controls:", (10, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, "N: Next segment | P: Previous segment", (10, help_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1)
        cv2.putText(frame, "S: Jump to segment start | E: Jump to segment end", (10, help_y + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "SPACE: Pause | LEFT/RIGHT: -/+5 frames", (10, help_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1)
        cv2.putText(frame, "UP/DOWN: -/+1 second | Q: Quit", (10, help_y + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1)

        return frame

    def verify(self):
        """Main verification loop"""

        cv2.namedWindow('Segment Verification', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Segment Verification', 1280, 720)

        paused = False
        current_frame_num = 0

        # Start at first segment
        if len(self.segments) > 0:
            current_frame_num = int(self.segments.iloc[0]['start_frame'])
            self.jump_to_frame(current_frame_num)

        print("\nStarting verification...")
        print(f"Review each segment and verify if start/end frames are correct.\n")

        while True:
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("End of video reached")
                    break

                current_frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                current_frame = frame.copy()
            else:
                frame = current_frame.copy()

            # Draw info
            frame = self.draw_segment_info(frame, current_frame_num)

            # Display
            cv2.imshow('Segment Verification', frame)

            # Handle keys
            key = cv2.waitKey(int(1000 / self.fps) if not paused else 1) & 0xFF

            if key == ord('q') or key == 27:  # Quit
                break

            elif key == ord(' '):  # Pause
                paused = not paused

            elif key == ord('n'):  # Next segment
                self.current_segment_idx = min(self.current_segment_idx + 1, len(self.segments) - 1)
                seg = self.segments.iloc[self.current_segment_idx]
                current_frame_num = int(seg['start_frame'])
                self.jump_to_frame(current_frame_num)
                paused = True
                print(f"→ Segment {self.current_segment_idx + 1}: {seg['movement_name']}")

            elif key == ord('p'):  # Previous segment
                self.current_segment_idx = max(self.current_segment_idx - 1, 0)
                seg = self.segments.iloc[self.current_segment_idx]
                current_frame_num = int(seg['start_frame'])
                self.jump_to_frame(current_frame_num)
                paused = True
                print(f"← Segment {self.current_segment_idx + 1}: {seg['movement_name']}")

            elif key == ord('s'):  # Jump to current segment start
                for idx, seg in self.segments.iterrows():
                    if seg['start_frame'] <= current_frame_num <= seg['end_frame']:
                        current_frame_num = int(seg['start_frame'])
                        self.jump_to_frame(current_frame_num)
                        paused = True
                        print(f"⏮ Jump to START of Movement {seg['movement_number']}")
                        break

            elif key == ord('e'):  # Jump to current segment end
                for idx, seg in self.segments.iterrows():
                    if seg['start_frame'] <= current_frame_num <= seg['end_frame']:
                        current_frame_num = int(seg['end_frame'])
                        self.jump_to_frame(current_frame_num)
                        paused = True
                        print(f"⏭ Jump to END of Movement {seg['movement_number']}")
                        break

            elif key == 81:  # Left arrow: -5 frames
                current_frame_num = max(0, current_frame_num - 5)
                self.jump_to_frame(current_frame_num)
                paused = True

            elif key == 83:  # Right arrow: +5 frames
                current_frame_num = min(self.total_frames - 1, current_frame_num + 5)
                self.jump_to_frame(current_frame_num)
                paused = True

            elif key == 82:  # Up arrow: -1 second
                current_frame_num = max(0, current_frame_num - int(self.fps))
                self.jump_to_frame(current_frame_num)
                paused = True

            elif key == 84:  # Down arrow: +1 second
                current_frame_num = min(self.total_frames - 1, current_frame_num + int(self.fps))
                self.jump_to_frame(current_frame_num)
                paused = True

        self.cap.release()
        cv2.destroyAllWindows()

        print("\nVerification complete!")


def main():
    parser = argparse.ArgumentParser(description='Verify movement segments')
    parser.add_argument('--video', type=str, required=True, help='Video file path')
    parser.add_argument('--segments', type=str, required=True, help='Segments CSV file')

    args = parser.parse_args()

    verifier = SegmentVerifier(args.video, args.segments)
    verifier.verify()


if __name__ == "__main__":
    main()


 #python verify_segments_v2.py --video "D:\All Docs\All Projects\Pycharm\poomsae_recognition\data\reference\videos\P001.mp4" --segments r"D:\All Docs\All Projects\Pycharm\poomsae_recognition\results\movement_segments\P011_segments_20260121_115122.csv"