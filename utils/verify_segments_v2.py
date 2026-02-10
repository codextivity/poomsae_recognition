"""
Visual Segment Verification Tool - With Korean Font Support

Usage:
    python verify_segments_v2.py --video P001.mp4 --segments P001_segments.csv
"""

import cv2
import pandas as pd
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


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

        # Load Korean font
        try:
            self.font_large = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 32)
            self.font_medium = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 24)
            self.font_small = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 18)
        except:
            try:
                self.font_large = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 32)
                self.font_medium = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 24)
                self.font_small = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 18)
            except:
                print("⚠️  Korean font not found, using default")
                self.font_large = ImageFont.load_default()
                self.font_medium = ImageFont.load_default()
                self.font_small = ImageFont.load_default()

        # Load segments
        self.segments = pd.read_csv(segments_csv)
        self.current_segment_idx = 0

        print(f"\n{'='*70}")
        print(f"SEGMENT VERIFICATION TOOL")
        print(f"{'='*70}")
        print(f"Video: {Path(video_path).name}")
        print(f"Segments: {len(self.segments)}")
        print(f"FPS: {self.fps:.2f}")
        print(f"Duration: {self.total_frames/self.fps:.2f}s")
        print(f"{'='*70}\n")

    def jump_to_frame(self, frame_num):
        """Jump to specific frame"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    def draw_text_pil(self, frame, text, position, font, color):
        """Draw text using PIL for Korean support"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)

        # Draw text
        draw.text(position, text, font=font, fill=color)

        # Convert back to BGR
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return frame

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
        cv2.rectangle(overlay, (0, 0), (self.width, 250), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        if current_segment is not None:
            # Determine color based on position in segment
            progress = (frame_num - current_segment['start_frame']) / max(1, (current_segment['end_frame'] - current_segment['start_frame']))

            if progress < 0.2:
                color_bgr = (0, 255, 255)  # Yellow = start
                color_rgb = (255, 255, 0)
                position_text = "▶ START"
            elif progress > 0.8:
                color_bgr = (0, 165, 255)  # Orange = end
                color_rgb = (255, 165, 0)
                position_text = "◼ END"
            else:
                color_bgr = (0, 255, 0)  # Green = middle
                color_rgb = (0, 255, 0)
                position_text = "● MIDDLE"

            # Draw Korean text using PIL
            text1 = f"Movement {current_segment['movement_number']}: {current_segment['movement_name']} {position_text}"
            frame = self.draw_text_pil(frame, text1, (10, 10), self.font_large, color_rgb)

            # Draw English text using OpenCV (faster)
            text2 = f"Segment: Frame {current_segment['start_frame']}-{current_segment['end_frame']}"
            text3 = f"Duration: {current_segment['duration']:.2f}s | Confidence: {current_segment['avg_confidence']*100:.1f}%"
            text4 = f"Current: Frame {frame_num} ({frame_num/self.fps:.2f}s)"

            cv2.putText(frame, text2, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, text3, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, text4, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Progress bar for current segment
            bar_width = self.width - 40
            bar_x, bar_y = 20, 180
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 30), (50, 50, 50), -1)

            progress_width = int(bar_width * progress)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + 30), color_bgr, -1)

            # Mark start and end
            cv2.line(frame, (bar_x, bar_y-5), (bar_x, bar_y + 35), (0, 255, 255), 4)  # Start marker
            cv2.line(frame, (bar_x + bar_width, bar_y-5), (bar_x + bar_width, bar_y + 35), (0, 165, 255), 4)  # End marker

            # Add labels
            cv2.putText(frame, "START", (bar_x-5, bar_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(frame, "END", (bar_x + bar_width-25, bar_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

        else:
            # No segment at this frame
            text = f"Frame {frame_num} ({frame_num/self.fps:.2f}s) - NO SEGMENT"
            cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Controls help at bottom
        help_y = self.height - 140
        cv2.rectangle(frame, (0, help_y-10), (self.width, self.height), (0, 0, 0), -1)

        cv2.putText(frame, "CONTROLS:", (10, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, "N: Next segment | P: Previous segment | SPACE: Pause/Play",
                   (10, help_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, "S: Segment START | E: Segment END | Q/ESC: Quit",
                   (10, help_y+55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, "LEFT/RIGHT: -/+5 frames | UP/DOWN: -/+1 second",
                   (10, help_y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, "1-9: Jump to segment number | 0: Jump to segment 10",
                   (10, help_y+105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        return frame

    def verify(self, save_video=False, output_path='verification_output.mp4'):
        """Main verification loop"""

        # Initialize video writer if saving
        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
            print(f"💾 Will save annotated video to: {output_path}\n")

        cv2.namedWindow('Segment Verification', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Segment Verification', 1280, 720)

        paused = False
        current_frame_num = 0

        # Start at first segment
        if len(self.segments) > 0:
            current_frame_num = int(self.segments.iloc[0]['start_frame'])
            self.jump_to_frame(current_frame_num)

        print("\n╔═══════════════════════════════════════════════════════════╗")
        print("║  VERIFICATION INSTRUCTIONS                                ║")
        print("╠═══════════════════════════════════════════════════════════╣")
        print("║  1. Press 'N' to go through each segment                  ║")
        print("║  2. Check if movement STARTS at the marked frame          ║")
        print("║  3. Press 'E' to jump to END of current segment           ║")
        print("║  4. Check if movement ENDS at the marked frame            ║")
        print("║  5. Use LEFT/RIGHT arrows for fine adjustments            ║")
        if save_video:
            print("║  6. Video will be saved with annotations                  ║")
        print("╚═══════════════════════════════════════════════════════════╝\n")

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

            # Save frame if enabled (only when not paused)
            if video_writer is not None and not paused:
                video_writer.write(frame)

            # Display
            cv2.imshow('Segment Verification', frame)

            # Handle keys
            key = cv2.waitKey(int(1000 / self.fps) if not paused else 1) & 0xFF

            if key == ord('q') or key == 27:  # Quit
                break

            elif key == ord(' '):  # Pause/Play
                paused = not paused
                status = "⏸ PAUSED" if paused else "▶ PLAYING"
                print(f"{status} at frame {current_frame_num} ({current_frame_num / self.fps:.2f}s)")

            elif key == ord('n'):  # Next segment
                self.current_segment_idx = min(self.current_segment_idx + 1, len(self.segments) - 1)
                seg = self.segments.iloc[self.current_segment_idx]
                current_frame_num = int(seg['start_frame'])
                self.jump_to_frame(current_frame_num)
                paused = True
                print(f"\n→ Movement {seg['movement_number']}: {seg['movement_name']}")
                print(f"  START frame: {seg['start_frame']} ({seg['start_time']:.2f}s)")

            elif key == ord('p'):  # Previous segment
                self.current_segment_idx = max(self.current_segment_idx - 1, 0)
                seg = self.segments.iloc[self.current_segment_idx]
                current_frame_num = int(seg['start_frame'])
                self.jump_to_frame(current_frame_num)
                paused = True
                print(f"\n← Movement {seg['movement_number']}: {seg['movement_name']}")
                print(f"  START frame: {seg['start_frame']} ({seg['start_time']:.2f}s)")

            elif key == ord('s'):  # Jump to current segment start
                for idx, seg in self.segments.iterrows():
                    if seg['start_frame'] <= current_frame_num <= seg['end_frame']:
                        current_frame_num = int(seg['start_frame'])
                        self.jump_to_frame(current_frame_num)
                        paused = True
                        print(f"⏮ START of Movement {seg['movement_number']} (frame {current_frame_num})")
                        break

            elif key == ord('e'):  # Jump to current segment end
                for idx, seg in self.segments.iterrows():
                    if seg['start_frame'] <= current_frame_num <= seg['end_frame']:
                        current_frame_num = int(seg['end_frame'])
                        self.jump_to_frame(current_frame_num)
                        paused = True
                        print(f"⏭ END of Movement {seg['movement_number']} (frame {current_frame_num})")
                        break

            elif key == 81:  # Left arrow: -5 frames
                current_frame_num = max(0, current_frame_num - 5)
                self.jump_to_frame(current_frame_num)
                paused = True
                print(f"⏪ Frame {current_frame_num} (-5 frames)")

            elif key == 83:  # Right arrow: +5 frames
                current_frame_num = min(self.total_frames - 1, current_frame_num + 5)
                self.jump_to_frame(current_frame_num)
                paused = True
                print(f"⏩ Frame {current_frame_num} (+5 frames)")

            elif key == 82:  # Up arrow: -1 second
                current_frame_num = max(0, current_frame_num - int(self.fps))
                self.jump_to_frame(current_frame_num)
                paused = True
                print(f"⏫ Frame {current_frame_num} (-1 second)")

            elif key == 84:  # Down arrow: +1 second
                current_frame_num = min(self.total_frames - 1, current_frame_num + int(self.fps))
                self.jump_to_frame(current_frame_num)
                paused = True
                print(f"⏬ Frame {current_frame_num} (+1 second)")

            # Number keys 1-9 to jump to segments 1-9
            elif ord('1') <= key <= ord('9'):
                seg_num = key - ord('0')
                if seg_num <= len(self.segments):
                    self.current_segment_idx = seg_num - 1
                    seg = self.segments.iloc[self.current_segment_idx]
                    current_frame_num = int(seg['start_frame'])
                    self.jump_to_frame(current_frame_num)
                    paused = True
                    print(f"\n→ Jump to Movement {seg_num}: {seg['movement_name']}")

            # '0' key for segment 10
            elif key == ord('0'):
                if len(self.segments) >= 10:
                    self.current_segment_idx = 9
                    seg = self.segments.iloc[9]
                    current_frame_num = int(seg['start_frame'])
                    self.jump_to_frame(current_frame_num)
                    paused = True
                    print(f"\n→ Jump to Movement 10: {seg['movement_name']}")

        self.cap.release()

        # Close video writer
        if video_writer is not None:
            video_writer.release()
            print(f"\n✅ Annotated video saved: {output_path}")

        cv2.destroyAllWindows()

        print("\n✓ Verification complete!")


def main():
    parser = argparse.ArgumentParser(description='Verify movement segments')
    parser.add_argument('--video', type=str, required=True, help='Video file path')
    parser.add_argument('--segments', type=str, required=True, help='Segments CSV file')
    parser.add_argument('--save-video', action='store_true', help='Save annotated video')
    parser.add_argument('--output', type=str, default='verification_output.mp4', help='Output video path')

    args = parser.parse_args()

    verifier = SegmentVerifier(args.video, args.segments)
    verifier.verify(save_video=args.save_video, output_path=args.output)


if __name__ == "__main__":
    main()
# python verify_segments_v2.py - -video "D:\All Docs\All Projects\Pycharm\poomsae_recognition\data\reference\videos\P001.mp4" - -segments "D:\All Docs\All Projects\Pycharm\poomsae_recognition\results\movement_segments\P011_segments_20260121_115122.csv