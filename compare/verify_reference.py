"""
Verify Reference Database - Visual Inspection Tool

Creates a visual grid showing key poses (start, middle, end) for each movement.
Helps verify that movement boundaries are correctly detected.

Usage:
    python verify_reference.py --reference master_22class.pkl --video P001.mp4
    python verify_reference.py --reference master_22class.pkl --video P001.mp4 --output verify.png
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import numpy as np
from pathlib import Path
import pickle
import argparse
from PIL import Image, ImageDraw, ImageFont


def extract_frame(video_path, frame_num):
    """Extract a single frame from video"""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def create_verification_grid(reference_path, video_path, output_path=None):
    """Create visual grid of key poses for verification"""

    # Load reference
    with open(reference_path, 'rb') as f:
        ref = pickle.load(f)

    print(f"Loaded reference: {ref['video_name']}")
    print(f"Movements: {len(ref['movements'])}")

    # Settings
    thumb_width = 320
    thumb_height = 240
    cols = 3  # start, middle, end
    rows = len(ref['movements'])

    # Create large canvas
    grid_width = thumb_width * cols + 200  # Extra space for labels
    grid_height = thumb_height * rows + 50  # Header space

    grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 40  # Dark gray background

    # Load font
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 16)
        font_small = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 12)
        font_header = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 20)
    except:
        font = font_small = font_header = ImageFont.load_default()

    # Convert to PIL for text
    pil_grid = Image.fromarray(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_grid)

    # Header
    draw.text((10, 10), "Movement", font=font_header, fill=(255, 255, 255))
    draw.text((200 + thumb_width//2 - 30, 15), "START", font=font_header, fill=(100, 255, 100))
    draw.text((200 + thumb_width + thumb_width//2 - 30, 15), "MIDDLE", font=font_header, fill=(255, 255, 100))
    draw.text((200 + thumb_width*2 + thumb_width//2 - 20, 15), "END", font=font_header, fill=(255, 100, 100))

    grid = cv2.cvtColor(np.array(pil_grid), cv2.COLOR_RGB2BGR)

    # Process each movement
    for i, mov in enumerate(ref['movements']):
        y_offset = 50 + i * thumb_height

        start_frame = mov['start_frame']
        end_frame = mov['end_frame']
        mid_frame = (start_frame + end_frame) // 2

        frames_to_show = [
            (start_frame, "START", (100, 255, 100)),
            (mid_frame, "MIDDLE", (255, 255, 100)),
            (end_frame, "END", (255, 100, 100))
        ]

        # Draw movement label
        pil_grid = Image.fromarray(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_grid)

        mov_name = mov['movement_name']
        mov_id = mov.get('movement_id', '?')
        duration = mov['duration']
        conf = mov.get('confidence', 0)

        # Movement info on left side
        draw.text((10, y_offset + 10), f"{i+1}. {mov_id}", font=font, fill=(255, 255, 255))
        draw.text((10, y_offset + 35), f"{mov_name[:20]}", font=font_small, fill=(200, 200, 200))
        draw.text((10, y_offset + 55), f"Frames: {start_frame}-{end_frame}", font=font_small, fill=(150, 150, 150))
        draw.text((10, y_offset + 75), f"Duration: {duration:.2f}s", font=font_small, fill=(150, 150, 150))
        draw.text((10, y_offset + 95), f"Conf: {conf*100:.1f}%", font=font_small,
                  fill=(100, 255, 100) if conf > 0.7 else (255, 255, 100) if conf > 0.5 else (255, 100, 100))

        grid = cv2.cvtColor(np.array(pil_grid), cv2.COLOR_RGB2BGR)

        # Extract and place frames
        for j, (frame_num, label, color) in enumerate(frames_to_show):
            x_offset = 200 + j * thumb_width

            frame = extract_frame(video_path, frame_num)
            if frame is not None:
                # Resize
                frame_resized = cv2.resize(frame, (thumb_width - 10, thumb_height - 30))

                # Add frame number
                cv2.putText(frame_resized, f"F{frame_num}", (5, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame_resized, f"{frame_num/ref['fps']:.2f}s", (5, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                # Border color based on position
                border_color = color[::-1]  # RGB to BGR
                cv2.rectangle(frame_resized, (0, 0), (thumb_width-11, thumb_height-31), border_color, 2)

                # Place on grid
                grid[y_offset + 5:y_offset + thumb_height - 25,
                     x_offset + 5:x_offset + thumb_width - 5] = frame_resized
            else:
                # No frame - draw placeholder
                cv2.rectangle(grid, (x_offset + 5, y_offset + 5),
                             (x_offset + thumb_width - 5, y_offset + thumb_height - 25), (100, 100, 100), -1)
                cv2.putText(grid, "No frame", (x_offset + 100, y_offset + thumb_height//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # Save or display
    if output_path:
        cv2.imwrite(str(output_path), grid)
        print(f"\nVerification grid saved: {output_path}")

    # Also save as scrollable HTML with images
    save_html_report(ref, video_path, output_path)

    return grid


def save_html_report(ref, video_path, output_path):
    """Save an HTML report with key frames for each movement"""

    output_dir = Path(output_path).parent if output_path else Path('compare/verification')
    output_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = output_dir / 'frames'
    frames_dir.mkdir(exist_ok=True)

    html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Reference Verification - {video_name}</title>
    <style>
        body {{ font-family: 'Malgun Gothic', sans-serif; background: #1a1a1a; color: #fff; padding: 20px; }}
        h1 {{ color: #4CAF50; }}
        .movement {{ background: #2a2a2a; margin: 20px 0; padding: 15px; border-radius: 8px; }}
        .movement-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
        .movement-name {{ font-size: 18px; color: #4CAF50; }}
        .movement-info {{ color: #888; font-size: 14px; }}
        .frames {{ display: flex; gap: 10px; }}
        .frame-box {{ text-align: center; }}
        .frame-box img {{ max-width: 400px; border: 3px solid #444; border-radius: 4px; }}
        .frame-label {{ margin-top: 5px; font-size: 12px; }}
        .start {{ border-color: #4CAF50 !important; }}
        .middle {{ border-color: #FFC107 !important; }}
        .end {{ border-color: #F44336 !important; }}
        .conf-high {{ color: #4CAF50; }}
        .conf-med {{ color: #FFC107; }}
        .conf-low {{ color: #F44336; }}
        .warning {{ background: #553300; padding: 10px; border-radius: 4px; margin-top: 10px; }}
    </style>
</head>
<body>
    <h1>Reference Verification: {video_name}</h1>
    <p>FPS: {fps} | Total Movements: {num_movements}</p>
    <hr>
""".format(
        video_name=ref['video_name'],
        fps=ref['fps'],
        num_movements=len(ref['movements'])
    )

    for i, mov in enumerate(ref['movements']):
        start_frame = mov['start_frame']
        end_frame = mov['end_frame']
        mid_frame = (start_frame + end_frame) // 2
        duration = mov['duration']
        conf = mov.get('confidence', 0)
        mov_id = mov.get('movement_id', '?')

        # Determine confidence class
        conf_class = 'conf-high' if conf > 0.7 else 'conf-med' if conf > 0.5 else 'conf-low'

        # Extract and save frames
        frame_files = []
        for frame_num, label in [(start_frame, 'start'), (mid_frame, 'middle'), (end_frame, 'end')]:
            frame = extract_frame(video_path, frame_num)
            if frame is not None:
                filename = f"mov{i+1:02d}_{label}_f{frame_num}.jpg"
                cv2.imwrite(str(frames_dir / filename), frame)
                frame_files.append((filename, frame_num, label))

        # Warnings
        warnings = []
        if duration > 3.0:
            warnings.append(f"Duration unusually long ({duration:.1f}s)")
        if duration < 0.3:
            warnings.append(f"Duration unusually short ({duration:.1f}s)")
        if conf < 0.5 and conf > 0:
            warnings.append(f"Low confidence ({conf*100:.1f}%)")

        html_content += f"""
    <div class="movement">
        <div class="movement-header">
            <span class="movement-name">{i+1}. [{mov_id}] {mov['movement_name']}</span>
            <span class="movement-info">
                Frames: {start_frame} - {end_frame} ({end_frame - start_frame} frames) |
                Duration: {duration:.2f}s |
                Confidence: <span class="{conf_class}">{conf*100:.1f}%</span>
            </span>
        </div>
        <div class="frames">
"""

        for filename, frame_num, label in frame_files:
            label_class = label
            html_content += f"""
            <div class="frame-box">
                <img src="frames/{filename}" class="{label_class}">
                <div class="frame-label">{label.upper()} (F{frame_num}, {frame_num/ref['fps']:.2f}s)</div>
            </div>
"""

        html_content += "        </div>\n"

        if warnings:
            html_content += f'        <div class="warning">⚠️ {" | ".join(warnings)}</div>\n'

        html_content += "    </div>\n"

    html_content += """
</body>
</html>
"""

    html_path = output_dir / 'verification_report.html'
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"HTML report saved: {html_path}")


def main():
    parser = argparse.ArgumentParser(description='Verify reference database')
    parser.add_argument('--reference', required=True, help='Reference PKL file')
    parser.add_argument('--video', required=True, help='Original video file')
    parser.add_argument('--output', default='compare/verification/verification_grid.png',
                        help='Output image path')

    args = parser.parse_args()

    create_verification_grid(args.reference, args.video, args.output)


if __name__ == "__main__":
    main()
