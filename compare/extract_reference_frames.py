"""
Extract and Save Reference Frames as Images

Saves all frames from each movement as individual images,
organized in folders named after the movement.

Usage:
    python extract_reference_frames.py \
        --video data/reference/annotations/P001.mp4 \
        --annotation data/reference/annotations/P001_annotations.json \
        --output compare/references/frames
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import numpy as np
from pathlib import Path
import argparse
import json
import sys
import re

sys.path.append(str(Path(__file__).parent.parent))
from preprocessing.create_windows import CLASS_MAPPING, CLASS_NAMES


def sanitize_folder_name(name):
    """Remove or replace invalid characters for folder names"""
    # Replace invalid characters with underscore
    invalid_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(invalid_chars, '_', name)
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(' .')
    return sanitized


def get_movement_id(movement_name):
    """Extract movement ID like '0_1', '14_2' from movement name"""
    parts = str(movement_name).strip().split('_')
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return f"{parts[0]}_{parts[1]}"
    return None


def extract_frames(video_path, annotation_path, output_dir, save_keypoints=False):
    """Extract and save frames for each movement"""

    video_path = Path(video_path)
    annotation_path = Path(annotation_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return

    if not annotation_path.exists():
        print(f"Annotation not found: {annotation_path}")
        return

    # Load annotation
    with open(annotation_path, 'r', encoding='utf-8') as f:
        annotation = json.load(f)

    annotations = annotation['annotations']

    print(f"\n{'='*70}")
    print("EXTRACTING REFERENCE FRAMES")
    print(f"{'='*70}")
    print(f"Video: {video_path.name}")
    print(f"Annotation: {annotation_path.name}")
    print(f"Output: {output_dir}")
    print(f"Movements: {len(annotations)}")
    print(f"{'='*70}\n")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {fps:.1f} FPS, {total_frames} frames\n")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize pose estimator if saving keypoints
    pose_estimator = None
    if save_keypoints:
        try:
            from rtmlib import BodyWithFeet
            pose_estimator = BodyWithFeet(
                to_openpose=False,
                mode='balanced',
                backend='onnxruntime',
                device='cuda'
            )
            print("RTMPose initialized for keypoint extraction\n")
        except ImportError:
            print("Warning: rtmlib not available, skipping keypoint extraction\n")
            save_keypoints = False

    # Process each movement
    total_saved = 0

    for i, ann in enumerate(annotations):
        movement_name = ann['movement']
        start_frame = ann['frame']

        # End frame is start of next movement (or end of video)
        if i + 1 < len(annotations):
            end_frame = annotations[i + 1]['frame'] - 1
        else:
            end_frame = min(start_frame + int(fps * 3), total_frames - 1)

        num_frames = end_frame - start_frame + 1

        # Create folder for this movement (use ID for folder name to avoid encoding issues)
        movement_id = get_movement_id(movement_name)
        if movement_id:
            folder_name = movement_id  # e.g., "0_1", "14_2"
        else:
            folder_name = f"mov_{i+1:02d}"

        movement_dir = output_dir / folder_name
        movement_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{i+1:02d}/22] {folder_name} - {movement_name}")
        print(f"        Frames: {start_frame} - {end_frame} ({num_frames} frames)")

        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        saved_count = 0
        keypoints_data = []

        for frame_num in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break

            # Save frame as image
            # Format: frame_XXXXX.jpg (5 digits for sorting)
            filename = f"frame_{frame_num:05d}.jpg"
            filepath = movement_dir / filename
            cv2.imwrite(str(filepath), frame)
            saved_count += 1

            # Extract keypoints if requested
            if save_keypoints and pose_estimator:
                keypoints, scores = pose_estimator(frame)
                if len(keypoints) > 0:
                    kp_data = {
                        'frame': frame_num,
                        'keypoints': keypoints[0].tolist(),
                        'scores': scores[0].tolist()
                    }
                    keypoints_data.append(kp_data)

        # Save keypoints JSON if extracted
        if save_keypoints and keypoints_data:
            kp_file = movement_dir / 'keypoints.json'
            with open(kp_file, 'w', encoding='utf-8') as f:
                json.dump(keypoints_data, f, indent=2)

        print(f"        Saved: {saved_count} images")
        total_saved += saved_count

    cap.release()

    # Create index file
    index_data = {
        'video': video_path.name,
        'annotation': annotation_path.name,
        'fps': fps,
        'total_frames': total_frames,
        'movements': []
    }

    for i, ann in enumerate(annotations):
        movement_name = ann['movement']
        movement_id = get_movement_id(movement_name)
        if movement_id:
            folder_name = movement_id
        else:
            folder_name = f"mov_{i+1:02d}"

        start_frame = ann['frame']

        if i + 1 < len(annotations):
            end_frame = annotations[i + 1]['frame'] - 1
        else:
            end_frame = min(start_frame + int(fps * 3), total_frames - 1)

        index_data['movements'].append({
            'index': i,
            'movement_id': movement_id,
            'name': movement_name,
            'folder': folder_name,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'num_frames': end_frame - start_frame + 1
        })

    index_file = output_dir / 'index.json'
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"Total images saved: {total_saved}")
    print(f"Output directory: {output_dir}")
    print(f"Index file: {index_file}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Extract reference frames as images')
    parser.add_argument('--video', required=True, help='Video file path')
    parser.add_argument('--annotation', required=True, help='Annotation JSON file path')
    parser.add_argument('--output', default='compare/references/frames',
                        help='Output directory for frames')
    parser.add_argument('--keypoints', action='store_true',
                        help='Also extract and save keypoints')

    args = parser.parse_args()

    extract_frames(args.video, args.annotation, args.output, args.keypoints)


if __name__ == "__main__":
    main()
