"""
Video Diagnostic Tool

Check if a video is suitable for testing before running full inference.

Usage:
    python diagnose_video.py --video path/to/video.mp4
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import sys

try:
    from rtmlib import BodyWithFeet

    RTMLIB_AVAILABLE = True
except ImportError:
    RTMLIB_AVAILABLE = False


def check_video(video_path):
    """Check video properties"""
    video_path = Path(video_path)

    if not video_path.exists():
        print(f"❌ ERROR: Video not found: {video_path}")
        return False

    print(f"\n{'=' * 70}")
    print(f"VIDEO DIAGNOSTICS: {video_path.name}")
    print(f"{'=' * 70}\n")

    # Open video
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"❌ ERROR: Cannot open video file")
        print(f"   Possible issues:")
        print(f"   - Corrupted file")
        print(f"   - Unsupported format")
        print(f"   - Missing codec")
        return False

    # Get properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    print("VIDEO PROPERTIES:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {duration:.2f} seconds")

    # Check minimum requirements
    print(f"\nREQUIREMENTS CHECK:")

    # 1. Duration check
    min_duration = 2.0  # Need at least 60 frames at 30fps
    if duration >= min_duration:
        print(f"  ✅ Duration: {duration:.2f}s (need ≥{min_duration}s)")
    else:
        print(f"  ❌ Duration: {duration:.2f}s (need ≥{min_duration}s)")
        print(f"     Video too short for testing!")
        print(f"     Model needs 60 frames (2 seconds) minimum")

    # 2. FPS check
    if 25 <= fps <= 35:
        print(f"  ✅ FPS: {fps:.2f} (optimal: 30)")
    elif fps > 0:
        print(f"  ⚠️  FPS: {fps:.2f} (unusual, may affect predictions)")
        print(f"     Model trained on 30fps videos")
    else:
        print(f"  ❌ FPS: {fps:.2f} (invalid)")

    # 3. Resolution check
    if width >= 640 and height >= 480:
        print(f"  ✅ Resolution: {width}x{height} (adequate)")
    else:
        print(f"  ⚠️  Resolution: {width}x{height} (low, may affect detection)")
        print(f"     Recommended: ≥640x480")

    # 4. Read first frame
    ret, first_frame = cap.read()
    if not ret:
        print(f"  ❌ Cannot read first frame")
        cap.release()
        return False
    else:
        print(f"  ✅ Can read frames")

    # 5. Check if RTMLib available
    print(f"\nPOSE DETECTION CHECK:")
    if not RTMLIB_AVAILABLE:
        print(f"  ❌ RTMLib not installed")
        print(f"     Run: pip install rtmlib")
        cap.release()
        return False
    else:
        print(f"  ✅ RTMLib available")

    # 6. Test pose detection on first frame
    print(f"  Testing pose detection on first frame...")
    try:
        pose_estimator = BodyWithFeet(
            mode='balanced',
            backend='onnxruntime',
            device='cpu'  # Use CPU for diagnostic
        )

        keypoints, scores = pose_estimator(first_frame)

        if len(keypoints) > 0:
            avg_score = scores[0].mean()
            print(f"  ✅ Person detected!")
            print(f"     Average confidence: {avg_score:.3f}")

            if avg_score < 0.3:
                print(f"     ⚠️  Low confidence - person may not be clearly visible")
        else:
            print(f"  ⚠️  No person detected in first frame")
            print(f"     Possible issues:")
            print(f"     - Person not in frame")
            print(f"     - Person too small")
            print(f"     - Poor lighting")
            print(f"     - Person occluded")
    except Exception as e:
        print(f"  ❌ Pose detection failed: {e}")
        cap.release()
        return False

    # 7. Sample frames throughout video
    print(f"\n  Sampling frames throughout video...")
    sample_frames = [int(total_frames * p) for p in [0.25, 0.5, 0.75]]
    detections = 0

    for frame_num in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            keypoints, scores = pose_estimator(frame)
            if len(keypoints) > 0:
                detections += 1

    detection_rate = detections / len(sample_frames) * 100
    print(f"  Person detected in {detections}/{len(sample_frames)} sampled frames ({detection_rate:.0f}%)")

    if detection_rate >= 66:
        print(f"  ✅ Good detection rate")
    elif detection_rate >= 33:
        print(f"  ⚠️  Medium detection rate - some frames may fail")
    else:
        print(f"  ❌ Low detection rate - video may not work well")

    cap.release()

    # Final verdict
    print(f"\n{'=' * 70}")
    print(f"VERDICT:")
    print(f"{'=' * 70}\n")

    if duration < min_duration:
        print(f"❌ VIDEO TOO SHORT")
        print(f"   Need at least {min_duration}s, have {duration:.2f}s")
        print(f"   Solution: Use a longer video")
        return False

    if detection_rate < 33:
        print(f"❌ POOR PERSON DETECTION")
        print(f"   Person only detected in {detection_rate:.0f}% of frames")
        print(f"   Solutions:")
        print(f"   - Ensure person is fully visible (head to feet)")
        print(f"   - Improve lighting")
        print(f"   - Move person closer to camera")
        print(f"   - Use plain background")
        return False

    if detection_rate < 66:
        print(f"⚠️  VIDEO MAY WORK BUT WITH ISSUES")
        print(f"   Detection rate: {detection_rate:.0f}%")
        print(f"   You can try, but results may be unreliable")
        print(f"\n   Proceed with: python test_on_video.py --video {video_path}")
        return True

    print(f"✅ VIDEO LOOKS GOOD!")
    print(f"   Duration: {duration:.2f}s ✓")
    print(f"   Detection: {detection_rate:.0f}% ✓")
    print(f"   Ready for testing!")
    print(f"\n   Run: python test_on_video.py --video {video_path}")

    return True


def main():
    parser = argparse.ArgumentParser(description='Diagnose video for testing')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')

    args = parser.parse_args()

    check_video(args.video)


if __name__ == "__main__":
    main()