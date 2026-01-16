"""
Quick Keypoint Extraction Verification

Checks:
1. FPS consistency across videos
2. Resolution/coordinate ranges
3. Missing detection frequency
4. Keypoint quality
"""

import pickle
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from configs.paths import Paths


def verify_keypoints():
    """Verify keypoint extraction quality"""

    keypoint_dir = Paths.KEYPOINTS_DIR
    keypoint_files = sorted(keypoint_dir.glob('*_keypoints.pkl'))

    if not keypoint_files:
        print("❌ No keypoint files found!")
        return

    print(f"\n{'=' * 70}")
    print(f"KEYPOINT EXTRACTION VERIFICATION")
    print(f"{'=' * 70}")
    print(f"Found {len(keypoint_files)} keypoint files\n")

    fps_list = []
    resolution_ranges = []
    missing_frames = []
    confidence_stats = []

    for kp_file in keypoint_files:
        with open(kp_file, 'rb') as f:
            data = pickle.load(f)

        keypoints = data['keypoints']  # (num_frames, 26, 3)
        fps = data['fps']

        # FPS
        fps_list.append(fps)

        # Coordinate ranges (to check resolution)
        x_coords = keypoints[:, :, 0]
        y_coords = keypoints[:, :, 1]

        x_min, x_max = x_coords[x_coords > 0].min(), x_coords.max()
        y_min, y_max = y_coords[y_coords > 0].min(), y_coords.max()

        resolution_ranges.append({
            'file': kp_file.name,
            'x_range': (x_min, x_max),
            'y_range': (y_min, y_max),
            'estimated_resolution': (int(x_max), int(y_max))
        })

        # Missing detections (all zeros)
        all_zero_frames = (keypoints == 0).all(axis=(1, 2)).sum()
        missing_frames.append({
            'file': kp_file.name,
            'total_frames': len(keypoints),
            'missing': all_zero_frames,
            'percentage': (all_zero_frames / len(keypoints)) * 100
        })

        # Confidence scores
        confidence = keypoints[:, :, 2]
        confidence_stats.append({
            'file': kp_file.name,
            'mean_conf': confidence[confidence > 0].mean(),
            'min_conf': confidence[confidence > 0].min(),
            'low_conf_percentage': (confidence < 0.3).sum() / confidence.size * 100
        })

    # FPS Analysis
    print(f"{'=' * 70}")
    print(f"FPS CONSISTENCY CHECK")
    print(f"{'=' * 70}\n")

    unique_fps = set(fps_list)

    if len(unique_fps) == 1:
        print(f"✅ All videos have same FPS: {fps_list[0]:.2f}")
    else:
        print(f"⚠️  WARNING: Inconsistent FPS detected!")
        print(f"   Unique FPS values: {sorted(unique_fps)}")
        print(f"   Files by FPS:")
        for fps_val in sorted(unique_fps):
            count = fps_list.count(fps_val)
            print(f"     {fps_val:.2f} fps: {count} files")

    # Resolution Analysis
    print(f"\n{'=' * 70}")
    print(f"RESOLUTION/COORDINATE RANGE CHECK")
    print(f"{'=' * 70}\n")

    unique_resolutions = set(r['estimated_resolution'] for r in resolution_ranges)

    if len(unique_resolutions) == 1:
        print(f"✅ All videos appear to have same resolution: {list(unique_resolutions)[0]}")
    else:
        print(f"⚠️  WARNING: Different resolutions detected!")
        print(f"   Unique resolutions: {sorted(unique_resolutions)}")
        print(f"\n   First 5 files:")
        for r in resolution_ranges[:5]:
            print(f"     {r['file']}: ~{r['estimated_resolution']}")

    # Missing Detection Analysis
    print(f"\n{'=' * 70}")
    print(f"MISSING DETECTION CHECK")
    print(f"{'=' * 70}\n")

    total_missing = sum(m['missing'] for m in missing_frames)
    total_frames = sum(m['total_frames'] for m in missing_frames)

    if total_missing == 0:
        print(f"✅ No missing detections! All frames have keypoints.")
    else:
        print(f"⚠️  Missing detections found:")
        print(f"   Total: {total_missing}/{total_frames} frames ({total_missing / total_frames * 100:.2f}%)")

        # Show worst files
        worst = sorted(missing_frames, key=lambda x: x['percentage'], reverse=True)[:5]
        print(f"\n   Worst files:")
        for m in worst:
            if m['missing'] > 0:
                print(f"     {m['file']}: {m['missing']}/{m['total_frames']} ({m['percentage']:.2f}%)")

    # Confidence Analysis
    print(f"\n{'=' * 70}")
    print(f"CONFIDENCE SCORE CHECK")
    print(f"{'=' * 70}\n")

    avg_confidence = np.mean([c['mean_conf'] for c in confidence_stats])

    print(f"Average confidence across all files: {avg_confidence:.3f}")

    if avg_confidence >= 0.8:
        print(f"✅ High confidence detections (≥0.8)")
    elif avg_confidence >= 0.6:
        print(f"⚠️  Medium confidence (0.6-0.8)")
    else:
        print(f"❌ Low confidence (<0.6) - keypoints may be unreliable")

    # Show files with low confidence
    low_conf_files = [c for c in confidence_stats if c['mean_conf'] < 0.6]
    if low_conf_files:
        print(f"\n   Files with low average confidence:")
        for c in low_conf_files:
            print(f"     {c['file']}: {c['mean_conf']:.3f}")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}\n")

    issues = []

    if len(unique_fps) > 1:
        issues.append("⚠️  Inconsistent FPS")
    if len(unique_resolutions) > 1:
        issues.append("⚠️  Different video resolutions")
    if total_missing > total_frames * 0.01:  # More than 1% missing
        issues.append(f"⚠️  {total_missing / total_frames * 100:.2f}% missing detections")
    if avg_confidence < 0.6:
        issues.append("❌ Low confidence scores")

    if not issues:
        print("✅ All checks passed!")
        print("\nYour keypoint extraction is consistent and high quality.")
        print("Ready to proceed with window generation.")
    else:
        print("Issues found:")
        for issue in issues:
            print(f"  {issue}")
        print("\nRecommendations:")
        if len(unique_fps) > 1:
            print("  - Convert all videos to same FPS (use fix_60fps_videos.py)")
        if len(unique_resolutions) > 1:
            print("  - Consider using normalized coordinates (hip-centered)")
        if total_missing > 0:
            print("  - Review videos with missing detections")
            print("  - Ensure person is visible and well-lit")

    print()


if __name__ == "__main__":
    verify_keypoints()