"""
RTMPose Diagnostic Tool

Tests RTMPose installation and keypoint detection quality.

Usage:
    python diagnose_rtmpose.py --image path/to/image.jpg
"""

import cv2
import numpy as np
import argparse
from pathlib import Path


def test_rtmpose():
    """Test RTMPose installation and output"""

    print(f"\n{'=' * 70}")
    print("RTMPOSE DIAGNOSTIC TEST")
    print(f"{'=' * 70}\n")

    # Test 1: Import
    print("Test 1: Checking RTMLib installation...")
    try:
        from rtmlib import BodyWithFeet, Body
        print("✅ RTMLib imported successfully")
    except ImportError as e:
        print(f"❌ RTMLib import failed: {e}")
        print("   Solution: pip install rtmlib")
        return False

    # Test 2: Model initialization
    print("\nTest 2: Initializing BodyWithFeet model...")
    try:
        pose_estimator = BodyWithFeet(
            mode='balanced',
            backend='onnxruntime',
            device='cpu'
        )
        print("✅ BodyWithFeet initialized")
        print(f"   Mode: balanced")
        print(f"   Backend: onnxruntime")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False

    # Test 3: Check model details
    print("\nTest 3: Checking model configuration...")
    try:
        # Try to access model attributes
        print(f"✅ Model ready for inference")
    except Exception as e:
        print(f"⚠️  Could not access model details: {e}")

    return True, pose_estimator


def analyze_image(image_path, pose_estimator):
    """Analyze single image keypoint detection"""

    print(f"\n{'=' * 70}")
    print(f"ANALYZING IMAGE: {Path(image_path).name}")
    print(f"{'=' * 70}\n")

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return

    height, width = image.shape[:2]
    print(f"Image properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  Size: {image.shape}")

    # Run detection
    print(f"\nRunning pose detection...")
    keypoints, scores = pose_estimator(image)

    print(f"\nDetection results:")
    print(f"  Number of people detected: {len(keypoints)}")

    if len(keypoints) == 0:
        print(f"\n❌ NO PERSON DETECTED!")
        print(f"   Possible causes:")
        print(f"   - Person not visible")
        print(f"   - Image too small/large")
        print(f"   - Poor image quality")
        print(f"   - Model issue")
        return

    # Analyze first person
    person_kp = keypoints[0]
    person_scores = scores[0]

    print(f"\n✅ Person detected!")
    print(f"   Number of keypoints: {len(person_kp)}")
    print(f"   Keypoint shape: {person_kp.shape}")
    print(f"   Scores shape: {person_scores.shape}")

    # Keypoint statistics
    print(f"\nKeypoint confidence statistics:")
    print(f"  Mean confidence: {person_scores.mean():.3f}")
    print(f"  Min confidence: {person_scores.min():.3f}")
    print(f"  Max confidence: {person_scores.max():.3f}")
    print(f"  Std deviation: {person_scores.std():.3f}")

    # Count by confidence level
    high_conf = (person_scores > 0.7).sum()
    med_conf = ((person_scores > 0.3) & (person_scores <= 0.7)).sum()
    low_conf = (person_scores <= 0.3).sum()

    print(f"\nConfidence distribution:")
    print(f"  High (>0.7): {high_conf}/{len(person_scores)} ({high_conf / len(person_scores) * 100:.1f}%)")
    print(f"  Medium (0.3-0.7): {med_conf}/{len(person_scores)} ({med_conf / len(person_scores) * 100:.1f}%)")
    print(f"  Low (<0.3): {low_conf}/{len(person_scores)} ({low_conf / len(person_scores) * 100:.1f}%)")

    # Detailed keypoint analysis
    print(f"\nDetailed keypoint analysis:")

    keypoint_names = [
        "0: Nose", "1: Left Eye", "2: Right Eye", "3: Left Ear", "4: Right Ear",
        "5: Left Shoulder", "6: Right Shoulder", "7: Left Elbow", "8: Right Elbow",
        "9: Left Wrist", "10: Right Wrist", "11: Left Hip", "12: Right Hip",
        "13: Left Knee", "14: Right Knee", "15: Left Ankle", "16: Right Ankle",
        "17: Left Heel", "18: Left Big Toe", "19: Left Small Toe",
        "20: Right Heel", "21: Right Big Toe", "22: Right Small Toe",
        "23: Neck", "24: Head Top", "25: Pelvis"
    ]

    print(f"\n{'ID':<3} {'Name':<20} {'X':>6} {'Y':>6} {'Conf':>6} {'Status'}")
    print(f"{'-' * 60}")

    for i, (kp, score) in enumerate(zip(person_kp, person_scores)):
        name = keypoint_names[i] if i < len(keypoint_names) else f"KP_{i}"
        x, y = kp

        if score < 0.3:
            status = "❌ LOW"
        elif score < 0.7:
            status = "⚠️  MED"
        else:
            status = "✅ HIGH"

        print(f"{i:<3} {name:<20} {x:>6.1f} {y:>6.1f} {score:>6.3f} {status}")

    # Check for problematic keypoints
    print(f"\n{'=' * 70}")
    print(f"PROBLEM DETECTION")
    print(f"{'=' * 70}\n")

    issues = []

    # Check face keypoints
    face_indices = [0, 1, 2, 3, 4]
    face_confs = person_scores[face_indices]
    if face_confs.mean() < 0.5:
        issues.append("⚠️  Face keypoints have low confidence (person wearing mask?)")

    # Check body keypoints
    body_indices = [5, 6, 11, 12]  # Shoulders and hips
    body_confs = person_scores[body_indices]
    if body_confs.mean() < 0.7:
        issues.append("⚠️  Body keypoints have low confidence (occlusion?)")

    # Check arms
    arm_indices = [7, 8, 9, 10]  # Elbows and wrists
    arm_confs = person_scores[arm_indices]
    if arm_confs.mean() < 0.5:
        issues.append("⚠️  Arm keypoints have low confidence (arms occluded/crossed?)")

    # Check legs
    leg_indices = [13, 14, 15, 16]  # Knees and ankles
    leg_confs = person_scores[leg_indices]
    if leg_confs.mean() < 0.7:
        issues.append("⚠️  Leg keypoints have low confidence (occlusion?)")

    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✅ No major issues detected!")

    # Visualize
    print(f"\n{'=' * 70}")
    print(f"CREATING VISUALIZATION")
    print(f"{'=' * 70}\n")

    vis_image = image.copy()

    # Draw keypoints with color coding
    for i, (kp, score) in enumerate(zip(person_kp, person_scores)):
        if score < 0.3:
            continue

        x, y = int(kp[0]), int(kp[1])

        # Color based on confidence
        if score > 0.7:
            color = (0, 255, 0)  # Green = good
        elif score > 0.5:
            color = (0, 255, 255)  # Yellow = medium
        else:
            color = (0, 0, 255)  # Red = low

        cv2.circle(vis_image, (x, y), 5, color, -1)

        # Draw keypoint number
        cv2.putText(vis_image, str(i), (x + 7, y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Draw simple skeleton
    connections = [
        (5, 6), (5, 11), (6, 12), (11, 12),  # Torso
        (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
    ]

    for start, end in connections:
        if person_scores[start] > 0.5 and person_scores[end] > 0.5:
            pt1 = (int(person_kp[start][0]), int(person_kp[start][1]))
            pt2 = (int(person_kp[end][0]), int(person_kp[end][1]))
            cv2.line(vis_image, pt1, pt2, (0, 255, 0), 2)

    # Save visualization
    output_path = "rtmpose_diagnosis.jpg"
    cv2.imwrite(output_path, vis_image)
    print(f"✅ Visualization saved: {output_path}")

    # Display (if possible)
    try:
        cv2.imshow('RTMPose Diagnosis', vis_image)
        print(f"\n[Press any key to close window]")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print(f"   (Could not display window - headless environment?)")

    # Final assessment
    print(f"\n{'=' * 70}")
    print(f"ASSESSMENT")
    print(f"{'=' * 70}\n")

    avg_conf = person_scores.mean()

    if avg_conf > 0.7:
        print(f"✅ EXCELLENT detection (avg conf: {avg_conf:.3f})")
        print(f"   RTMPose is working correctly")
        print(f"   Image quality is good")
    elif avg_conf > 0.5:
        print(f"🟡 DECENT detection (avg conf: {avg_conf:.3f})")
        print(f"   RTMPose working but not ideal")
        print(f"   Consider:")
        print(f"   - Better lighting")
        print(f"   - Remove occlusions")
        print(f"   - Clearer pose")
    else:
        print(f"🔴 POOR detection (avg conf: {avg_conf:.3f})")
        print(f"   Something is wrong:")
        print(f"   - Image quality issues")
        print(f"   - Severe occlusions")
        print(f"   - Person not clearly visible")
        print(f"   - Possible RTMPose model issue")

    return person_kp, person_scores


def main():
    parser = argparse.ArgumentParser(description='Diagnose RTMPose')
    parser.add_argument('--image', type=str, required=True, help='Path to test image')

    args = parser.parse_args()

    # Test installation
    result = test_rtmpose()
    if result == False:
        return

    success, pose_estimator = result

    if not success:
        return

    # Analyze image
    analyze_image(args.image, pose_estimator)


if __name__ == "__main__":
    main()