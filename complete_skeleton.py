"""
Complete Skeleton Drawing - All Standard Connections
Halpe26 Format with Full Body Connectivity

This follows standard pose estimation skeleton structure.
"""

import cv2
import numpy as np


def draw_complete_skeleton(frame, keypoints, confidence_threshold=0.3):
    """
    Draw COMPLETE skeleton with all standard connections

    Halpe26 keypoints:
    0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear
    5: Left Shoulder, 6: Right Shoulder, 7: Left Elbow, 8: Right Elbow
    9: Left Wrist, 10: Right Wrist, 11: Left Hip, 12: Right Hip
    13: Left Knee, 14: Right Knee, 15: Left Ankle, 16: Right Ankle
    17: Left Heel, 18: Left Big Toe, 19: Left Small Toe
    20: Right Heel, 21: Right Big Toe, 22: Right Small Toe
    23: Neck, 24: Head Top, 25: Pelvis

    Args:
        frame: Image to draw on
        keypoints: (26, 3) array [x, y, confidence]
        confidence_threshold: Minimum confidence to draw
    """

    # ALL standard skeleton connections
    connections = [
        # === HEAD/FACE ===
        (0, 1),  # Nose to Left Eye
        (0, 2),  # Nose to Right Eye
        (1, 3),  # Left Eye to Left Ear
        (2, 4),  # Right Eye to Right Ear
        (0, 24),  # Nose to Head Top

        # === NECK ===
        (0, 23),  # Nose to Neck (or use Head Top to Neck)
        (23, 24),  # Neck to Head Top

        # === SHOULDERS ===
        (23, 5),  # Neck to Left Shoulder
        (23, 6),  # Neck to Right Shoulder
        (5, 6),  # Left Shoulder to Right Shoulder

        # === TORSO ===
        (5, 11),  # Left Shoulder to Left Hip
        (6, 12),  # Right Shoulder to Right Hip
        (11, 12),  # Left Hip to Right Hip
        (23, 25),  # Neck to Pelvis (spine)
        (25, 11),  # Pelvis to Left Hip
        (25, 12),  # Pelvis to Right Hip

        # === LEFT ARM ===
        (5, 7),  # Left Shoulder to Left Elbow
        (7, 9),  # Left Elbow to Left Wrist

        # === RIGHT ARM ===
        (6, 8),  # Right Shoulder to Right Elbow
        (8, 10),  # Right Elbow to Right Wrist

        # === LEFT LEG ===
        (11, 13),  # Left Hip to Left Knee
        (13, 15),  # Left Knee to Left Ankle

        # === RIGHT LEG ===
        (12, 14),  # Right Hip to Right Knee
        (14, 16),  # Right Knee to Right Ankle

        # === LEFT FOOT ===
        (15, 17),  # Left Ankle to Left Heel
        (15, 18),  # Left Ankle to Left Big Toe
        (15, 19),  # Left Ankle to Left Small Toe
        (17, 18),  # Left Heel to Left Big Toe
        (18, 19),  # Left Big Toe to Left Small Toe

        # === RIGHT FOOT ===
        (16, 20),  # Right Ankle to Right Heel
        (16, 21),  # Right Ankle to Right Big Toe
        (16, 22),  # Right Ankle to Right Small Toe
        (20, 21),  # Right Heel to Right Big Toe
        (21, 22),  # Right Big Toe to Right Small Toe
    ]

    # Color scheme for different body parts
    def get_color(start_idx, end_idx):
        """Get color based on body part"""
        # Head/Face - Magenta
        if max(start_idx, end_idx) <= 4 or start_idx == 24 or end_idx == 24:
            return (255, 0, 255)
        # Neck/Spine - Yellow
        elif (start_idx == 23 or end_idx == 23 or
              start_idx == 25 or end_idx == 25):
            return (0, 255, 255)
        # Left Arm - Cyan
        elif start_idx in [5, 7, 9] and end_idx in [5, 7, 9]:
            return (255, 255, 0)
        # Right Arm - Yellow-Green
        elif start_idx in [6, 8, 10] and end_idx in [6, 8, 10]:
            return (0, 255, 128)
        # Left Leg - Blue
        elif start_idx in [11, 13, 15] and end_idx in [11, 13, 15]:
            return (255, 0, 0)
        # Right Leg - Red
        elif start_idx in [12, 14, 16] and end_idx in [12, 14, 16]:
            return (0, 0, 255)
        # Left Foot - Light Blue
        elif start_idx in [15, 17, 18, 19] and end_idx in [15, 17, 18, 19]:
            return (255, 128, 0)
        # Right Foot - Orange
        elif start_idx in [16, 20, 21, 22] and end_idx in [16, 20, 21, 22]:
            return (0, 128, 255)
        # Torso - Green
        else:
            return (0, 255, 0)

    # Draw all connections
    for start_idx, end_idx in connections:
        # Check indices valid
        if start_idx >= len(keypoints) or end_idx >= len(keypoints):
            continue

        start_pt = keypoints[start_idx]
        end_pt = keypoints[end_idx]

        # Get confidence
        start_conf = start_pt[2] if len(start_pt) > 2 else 1.0
        end_conf = end_pt[2] if len(end_pt) > 2 else 1.0

        # Lower threshold for feet and face (often lower confidence)
        if start_idx >= 17 or end_idx >= 17 or start_idx <= 4 or end_idx <= 4:
            threshold = 0.2
        else:
            threshold = confidence_threshold

        # Check confidence
        if start_conf < threshold or end_conf < threshold:
            continue

        # Check coordinates valid
        if (start_pt[0] <= 0 or start_pt[1] <= 0 or
                end_pt[0] <= 0 or end_pt[1] <= 0):
            continue

        # Convert to integer coordinates
        pt1 = (int(start_pt[0]), int(start_pt[1]))
        pt2 = (int(end_pt[0]), int(end_pt[1]))

        # Get color for this connection
        color = get_color(start_idx, end_idx)

        # Line thickness based on confidence
        avg_conf = (start_conf + end_conf) / 2
        thickness = max(1, int(avg_conf * 3))

        # Draw line
        cv2.line(frame, pt1, pt2, color, thickness)

    # Draw keypoints on top
    for idx, kp in enumerate(keypoints):
        # Get confidence
        conf = kp[2] if len(kp) > 2 else 1.0

        # Lower threshold for feet and face
        threshold = 0.2 if (idx >= 17 or idx <= 4) else confidence_threshold

        if conf < threshold:
            continue

        # Check coordinates
        if kp[0] <= 0 or kp[1] <= 0:
            continue

        pt = (int(kp[0]), int(kp[1]))

        # Circle size based on confidence
        radius = max(3, int(conf * 5))

        # Color based on confidence
        if conf > 0.7:
            color = (0, 255, 0)  # Green = high confidence
        elif conf > 0.5:
            color = (0, 255, 255)  # Yellow = medium
        else:
            color = (0, 165, 255)  # Orange = low but visible

        # Draw circle
        cv2.circle(frame, pt, radius, color, -1)

        # Draw white border for visibility
        cv2.circle(frame, pt, radius + 1, (255, 255, 255), 1)

    return frame


def draw_complete_skeleton_simple(frame, keypoints, confidence_threshold=0.3):
    """
    Complete skeleton but single color (simpler, cleaner)
    Good for presentations or when color-coding not needed
    """

    connections = [
        # Head
        (0, 1), (0, 2), (1, 3), (2, 4), (0, 24),
        # Neck
        (0, 23), (23, 24), (23, 5), (23, 6),
        # Shoulders and torso
        (5, 6), (5, 11), (6, 12), (11, 12),
        (23, 25), (25, 11), (25, 12),
        # Arms
        (5, 7), (7, 9), (6, 8), (8, 10),
        # Legs
        (11, 13), (13, 15), (12, 14), (14, 16),
        # Feet
        (15, 17), (15, 18), (15, 19), (17, 18), (18, 19),
        (16, 20), (16, 21), (16, 22), (20, 21), (21, 22),
    ]

    # Draw all connections in green
    for start_idx, end_idx in connections:
        if start_idx >= len(keypoints) or end_idx >= len(keypoints):
            continue

        start_pt = keypoints[start_idx]
        end_pt = keypoints[end_idx]

        start_conf = start_pt[2] if len(start_pt) > 2 else 1.0
        end_conf = end_pt[2] if len(end_pt) > 2 else 1.0

        # Lower threshold for extremities
        threshold = 0.2 if (start_idx >= 17 or end_idx >= 17 or
                            start_idx <= 4 or end_idx <= 4) else confidence_threshold

        if start_conf < threshold or end_conf < threshold:
            continue

        if (start_pt[0] <= 0 or start_pt[1] <= 0 or
                end_pt[0] <= 0 or end_pt[1] <= 0):
            continue

        pt1 = (int(start_pt[0]), int(start_pt[1]))
        pt2 = (int(end_pt[0]), int(end_pt[1]))

        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    # Draw keypoints
    for idx, kp in keypoints:
        conf = kp[2] if len(kp) > 2 else 1.0

        threshold = 0.2 if (idx >= 17 or idx <= 4) else confidence_threshold

        if conf < threshold or kp[0] <= 0 or kp[1] <= 0:
            continue

        pt = (int(kp[0]), int(kp[1]))
        cv2.circle(frame, pt, 4, (0, 0, 255), -1)

    return frame


def draw_skeleton_with_labels(frame, keypoints, confidence_threshold=0.3):
    """
    Complete skeleton with keypoint number labels
    Good for debugging and understanding keypoint positions
    """

    # Draw skeleton first
    frame = draw_complete_skeleton(frame, keypoints, confidence_threshold)

    # Add labels
    keypoint_names = [
        "Nose", "LEye", "REye", "LEar", "REar",
        "LShoulder", "RShoulder", "LElbow", "RElbow",
        "LWrist", "RWrist", "LHip", "RHip",
        "LKnee", "RKnee", "LAnkle", "RAnkle",
        "LHeel", "LBigToe", "LSmallToe",
        "RHeel", "RBigToe", "RSmallToe",
        "Neck", "HeadTop", "Pelvis"
    ]

    for idx, kp in enumerate(keypoints):
        conf = kp[2] if len(kp) > 2 else 1.0

        if conf < 0.2 or kp[0] <= 0 or kp[1] <= 0:
            continue

        pt = (int(kp[0]), int(kp[1]))

        # Draw label
        label = f"{idx}"  # Just number, or use keypoint_names[idx] for name
        cv2.putText(frame, label, (pt[0] + 8, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return frame


# Keypoint reference
HALPE26_KEYPOINTS = {
    'FACE': [0, 1, 2, 3, 4],
    'TORSO': [5, 6, 11, 12, 23, 25],
    'LEFT_ARM': [5, 7, 9],
    'RIGHT_ARM': [6, 8, 10],
    'LEFT_LEG': [11, 13, 15],
    'RIGHT_LEG': [12, 14, 16],
    'LEFT_FOOT': [15, 17, 18, 19],
    'RIGHT_FOOT': [16, 20, 21, 22],
}

if __name__ == "__main__":
    print("Complete Skeleton Drawing Functions")
    print("\nAvailable functions:")
    print("  1. draw_complete_skeleton() - Full color-coded skeleton")
    print("  2. draw_complete_skeleton_simple() - Single color, clean")
    print("  3. draw_skeleton_with_labels() - With keypoint numbers")
    print("\nAll standard connections included:")
    print("  ✓ Face (eyes, ears, nose)")
    print("  ✓ Neck and spine")
    print("  ✓ Shoulders and torso")
    print("  ✓ Arms (shoulder-elbow-wrist)")
    print("  ✓ Legs (hip-knee-ankle)")
    print("  ✓ Feet (heel, big toe, small toe)")