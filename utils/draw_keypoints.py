import cv2
def draw_skeleton(self, frame, keypoints):
    """Draw skeleton with COMPLETE Halpe26 connections"""

    # COMPLETE connections for Halpe26 format
    connections = [
        # Head & Neck
        (0, 18),  # Nose → Neck
        (17, 18),  # Head top → Neck

        # Face (optional - can remove if too cluttered)
        (0, 1), (0, 2),  # Nose → Eyes
        (1, 3), (2, 4),  # Eyes → Ears

        # Neck to Shoulders
        (18, 5), (18, 6),  # Neck → Left/Right Shoulder

        # Torso
        (5, 6),  # Shoulder to Shoulder
        (5, 11), (6, 12),  # Shoulders → Hips
        (11, 12),  # Hip to Hip
        (18, 19),  # Neck → Hip Center (spine)

        # Left Arm
        (5, 7), (7, 9),  # Shoulder → Elbow → Wrist

        # Right Arm
        (6, 8), (8, 10),  # Shoulder → Elbow → Wrist

        # Left Leg
        (11, 13), (13, 15),  # Hip → Knee → Ankle

        # Right Leg
        (12, 14), (14, 16),  # Hip → Knee → Ankle

        # Left Foot
        (15, 20),  # Ankle → Big Toe
        (15, 22),  # Ankle → Small Toe
        (15, 24),  # Ankle → Heel
        (20, 22),  # Big Toe → Small Toe

        # Right Foot
        (16, 21),  # Ankle → Big Toe
        (16, 23),  # Ankle → Small Toe
        (16, 25),  # Ankle → Heel
        (21, 23),  # Big Toe → Small Toe
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
