"""LSTM Model Configuration for Short/Fast Movements

Specialized model for detecting movements 6_1, 12_1, 14_1, 16_1
which have durations as short as 5-12 frames.
"""


class LSTMConfigShort:
    # Model architecture (smaller for faster inference)
    INPUT_SIZE = 78  # 26 keypoints × 3 (x, y, confidence)
    HIDDEN_SIZE = 64  # Smaller than main model
    NUM_LAYERS = 2
    DROPOUT = 0.3
    BIDIRECTIONAL = True

    # Output - only 4 short movements + 1 "other" class
    NUM_CLASSES = 5
    CLASS_NAMES = ['6_1', '12_1', '14_1', '16_1', 'other']

    # Maps original movement indices to short model classes
    # In the 22-class system:
    #   Class 6:  6_1 (오른 지르기) - right punch
    #   Class 12: 12_1 (왼 지르기) - left punch
    #   Class 14: 14_1 (오른발 앞차기) - right front kick
    #   Class 17: 16_1 (왼발 앞차기) - left front kick
    SHORT_MOVEMENT_IDS = [6, 12, 14, 17]  # 0-indexed class indices in 22-class system

    # Sequence parameters - optimized for short movements
    SEQUENCE_LENGTH = 16  # ~0.53 seconds at 30 FPS
    STRIDE = 2  # Dense sampling for short movements

    # Feature engineering
    USE_JOINT_ANGLES = False
    USE_VELOCITIES = False
    USE_ACCELERATIONS = False

    # Normalization
    NORMALIZE_COORDINATES = True
    REFERENCE_KEYPOINT = 19  # Hip center
