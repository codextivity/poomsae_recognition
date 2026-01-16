"""LSTM Model Configuration"""


class LSTMConfig:
    # Model architecture
    INPUT_SIZE = 78  # 26 keypoints × 3 (x, y, confidence)
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT = 0.3
    BIDIRECTIONAL = True

    # Output
    NUM_CLASSES = 20  # For Taegeuk 1

    # Sequence parameters
    SEQUENCE_LENGTH = 60  # 3 seconds at 30 FPS
    STRIDE = 10  # 0.33 seconds

    # Feature engineering
    USE_JOINT_ANGLES = False
    USE_VELOCITIES = False
    USE_ACCELERATIONS = False

    # Normalization
    NORMALIZE_COORDINATES = True
    REFERENCE_KEYPOINT = 19  # Hip center