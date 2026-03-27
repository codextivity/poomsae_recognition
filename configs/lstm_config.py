"""LSTM Model Configuration"""


class LSTMConfig:
    # Model architecture
    INPUT_SIZE = 78  # 26 keypoints × 3 (x, y, confidence)
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    DROPOUT = 0.5
    BIDIRECTIONAL = True

    # Output
    NUM_CLASSES = 22  # Default fallback; active scripts may override from dataset/checkpoint metadata

    # Sequence parameters
    SEQUENCE_LENGTH = 16  # 0.8 seconds at 30 FPS (optimized for short movements)
    STRIDE = 2  # Dense sampling for better short movement detection

    # Feature engineering
    USE_JOINT_ANGLES = False
    USE_VELOCITIES = False
    USE_ACCELERATIONS = False

    # Normalization
    NORMALIZE_COORDINATES = True
    REFERENCE_KEYPOINT = 19  # Hip center
