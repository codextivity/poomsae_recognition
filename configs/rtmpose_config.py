"""RTMPose Configuration - Halpe 26 keypoints"""


class RTMPoseConfig:
    # Model settings
    MODEL_NAME = 'rtmpose-m'
    CHECKPOINT_PATH = 'checkpoints/rtmpose_halpe26.pth'
    CONFIG_PATH = 'configs/rtmpose_halpe26_config.py'

    # Halpe 26 keypoint indices
    KEYPOINT_NAMES = [
        'nose',  # 0
        'left_eye',  # 1
        'right_eye',  # 2
        'left_ear',  # 3
        'right_ear',  # 4
        'left_shoulder',  # 5
        'right_shoulder',  # 6
        'left_elbow',  # 7
        'right_elbow',  # 8
        'left_wrist',  # 9
        'right_wrist',  # 10
        'left_hip',  # 11
        'right_hip',  # 12
        'left_knee',  # 13
        'right_knee',  # 14
        'left_ankle',  # 15
        'right_ankle',  # 16
        'head',  # 17
        'neck',  # 18
        'hip',  # 19
        'left_big_toe',  # 20
        'right_big_toe',  # 21
        'left_small_toe',  # 22
        'right_small_toe',  # 23
        'left_heel',  # 24
        'right_heel'  # 25
    ]

    # Each keypoint has (x, y, confidence)
    NUM_KEYPOINTS = 26
    FEATURES_PER_KEYPOINT = 3  # x, y, confidence
    TOTAL_FEATURES = NUM_KEYPOINTS * FEATURES_PER_KEYPOINT  # 78

    # Key landmarks for Taekwondo
    IMPORTANT_KEYPOINTS = {
        'wrists': [9, 10],
        'elbows': [7, 8],
        'shoulders': [5, 6],
        'hips': [11, 12],
        'knees': [13, 14],
        'ankles': [15, 16],
        'toes': [20, 21, 22, 23]
    }

    # Detection parameters
    BBOX_THRESHOLD = 0.3
    KEYPOINT_THRESHOLD = 0.3
    NMS_THRESHOLD = 0.65