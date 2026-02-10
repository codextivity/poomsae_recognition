"""Training Configuration"""


class TrainingConfig:
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5

    # Optimization
    OPTIMIZER = 'Adam'
    SCHEDULER = 'ReduceLROnPlateau'
    SCHEDULER_PATIENCE = 10
    SCHEDULER_FACTOR = 0.5

    # Loss function
    LOSS_FUNCTION = 'CrossEntropyLoss'
    LABEL_SMOOTHING = 0.1

    # Class weighting for short movements
    # These classes have shorter durations and fewer samples
    # Give them higher weight so model pays more attention
    SHORT_MOVEMENT_CLASSES = [6, 12, 14, 17]  # 6_1, 12_1, 14_1, 16_1
    SHORT_MOVEMENT_WEIGHT_MULTIPLIER = 3.0  # 3x penalty for misclassifying these

    # Early stopping
    EARLY_STOPPING_PATIENCE = 20
    MIN_DELTA = 0.001

    # Data split
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15

    # Data augmentation
    AUGMENTATION = True
    TIME_WARPING = True
    NOISE_INJECTION = True
    ROTATION_AUGMENT = True

    # Checkpointing
    SAVE_EVERY = 5  # epochs
    KEEP_BEST_ONLY = True

    # Device
    DEVICE = 'cuda'  # or 'cpu'
    NUM_WORKERS = 4
    PIN_MEMORY = True