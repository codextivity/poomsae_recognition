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