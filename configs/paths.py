"""All file paths in one place"""
from pathlib import Path


class Paths:
    # Base directory
    ROOT = Path(__file__).parent.parent

    # Data directories
    DATA_ROOT = ROOT / 'data'
    RAW_VIDEOS = DATA_ROOT / 'raw' / 'videos'
    RAW_ANNOTATIONS = DATA_ROOT / 'raw' / 'JSON'

    PROCESSED_ROOT = DATA_ROOT / 'processed'
    KEYPOINTS_DIR = PROCESSED_ROOT / 'keypoints'
    MOVEMENTS_DIR = PROCESSED_ROOT / 'movements'
    WINDOWS_DIR = PROCESSED_ROOT / 'windows'

    REFERENCE_ROOT = DATA_ROOT / 'reference'
    REFERENCE_VIDEOS = REFERENCE_ROOT / 'videos'
    REFERENCE_KEYPOINTS = REFERENCE_ROOT / 'keypoints'

    DATASETS_DIR = DATA_ROOT / 'datasets'

    # Model directories
    MODELS_DIR = ROOT / 'models'
    CHECKPOINTS_DIR = ROOT / 'checkpoints/22_classes_model_mediapipe'
    RTMPOSE_MODELS_DIR = CHECKPOINTS_DIR / 'rtmpose_models'

    # Results
    RESULTS_DIR = ROOT / 'results'
    PLOTS_DIR = RESULTS_DIR / 'plots'
    METRICS_DIR = RESULTS_DIR / 'metrics'

    # Ensure directories exist
    @classmethod
    def create_directories(cls):
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if isinstance(attr, Path) and '_DIR' in attr_name:
                attr.mkdir(parents=True, exist_ok=True)