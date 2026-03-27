"""Switchable policy options for data prep, training sampling, and inference."""
import os

from configs.class_metadata import DEFAULT_SHORT_MOVEMENT_IDS


class PolicyConfig:
    """
    Central place to switch behaviors without editing pipeline logic.

    Profiles:
    - "baseline": keep existing behavior
    - "short_aware": keep more short-movement windows and use quality-aware sampling
    """

    PROFILE = os.getenv("POOMSAE_POLICY_PROFILE", "short_aware")

    # Movement IDs that receive short-movement handling when present.
    SHORT_MOVEMENT_IDS = tuple(DEFAULT_SHORT_MOVEMENT_IDS)
    SHORT_CLASS_INDICES = set()

    # ----------------------------
    # Window selection (preprocess)
    # ----------------------------
    # Baseline behavior: keep only high+medium.
    KEEP_LOW_QUALITY_WINDOWS = False
    KEEP_LOW_FOR_SHORT_CLASSES_ONLY = True
    LOW_QUALITY_MIN_OVERLAP_PCT = 30.0

    # ----------------------------
    # Sampler weighting (training)
    # ----------------------------
    USE_QUALITY_AWARE_SAMPLING = False
    SHORT_CLASS_WEIGHT_MULTIPLIER = 3.0
    QUALITY_WEIGHT_MULTIPLIERS = {
        "high": 1.0,
        "medium": 0.7,
        "low": 0.35,
        "none": 0.35,
    }

    # ----------------------------
    # Inference sequence validation
    # ----------------------------
    ALLOW_SKIP_BY_ONE = True
    ALLOW_FUTURE_SKIP = True

    # Prediction smoothing options (kept equal to current defaults)
    PREDICTION_HISTORY_SIZE = 5
    MIN_HISTORY_FOR_SMOOTHING = 3

    @classmethod
    def apply_profile(cls):
        """Set options based on PROFILE while keeping explicit overrides easy."""
        profile = str(cls.PROFILE).strip().lower()

        if profile == "custom":
            # Keep values exactly as written in this file.
            return

        if profile == "baseline":
            cls.KEEP_LOW_QUALITY_WINDOWS = True
            cls.KEEP_LOW_FOR_SHORT_CLASSES_ONLY = True
            cls.LOW_QUALITY_MIN_OVERLAP_PCT = 30.0

            cls.USE_QUALITY_AWARE_SAMPLING = True
            cls.SHORT_CLASS_WEIGHT_MULTIPLIER = 2.0
            cls.QUALITY_WEIGHT_MULTIPLIERS = {
                "high": 1.0,
                "medium": 0.7,
                "low": 0.35,
                "none": 0.35,
            }

            cls.ALLOW_SKIP_BY_ONE = False
            cls.ALLOW_FUTURE_SKIP = False
            cls.PREDICTION_HISTORY_SIZE = 3
            cls.MIN_HISTORY_FOR_SMOOTHING = 2
        else:
            # Short-aware profile: keep more boundary windows for short classes
            # while down-weighting noisier samples during training.
            cls.KEEP_LOW_QUALITY_WINDOWS = True
            cls.KEEP_LOW_FOR_SHORT_CLASSES_ONLY = True
            cls.LOW_QUALITY_MIN_OVERLAP_PCT = 20.0

            cls.USE_QUALITY_AWARE_SAMPLING = True
            cls.SHORT_CLASS_WEIGHT_MULTIPLIER = 4.0
            cls.QUALITY_WEIGHT_MULTIPLIERS = {
                "high": 1.0,
                "medium": 0.85,
                "low": 0.55,
                "none": 0.25,
            }

            cls.ALLOW_SKIP_BY_ONE = True
            cls.ALLOW_FUTURE_SKIP = True
            cls.PREDICTION_HISTORY_SIZE = 5
            cls.MIN_HISTORY_FOR_SMOOTHING = 3
