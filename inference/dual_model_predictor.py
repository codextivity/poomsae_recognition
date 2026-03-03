"""
Dual Model Predictor for Poomsae Recognition

Combines:
1. Main model (32 frames) - for most movements
2. Short model (16 frames) - specialized for fast movements (6_1, 12_1, 14_1, 16_1)

Fusion strategy:
- For movements 6, 12, 14, 16: use short model if confidence > threshold
- For other movements: use main model
- Confidence-weighted ensemble for edge cases
"""

import torch
import numpy as np
from pathlib import Path
from collections import deque
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.lstm_classifier import PoomsaeLSTM
from configs.lstm_config import LSTMConfig
from configs.lstm_config_short import LSTMConfigShort
from configs.paths import Paths


class DualModelPredictor:
    """
    Dual model predictor that combines main and short movement models.

    The short model is specialized for detecting fast movements that
    the main model often misses:
    - Movement 6 (6_1): Right punch
    - Movement 12 (12_1): Left punch
    - Movement 14 (14_1): Right front kick
    - Movement 16 (16_1): Left front kick
    """

    # Map short model class indices to main model indices (22-class system)
    # Short model classes: 0=6_1, 1=12_1, 2=14_1, 3=16_1, 4=other
    SHORT_TO_MAIN = {
        0: 6,   # 6_1 -> Class 6 (오른 지르기)
        1: 12,  # 12_1 -> Class 12 (왼 지르기)
        2: 14,  # 14_1 -> Class 14 (오른발 앞차기)
        3: 17,  # 16_1 -> Class 17 (왼발 앞차기)
    }

    # Which main model movements can be overridden by short model
    SHORT_MOVEMENT_INDICES = {6, 12, 14, 17}

    def __init__(
        self,
        main_model_path,
        short_model_path=None,
        device='cuda',
        short_confidence_threshold=0.5,
        fusion_mode='priority'
    ):
        """
        Args:
            main_model_path: Path to main model checkpoint
            short_model_path: Path to short model checkpoint (optional)
            device: 'cuda' or 'cpu'
            short_confidence_threshold: Minimum confidence to use short model
            fusion_mode: 'priority' (short model overrides for fast movements)
                        or 'ensemble' (weighted combination)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.short_confidence_threshold = short_confidence_threshold
        self.fusion_mode = fusion_mode

        # Load main model
        self.main_config = LSTMConfig()
        self.main_model = PoomsaeLSTM(self.main_config)
        checkpoint = torch.load(main_model_path, map_location=self.device)
        self.main_model.load_state_dict(checkpoint['model_state_dict'])
        self.main_model.to(self.device)
        self.main_model.eval()
        print(f"[OK] Main model loaded (32 frames)")

        # Load short model if available
        self.short_model = None
        if short_model_path and Path(short_model_path).exists():
            self.short_config = LSTMConfigShort()
            self.short_model = PoomsaeLSTM(self.short_config)
            short_checkpoint = torch.load(short_model_path, map_location=self.device)
            self.short_model.load_state_dict(short_checkpoint['model_state_dict'])
            self.short_model.to(self.device)
            self.short_model.eval()
            print(f"[OK] Short model loaded (16 frames)")
        else:
            print("[!] Short model not found - using main model only")

        # Buffers for different window sizes
        self.main_buffer = deque(maxlen=self.main_config.SEQUENCE_LENGTH)  # 32 frames
        self.short_buffer = deque(maxlen=16) if self.short_model else None  # 16 frames

        # Normalization stats
        self.main_mean = np.zeros(78)
        self.main_std = np.ones(78)
        self.short_mean = np.zeros(78)
        self.short_std = np.ones(78)

        self.load_normalization_stats()

    def load_normalization_stats(self):
        """Load normalization statistics for both models"""
        # Main model stats
        main_stats_file = Paths.CHECKPOINTS_DIR / 'normalization_stats.pkl'
        if main_stats_file.exists():
            import pickle
            with open(main_stats_file, 'rb') as f:
                stats = pickle.load(f)
            self.main_mean = stats['mean']
            self.main_std = stats['std']
            print("[OK] Loaded main model normalization stats")

        # Short model stats (if different file exists)
        short_stats_file = Paths.CHECKPOINTS_DIR / 'normalization_stats_short.pkl'
        if short_stats_file.exists():
            import pickle
            with open(short_stats_file, 'rb') as f:
                stats = pickle.load(f)
            self.short_mean = stats['mean']
            self.short_std = stats['std']
            print("[OK] Loaded short model normalization stats")
        else:
            # Use main stats for short model
            self.short_mean = self.main_mean
            self.short_std = self.main_std

    def add_keypoints(self, keypoints):
        """
        Add normalized keypoints to buffers.

        Args:
            keypoints: Normalized keypoints array (26, 3)
        """
        self.main_buffer.append(keypoints)
        if self.short_buffer is not None:
            self.short_buffer.append(keypoints)

    def predict_main(self):
        """Get prediction from main model"""
        if len(self.main_buffer) < self.main_config.SEQUENCE_LENGTH:
            return None, 0.0, np.zeros(20)

        window = np.array(list(self.main_buffer))
        window_flat = window.reshape(window.shape[0], -1)
        window_norm = (window_flat - self.main_mean) / self.main_std

        x = torch.FloatTensor(window_norm).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.main_model(x)
            probs = torch.softmax(out, dim=1)
            conf, pred = torch.max(probs, 1)

        return pred.item(), conf.item(), probs.cpu().numpy()[0]

    def predict_short(self):
        """Get prediction from short model"""
        if self.short_model is None or len(self.short_buffer) < 16:
            return None, 0.0, np.zeros(5)

        window = np.array(list(self.short_buffer))
        window_flat = window.reshape(window.shape[0], -1)
        window_norm = (window_flat - self.short_mean) / self.short_std

        x = torch.FloatTensor(window_norm).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.short_model(x)
            probs = torch.softmax(out, dim=1)
            conf, pred = torch.max(probs, 1)

        return pred.item(), conf.item(), probs.cpu().numpy()[0]

    def predict(self):
        """
        Get fused prediction from both models.

        Returns:
            predicted_class: int (0-19)
            confidence: float
            probabilities: np.array (20,)
            source: str ('main', 'short', 'fused')
        """
        # Get main model prediction
        main_pred, main_conf, main_probs = self.predict_main()

        if main_pred is None:
            return None, 0.0, np.zeros(20), 'buffering'

        # If no short model, return main prediction
        if self.short_model is None:
            return main_pred, main_conf, main_probs, 'main'

        # Get short model prediction
        short_pred, short_conf, short_probs = self.predict_short()

        if short_pred is None:
            return main_pred, main_conf, main_probs, 'main'

        # Fusion logic
        if self.fusion_mode == 'priority':
            return self._fuse_priority(
                main_pred, main_conf, main_probs,
                short_pred, short_conf, short_probs
            )
        else:
            return self._fuse_ensemble(
                main_pred, main_conf, main_probs,
                short_pred, short_conf, short_probs
            )

    def _fuse_priority(self, main_pred, main_conf, main_probs,
                       short_pred, short_conf, short_probs):
        """
        Priority fusion: Short model overrides for its target movements.
        """
        # Check if short model predicts a short movement with high confidence
        if short_pred < 4:  # Not 'other' class
            short_main_idx = self.SHORT_TO_MAIN[short_pred]

            if short_conf >= self.short_confidence_threshold:
                # Short model confident about a fast movement
                # Create modified probability array
                fused_probs = main_probs.copy()
                # Boost the short movement's probability
                fused_probs[short_main_idx] = max(fused_probs[short_main_idx], short_conf)
                # Renormalize
                fused_probs = fused_probs / fused_probs.sum()

                return short_main_idx, short_conf, fused_probs, 'short'

        # Default to main model
        return main_pred, main_conf, main_probs, 'main'

    def _fuse_ensemble(self, main_pred, main_conf, main_probs,
                       short_pred, short_conf, short_probs):
        """
        Ensemble fusion: Weighted combination for short movements.
        """
        fused_probs = main_probs.copy()

        # For each short movement, combine probabilities
        for short_idx, main_idx in self.SHORT_TO_MAIN.items():
            if short_idx < len(short_probs):
                # Weight by confidence
                weight = 0.6 if short_conf > self.short_confidence_threshold else 0.3
                fused_probs[main_idx] = (
                    (1 - weight) * main_probs[main_idx] +
                    weight * short_probs[short_idx]
                )

        # Renormalize
        fused_probs = fused_probs / fused_probs.sum()

        # Get final prediction
        pred = np.argmax(fused_probs)
        conf = fused_probs[pred]

        source = 'fused' if pred in self.SHORT_MOVEMENT_INDICES else 'main'
        return pred, conf, fused_probs, source

    def reset(self):
        """Reset buffers for new video"""
        self.main_buffer.clear()
        if self.short_buffer is not None:
            self.short_buffer.clear()

    def get_stats(self):
        """Get buffer statistics"""
        return {
            'main_buffer_size': len(self.main_buffer),
            'short_buffer_size': len(self.short_buffer) if self.short_buffer else 0,
            'main_ready': len(self.main_buffer) >= self.main_config.SEQUENCE_LENGTH,
            'short_ready': (self.short_buffer is not None and len(self.short_buffer) >= 16)
        }


def test_dual_predictor():
    """Test the dual model predictor"""
    main_path = Paths.CHECKPOINTS_DIR / 'lstm_taegeuk1_best.pth'
    short_path = Paths.CHECKPOINTS_DIR / 'lstm_short_best.pth'

    if not main_path.exists():
        print("[!] Main model not found")
        return

    predictor = DualModelPredictor(
        main_model_path=main_path,
        short_model_path=short_path if short_path.exists() else None,
        device='cuda'
    )

    # Simulate some predictions
    print("\nSimulating predictions with random keypoints...")
    for i in range(50):
        fake_keypoints = np.random.randn(26, 3).astype(np.float32)
        predictor.add_keypoints(fake_keypoints)

        pred, conf, probs, source = predictor.predict()
        if pred is not None:
            print(f"Frame {i}: Movement {pred+1}, Conf: {conf:.2f}, Source: {source}")


if __name__ == "__main__":
    test_dual_predictor()
