"""Data normalization and augmentation"""

import numpy as np
from configs.lstm_config import LSTMConfig
from configs.paths import Paths

class KeypointNormalizer:
    def __init__(self):
        self.config = LSTMConfig()
        self.reference_point = self.config.REFERENCE_KEYPOINT  # Hip center (19)

    def normalize_coordinates(self, keypoints):
        """
        Normalize keypoints relative to reference point (hip)

        Args:
            keypoints: (seq_len, 26, 3) or (batch, seq_len, 26, 3)

        Returns:
            normalized: Same shape as input
        """
        if keypoints.ndim == 3:
            # Single sequence
            return self._normalize_single(keypoints)
        elif keypoints.ndim == 4:
            # Batch of sequences
            return np.array([self._normalize_single(seq) for seq in keypoints])
        else:
            raise ValueError(f"Expected 3D or 4D array, got {keypoints.ndim}D")

    def _normalize_single(self, keypoints):
        """Normalize single sequence"""
        # keypoints shape: (seq_len, 26, 3)
        normalized = keypoints.copy()

        # Extract reference point (hip center) for each frame
        reference = keypoints[:, self.reference_point, :2]  # (seq_len, 2) - x, y only

        # Normalize x, y coordinates relative to hip
        for i in range(26):
            normalized[:, i, 0] -= reference[:, 0]  # x
            normalized[:, i, 1] -= reference[:, 1]  # y
            # Keep confidence as is (column 2)

        # Scale to unit variance (optional)
        # Calculate bounding box size
        for t in range(len(keypoints)):
            points = keypoints[t, :, :2]  # (26, 2)

            # Skip frames with no detections
            if np.all(points == 0):
                continue

            # Calculate scale (distance from hip to furthest point)
            distances = np.linalg.norm(points - reference[t], axis=1)
            max_distance = np.max(distances)

            if max_distance > 0:
                normalized[t, :, 0] /= max_distance  # x
                normalized[t, :, 1] /= max_distance  # y

        return normalized

    def calculate_joint_angles(self, keypoints):
        """
        Calculate important joint angles

        Args:
            keypoints: (seq_len, 26, 3)

        Returns:
            angles: (seq_len, num_angles)
        """
        seq_len = len(keypoints)

        # Define joint triplets (point1, joint, point2)
        joint_triplets = {
            'left_elbow': (5, 7, 9),      # shoulder, elbow, wrist
            'right_elbow': (6, 8, 10),
            'left_knee': (11, 13, 15),    # hip, knee, ankle
            'right_knee': (12, 14, 16),
            'left_shoulder': (18, 5, 7),  # neck, shoulder, elbow
            'right_shoulder': (18, 6, 8),
            'left_hip': (5, 11, 13),      # shoulder, hip, knee
            'right_hip': (6, 12, 14)
        }

        angles = []

        for t in range(seq_len):
            frame_angles = []

            for joint_name, (p1_idx, p2_idx, p3_idx) in joint_triplets.items():
                p1 = keypoints[t, p1_idx, :2]
                p2 = keypoints[t, p2_idx, :2]
                p3 = keypoints[t, p3_idx, :2]

                # Calculate angle
                angle = self._calculate_angle(p1, p2, p3)
                frame_angles.append(angle)

            angles.append(frame_angles)

        return np.array(angles)  # (seq_len, 8)

    def _calculate_angle(self, p1, p2, p3):
        """Calculate angle at p2"""
        v1 = p1 - p2
        v2 = p3 - p2

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

        return np.degrees(angle)

    def calculate_velocities(self, keypoints):
        """
        Calculate frame-to-frame velocities

        Args:
            keypoints: (seq_len, 26, 3)

        Returns:
            velocities: (seq_len-1, 26, 2)
        """
        velocities = np.diff(keypoints[:, :, :2], axis=0)  # Only x, y
        return velocities

    def augment_sequence(self, keypoints):
        """
        Data augmentation

        Techniques:
        - Horizontal flip
        - Small rotation
        - Temporal scaling (speed up/slow down)
        - Noise injection
        """
        augmented = []

        # Original
        augmented.append(keypoints)

        # Horizontal flip
        flipped = self._horizontal_flip(keypoints)
        augmented.append(flipped)

        # Small rotation (±5 degrees)
        for angle in [-5, 5]:
            rotated = self._rotate(keypoints, angle)
            augmented.append(rotated)

        # Temporal scaling (0.9x, 1.1x speed)
        for scale in [0.9, 1.1]:
            scaled = self._temporal_scale(keypoints, scale)
            augmented.append(scaled)

        return augmented

    def _horizontal_flip(self, keypoints):
        """Flip horizontally and swap left/right keypoints"""
        flipped = keypoints.copy()

        # Flip x coordinates
        flipped[:, :, 0] = -flipped[:, :, 0]

        # Swap left/right pairs
        swap_pairs = [
            (1, 2),   # left_eye, right_eye
            (3, 4),   # left_ear, right_ear
            (5, 6),   # left_shoulder, right_shoulder
            (7, 8),   # left_elbow, right_elbow
            (9, 10),  # left_wrist, right_wrist
            (11, 12), # left_hip, right_hip
            (13, 14), # left_knee, right_knee
            (15, 16), # left_ankle, right_ankle
            (20, 21), # left_big_toe, right_big_toe
            (22, 23), # left_small_toe, right_small_toe
            (24, 25)  # left_heel, right_heel
        ]

        for left_idx, right_idx in swap_pairs:
            flipped[:, [left_idx, right_idx]] = flipped[:, [right_idx, left_idx]]

        return flipped

    def _rotate(self, keypoints, angle_degrees):
        """Rotate all keypoints by angle"""
        angle_rad = np.radians(angle_degrees)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])

        rotated = keypoints.copy()

        # Rotate x, y coordinates
        for t in range(len(keypoints)):
            for i in range(26):
                xy = keypoints[t, i, :2]
                rotated[t, i, :2] = rotation_matrix @ xy

        return rotated

    def _temporal_scale(self, keypoints, scale):
        """Scale temporal dimension (speed up/slow down)"""
        from scipy.interpolate import interp1d

        original_length = len(keypoints)
        new_length = int(original_length / scale)

        # Interpolate
        x_old = np.linspace(0, 1, original_length)
        x_new = np.linspace(0, 1, new_length)

        scaled = np.zeros((new_length, 26, 3))

        for i in range(26):
            for j in range(3):
                interpolator = interp1d(x_old, keypoints[:, i, j], kind='cubic')
                scaled[:, i, j] = interpolator(x_new)

        # Pad or truncate to original length
        if new_length < original_length:
            # Pad with last frame
            padding = np.repeat(scaled[-1:], original_length - new_length, axis=0)
            scaled = np.concatenate([scaled, padding], axis=0)
        else:
            # Truncate
            scaled = scaled[:original_length]

        return scaled


if __name__ == "__main__":
    # Test normalization
    normalizer = KeypointNormalizer()

    # Load sample data
    sample = np.load(Paths.WINDOWS_DIR / 'student1_taegeuk1_windows.npz')
    X = sample['X']  # (num_windows, 90, 78)

    # Reshape to (num_windows, 90, 26, 3)
    X_reshaped = X.reshape(X.shape[0], X.shape[1], 26, 3)

    # Normalize
    X_normalized = normalizer.normalize_coordinates(X_reshaped)

    print(f"Original shape: {X_reshaped.shape}")
    print(f"Normalized shape: {X_normalized.shape}")
    print(f"Sample normalized values:\n{X_normalized[0, 0, :3]}")