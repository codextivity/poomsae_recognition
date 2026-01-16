"""LSTM Model for Movement Classification"""

import torch
import torch.nn as nn
from configs.lstm_config import LSTMConfig


class PoomsaeLSTM(nn.Module):
    def __init__(self, config=None):
        super(PoomsaeLSTM, self).__init__()

        if config is None:
            config = LSTMConfig()

        self.config = config

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.INPUT_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0,
            bidirectional=config.BIDIRECTIONAL
        )

        # Calculate LSTM output size
        lstm_output_size = config.HIDDEN_SIZE * 2 if config.BIDIRECTIONAL else config.HIDDEN_SIZE

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, config.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_SIZE, config.NUM_CLASSES)
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, sequence_length, input_size)
               e.g., (32, 90, 78)

        Returns:
            output: (batch_size, num_classes)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use final hidden state from last layer
        if self.config.BIDIRECTIONAL:
            # Concatenate forward and backward final hidden states
            hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            hidden = h_n[-1]

        # Fully connected
        output = self.fc(hidden)

        return output

    def get_attention_weights(self, x):
        """
        Optional: Get attention over sequence
        Useful for visualization
        """
        # This is a placeholder for future attention mechanism
        pass


class PoomsaeLSTMWithAttention(nn.Module):
    """LSTM with attention mechanism"""

    def __init__(self, config=None):
        super(PoomsaeLSTMWithAttention, self).__init__()

        if config is None:
            config = LSTMConfig()

        self.config = config

        # LSTM
        self.lstm = nn.LSTM(
            input_size=config.INPUT_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0,
            bidirectional=config.BIDIRECTIONAL
        )

        lstm_output_size = config.HIDDEN_SIZE * 2 if config.BIDIRECTIONAL else config.HIDDEN_SIZE

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.Tanh(),
            nn.Linear(lstm_output_size // 2, 1)
        )

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, config.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_SIZE, config.NUM_CLASSES)
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, sequence_length, input_size)

        Returns:
            output: (batch_size, num_classes)
            attention_weights: (batch_size, sequence_length)
        """
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size*2)

        # Attention
        attention_scores = self.attention(lstm_out).squeeze(-1)  # (batch, seq_len)
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch, seq_len)

        # Weighted sum
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, seq_len)
            lstm_out  # (batch, seq_len, hidden_size*2)
        ).squeeze(1)  # (batch, hidden_size*2)

        # Classify
        output = self.fc(context)

        return output, attention_weights


class FeatureExtractor(nn.Module):
    """Extract additional features from keypoints"""

    def __init__(self):
        super(FeatureExtractor, self).__init__()

        # Define which keypoints to use for each feature
        self.angle_triplets = {
            'left_elbow': (5, 7, 9),
            'right_elbow': (6, 8, 10),
            'left_knee': (11, 13, 15),
            'right_knee': (12, 14, 16)
        }

    def forward(self, keypoints):
        """
        Args:
            keypoints: (batch, seq_len, 78) - flattened
                      or (batch, seq_len, 26, 3) - structured

        Returns:
            features: (batch, seq_len, num_features)
        """
        # Reshape if needed
        if keypoints.shape[-1] == 78:
            keypoints = keypoints.reshape(keypoints.shape[0], keypoints.shape[1], 26, 3)

        batch_size, seq_len = keypoints.shape[:2]

        # Calculate angles
        angles = self._calculate_angles(keypoints)

        # Calculate velocities
        velocities = self._calculate_velocities(keypoints)

        # Combine
        features = torch.cat([
            keypoints.reshape(batch_size, seq_len, -1),
            angles,
            velocities
        ], dim=-1)

        return features

    def _calculate_angles(self, keypoints):
        """Calculate joint angles"""
        batch_size, seq_len = keypoints.shape[:2]
        num_angles = len(self.angle_triplets)

        angles = torch.zeros(batch_size, seq_len, num_angles, device=keypoints.device)

        for i, (p1_idx, p2_idx, p3_idx) in enumerate(self.angle_triplets.values()):
            p1 = keypoints[:, :, p1_idx, :2]
            p2 = keypoints[:, :, p2_idx, :2]
            p3 = keypoints[:, :, p3_idx, :2]

            # Vectors
            v1 = p1 - p2
            v2 = p3 - p2

            # Angle
            cos_angle = (v1 * v2).sum(dim=-1) / (
                    torch.norm(v1, dim=-1) * torch.norm(v2, dim=-1) + 1e-8
            )
            angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))

            angles[:, :, i] = torch.rad2deg(angle)

        return angles

    def _calculate_velocities(self, keypoints):
        """Calculate frame-to-frame velocities"""
        # Velocity is difference between consecutive frames
        velocities = keypoints[:, 1:, :, :2] - keypoints[:, :-1, :, :2]

        # Pad first frame with zeros
        zero_pad = torch.zeros(
            keypoints.shape[0], 1, keypoints.shape[2], 2,
            device=keypoints.device
        )
        velocities = torch.cat([zero_pad, velocities], dim=1)

        # Flatten
        velocities = velocities.reshape(keypoints.shape[0], keypoints.shape[1], -1)

        return velocities


if __name__ == "__main__":
    # Test model
    config = LSTMConfig()

    # Create model
    model = PoomsaeLSTM(config)

    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 4
    seq_len = 90
    input_size = 78

    x = torch.randn(batch_size, seq_len, input_size)
    output = model(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: ({batch_size}, {config.NUM_CLASSES})")