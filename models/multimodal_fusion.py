"""
Multimodal Fusion Model: CNN + LSTM for combining posture, facial, audio, and sentiment features.
Implements Table 2 specification for unified interview assessment.
"""

import torch
import torch.nn as nn
import numpy as np


class MultimodalFusionModel(nn.Module):
    """
    CNN + LSTM architecture for fusing multimodal interview features.

    Input: Concatenated features from:
    - Posture (OpenPose + MediaPipe landmarks): ~50-100 dims
    - Facial embeddings (FaceNet): 128 dims
    - Audio MFCCs: ~39 dims (13 MFCCs × 3 stats)
    - Sentiment embedding: 768 dims (BERT)

    Output: Single score 0-100
    """

    def __init__(self, input_dim=1000, hidden_dim=256, lstm_dim=128):
        """
        Args:
            input_dim: Total dimension of concatenated features (default ~1000)
            hidden_dim: Hidden dimension for CNN layers
            lstm_dim: Hidden dimension for LSTM
        """
        super(MultimodalFusionModel, self).__init__()

        # CNN layers for spatial feature processing
        self.cnn_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim // 2, lstm_dim),
            nn.ReLU(),
        )

        # LSTM for temporal/sequential feature processing
        self.lstm = nn.LSTM(
            input_size=lstm_dim,
            hidden_size=lstm_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # Dense layers for final scoring
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x: Tensor of shape (batch_size, sequence_length, input_dim) or (batch_size, input_dim)

        Returns:
            Score tensor of shape (batch_size, 1) with values 0-100
        """
        # Handle 2D input (single frame) by adding sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, input_dim) → (batch_size, 1, input_dim)

        # CNN processing
        batch_size, seq_len, input_dim = x.shape
        x_flat = x.view(-1, input_dim)  # Flatten to (batch_size * seq_len, input_dim)
        x_cnn = self.cnn_layers(x_flat)  # (batch_size * seq_len, lstm_dim)
        x_cnn = x_cnn.view(batch_size, seq_len, -1)  # Reshape back

        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x_cnn)  # lstm_out: (batch_size, seq_len, lstm_dim)

        # Use final hidden state for scoring
        x_lstm = h_n[-1]  # (batch_size, lstm_dim) - last layer, final timestep

        # Dense layers and scaling
        score = self.fc_layers(x_lstm)  # (batch_size, 1)
        score = self.sigmoid(score) * 100  # Scale to 0-100

        return score.squeeze(-1)  # (batch_size,)


class FeatureBuffer:
    """
    Buffer for collecting and normalizing multimodal features across frames.
    """

    def __init__(self, max_frames=30):
        """
        Args:
            max_frames: Maximum number of frames to buffer
        """
        self.max_frames = max_frames
        self.posture_features = []
        self.facial_features = []
        self.audio_features = None
        self.sentiment_features = None

    def add_posture(self, landmarks_flat):
        """Add quantized posture/pose landmarks."""
        if len(self.posture_features) < self.max_frames:
            self.posture_features.append(landmarks_flat)

    def add_facial(self, embedding):
        """Add facial embedding (128-dim from FaceNet)."""
        if len(self.facial_features) < self.max_frames:
            self.facial_features.append(embedding)

    def set_audio_features(self, mfcc_stats):
        """Set audio MFCC statistics (~39 dims: 13 mfccs × 3 stats)."""
        self.audio_features = mfcc_stats

    def set_sentiment_features(self, sentiment_embedding):
        """Set sentiment embedding (~768 dims from BERT)."""
        self.sentiment_features = sentiment_embedding

    def get_concatenated_features(self):
        """
        Concatenate all collected features into a single tensor.

        Returns:
            Tensor of shape (num_frames, total_feature_dim) or (1, total_feature_dim) if single
        """
        features = []

        # Posture features (averaged or stacked)
        if self.posture_features:
            posture_array = np.array(self.posture_features)  # (num_frames, posture_dim)
            features.append(torch.FloatTensor(posture_array))

        # Facial features (averaged per frame)
        if self.facial_features:
            facial_array = np.array(self.facial_features)  # (num_frames, 128)
            features.append(torch.FloatTensor(facial_array))

        # Audio features (broadcast to all frames)
        if self.audio_features is not None:
            num_frames = len(self.posture_features) if self.posture_features else 1
            audio_expanded = torch.FloatTensor(self.audio_features).unsqueeze(0).repeat(num_frames, 1)
            features.append(audio_expanded)

        # Sentiment features (broadcast to all frames)
        if self.sentiment_features is not None:
            num_frames = len(self.posture_features) if self.posture_features else 1
            sentiment_expanded = torch.FloatTensor(self.sentiment_features).unsqueeze(0).repeat(num_frames, 1)
            features.append(sentiment_expanded)

        # Concatenate all features along feature dimension
        if features:
            concatenated = torch.cat(features, dim=1)
            return concatenated
        else:
            return torch.zeros(1, 1)  # Fallback empty tensor

    def clear(self):
        """Reset all buffers."""
        self.posture_features = []
        self.facial_features = []
        self.audio_features = None
        self.sentiment_features = None


def create_fusion_model(device='cpu'):
    """
    Factory function to create and initialize multimodal fusion model.

    Args:
        device: 'cpu' or 'cuda'

    Returns:
        MultimodalFusionModel instance on specified device
    """
    model = MultimodalFusionModel(input_dim=1000, hidden_dim=256, lstm_dim=128)
    model = model.to(device)
    model.eval()  # Set to evaluation mode by default
    return model
