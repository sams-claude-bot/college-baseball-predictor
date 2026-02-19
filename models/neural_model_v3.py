#!/usr/bin/env python3
"""
Neural Network Model v3

Improvements over v1:
- Residual/skip connections in the network architecture
- Platt scaling calibration via sklearn (fitted on validation data, saved in checkpoint)
- Supports enhanced features via EnhancedFeatureComputer
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import pickle

from models.base_model import BaseModel
from models.nn_features import FeatureComputer

MODEL_PATH = Path(__file__).parent.parent / "data" / "nn_model_v3.pt"


class ResidualBlock(nn.Module):
    """Residual block with projection shortcut."""

    def __init__(self, in_features, out_features, dropout=0.3):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features),
        )
        # Projection shortcut when dimensions differ
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.relu(self.main(x) + self.shortcut(x)))


class BaseballNetV3(nn.Module):
    """
    Neural network with residual connections.
    Architecture: input → 256 (residual) → 128 (residual) → 64 (residual) → 1
    """

    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.block1 = ResidualBlock(input_size, 256, dropout=0.3)
        self.block2 = ResidualBlock(256, 128, dropout=0.2)
        self.block3 = ResidualBlock(128, 64, dropout=0.1)
        self.head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x).squeeze(-1)


class NeuralModelV3(BaseModel):
    """Neural network v3 with residual blocks and Platt scaling calibration."""

    name = "neural_v3"
    version = "3.0"
    description = "PyTorch neural network v3 (residual blocks + Platt calibration)"

    def __init__(self, use_model_predictions=False, use_enhanced_features=False):
        if use_enhanced_features:
            from models.nn_features_enhanced import EnhancedFeatureComputer
            self.feature_computer = EnhancedFeatureComputer(
                use_model_predictions=use_model_predictions
            )
        else:
            self.feature_computer = FeatureComputer(
                use_model_predictions=use_model_predictions
            )
        self.input_size = self.feature_computer.get_num_features()
        self.model = BaseballNetV3(self.input_size)
        self.model.eval()
        self._loaded = False
        self._calibrator = None  # sklearn LogisticRegression for Platt scaling
        self._feature_mean = None
        self._feature_std = None

        if MODEL_PATH.exists():
            try:
                checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    saved_size = checkpoint.get('input_size', self.input_size)
                    if saved_size != self.input_size:
                        self.input_size = saved_size
                        self.model = BaseballNetV3(saved_size)
                        self.model.eval()
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self._feature_mean = checkpoint.get('feature_mean')
                    self._feature_std = checkpoint.get('feature_std')
                    # Load calibrator if saved
                    if 'calibrator' in checkpoint and checkpoint['calibrator'] is not None:
                        self._calibrator = pickle.loads(checkpoint['calibrator'])
                self._loaded = True
            except Exception as e:
                print(f"Warning: Could not load neural v3 model: {e}")

    def _calibrate(self, raw_prob):
        """Apply Platt scaling calibration if fitted, else return raw."""
        if self._calibrator is not None:
            # LogisticRegression expects log-odds as input
            import math
            logit = math.log(max(raw_prob, 1e-6) / max(1 - raw_prob, 1e-6))
            return float(self._calibrator.predict_proba([[logit]])[0, 1])
        return raw_prob

    def is_trained(self):
        return self._loaded

    def predict_game(self, home_team_id, away_team_id, neutral_site=False):
        if not self._loaded:
            return {
                'model': self.name,
                'home_win_probability': 0.5,
                'away_win_probability': 0.5,
                'projected_home_runs': 4.5,
                'projected_away_runs': 4.5,
                'projected_total': 9.0,
                'confidence': 'none (model not trained)',
                'run_line': self.calculate_run_line(4.5, 4.5),
            }

        features = self.feature_computer.compute_features(
            home_team_id, away_team_id, neutral_site=neutral_site
        )

        if len(features) > self.input_size:
            features = features[:self.input_size]
        elif len(features) < self.input_size:
            features = np.pad(features, (0, self.input_size - len(features)))

        if self._feature_mean is not None and self._feature_std is not None:
            features = (features - self._feature_mean) / (self._feature_std + 1e-8)

        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            raw_prob = self.model(x).item()

        prob = self._calibrate(raw_prob)
        prob = max(0.05, min(0.95, prob))

        avg_runs = 6.0
        run_diff = (prob - 0.5) * 6.0
        home_runs = max(avg_runs + run_diff / 2, 1.0)
        away_runs = max(avg_runs - run_diff / 2, 1.0)

        deviation = abs(prob - 0.5)
        confidence = 'high' if deviation > 0.25 else ('medium' if deviation > 0.15 else 'low')

        return {
            'model': self.name,
            'home_win_probability': round(prob, 4),
            'away_win_probability': round(1 - prob, 4),
            'projected_home_runs': round(home_runs, 2),
            'projected_away_runs': round(away_runs, 2),
            'projected_total': round(home_runs + away_runs, 2),
            'confidence': confidence,
            'run_line': self.calculate_run_line(home_runs, away_runs),
        }
