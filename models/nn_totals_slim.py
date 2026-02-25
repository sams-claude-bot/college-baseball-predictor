#!/usr/bin/env python3
"""
Slim Neural Network for Total Runs Regression — v3

Uses the same 58 v3 features as the slim win model.
Predicts total runs (home + away) as a regression target.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from models.base_model import BaseModel
from models.nn_features_slim import SlimFeatureComputer, ResidualBlock, NUM_FEATURES

MODEL_PATH = Path(__file__).parent.parent / "data" / "nn_slim_totals.pt"
FINETUNED_PATH = Path(__file__).parent.parent / "data" / "nn_slim_totals_finetuned.pt"


class TotalsNet(nn.Module):
    """Regression network for predicting total runs — v3 with GELU."""
    def __init__(self, input_size=NUM_FEATURES):
        super().__init__()
        self.input_size = input_size
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
            ResidualBlock(128, dropout=0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            # No activation — raw regression output
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SlimTotalsModel(BaseModel):
    """Neural network totals model implementing BaseModel interface."""

    name = "nn_slim_totals"
    version = "3.0"
    description = "Slim NN regression for total runs (58 features, v3 w/ NCAA stats)"

    def __init__(self):
        self.feature_computer = SlimFeatureComputer()
        self.input_size = self.feature_computer.get_num_features()
        self.model = TotalsNet(self.input_size)
        self.model.eval()
        self._loaded = False
        self._feature_mean = None
        self._feature_std = None

        load_path = FINETUNED_PATH if FINETUNED_PATH.exists() else MODEL_PATH
        if load_path.exists():
            try:
                checkpoint = torch.load(load_path, map_location='cpu', weights_only=False)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    saved_size = checkpoint.get('input_size', self.input_size)
                    if saved_size != self.input_size:
                        self.input_size = saved_size
                        self.model = TotalsNet(saved_size)
                        self.model.eval()
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self._feature_mean = checkpoint.get('feature_mean')
                    self._feature_std = checkpoint.get('feature_std')
                    self._target_mean = checkpoint.get('target_mean', 0.0)
                    self._target_std = checkpoint.get('target_std', 1.0)
                else:
                    self.model.load_state_dict(checkpoint)
                self._loaded = True
            except Exception as e:
                print(f"Warning: Could not load slim totals model: {e}")

    def is_trained(self):
        return self._loaded

    def predict_game(self, home_team_id, away_team_id, neutral_site=False,
                     game_id=None, weather_data=None):
        if not self._loaded:
            return {
                'model': self.name,
                'projected_total': 9.0,
                'projected_home_runs': 4.5,
                'projected_away_runs': 4.5,
                'confidence': 'none (model not trained)',
            }

        features = self.feature_computer.compute_features(
            home_team_id, away_team_id,
            neutral_site=neutral_site, game_id=game_id,
            weather_data=weather_data
        )

        if len(features) > self.input_size:
            features = features[:self.input_size]
        elif len(features) < self.input_size:
            features = np.pad(features, (0, self.input_size - len(features)))

        if self._feature_mean is not None and self._feature_std is not None:
            features = (features - self._feature_mean) / (self._feature_std + 1e-8)

        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            raw_total = self.model(x).item()

        # Denormalize if target was normalized during training
        if hasattr(self, '_target_mean') and hasattr(self, '_target_std'):
            projected_total = raw_total * self._target_std + self._target_mean
        else:
            projected_total = raw_total

        projected_total = max(projected_total, 2.0)

        # Split into home/away based on RPG features
        # In v3: indices 6 and 30 are home/away RPG (15 base + 9 NCAA = 24 per team)
        raw_features = self.feature_computer.compute_features(
            home_team_id, away_team_id, neutral_site=neutral_site
        )
        home_rpg_raw = raw_features[6]   # home_runs_per_game
        away_rpg_raw = raw_features[30]  # away_runs_per_game (15+9+6=30)
        total_rpg = home_rpg_raw + away_rpg_raw
        if total_rpg > 0:
            home_ratio = home_rpg_raw / total_rpg
        else:
            home_ratio = 0.5

        home_runs = projected_total * home_ratio
        away_runs = projected_total * (1 - home_ratio)

        return {
            'model': self.name,
            'projected_total': round(projected_total, 1),
            'projected_home_runs': round(home_runs, 1),
            'projected_away_runs': round(away_runs, 1),
            'confidence': 'high' if abs(projected_total - 9.0) > 3 else
                         'medium' if abs(projected_total - 9.0) > 1.5 else 'low',
        }
