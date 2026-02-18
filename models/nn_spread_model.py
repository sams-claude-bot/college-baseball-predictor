#!/usr/bin/env python3
"""
Neural Network Spread Model

Predicts margin of victory (home_score - away_score) for run line betting.
Uses same feature pipeline as win probability NN.
Outputs projected margin + probability of covering -1.5.

TRAINING DATA: Current season games only (2026, ~300-400+ games)
RATIONALE: Win margins depend heavily on current team strength, roster
composition, and season-specific performance. Historical data from
different teams/rosters would introduce noise. This model improves
as the current season progresses and more games are played.
"""

import sys
import math
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from models.base_model import BaseModel
from models.nn_features import FeatureComputer

MODEL_PATH = Path(__file__).parent.parent / "data" / "nn_spread_model.pt"
FINETUNED_PATH = Path(__file__).parent.parent / "data" / "nn_spread_model_finetuned.pt"


class SpreadNet(nn.Module):
    """
    Neural network for predicting game margin.

    Architecture matches TotalsNet but output has no ReLU (margin can be negative).
    Outputs mean margin and log-variance for uncertainty.
    """

    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.shared = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.mean_head = nn.Linear(32, 1)  # no activation - margin can be negative
        self.logvar_head = nn.Linear(32, 1)

    def forward(self, x):
        h = self.shared(x)
        mean = self.mean_head(h).squeeze(-1)
        logvar = self.logvar_head(h).squeeze(-1)
        return mean, logvar


class NNSpreadModel(BaseModel):
    """Neural network spread model implementing BaseModel interface."""

    name = "nn_spread"
    version = "1.0"
    description = "PyTorch neural network for run line spreads"

    def __init__(self, use_model_predictions=False):
        self.feature_computer = FeatureComputer(
            use_model_predictions=use_model_predictions
        )
        self.input_size = self.feature_computer.get_num_features()
        self.model = SpreadNet(self.input_size)
        self.model.eval()
        self._loaded = False
        self._feature_mean = None
        self._feature_std = None

        # Prefer finetuned weights if available, fall back to base
        _load_path = FINETUNED_PATH if FINETUNED_PATH.exists() else MODEL_PATH
        if _load_path.exists():
            try:
                checkpoint = torch.load(_load_path, map_location='cpu',
                                        weights_only=False)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    saved_size = checkpoint.get('input_size', self.input_size)
                    if saved_size != self.input_size:
                        self.input_size = saved_size
                        self.model = SpreadNet(saved_size)
                        self.model.eval()
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self._feature_mean = checkpoint.get('feature_mean')
                    self._feature_std = checkpoint.get('feature_std')
                self._loaded = True
            except Exception as e:
                print(f"Warning: Could not load nn_spread weights: {e}")

    def is_trained(self):
        if not self._loaded:
            return False
        # Check feature dimension compatibility
        if self._feature_mean is not None:
            from models.nn_features import FeatureComputer
            fc = FeatureComputer(use_model_predictions=False)
            expected_features = fc.get_num_features()
            model_features = len(self._feature_mean)
            if expected_features != model_features:
                return False
        return True

    def predict_game(self, home_team_id, away_team_id, neutral_site=False,
                     spread=-1.5):
        """
        Predict margin for a game.

        Returns dict with:
        - projected_margin: float (positive = home wins by that much)
        - cover_prob: float (prob of home covering the spread)
        - projected_home_runs / projected_away_runs
        - home_win_probability: derived from margin
        """
        if not self._loaded:
            return {
                'model': self.name,
                'projected_margin': 0.0,
                'cover_prob': 0.5,
                'home_win_probability': 0.5,
                'away_win_probability': 0.5,
                'projected_home_runs': 4.5,
                'projected_away_runs': 4.5,
                'projected_total': 9.0,
                'confidence': 'none (model not trained)',
            }

        features = self.feature_computer.compute_features(
            home_team_id, away_team_id, neutral_site=neutral_site
        )

        # Handle NaN/inf
        import numpy as np
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Truncate or pad features to match model's expected input size
        if len(features) > self.input_size:
            features = features[:self.input_size]
        elif len(features) < self.input_size:
            features = np.pad(features, (0, self.input_size - len(features)))

        if self._feature_mean is not None and self._feature_std is not None:
            features = (features - self._feature_mean) / (self._feature_std + 1e-8)
            # Clip to prevent extreme predictions from distribution mismatch
            features = np.clip(features, -4.0, 4.0)

        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            mean, logvar = self.model(x)
            projected_margin = mean.item()
            variance = torch.exp(logvar).item()

        std = math.sqrt(max(variance, 0.01))

        # Probability of home covering spread (winning by more than |spread|)
        # P(margin > spread) = P(Z > (spread - mean) / std)
        z = (spread - projected_margin) / std
        cover_prob = 1.0 - 0.5 * (1 + math.erf(z / math.sqrt(2)))

        # Estimate individual team runs
        # Use average total of ~9 runs as baseline, split by margin
        avg_total = 9.0
        home_runs = (avg_total + projected_margin) / 2.0
        away_runs = (avg_total - projected_margin) / 2.0
        home_runs = max(home_runs, 0.5)
        away_runs = max(away_runs, 0.5)

        # Win probability from margin
        z_win = -projected_margin / std
        home_win_prob = 1.0 - 0.5 * (1 + math.erf(z_win / math.sqrt(2)))
        home_win_prob = max(0.05, min(0.95, home_win_prob))

        return {
            'model': self.name,
            'projected_margin': round(projected_margin, 2),
            'cover_prob': round(max(0.01, min(0.99, cover_prob)), 4),
            'std': round(std, 2),
            'home_win_probability': round(home_win_prob, 4),
            'away_win_probability': round(1 - home_win_prob, 4),
            'projected_home_runs': round(home_runs, 2),
            'projected_away_runs': round(away_runs, 2),
            'projected_total': round(home_runs + away_runs, 2),
            'confidence': 'high' if std < 3.0 else 'medium' if std < 5.0 else 'low',
        }


class SpreadTrainer:
    """Handles training the SpreadNet model."""

    def __init__(self, input_size, lr=0.001, batch_size=64, epochs=100,
                 patience=15, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SpreadNet(input_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.feature_mean = None
        self.feature_std = None
        self.input_size = input_size

    def gaussian_nll_loss(self, mean, logvar, target):
        var = torch.exp(logvar) + 1e-6
        return 0.5 * torch.mean(torch.log(var) + (target - mean) ** 2 / var)

    def normalize_features(self, X_train):
        self.feature_mean = X_train.mean(axis=0)
        self.feature_std = X_train.std(axis=0)
        self.feature_std[self.feature_std < 1e-8] = 1.0
        return (X_train - self.feature_mean) / self.feature_std

    def apply_normalization(self, X):
        if self.feature_mean is None:
            return X
        return (X - self.feature_mean) / self.feature_std

    def train(self, X_train, y_train, X_val, y_val):
        X_train = self.normalize_features(X_train)
        X_val = self.apply_normalization(X_val)

        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        history = {'train_loss': [], 'val_loss': [], 'val_mae': []}

        for epoch in range(self.epochs):
            self.model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                mean, logvar = self.model(X_batch)
                loss = self.gaussian_nll_loss(mean, logvar, y_batch)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

            avg_train_loss = sum(train_losses) / len(train_losses)

            self.model.eval()
            with torch.no_grad():
                val_mean, val_logvar = self.model(X_val_t)
                val_loss = self.gaussian_nll_loss(val_mean, val_logvar, y_val_t).item()
                val_mae = torch.abs(val_mean - y_val_t).mean().item()

            self.scheduler.step(val_loss)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['val_mae'].append(val_mae)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{self.epochs} | "
                      f"Train Loss: {avg_train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val MAE: {val_mae:.2f} runs | "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint()
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        self._load_best_checkpoint()
        return history

    def _save_checkpoint(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std,
            'input_size': self.input_size,
        }, MODEL_PATH)

    def _load_best_checkpoint(self):
        if MODEL_PATH.exists():
            checkpoint = torch.load(MODEL_PATH, map_location=self.device,
                                    weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])

    def evaluate(self, X_test, y_test, spread=-1.5):
        """Evaluate with MAE and run line accuracy."""
        X_test = self.apply_normalization(X_test)
        X_t = torch.tensor(X_test, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            mean, logvar = self.model(X_t)
            preds = mean.cpu().numpy()

        errors = preds - y_test
        mae = np.abs(errors).mean()
        rmse = np.sqrt((errors ** 2).mean())

        # Run line accuracy: did we correctly predict covering -1.5?
        pred_covers = preds > spread  # home covers if margin > -1.5 (i.e., > spread)
        actual_covers = y_test > spread
        rl_correct = (pred_covers == actual_covers).sum()

        return {
            'mae': round(float(mae), 3),
            'rmse': round(float(rmse), 3),
            'mean_predicted': round(float(preds.mean()), 2),
            'mean_actual': round(float(y_test.mean()), 2),
            'n_samples': len(y_test),
            'rl_accuracy': round(float(rl_correct / len(y_test)), 4),
            'rl_correct': int(rl_correct),
            'rl_total': len(y_test),
        }
