#!/usr/bin/env python3
"""
Neural Network Prediction Model

PyTorch-based model that combines team stats, Elo ratings, and predictions
from other models to predict game outcomes. Implements BaseModel interface
for ensemble integration.

TRAINING DATA: Current season games only (when trained)
RATIONALE: Win probability depends heavily on current team strength and
roster composition. This is a moneyline model that should be trained on
current season data only.

NOTE: This model is currently excluded from the ensemble and tracked
independently. See ensemble_model.py for details.
"""

import sys
import math
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from models.base_model import BaseModel
from models.nn_features_slim import SlimFeatureComputer, SlimBaseballNet, NUM_FEATURES as SLIM_NUM_FEATURES

# Paths — now uses slim model trained on real historical features only
MODEL_PATH = Path(__file__).parent.parent / "data" / "nn_slim_model.pt"
FINETUNED_PATH = Path(__file__).parent.parent / "data" / "nn_slim_model_finetuned.pt"


class BaseballNet(nn.Module):
    """
    Neural network for predicting home win probability.

    Architecture:
        Input (~40-50 features)
        → Linear(input, 128) + BatchNorm + ReLU + Dropout(0.3)
        → Linear(128, 64) + BatchNorm + ReLU + Dropout(0.2)
        → Linear(64, 32) + ReLU + Dropout(0.1)
        → Linear(32, 1) + Sigmoid
    """

    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.net = nn.Sequential(
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

            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class NeuralModel(BaseModel):
    """Neural network model implementing the BaseModel interface."""

    name = "neural"
    version = "1.0"
    description = "PyTorch neural network (model stacking ensemble)"

    def __init__(self, use_model_predictions=False):
        self.feature_computer = SlimFeatureComputer()
        self.input_size = self.feature_computer.get_num_features()
        self.model = SlimBaseballNet(self.input_size)
        self.model.eval()
        self._loaded = False

        # Try to load saved weights (prefer finetuned if available)
        _load_path = FINETUNED_PATH if FINETUNED_PATH.exists() else MODEL_PATH
        if _load_path.exists():
            try:
                checkpoint = torch.load(_load_path, map_location='cpu',
                                        weights_only=False)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    # Rebuild model with saved input_size if different
                    saved_size = checkpoint.get('input_size', self.input_size)
                    if saved_size != self.input_size:
                        self.input_size = saved_size
                        self.model = SlimBaseballNet(saved_size)
                        self.model.eval()
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self._feature_mean = checkpoint.get('feature_mean')
                    self._feature_std = checkpoint.get('feature_std')
                else:
                    self.model.load_state_dict(checkpoint)
                    self._feature_mean = None
                    self._feature_std = None
                self._loaded = True
            except Exception as e:
                print(f"Warning: Could not load neural model weights: {e}")
                self._feature_mean = None
                self._feature_std = None
        else:
            self._feature_mean = None
            self._feature_std = None

    # Calibration parameters (updated periodically from prediction history)
    # Stretches probabilities away from 0.5 to match observed win rates
    CALIBRATION_STRENGTH = 1.5  # >1 stretches, <1 compresses, 1.0 = no change

    def _calibrate(self, prob):
        """
        Apply calibration to raw model output.
        
        Uses power-based stretching: maps 0.5 to 0.5, but stretches
        predictions toward 0 and 1 based on CALIBRATION_STRENGTH.
        """
        if self.CALIBRATION_STRENGTH == 1.0:
            return prob
        
        # Center around 0.5, apply power scaling, re-center
        centered = prob - 0.5  # range [-0.5, 0.5]
        sign = 1 if centered >= 0 else -1
        magnitude = abs(centered) * 2  # range [0, 1]
        
        # Power < 1 stretches toward extremes
        stretched = magnitude ** (1.0 / self.CALIBRATION_STRENGTH)
        
        return 0.5 + sign * stretched / 2

    def is_trained(self):
        """Check if model weights have been loaded."""
        return self._loaded

    def predict_game(self, home_team_id, away_team_id, neutral_site=False):
        """
        Predict a single game.

        Returns dict with:
        - home_win_probability: float 0-1
        - projected_home_runs: float
        - projected_away_runs: float
        - model: str
        - confidence: str
        """
        if not self._loaded:
            # Fallback to 50/50 if not trained
            return {
                'model': self.name,
                'home_win_probability': 0.5,
                'away_win_probability': 0.5,
                'projected_home_runs': None,
                'projected_away_runs': None,
                'projected_total': 9.0,
                'confidence': 'none (model not trained)',
                'run_line': self.calculate_run_line(4.5, 4.5),
            }

        features = self.feature_computer.compute_features(
            home_team_id, away_team_id, neutral_site=neutral_site
        )

        # Truncate or pad features to match model's expected input size
        if len(features) > self.input_size:
            features = features[:self.input_size]
        elif len(features) < self.input_size:
            features = np.pad(features, (0, self.input_size - len(features)))

        # Normalize if we have training stats
        if self._feature_mean is not None and self._feature_std is not None:
            features = (features - self._feature_mean) / (self._feature_std + 1e-8)

        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            raw_prob = self.model(x).item()

        # Calibration: stretch predictions toward extremes
        # Based on empirical observation that model is under-confident
        # Applies sigmoid-based stretching centered at 0.5
        prob = self._calibrate(raw_prob)
        prob = max(0.05, min(0.95, prob))

        # Estimate runs from probability
        # College baseball averages ~6 runs per team (12 total)
        avg_runs = 6.0
        run_diff = (prob - 0.5) * 6.0  # scale prob diff to run diff
        home_runs = avg_runs + run_diff / 2
        away_runs = avg_runs - run_diff / 2
        # Floor at 1 run
        home_runs = max(home_runs, 1.0)
        away_runs = max(away_runs, 1.0)

        # Confidence based on how far from 0.5
        deviation = abs(prob - 0.5)
        if deviation > 0.25:
            confidence = 'high'
        elif deviation > 0.15:
            confidence = 'medium'
        else:
            confidence = 'low'

        return {
            'model': self.name,
            'home_win_probability': round(prob, 4),
            'away_win_probability': round(1 - prob, 4),
            'projected_home_runs': None,
            'projected_away_runs': None,
            'projected_total': round(home_runs + away_runs, 2),
            'confidence': confidence,
            'run_line': self.calculate_run_line(home_runs, away_runs),
        }


class Trainer:
    """Handles training the BaseballNet model."""

    def __init__(self, input_size, lr=0.001, batch_size=64, epochs=100,
                 patience=10, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BaseballNet(input_size).to(self.device)
        self.criterion = nn.BCELoss()
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

    def normalize_features(self, X_train):
        """Compute and store normalization stats from training data."""
        self.feature_mean = X_train.mean(axis=0)
        self.feature_std = X_train.std(axis=0)
        # Avoid division by zero for constant features
        self.feature_std[self.feature_std < 1e-8] = 1.0
        return (X_train - self.feature_mean) / self.feature_std

    def apply_normalization(self, X):
        """Apply stored normalization to new data."""
        if self.feature_mean is None:
            return X
        return (X - self.feature_mean) / self.feature_std

    def train(self, X_train, y_train, X_val, y_val):
        """
        Train the model.

        Args:
            X_train: numpy array (N, features)
            y_train: numpy array (N,) binary labels
            X_val: numpy array (M, features)
            y_val: numpy array (M,) binary labels

        Returns:
            dict with training history
        """
        # Normalize
        X_train = self.normalize_features(X_train)
        X_val = self.apply_normalization(X_val)

        # Convert to tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                preds = self.model(X_batch)
                loss = self.criterion(preds, y_batch)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

            avg_train_loss = sum(train_losses) / len(train_losses)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(X_val_t)
                val_loss = self.criterion(val_preds, y_val_t).item()
                val_acc = ((val_preds > 0.5).float() == y_val_t).float().mean().item()

            self.scheduler.step(val_loss)

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(f"Epoch {epoch+1}/{self.epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint()
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        self._load_best_checkpoint()
        return history

    def _save_checkpoint(self):
        """Save best model weights."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std,
            'input_size': self.model.input_size,
        }, MODEL_PATH)

    def _load_best_checkpoint(self):
        """Load best saved weights."""
        if MODEL_PATH.exists():
            checkpoint = torch.load(MODEL_PATH, map_location=self.device,
                                    weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])

    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set.

        Returns dict with accuracy, log_loss, calibration metrics.
        """
        X_test = self.apply_normalization(X_test)
        X_t = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_test, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_t)
            loss = self.criterion(preds, y_t).item()
            acc = ((preds > 0.5).float() == y_t).float().mean().item()
            preds_np = preds.cpu().numpy()

        # Calibration: bin predictions and compare to actual rates
        bins = np.linspace(0, 1, 11)
        calibration = []
        for i in range(len(bins) - 1):
            mask = (preds_np >= bins[i]) & (preds_np < bins[i+1])
            if mask.sum() > 0:
                predicted_avg = preds_np[mask].mean()
                actual_avg = y_test[mask].mean()
                calibration.append({
                    'bin': f'{bins[i]:.1f}-{bins[i+1]:.1f}',
                    'count': int(mask.sum()),
                    'predicted': round(float(predicted_avg), 3),
                    'actual': round(float(actual_avg), 3),
                })

        return {
            'accuracy': round(acc, 4),
            'log_loss': round(loss, 4),
            'n_samples': len(y_test),
            'calibration': calibration,
        }
