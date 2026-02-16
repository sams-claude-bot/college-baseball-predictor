#!/usr/bin/env python3
"""
Day-of-Week Aware Neural Network Totals Model

Incorporates day-of-week as a learned embedding to capture scheduling
patterns in college baseball scoring (Friday aces vs Sunday bullpen days).

Architecture: Learned 4-dim DoW embedding concatenated with standard features,
fed through the same architecture as the regular totals NN.
"""

import sys
import math
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from models.base_model import BaseModel
from models.nn_features import FeatureComputer

MODEL_PATH = Path(__file__).parent.parent / "data" / "nn_dow_totals_model.pt"
FINETUNED_PATH = Path(__file__).parent.parent / "data" / "nn_dow_totals_model_finetuned.pt"


class DoWTotalsNet(nn.Module):
    """
    Neural network for predicting total runs with day-of-week awareness.

    Uses a learned 4-dimensional embedding for day of week (0=Sunday..6=Saturday)
    concatenated with standard game features.
    """

    def __init__(self, base_input_size, dow_embed_dim=4):
        super().__init__()
        self.base_input_size = base_input_size
        self.dow_embed_dim = dow_embed_dim
        self.input_size = base_input_size  # stored for checkpoint compat

        self.dow_embedding = nn.Embedding(7, dow_embed_dim)

        total_input = base_input_size + dow_embed_dim

        self.shared = nn.Sequential(
            nn.Linear(total_input, 128),
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
        self.mean_head = nn.Linear(32, 1)
        self.logvar_head = nn.Linear(32, 1)

    def forward(self, x, dow):
        """
        Args:
            x: (batch, base_input_size) standard features
            dow: (batch,) int tensor, day of week 0=Sunday..6=Saturday
        """
        dow_emb = self.dow_embedding(dow)  # (batch, dow_embed_dim)
        combined = torch.cat([x, dow_emb], dim=1)
        h = self.shared(combined)
        mean = torch.relu(self.mean_head(h)).squeeze(-1)
        logvar = self.logvar_head(h).squeeze(-1)
        return mean, logvar


class NNDoWTotalsModel(BaseModel):
    """Day-of-week aware neural network totals model implementing BaseModel interface."""

    name = "nn_dow_totals"
    version = "1.0"
    description = "PyTorch NN for run totals with day-of-week embedding"

    def __init__(self, use_model_predictions=False):
        self.feature_computer = FeatureComputer(
            use_model_predictions=use_model_predictions
        )
        self.base_input_size = self.feature_computer.get_num_features()
        self.model = DoWTotalsNet(self.base_input_size)
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
                    saved_size = checkpoint.get('input_size', self.base_input_size)
                    dow_embed_dim = checkpoint.get('dow_embed_dim', 4)
                    if saved_size != self.base_input_size:
                        self.base_input_size = saved_size
                    self.model = DoWTotalsNet(saved_size, dow_embed_dim)
                    self.model.eval()
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self._feature_mean = checkpoint.get('feature_mean')
                    self._feature_std = checkpoint.get('feature_std')
                self._loaded = True
            except Exception as e:
                print(f"Warning: Could not load nn_dow_totals weights: {e}")

    def is_trained(self):
        return self._loaded

    def predict_game(self, home_team_id, away_team_id, neutral_site=False,
                     over_under_line=None, game_date=None):
        """
        Predict total runs for a game, incorporating day-of-week.

        Args:
            game_date: date string 'YYYY-MM-DD' or datetime. If None, uses today.
        """
        # Determine day of week
        if game_date is None:
            dow = datetime.now().weekday()  # Monday=0..Sunday=6
            # Convert to strftime %w style: Sunday=0..Saturday=6
            dow = (dow + 1) % 7
        elif isinstance(game_date, str):
            dt = datetime.strptime(game_date, '%Y-%m-%d')
            dow = int(dt.strftime('%w'))
        elif isinstance(game_date, datetime):
            dow = int(game_date.strftime('%w'))
        else:
            dow = 5  # default Friday

        if not self._loaded:
            # Fallback: use day-of-week averages from historical data
            dow_avgs = {0: 13.81, 1: 12.78, 2: 13.70, 3: 13.58,
                        4: 12.65, 5: 12.36, 6: 13.18}
            avg = dow_avgs.get(dow, 13.0)
            return {
                'model': self.name,
                'projected_total': avg,
                'over_prob': 0.5,
                'under_prob': 0.5,
                'home_win_probability': 0.5,
                'away_win_probability': 0.5,
                'projected_home_runs': avg * 0.52,
                'projected_away_runs': avg * 0.48,
                'confidence': 'none (model not trained)',
                'day_of_week': dow,
            }

        features = self.feature_computer.compute_features(
            home_team_id, away_team_id, neutral_site=neutral_site
        )

        if self._feature_mean is not None and self._feature_std is not None:
            features = (features - self._feature_mean) / (self._feature_std + 1e-8)

        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            dow_t = torch.tensor([dow], dtype=torch.long)
            mean, logvar = self.model(x, dow_t)
            projected_total = mean.item()
            variance = torch.exp(logvar).item()

        std = math.sqrt(max(variance, 0.01))
        projected_total = max(projected_total, 0.0)

        if over_under_line is not None and over_under_line > 0:
            z = (over_under_line - projected_total) / std
            under_prob = 0.5 * (1 + math.erf(z / math.sqrt(2)))
            over_prob = 1.0 - under_prob
        else:
            over_prob = 0.5
            under_prob = 0.5

        home_runs = projected_total * 0.52
        away_runs = projected_total * 0.48

        return {
            'model': self.name,
            'projected_total': round(projected_total, 2),
            'over_prob': round(max(0.01, min(0.99, over_prob)), 4),
            'under_prob': round(max(0.01, min(0.99, under_prob)), 4),
            'std': round(std, 2),
            'home_win_probability': 0.5,
            'away_win_probability': 0.5,
            'projected_home_runs': round(home_runs, 2),
            'projected_away_runs': round(away_runs, 2),
            'confidence': 'high' if std < 3.0 else 'medium' if std < 5.0 else 'low',
            'day_of_week': dow,
        }


class DoWTotalsTrainer:
    """Handles training the DoWTotalsNet model."""

    def __init__(self, input_size, dow_embed_dim=4, lr=0.001, batch_size=64,
                 epochs=100, patience=15, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DoWTotalsNet(input_size, dow_embed_dim).to(self.device)
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
        self.dow_embed_dim = dow_embed_dim

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

    def train(self, X_train, dow_train, y_train, X_val, dow_val, y_val):
        """
        Train with features + day-of-week arrays.

        Args:
            X_train/X_val: (N, features) numpy arrays
            dow_train/dow_val: (N,) int arrays, day of week 0-6
            y_train/y_val: (N,) float arrays, total runs
        """
        X_train = self.normalize_features(X_train)
        X_val = self.apply_normalization(X_val)

        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        dow_train_t = torch.tensor(dow_train, dtype=torch.long).to(self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        dow_val_t = torch.tensor(dow_val, dtype=torch.long).to(self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        train_dataset = torch.utils.data.TensorDataset(X_train_t, dow_train_t, y_train_t)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        history = {'train_loss': [], 'val_loss': [], 'val_mae': []}

        for epoch in range(self.epochs):
            self.model.train()
            train_losses = []
            for X_batch, dow_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                mean, logvar = self.model(X_batch, dow_batch)
                loss = self.gaussian_nll_loss(mean, logvar, y_batch)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

            avg_train_loss = sum(train_losses) / len(train_losses)

            self.model.eval()
            with torch.no_grad():
                val_mean, val_logvar = self.model(X_val_t, dow_val_t)
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
            'dow_embed_dim': self.dow_embed_dim,
        }, MODEL_PATH)

    def _load_best_checkpoint(self):
        if MODEL_PATH.exists():
            checkpoint = torch.load(MODEL_PATH, map_location=self.device,
                                    weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])

    def evaluate(self, X_test, dow_test, y_test, over_under_lines=None):
        """Evaluate with MAE, RMSE, and over/under accuracy."""
        X_test = self.apply_normalization(X_test)
        X_t = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        dow_t = torch.tensor(dow_test, dtype=torch.long).to(self.device)

        self.model.eval()
        with torch.no_grad():
            mean, logvar = self.model(X_t, dow_t)
            preds = mean.cpu().numpy()

        errors = preds - y_test
        mae = np.abs(errors).mean()
        rmse = np.sqrt((errors ** 2).mean())

        result = {
            'mae': round(float(mae), 3),
            'rmse': round(float(rmse), 3),
            'mean_predicted': round(float(preds.mean()), 2),
            'mean_actual': round(float(y_test.mean()), 2),
            'n_samples': len(y_test),
            'predictions': preds,
        }

        if over_under_lines is not None:
            correct = 0
            total = 0
            for i in range(len(y_test)):
                line = over_under_lines[i]
                if line and line > 0:
                    pred_over = preds[i] > line
                    actual_over = y_test[i] > line
                    if pred_over == actual_over:
                        correct += 1
                    total += 1
            if total > 0:
                result['ou_accuracy'] = round(correct / total, 4)
                result['ou_correct'] = correct
                result['ou_total'] = total

        return result

    def evaluate_by_dow(self, X_test, dow_test, y_test, over_under_lines=None):
        """Evaluate broken down by day of week."""
        X_test_norm = self.apply_normalization(X_test)
        X_t = torch.tensor(X_test_norm, dtype=torch.float32).to(self.device)
        dow_t = torch.tensor(dow_test, dtype=torch.long).to(self.device)

        self.model.eval()
        with torch.no_grad():
            mean, _ = self.model(X_t, dow_t)
            preds = mean.cpu().numpy()

        dow_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday',
                     'Thursday', 'Friday', 'Saturday']
        results = {}

        for d in range(7):
            mask = dow_test == d
            if mask.sum() == 0:
                continue
            d_preds = preds[mask]
            d_actual = y_test[mask]
            d_errors = d_preds - d_actual

            entry = {
                'n': int(mask.sum()),
                'mae': round(float(np.abs(d_errors).mean()), 3),
                'rmse': round(float(np.sqrt((d_errors ** 2).mean())), 3),
                'mean_pred': round(float(d_preds.mean()), 2),
                'mean_actual': round(float(d_actual.mean()), 2),
            }

            if over_under_lines is not None:
                d_lines = over_under_lines[mask]
                correct = 0
                total = 0
                for i in range(len(d_actual)):
                    if d_lines[i] and d_lines[i] > 0:
                        if (d_preds[i] > d_lines[i]) == (d_actual[i] > d_lines[i]):
                            correct += 1
                        total += 1
                if total > 0:
                    entry['ou_accuracy'] = round(correct / total, 4)
                    entry['ou_n'] = total

            results[dow_names[d]] = entry

        return results
