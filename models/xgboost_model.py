#!/usr/bin/env python3
"""
XGBoost Gradient Boosting Models

Provides gradient boosting models for:
- Moneyline prediction (classification: home win/loss)
- Totals prediction (regression: total runs)
- Spread prediction (regression: margin)

Uses GPU acceleration with tree_method='hist' and device='cuda'.

Training uses recency weighting via exponential decay:
    weight = max(0.1, exp(-0.002 * days_ago))
This weights recent games more heavily while still learning from
historical data. Games from today have weight ~1.0, games from
2 years ago have weight ~0.2.
"""

import sys
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from models.base_model import BaseModel
from models.nn_features import FeatureComputer

DATA_DIR = Path(__file__).parent.parent / "data"
MONEYLINE_PATH = DATA_DIR / "xgb_moneyline.pkl"
TOTALS_PATH = DATA_DIR / "xgb_totals.pkl"
SPREAD_PATH = DATA_DIR / "xgb_spread.pkl"


class XGBMoneylineModel(BaseModel):
    """XGBoost classifier for moneyline (win/loss) predictions."""

    name = "xgb_moneyline"
    version = "1.0"
    description = "XGBoost gradient boosting for moneyline predictions"

    def __init__(self, use_model_predictions=False):
        self.feature_computer = FeatureComputer(
            use_model_predictions=use_model_predictions
        )
        self.model = None
        self.feature_mean = None
        self.feature_std = None
        self._loaded = False
        self._load_model()

    def _load_model(self):
        if not XGB_AVAILABLE:
            return
        if MONEYLINE_PATH.exists():
            try:
                with open(MONEYLINE_PATH, 'rb') as f:
                    checkpoint = pickle.load(f)
                self.model = checkpoint['model']
                self.feature_mean = checkpoint.get('feature_mean')
                self.feature_std = checkpoint.get('feature_std')
                self._loaded = True
            except Exception as e:
                print(f"Warning: Could not load XGBoost moneyline model: {e}")

    def is_trained(self):
        return self._loaded and self.model is not None

    def predict_game(self, home_team_id, away_team_id, neutral_site=False, **kwargs):
        """Predict home win probability."""
        if not self.is_trained():
            return {
                'model': self.name,
                'home_win_probability': 0.5,
                'away_win_probability': 0.5,
                'projected_home_runs': 4.5,
                'projected_away_runs': 4.5,
                'confidence': 'none (model not trained)',
            }

        features = self.feature_computer.compute_features(
            home_team_id, away_team_id, neutral_site=neutral_site
        )
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        if self.feature_mean is not None and self.feature_std is not None:
            # Handle feature size mismatch (new features added but model not retrained)
            saved_size = len(self.feature_mean)
            if len(features) > saved_size:
                features = features[:saved_size]
            elif len(features) < saved_size:
                features = np.pad(features, (0, saved_size - len(features)))
            features = (features - self.feature_mean) / (self.feature_std + 1e-8)
            features = np.clip(features, -5.0, 5.0)

        X = features.reshape(1, -1)
        home_prob = float(self.model.predict_proba(X)[0, 1])
        home_prob = max(0.05, min(0.95, home_prob))

        return {
            'model': self.name,
            'home_win_probability': round(home_prob, 4),
            'away_win_probability': round(1 - home_prob, 4),
            'projected_home_runs': 4.5,
            'projected_away_runs': 4.5,
            'confidence': 'high' if abs(home_prob - 0.5) > 0.2 else 'medium',
        }


class XGBTotalsModel(BaseModel):
    """XGBoost regressor for total runs predictions."""

    name = "xgb_totals"
    version = "1.0"
    description = "XGBoost gradient boosting for run totals (over/under)"

    def __init__(self, use_model_predictions=False):
        self.feature_computer = FeatureComputer(
            use_model_predictions=use_model_predictions
        )
        self.model = None
        self.feature_mean = None
        self.feature_std = None
        self._loaded = False
        self._load_model()

    def _load_model(self):
        if not XGB_AVAILABLE:
            return
        if TOTALS_PATH.exists():
            try:
                with open(TOTALS_PATH, 'rb') as f:
                    checkpoint = pickle.load(f)
                self.model = checkpoint['model']
                self.feature_mean = checkpoint.get('feature_mean')
                self.feature_std = checkpoint.get('feature_std')
                self._loaded = True
            except Exception as e:
                print(f"Warning: Could not load XGBoost totals model: {e}")

    def is_trained(self):
        return self._loaded and self.model is not None

    def predict_game(self, home_team_id, away_team_id, neutral_site=False,
                     over_under_line=None, **kwargs):
        """Predict total runs."""
        if not self.is_trained():
            return {
                'model': self.name,
                'projected_total': 9.0,
                'over_prob': 0.5,
                'under_prob': 0.5,
                'home_win_probability': 0.5,
                'away_win_probability': 0.5,
                'projected_home_runs': 4.5,
                'projected_away_runs': 4.5,
                'confidence': 'none (model not trained)',
            }

        features = self.feature_computer.compute_features(
            home_team_id, away_team_id, neutral_site=neutral_site
        )
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        if self.feature_mean is not None and self.feature_std is not None:
            # Handle feature size mismatch (new features added but model not retrained)
            saved_size = len(self.feature_mean)
            if len(features) > saved_size:
                features = features[:saved_size]
            elif len(features) < saved_size:
                features = np.pad(features, (0, saved_size - len(features)))
            features = (features - self.feature_mean) / (self.feature_std + 1e-8)
            features = np.clip(features, -5.0, 5.0)

        X = features.reshape(1, -1)
        projected_total = float(self.model.predict(X)[0])
        projected_total = max(0.0, projected_total)

        # Estimate over/under probability using model's variance estimate
        # XGBoost doesn't directly output variance, so use a fixed std based on training
        std = 3.5  # Typical std for college baseball totals
        
        if over_under_line is not None and over_under_line > 0:
            import math
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
            'home_win_probability': 0.5,
            'away_win_probability': 0.5,
            'projected_home_runs': round(home_runs, 2),
            'projected_away_runs': round(away_runs, 2),
            'confidence': 'high' if std < 3.0 else 'medium',
        }


class XGBSpreadModel(BaseModel):
    """XGBoost regressor for spread (margin) predictions."""

    name = "xgb_spread"
    version = "1.0"
    description = "XGBoost gradient boosting for spread predictions"

    def __init__(self, use_model_predictions=False):
        self.feature_computer = FeatureComputer(
            use_model_predictions=use_model_predictions
        )
        self.model = None
        self.feature_mean = None
        self.feature_std = None
        self._loaded = False
        self._load_model()

    def _load_model(self):
        if not XGB_AVAILABLE:
            return
        if SPREAD_PATH.exists():
            try:
                with open(SPREAD_PATH, 'rb') as f:
                    checkpoint = pickle.load(f)
                self.model = checkpoint['model']
                self.feature_mean = checkpoint.get('feature_mean')
                self.feature_std = checkpoint.get('feature_std')
                self._loaded = True
            except Exception as e:
                print(f"Warning: Could not load XGBoost spread model: {e}")

    def is_trained(self):
        return self._loaded and self.model is not None

    def predict_game(self, home_team_id, away_team_id, neutral_site=False,
                     spread_line=None, **kwargs):
        """Predict home team margin (positive = home wins by X)."""
        if not self.is_trained():
            return {
                'model': self.name,
                'predicted_margin': 0.0,
                'home_cover_prob': 0.5,
                'away_cover_prob': 0.5,
                'home_win_probability': 0.5,
                'away_win_probability': 0.5,
                'projected_home_runs': 4.5,
                'projected_away_runs': 4.5,
                'confidence': 'none (model not trained)',
            }

        features = self.feature_computer.compute_features(
            home_team_id, away_team_id, neutral_site=neutral_site
        )
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        if self.feature_mean is not None and self.feature_std is not None:
            # Handle feature size mismatch (new features added but model not retrained)
            saved_size = len(self.feature_mean)
            if len(features) > saved_size:
                features = features[:saved_size]
            elif len(features) < saved_size:
                features = np.pad(features, (0, saved_size - len(features)))
            features = (features - self.feature_mean) / (self.feature_std + 1e-8)
            features = np.clip(features, -5.0, 5.0)

        X = features.reshape(1, -1)
        predicted_margin = float(self.model.predict(X)[0])

        # Convert margin to win probability (logistic approximation)
        import math
        home_prob = 1.0 / (1.0 + math.exp(-predicted_margin / 3.0))
        home_prob = max(0.05, min(0.95, home_prob))

        # Cover probability given a spread line
        std = 4.0  # Typical std for margin
        if spread_line is not None:
            adjusted_margin = predicted_margin + spread_line  # spread_line is usually negative for favorite
            z = adjusted_margin / std
            home_cover_prob = 0.5 * (1 + math.erf(z / math.sqrt(2)))
        else:
            home_cover_prob = 0.5

        return {
            'model': self.name,
            'predicted_margin': round(predicted_margin, 2),
            'home_cover_prob': round(max(0.01, min(0.99, home_cover_prob)), 4),
            'away_cover_prob': round(max(0.01, min(0.99, 1 - home_cover_prob)), 4),
            'home_win_probability': round(home_prob, 4),
            'away_win_probability': round(1 - home_prob, 4),
            'projected_home_runs': 4.5,
            'projected_away_runs': 4.5,
            'confidence': 'high' if abs(predicted_margin) > 3 else 'medium',
        }


class XGBTrainer:
    """Trains XGBoost models with GPU acceleration."""

    def __init__(self, task='classification', use_gpu=True, n_estimators=800,
                 max_depth=8, learning_rate=0.05, early_stopping_rounds=50):
        self.task = task
        self.use_gpu = use_gpu
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None
        self.feature_mean = None
        self.feature_std = None

    def _get_params(self):
        """Get XGBoost parameters with GPU support."""
        params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
        }
        
        if self.use_gpu:
            params['tree_method'] = 'hist'
            params['device'] = 'cuda'
        
        return params

    def normalize_features(self, X_train):
        """Compute normalization stats from training data."""
        self.feature_mean = X_train.mean(axis=0)
        self.feature_std = X_train.std(axis=0)
        self.feature_std[self.feature_std < 1e-8] = 1.0
        return (X_train - self.feature_mean) / self.feature_std

    def apply_normalization(self, X):
        """Apply normalization to new data."""
        if self.feature_mean is None:
            return X
        return (X - self.feature_mean) / self.feature_std

    def train(self, X_train, y_train, X_val, y_val, sample_weight=None):
        """
        Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels/targets
            X_val: Validation features
            y_val: Validation labels/targets
            sample_weight: Optional sample weights for recency weighting
        """
        X_train_norm = self.normalize_features(X_train)
        X_val_norm = self.apply_normalization(X_val)

        params = self._get_params()

        if self.task == 'classification':
            self.model = xgb.XGBClassifier(**params)
            eval_metric = 'logloss'
        else:
            self.model = xgb.XGBRegressor(**params)
            eval_metric = 'rmse'

        self.model.fit(
            X_train_norm, y_train,
            sample_weight=sample_weight,
            eval_set=[(X_val_norm, y_val)],
            verbose=50
        )

        # Get best iteration
        best_iter = self.model.best_iteration if hasattr(self.model, 'best_iteration') else self.n_estimators
        print(f"Best iteration: {best_iter}")

        return self.model

    def save(self, path):
        """Save model and normalization stats."""
        checkpoint = {
            'model': self.model,
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std,
            'task': self.task,
        }
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"Saved model to {path}")

    def evaluate(self, X_test, y_test):
        """Evaluate model on test set."""
        X_test_norm = self.apply_normalization(X_test)

        if self.task == 'classification':
            y_pred = self.model.predict(X_test_norm)
            y_prob = self.model.predict_proba(X_test_norm)[:, 1]
            accuracy = (y_pred == y_test).mean()
            
            # Log loss
            from sklearn.metrics import log_loss
            logloss = log_loss(y_test, y_prob)
            
            return {
                'accuracy': round(float(accuracy), 4),
                'log_loss': round(float(logloss), 4),
                'n_samples': len(y_test),
            }
        else:
            y_pred = self.model.predict(X_test_norm)
            errors = y_pred - y_test
            mae = np.abs(errors).mean()
            rmse = np.sqrt((errors ** 2).mean())
            
            return {
                'mae': round(float(mae), 4),
                'rmse': round(float(rmse), 4),
                'n_samples': len(y_test),
            }

    def get_feature_importance(self, feature_names=None):
        """Get feature importance scores."""
        if self.model is None:
            return {}
        
        importance = self.model.feature_importances_
        if feature_names is None:
            feature_names = [f'f{i}' for i in range(len(importance))]
        
        # Sort by importance
        sorted_idx = np.argsort(importance)[::-1]
        top_features = {
            feature_names[i]: round(float(importance[i]), 4)
            for i in sorted_idx[:20]
        }
        
        return top_features


if __name__ == "__main__":
    # Quick test
    print("XGBoost Available:", XGB_AVAILABLE)
    
    model = XGBMoneylineModel()
    print(f"Moneyline model loaded: {model.is_trained()}")
    
    model = XGBTotalsModel()
    print(f"Totals model loaded: {model.is_trained()}")
    
    model = XGBSpreadModel()
    print(f"Spread model loaded: {model.is_trained()}")
