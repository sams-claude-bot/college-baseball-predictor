#!/usr/bin/env python3
"""
XGBoost v2 Models — No normalization, calibration support.
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
MONEYLINE_V2_PATH = DATA_DIR / "xgb_moneyline_v2.pkl"
TOTALS_V2_PATH = DATA_DIR / "xgb_totals_v2.pkl"
SPREAD_V2_PATH = DATA_DIR / "xgb_spread_v2.pkl"


class XGBMoneylineModelV2(BaseModel):
    """XGBoost v2 classifier for moneyline — no normalization, optional calibration."""

    name = "xgb_moneyline_v2"
    version = "2.0"
    description = "XGBoost v2 gradient boosting for moneyline (no norm, calibrated)"

    def __init__(self, use_model_predictions=False):
        self.feature_computer = FeatureComputer(use_model_predictions=use_model_predictions)
        self.model = None
        self.calibrator = None
        self.n_features = None
        self._loaded = False
        self._load_model()

    def _load_model(self):
        if not XGB_AVAILABLE:
            return
        if MONEYLINE_V2_PATH.exists():
            try:
                with open(MONEYLINE_V2_PATH, 'rb') as f:
                    checkpoint = pickle.load(f)
                self.model = checkpoint['model']
                self.calibrator = checkpoint.get('calibrator')
                self.n_features = checkpoint.get('n_features')
                self._loaded = True
            except Exception as e:
                print(f"Warning: Could not load XGBoost v2 moneyline model: {e}")

    def is_trained(self):
        return self._loaded and self.model is not None

    def _prepare_features(self, home_team_id, away_team_id, neutral_site):
        features = self.feature_computer.compute_features(
            home_team_id, away_team_id, neutral_site=neutral_site
        )
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        if self.n_features is not None:
            if len(features) > self.n_features:
                features = features[:self.n_features]
            elif len(features) < self.n_features:
                features = np.pad(features, (0, self.n_features - len(features)))
        return features.reshape(1, -1)

    def predict_game(self, home_team_id, away_team_id, neutral_site=False, **kwargs):
        if not self.is_trained():
            return {
                'model': self.name, 'home_win_probability': 0.5,
                'away_win_probability': 0.5, 'projected_home_runs': 4.5,
                'projected_away_runs': 4.5, 'confidence': 'none (model not trained)',
            }

        X = self._prepare_features(home_team_id, away_team_id, neutral_site)

        if self.calibrator is not None:
            home_prob = float(self.calibrator.predict_proba(X)[0, 1])
        else:
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


class XGBTotalsModelV2(BaseModel):
    """XGBoost v2 regressor for total runs — no normalization."""

    name = "xgb_totals_v2"
    version = "2.0"
    description = "XGBoost v2 for run totals (no norm)"

    def __init__(self, use_model_predictions=False):
        self.feature_computer = FeatureComputer(use_model_predictions=use_model_predictions)
        self.model = None
        self.n_features = None
        self._loaded = False
        self._load_model()

    def _load_model(self):
        if not XGB_AVAILABLE:
            return
        if TOTALS_V2_PATH.exists():
            try:
                with open(TOTALS_V2_PATH, 'rb') as f:
                    checkpoint = pickle.load(f)
                self.model = checkpoint['model']
                self.n_features = checkpoint.get('n_features')
                self._loaded = True
            except Exception as e:
                print(f"Warning: Could not load XGBoost v2 totals model: {e}")

    def is_trained(self):
        return self._loaded and self.model is not None

    def predict_game(self, home_team_id, away_team_id, neutral_site=False,
                     over_under_line=None, **kwargs):
        if not self.is_trained():
            return {
                'model': self.name, 'projected_total': 9.0,
                'over_prob': 0.5, 'under_prob': 0.5,
                'home_win_probability': 0.5, 'away_win_probability': 0.5,
                'projected_home_runs': 4.5, 'projected_away_runs': 4.5,
                'confidence': 'none (model not trained)',
            }

        features = self.feature_computer.compute_features(
            home_team_id, away_team_id, neutral_site=neutral_site
        )
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        if self.n_features is not None:
            if len(features) > self.n_features:
                features = features[:self.n_features]
            elif len(features) < self.n_features:
                features = np.pad(features, (0, self.n_features - len(features)))

        X = features.reshape(1, -1)
        projected_total = max(0.0, float(self.model.predict(X)[0]))
        std = 3.5

        if over_under_line is not None and over_under_line > 0:
            import math
            z = (over_under_line - projected_total) / std
            under_prob = 0.5 * (1 + math.erf(z / math.sqrt(2)))
            over_prob = 1.0 - under_prob
        else:
            over_prob = under_prob = 0.5

        return {
            'model': self.name,
            'projected_total': round(projected_total, 2),
            'over_prob': round(max(0.01, min(0.99, over_prob)), 4),
            'under_prob': round(max(0.01, min(0.99, under_prob)), 4),
            'home_win_probability': 0.5, 'away_win_probability': 0.5,
            'projected_home_runs': round(projected_total * 0.52, 2),
            'projected_away_runs': round(projected_total * 0.48, 2),
            'confidence': 'medium',
        }


class XGBSpreadModelV2(BaseModel):
    """XGBoost v2 regressor for spread — no normalization."""

    name = "xgb_spread_v2"
    version = "2.0"
    description = "XGBoost v2 for spread (no norm)"

    def __init__(self, use_model_predictions=False):
        self.feature_computer = FeatureComputer(use_model_predictions=use_model_predictions)
        self.model = None
        self.n_features = None
        self._loaded = False
        self._load_model()

    def _load_model(self):
        if not XGB_AVAILABLE:
            return
        if SPREAD_V2_PATH.exists():
            try:
                with open(SPREAD_V2_PATH, 'rb') as f:
                    checkpoint = pickle.load(f)
                self.model = checkpoint['model']
                self.n_features = checkpoint.get('n_features')
                self._loaded = True
            except Exception as e:
                print(f"Warning: Could not load XGBoost v2 spread model: {e}")

    def is_trained(self):
        return self._loaded and self.model is not None

    def predict_game(self, home_team_id, away_team_id, neutral_site=False,
                     spread_line=None, **kwargs):
        if not self.is_trained():
            return {
                'model': self.name, 'predicted_margin': 0.0,
                'home_cover_prob': 0.5, 'away_cover_prob': 0.5,
                'home_win_probability': 0.5, 'away_win_probability': 0.5,
                'projected_home_runs': 4.5, 'projected_away_runs': 4.5,
                'confidence': 'none (model not trained)',
            }

        features = self.feature_computer.compute_features(
            home_team_id, away_team_id, neutral_site=neutral_site
        )
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        if self.n_features is not None:
            if len(features) > self.n_features:
                features = features[:self.n_features]
            elif len(features) < self.n_features:
                features = np.pad(features, (0, self.n_features - len(features)))

        X = features.reshape(1, -1)
        predicted_margin = float(self.model.predict(X)[0])

        import math
        home_prob = 1.0 / (1.0 + math.exp(-predicted_margin / 3.0))
        home_prob = max(0.05, min(0.95, home_prob))

        std = 4.0
        if spread_line is not None:
            adjusted_margin = predicted_margin + spread_line
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
            'projected_home_runs': 4.5, 'projected_away_runs': 4.5,
            'confidence': 'high' if abs(predicted_margin) > 3 else 'medium',
        }
