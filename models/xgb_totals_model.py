#!/usr/bin/env python3
"""
XGBoost Totals Model — v2

Trained model for predicting total runs in college baseball games.
Uses team batting/pitching quality features, weather, market lines,
and game context. Also includes a market-informed deviation model
that predicts actual_total - line.

Features:
  - Team batting quality (runs_per_game, OPS, wOBA, wRC+, ISO, K%, BB%, HR/G, elite/solid/weak bats)
  - Team pitching quality (staff ERA, rotation ERA, bullpen ERA, WHIP, K/9, BB/9, FIP, quality/shutdown/liability arms)
  - Weather (temp, wind, humidity, dome)
  - Market line (betting O/U — powerful feature)
  - Game context (DOW, conference game, neutral site)
"""

import sys
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.database import get_connection

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_PATH = DATA_DIR / "xgb_totals_v2.pkl"

# Feature names in order
BATTING_FEATURES = [
    'runs_per_game', 'lineup_ops', 'lineup_woba', 'lineup_wrc_plus',
    'lineup_iso', 'lineup_k_pct', 'lineup_bb_pct', 'hr_per_game',
    'elite_bats', 'solid_bats', 'weak_bats',
]

PITCHING_FEATURES = [
    'staff_era', 'rotation_era', 'bullpen_era', 'staff_whip',
    'rotation_k_per_9', 'bullpen_k_per_9', 'rotation_bb_per_9',
    'bullpen_bb_per_9', 'quality_arms', 'shutdown_arms',
    'liability_arms', 'staff_fip',
]

WEATHER_FEATURES = ['temp_f', 'wind_speed_mph', 'humidity_pct', 'is_dome']

FEATURE_NAMES = []
# Sum features (home + away)
for feat in BATTING_FEATURES:
    FEATURE_NAMES.append(f'sum_bat_{feat}')
for feat in PITCHING_FEATURES:
    FEATURE_NAMES.append(f'sum_pitch_{feat}')
# Matchup features
FEATURE_NAMES.extend([
    'home_bat_ops_vs_away_pitch_era',
    'away_bat_ops_vs_home_pitch_era',
    'home_bat_woba_vs_away_pitch_era',
    'away_bat_woba_vs_home_pitch_era',
])
# K rate interaction
FEATURE_NAMES.extend([
    'sum_k_rate_interaction',  # sum of (pitching K/9 * batting K%)
])
# Weather
FEATURE_NAMES.extend(WEATHER_FEATURES)
# Market
FEATURE_NAMES.append('market_line')
FEATURE_NAMES.append('has_market_line')
# Context
FEATURE_NAMES.extend(['dow', 'is_conference', 'is_neutral_site'])

NUM_FEATURES = len(FEATURE_NAMES)


def get_team_batting(team_id, conn=None):
    """Load team batting quality from DB."""
    close = False
    if conn is None:
        conn = get_connection()
        close = True
    c = conn.cursor()
    c.execute('SELECT * FROM team_batting_quality WHERE team_id = ?', (team_id,))
    row = c.fetchone()
    if close:
        conn.close()
    if row is None:
        return None
    return dict(row)


def get_team_pitching(team_id, conn=None):
    """Load team pitching quality from DB."""
    close = False
    if conn is None:
        conn = get_connection()
        close = True
    c = conn.cursor()
    c.execute('SELECT * FROM team_pitching_quality WHERE team_id = ?', (team_id,))
    row = c.fetchone()
    if close:
        conn.close()
    if row is None:
        return None
    return dict(row)


def build_features(home_bat, away_bat, home_pitch, away_pitch,
                   weather=None, line=None, game_date=None,
                   is_conference=0, is_neutral=0):
    """Build feature vector for a game.

    Args:
        home_bat/away_bat: dicts from team_batting_quality
        home_pitch/away_pitch: dicts from team_pitching_quality
        weather: dict with temp_f, wind_speed_mph, humidity_pct, is_dome
        line: betting over/under line (or None)
        game_date: date string YYYY-MM-DD (for DOW)
        is_conference: 1/0
        is_neutral: 1/0

    Returns:
        numpy array of features
    """
    features = []

    # Sum batting features (both teams)
    for feat in BATTING_FEATURES:
        hv = home_bat.get(feat, 0) or 0
        av = away_bat.get(feat, 0) or 0
        features.append(float(hv) + float(av))

    # Sum pitching features (higher = worse pitching = more runs)
    for feat in PITCHING_FEATURES:
        hv = home_pitch.get(feat, 0) or 0
        av = away_pitch.get(feat, 0) or 0
        features.append(float(hv) + float(av))

    # Matchup features: batting quality vs opposing pitching
    h_ops = float(home_bat.get('lineup_ops', 0.700) or 0.700)
    a_ops = float(away_bat.get('lineup_ops', 0.700) or 0.700)
    h_woba = float(home_bat.get('lineup_woba', 0.320) or 0.320)
    a_woba = float(away_bat.get('lineup_woba', 0.320) or 0.320)
    h_pitch_era = float(home_pitch.get('staff_era', 4.5) or 4.5)
    a_pitch_era = float(away_pitch.get('staff_era', 4.5) or 4.5)

    features.append(h_ops * a_pitch_era)  # good home batting vs bad away pitching = more runs
    features.append(a_ops * h_pitch_era)
    features.append(h_woba * a_pitch_era)
    features.append(a_woba * h_pitch_era)

    # K rate interaction: high K pitching vs high K batting = fewer runs
    h_bat_k = float(home_bat.get('lineup_k_pct', 0.20) or 0.20)
    a_bat_k = float(away_bat.get('lineup_k_pct', 0.20) or 0.20)
    h_pitch_k9 = float(home_pitch.get('staff_k_per_9', 8.0) or 8.0)
    a_pitch_k9 = float(away_pitch.get('staff_k_per_9', 8.0) or 8.0)
    features.append((h_pitch_k9 * a_bat_k) + (a_pitch_k9 * h_bat_k))

    # Weather
    if weather:
        features.append(float(weather.get('temp_f', 65) or 65))
        features.append(float(weather.get('wind_speed_mph', 6) or 6))
        features.append(float(weather.get('humidity_pct', 55) or 55))
        features.append(float(weather.get('is_dome', 0) or 0))
    else:
        features.extend([65.0, 6.0, 55.0, 0.0])

    # Market line
    if line is not None and line > 0:
        features.append(float(line))
        features.append(1.0)
    else:
        features.append(12.0)  # fill with league average
        features.append(0.0)

    # Game context
    dow = 2  # default to Tuesday
    if game_date:
        try:
            from datetime import datetime
            dt = datetime.strptime(game_date[:10], '%Y-%m-%d')
            dow = dt.weekday()
        except (ValueError, TypeError):
            pass
    features.append(float(dow))
    features.append(float(is_conference))
    features.append(float(is_neutral))

    return np.array(features, dtype=np.float32)


class XGBTotalsModel:
    """XGBoost totals model with regressor + O/U classifier + deviation model."""

    def __init__(self):
        self._loaded = False
        self.regressor = None
        self.classifier = None
        self.deviation_model = None
        self.feature_names = list(FEATURE_NAMES)
        self._load()

    def _load(self):
        if MODEL_PATH.exists():
            try:
                with open(MODEL_PATH, 'rb') as f:
                    data = pickle.load(f)
                self.regressor = data.get('regressor')
                self.classifier = data.get('classifier')
                self.deviation_model = data.get('deviation_model')
                self._loaded = True
            except Exception as e:
                print(f"Warning: Could not load XGB totals model: {e}")

    def is_trained(self):
        return self._loaded and self.regressor is not None

    def predict(self, home_team_id, away_team_id, line=None,
                game_date=None, game_id=None):
        """Predict total runs for a game.

        Returns dict with projected_total, over_under prediction if line given,
        and deviation-based prediction if deviation model is available.
        """
        conn = get_connection()

        home_bat = get_team_batting(home_team_id, conn)
        away_bat = get_team_batting(away_team_id, conn)
        home_pitch = get_team_pitching(home_team_id, conn)
        away_pitch = get_team_pitching(away_team_id, conn)

        if not all([home_bat, away_bat, home_pitch, away_pitch]):
            conn.close()
            return {
                'model': 'xgb_totals',
                'projected_total': None,
                'error': 'missing team quality data',
            }

        # Get weather
        weather = None
        if game_id:
            c = conn.cursor()
            c.execute("""SELECT temp_f, wind_speed_mph, humidity_pct, is_dome
                         FROM game_weather WHERE game_id = ?""", (game_id,))
            row = c.fetchone()
            if row:
                weather = dict(row)

        # Get game context
        is_conference = 0
        is_neutral = 0
        if game_id:
            c = conn.cursor()
            c.execute("SELECT is_conference_game, is_neutral_site, date FROM games WHERE id = ?",
                      (game_id,))
            row = c.fetchone()
            if row:
                is_conference = row['is_conference_game'] or 0
                is_neutral = row['is_neutral_site'] or 0
                if game_date is None:
                    game_date = row['date']

        conn.close()

        features = build_features(
            home_bat, away_bat, home_pitch, away_pitch,
            weather=weather, line=line, game_date=game_date,
            is_conference=is_conference, is_neutral=is_neutral,
        )

        result = {
            'model': 'xgb_totals',
            'features_used': NUM_FEATURES,
        }

        # Raw regression prediction
        if self.regressor:
            pred_total = float(self.regressor.predict(features.reshape(1, -1))[0])
            pred_total = max(pred_total, 2.0)
            result['projected_total'] = round(pred_total, 1)

        # Deviation-based prediction (predicts actual - line)
        if self.deviation_model and line is not None and line > 0:
            deviation = float(self.deviation_model.predict(features.reshape(1, -1))[0])
            result['deviation_prediction'] = round(deviation, 2)
            result['deviation_adjusted_total'] = round(line + deviation, 1)

        # O/U classifier
        if self.classifier and line is not None and line > 0:
            ou_prob = self.classifier.predict_proba(features.reshape(1, -1))[0]
            result['over_prob'] = round(float(ou_prob[1]), 4)
            result['under_prob'] = round(float(ou_prob[0]), 4)
            result['ou_prediction'] = 'OVER' if ou_prob[1] > 0.5 else 'UNDER'

        return result
