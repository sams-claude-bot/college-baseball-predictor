#!/usr/bin/env python3
"""
Train XGBoost and LightGBM v2 — Improved pipeline with:
  a) 2026 games included in training
  b) Optuna Bayesian hyperparameter tuning (--tune)
  c) No normalization (raw features for trees)
  d) Stronger regularization defaults
  e) DART booster option (--dart)
  f) Platt/isotonic calibration (--calibrate)
  g) Conference target encoding (--conference-encoding)

Training policy: use ALL data for train/val (temporal CV), no holdout test set.
"""

import sys
import time
import argparse
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime, date
from math import exp
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.database import get_connection
from models.nn_features import HistoricalFeatureComputer, FeatureComputer

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, accuracy_score

DATA_DIR = Path(__file__).parent.parent / "data"

# v2 checkpoint paths
XGB_ML_V2 = DATA_DIR / "xgb_moneyline_v2.pkl"
XGB_TOT_V2 = DATA_DIR / "xgb_totals_v2.pkl"
XGB_SPR_V2 = DATA_DIR / "xgb_spread_v2.pkl"
LGB_ML_V2 = DATA_DIR / "lgb_moneyline_v2.pkl"
LGB_TOT_V2 = DATA_DIR / "lgb_totals_v2.pkl"
LGB_SPR_V2 = DATA_DIR / "lgb_spread_v2.pkl"

# --- Stronger regularization defaults (improvement d) ---
DEFAULT_XGB_PARAMS = {
    'n_estimators': 800,
    'max_depth': 5,
    'learning_rate': 0.05,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.5,
    'reg_lambda': 2.0,
    'random_state': 42,
    'n_jobs': -1,
}

DEFAULT_LGB_PARAMS = {
    'n_estimators': 800,
    'max_depth': 5,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_child_samples': 5,
    'subsample': 0.8,
    'subsample_freq': 1,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.5,
    'reg_lambda': 2.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}


def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            return True
    except ImportError:
        pass
    print("No GPU detected, using CPU")
    return False


def compute_recency_weights(game_dates, decay_rate=0.002, min_weight=0.1):
    today = date.today()
    weights = []
    for d in game_dates:
        if isinstance(d, str):
            d = datetime.strptime(d, '%Y-%m-%d').date()
        elif isinstance(d, datetime):
            d = d.date()
        days_ago = (today - d).days
        weights.append(max(min_weight, exp(-decay_rate * days_ago)))
    weights = np.array(weights, dtype=np.float32)
    print(f"Recency weights: min={weights.min():.3f}, max={weights.max():.3f}, mean={weights.mean():.3f}")
    return weights


# --- Conference target encoding (improvement g) ---
def build_conference_encoding(conn):
    """Build conference-pair win rate encoding from historical + 2026 data."""
    c = conn.cursor()

    # Get team -> conference mapping
    c.execute("SELECT id, conference FROM teams WHERE conference IS NOT NULL")
    team_conf = {row['id']: row['conference'] for row in c.fetchall()}

    # Also map historical team names to conferences (best effort via teams table name/nickname)
    c.execute("SELECT id, name, nickname, conference FROM teams WHERE conference IS NOT NULL")
    name_conf = {}
    for row in c.fetchall():
        name_conf[row['id']] = row['conference']
        if row['name']:
            name_conf[row['name']] = row['conference']

    # Gather conference matchup stats from historical games
    conf_wins = defaultdict(lambda: [0, 0])  # (home_conf, away_conf) -> [home_wins, total]

    c.execute("SELECT home_team, away_team, home_score, away_score FROM historical_games WHERE home_score IS NOT NULL")
    for row in c.fetchall():
        hc = name_conf.get(row['home_team'])
        ac = name_conf.get(row['away_team'])
        if hc and ac and hc != ac:
            key = (hc, ac)
            conf_wins[key][1] += 1
            if row['home_score'] > row['away_score']:
                conf_wins[key][0] += 1

    # Also from 2026 games
    c.execute("SELECT home_team_id, away_team_id, home_score, away_score FROM games WHERE status='final' AND home_score IS NOT NULL")
    for row in c.fetchall():
        hc = team_conf.get(row['home_team_id'])
        ac = team_conf.get(row['away_team_id'])
        if hc and ac and hc != ac:
            key = (hc, ac)
            conf_wins[key][1] += 1
            if row['home_score'] > row['away_score']:
                conf_wins[key][0] += 1

    # Compute win rates with smoothing
    conf_rates = {}
    for key, (wins, total) in conf_wins.items():
        conf_rates[key] = (wins + 5) / (total + 10)  # Bayesian smoothing toward 0.5

    return team_conf, name_conf, conf_rates


def get_conference_features(home_id, away_id, team_conf, name_conf, conf_rates):
    """Return [home_conf_winrate_vs_away_conf, away_conf_winrate_vs_home_conf]."""
    hc = team_conf.get(home_id) or name_conf.get(home_id)
    ac = team_conf.get(away_id) or name_conf.get(away_id)
    if not hc or not ac or hc == ac:
        return [0.5, 0.5]
    home_rate = conf_rates.get((hc, ac), 0.5)
    away_rate = conf_rates.get((ac, hc), 0.5)
    return [home_rate, away_rate]


# --- Data loading (improvement a: include 2026 games) ---
def load_all_data(min_games=20, use_conference_encoding=False):
    """Load historical + 2026 final games. Returns X, y_totals, y_spread, dates."""
    print("\n" + "=" * 60)
    print("Loading historical + 2026 game data (v2)...")
    print("=" * 60)

    conn = get_connection()
    c = conn.cursor()

    # Conference encoding setup
    conf_data = None
    if use_conference_encoding:
        print("Building conference target encoding...")
        conf_data = build_conference_encoding(conn)

    # --- Historical games ---
    c.execute("""
        SELECT h.id, h.home_team, h.away_team, h.home_score, h.away_score,
               h.date, h.season, h.neutral_site,
               w.temp_f, w.humidity_pct, w.wind_speed_mph, w.wind_direction_deg,
               w.precip_prob_pct, w.is_dome
        FROM historical_games h
        LEFT JOIN historical_game_weather w ON h.id = w.game_id
        WHERE h.home_score IS NOT NULL AND h.away_score IS NOT NULL
        ORDER BY h.date ASC
    """)
    hist_rows = c.fetchall()
    print(f"Historical games: {len(hist_rows)}")

    # --- 2026 games (improvement a) ---
    c.execute("""
        SELECT id, home_team_id, away_team_id, home_score, away_score,
               date, is_neutral_site
        FROM games
        WHERE status = 'final' AND home_score IS NOT NULL AND away_score IS NOT NULL
        ORDER BY date ASC
    """)
    current_rows = c.fetchall()
    print(f"2026 final games: {len(current_rows)}")
    conn.close()

    # Process historical with HistoricalFeatureComputer
    fc_hist = HistoricalFeatureComputer()
    features, totals, spreads, dates = [], [], [], []

    for row in hist_rows:
        game_row = {
            'home_team': row['home_team'], 'away_team': row['away_team'],
            'home_score': row['home_score'], 'away_score': row['away_score'],
            'date': row['date'], 'season': row['season'],
            'neutral_site': row['neutral_site'] or 0,
        }
        weather_row = None
        if row['temp_f'] is not None:
            weather_row = {
                'temp_f': row['temp_f'], 'humidity_pct': row['humidity_pct'],
                'wind_speed_mph': row['wind_speed_mph'],
                'wind_direction_deg': row['wind_direction_deg'],
                'precip_prob_pct': row['precip_prob_pct'],
                'is_dome': row['is_dome'],
            }

        feat, _ = fc_hist.compute_game_features(game_row, weather_row)
        fc_hist.update_state(game_row)

        home_games = fc_hist.team_stats[row['home_team']]['games']
        away_games = fc_hist.team_stats[row['away_team']]['games']
        if home_games >= min_games and away_games >= min_games:
            if use_conference_encoding and conf_data:
                conf_feat = get_conference_features(
                    row['home_team'], row['away_team'],
                    conf_data[0], conf_data[1], conf_data[2])
                feat = np.concatenate([feat, conf_feat])
            features.append(feat)
            totals.append(row['home_score'] + row['away_score'])
            spreads.append(row['home_score'] - row['away_score'])
            dates.append(row['date'])

    # Process 2026 games with FeatureComputer (live data, no leakage concern for current season)
    fc_current = FeatureComputer(use_model_predictions=False)
    for row in current_rows:
        try:
            feat = fc_current.compute_features(
                row['home_team_id'], row['away_team_id'],
                neutral_site=bool(row['is_neutral_site']))
            feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
            if use_conference_encoding and conf_data:
                conf_feat = get_conference_features(
                    row['home_team_id'], row['away_team_id'],
                    conf_data[0], conf_data[1], conf_data[2])
                feat = np.concatenate([feat, conf_feat])
            features.append(feat)
            totals.append(row['home_score'] + row['away_score'])
            spreads.append(row['home_score'] - row['away_score'])
            dates.append(row['date'])
        except Exception as e:
            # Skip games where feature computation fails
            continue

    # Pad features to same length (hist vs current may differ)
    max_len = max(len(f) for f in features) if features else 0
    features = [np.pad(f, (0, max_len - len(f))) if len(f) < max_len else f[:max_len]
                for f in features]

    X = np.array(features, dtype=np.float32)
    y_totals = np.array(totals, dtype=np.float32)
    y_spread = np.array(spreads, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Total training samples: {len(X)}, features: {X.shape[1] if len(X) > 0 else 0}")
    return X, y_totals, y_spread, dates


def load_2026_only_data(use_conference_encoding=False):
    """Load ONLY 2026 final games (no historical). Returns X, y_totals, y_spread, dates."""
    print("\n" + "=" * 60)
    print("Loading 2026-ONLY game data (no historical)...")
    print("=" * 60)

    conn = get_connection()
    c = conn.cursor()

    conf_data = None
    if use_conference_encoding:
        print("Building conference target encoding...")
        conf_data = build_conference_encoding(conn)

    c.execute("""
        SELECT id, home_team_id, away_team_id, home_score, away_score,
               date, is_neutral_site
        FROM games
        WHERE status = 'final' AND home_score IS NOT NULL AND away_score IS NOT NULL
              AND date >= '2026-01-01'
        ORDER BY date ASC
    """)
    current_rows = c.fetchall()
    print(f"2026 final games: {len(current_rows)}")
    conn.close()

    fc_current = FeatureComputer(use_model_predictions=False)
    features, totals, spreads, dates = [], [], [], []

    for row in current_rows:
        try:
            feat = fc_current.compute_features(
                row['home_team_id'], row['away_team_id'],
                neutral_site=bool(row['is_neutral_site']))
            feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
            if use_conference_encoding and conf_data:
                conf_feat = get_conference_features(
                    row['home_team_id'], row['away_team_id'],
                    conf_data[0], conf_data[1], conf_data[2])
                feat = np.concatenate([feat, conf_feat])
            features.append(feat)
            totals.append(row['home_score'] + row['away_score'])
            spreads.append(row['home_score'] - row['away_score'])
            dates.append(row['date'])
        except Exception as e:
            continue

    if not features:
        print("ERROR: No features computed!")
        sys.exit(1)

    max_len = max(len(f) for f in features)
    features = [np.pad(f, (0, max_len - len(f))) if len(f) < max_len else f[:max_len]
                for f in features]

    X = np.array(features, dtype=np.float32)
    y_totals = np.array(totals, dtype=np.float32)
    y_spread = np.array(spreads, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Total training samples: {len(X)}, features: {X.shape[1] if len(X) > 0 else 0}")
    return X, y_totals, y_spread, dates


def temporal_cv_split(X, y, weights, n_folds=5):
    """Generate temporal CV fold indices. Each fold uses earlier data for train, later for val."""
    n = len(X)
    fold_size = n // (n_folds + 1)
    folds = []
    for i in range(n_folds):
        val_start = fold_size * (i + 1)
        val_end = min(val_start + fold_size, n)
        train_idx = list(range(0, val_start))
        val_idx = list(range(val_start, val_end))
        if len(train_idx) > 0 and len(val_idx) > 0:
            folds.append((train_idx, val_idx))
    return folds


# --- Optuna tuning (improvement b) ---
def tune_xgb(X, y, weights, task='classification', use_gpu=True, use_dart=False, n_trials=100):
    """Bayesian hyperparameter tuning for XGBoost with Optuna."""
    if not OPTUNA_AVAILABLE:
        print("Optuna not available, using defaults")
        return DEFAULT_XGB_PARAMS.copy()

    folds = temporal_cv_split(X, y, weights)
    print(f"Tuning XGBoost ({task}) with {n_trials} trials, {len(folds)} temporal folds...")

    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42, 'n_jobs': -1,
        }
        if use_gpu:
            params['tree_method'] = 'hist'
            params['device'] = 'cuda'
        if use_dart:
            params['booster'] = 'dart'
            params['rate_drop'] = trial.suggest_float('rate_drop', 0.01, 0.3)
            params['skip_drop'] = trial.suggest_float('skip_drop', 0.3, 0.7)

        scores = []
        for train_idx, val_idx in folds:
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_va = X[val_idx], y[val_idx]
            w_tr = weights[train_idx] if weights is not None else None

            if task == 'classification':
                model = xgb.XGBClassifier(**params)
                model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_va, y_va)], verbose=0)
                pred = model.predict_proba(X_va)[:, 1]
                scores.append(log_loss(y_va, pred))
            else:
                model = xgb.XGBRegressor(**params)
                model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_va, y_va)], verbose=0)
                pred = model.predict(X_va)
                scores.append(float(np.sqrt(((pred - y_va) ** 2).mean())))

        return np.mean(scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = study.best_params
    best['random_state'] = 42
    best['n_jobs'] = -1
    if use_gpu:
        best['tree_method'] = 'hist'
        best['device'] = 'cuda'
    print(f"Best XGB params: {best}")
    return best


def tune_lgb(X, y, weights, task='classification', use_gpu=True, use_dart=False, n_trials=100):
    """Bayesian hyperparameter tuning for LightGBM with Optuna."""
    if not OPTUNA_AVAILABLE:
        print("Optuna not available, using defaults")
        return DEFAULT_LGB_PARAMS.copy()

    folds = temporal_cv_split(X, y, weights)
    print(f"Tuning LightGBM ({task}) with {n_trials} trials, {len(folds)} temporal folds...")

    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 20),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'subsample_freq': 1,
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42, 'n_jobs': -1, 'verbose': -1,
        }
        if use_gpu:
            params['device'] = 'gpu'
            params['gpu_use_dp'] = False
        if use_dart:
            params['boosting_type'] = 'dart'
            params['drop_rate'] = trial.suggest_float('drop_rate', 0.01, 0.3)
            params['skip_drop'] = trial.suggest_float('skip_drop', 0.3, 0.7)

        scores = []
        for train_idx, val_idx in folds:
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_va = X[val_idx], y[val_idx]
            w_tr = weights[train_idx] if weights is not None else None

            if task == 'classification':
                model = lgb.LGBMClassifier(**params)
                model.fit(X_tr, y_tr, sample_weight=w_tr,
                          eval_set=[(X_va, y_va)], eval_metric='logloss',
                          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
                pred = model.predict_proba(X_va)[:, 1]
                scores.append(log_loss(y_va, pred))
            else:
                model = lgb.LGBMRegressor(**params)
                model.fit(X_tr, y_tr, sample_weight=w_tr,
                          eval_set=[(X_va, y_va)], eval_metric='rmse',
                          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
                pred = model.predict(X_va)
                scores.append(float(np.sqrt(((pred - y_va) ** 2).mean())))

        return np.mean(scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = study.best_params
    best['random_state'] = 42
    best['n_jobs'] = -1
    best['verbose'] = -1
    best['subsample_freq'] = 1
    if use_gpu:
        best['device'] = 'gpu'
        best['gpu_use_dp'] = False
    print(f"Best LGB params: {best}")
    return best


def add_dart_params_xgb(params):
    params['booster'] = 'dart'
    params.setdefault('rate_drop', 0.1)
    params.setdefault('skip_drop', 0.5)
    return params


def add_dart_params_lgb(params):
    params['boosting_type'] = 'dart'
    params.setdefault('drop_rate', 0.1)
    params.setdefault('skip_drop', 0.5)
    return params


def save_checkpoint(path, model, task, calibrator=None, n_features=None):
    checkpoint = {
        'model': model,
        'task': task,
        'calibrator': calibrator,
        'n_features': n_features,
        'version': '2.0',
        'trained_at': datetime.now().isoformat(),
    }
    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"Saved: {path}")


def train_and_save(X, y, dates, task, framework, path, params, use_gpu,
                   use_dart, calibrate, weights):
    """Train a single model, optionally calibrate, and save."""
    n = len(X)
    # Use last 15% as validation for calibration/early stopping
    val_size = max(int(n * 0.15), 50)
    X_train, y_train = X[:-val_size], y[:-val_size]
    X_val, y_val = X[-val_size:], y[-val_size:]
    w_train = weights[:-val_size] if weights is not None else None

    if use_dart and framework == 'xgb':
        params = add_dart_params_xgb(params.copy())
    elif use_dart and framework == 'lgb':
        params = add_dart_params_lgb(params.copy())

    if framework == 'xgb':
        if use_gpu:
            params['tree_method'] = 'hist'
            params['device'] = 'cuda'
        if task == 'classification':
            model = xgb.XGBClassifier(**params)
        else:
            model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, sample_weight=w_train,
                  eval_set=[(X_val, y_val)], verbose=50)
    else:  # lgb
        if use_gpu:
            params['device'] = 'gpu'
            params['gpu_use_dp'] = False
        if task == 'classification':
            model = lgb.LGBMClassifier(**params)
            eval_metric = 'logloss'
        else:
            model = lgb.LGBMRegressor(**params)
            eval_metric = 'rmse'
        model.fit(X_train, y_train, sample_weight=w_train,
                  eval_set=[(X_val, y_val)], eval_metric=eval_metric,
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)])

    # Evaluate on val
    if task == 'classification':
        pred = model.predict_proba(X_val)[:, 1]
        ll = log_loss(y_val, pred)
        acc = accuracy_score(y_val, model.predict(X_val))
        print(f"  Val accuracy: {acc:.4f}, log_loss: {ll:.4f}")
    else:
        pred = model.predict(X_val)
        rmse = float(np.sqrt(((pred - y_val) ** 2).mean()))
        mae = float(np.abs(pred - y_val).mean())
        print(f"  Val RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # Calibration (improvement f)
    calibrator = None
    if calibrate and task == 'classification':
        print("  Fitting calibrator (isotonic)...")
        calibrator = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
        calibrator.fit(X_val, y_val)
        cal_pred = calibrator.predict_proba(X_val)[:, 1]
        cal_ll = log_loss(y_val, cal_pred)
        print(f"  Calibrated log_loss: {cal_ll:.4f}")

    save_checkpoint(path, model, task, calibrator=calibrator, n_features=X.shape[1])

    # Feature importance
    importance = model.feature_importances_
    top_idx = np.argsort(importance)[::-1][:10]
    print("  Top 10 features:", [(f"f{i}", round(float(importance[i]), 4)) for i in top_idx])

    return model, calibrator


def main():
    parser = argparse.ArgumentParser(description='Train XGBoost/LightGBM v2 models')
    parser.add_argument('--no-gpu', action='store_true')
    parser.add_argument('--moneyline-only', action='store_true')
    parser.add_argument('--totals-only', action='store_true')
    parser.add_argument('--spread-only', action='store_true')
    parser.add_argument('--tune', action='store_true', help='Optuna hyperparameter tuning')
    parser.add_argument('--n-trials', type=int, default=100, help='Optuna trials')
    parser.add_argument('--dart', action='store_true', help='Use DART booster')
    parser.add_argument('--calibrate', action='store_true', help='Platt/isotonic calibration')
    parser.add_argument('--conference-encoding', action='store_true', help='Conference target encoding')
    parser.add_argument('--xgb-only', action='store_true')
    parser.add_argument('--lgb-only', action='store_true')
    parser.add_argument('--2026-only', action='store_true', dest='only_2026',
                        help='Train on 2026 data only (no historical)')
    parser.add_argument('--model-suffix', type=str, default=None,
                        help='Custom suffix for model filenames')
    args = parser.parse_args()

    print("=" * 60)
    print("GRADIENT BOOSTING v2 TRAINING")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Flags: tune={args.tune}, dart={args.dart}, calibrate={args.calibrate}, "
          f"conference={args.conference_encoding}")
    print("=" * 60)

    use_gpu = not args.no_gpu and check_gpu()

    # Load data
    if args.only_2026:
        X, y_totals, y_spread, dates = load_2026_only_data(
            use_conference_encoding=args.conference_encoding)
    else:
        X, y_totals, y_spread, dates = load_all_data(
            use_conference_encoding=args.conference_encoding)
    y_ml = (y_spread > 0).astype(np.int32)
    weights = compute_recency_weights(dates)

    train_all = not (args.moneyline_only or args.totals_only or args.spread_only)
    do_xgb = not args.lgb_only
    do_lgb = not args.xgb_only

    tasks = []
    if train_all or args.moneyline_only:
        tasks.append(('moneyline', 'classification', y_ml))
    if train_all or args.totals_only:
        tasks.append(('totals', 'regression', y_totals))
    if train_all or args.spread_only:
        tasks.append(('spread', 'regression', y_spread))

    # Build path map, with optional custom suffix
    suffix = args.model_suffix
    if suffix:
        path_map = {
            ('xgb', 'moneyline'): DATA_DIR / f"xgb_moneyline{suffix}",
            ('xgb', 'totals'): DATA_DIR / f"xgb_totals{suffix}",
            ('xgb', 'spread'): DATA_DIR / f"xgb_spread{suffix}",
            ('lgb', 'moneyline'): DATA_DIR / f"lgb_moneyline{suffix}",
            ('lgb', 'totals'): DATA_DIR / f"lgb_totals{suffix}",
            ('lgb', 'spread'): DATA_DIR / f"lgb_spread{suffix}",
        }
    else:
        path_map = {
            ('xgb', 'moneyline'): XGB_ML_V2,
            ('xgb', 'totals'): XGB_TOT_V2,
            ('xgb', 'spread'): XGB_SPR_V2,
            ('lgb', 'moneyline'): LGB_ML_V2,
            ('lgb', 'totals'): LGB_TOT_V2,
            ('lgb', 'spread'): LGB_SPR_V2,
        }

    for name, task, y in tasks:
        for fw in ['xgb', 'lgb']:
            if fw == 'xgb' and not do_xgb:
                continue
            if fw == 'lgb' and not do_lgb:
                continue
            if fw == 'xgb' and not XGB_AVAILABLE:
                print(f"Skipping XGBoost {name} (not installed)")
                continue
            if fw == 'lgb' and not LGB_AVAILABLE:
                print(f"Skipping LightGBM {name} (not installed)")
                continue

            print(f"\n{'='*60}")
            print(f"Training {fw.upper()} {name.upper()} (v2)")
            print(f"{'='*60}")

            start = time.time()
            if args.tune:
                if fw == 'xgb':
                    params = tune_xgb(X, y, weights, task, use_gpu,
                                      args.dart, args.n_trials)
                else:
                    params = tune_lgb(X, y, weights, task, use_gpu,
                                      args.dart, args.n_trials)
            else:
                params = (DEFAULT_XGB_PARAMS if fw == 'xgb' else DEFAULT_LGB_PARAMS).copy()

            path = path_map[(fw, name)]
            train_and_save(X, y, dates, task, fw, path, params, use_gpu,
                           args.dart, args.calibrate, weights)
            print(f"  Time: {time.time()-start:.1f}s")

    print(f"\n{'='*60}")
    print(f"v2 Training complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
