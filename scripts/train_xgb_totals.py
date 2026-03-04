#!/usr/bin/env python3
"""
Train XGBoost totals model — v2

Trains three models:
1. XGBRegressor for raw total runs prediction
2. XGBClassifier for OVER/UNDER classification
3. XGBRegressor for market deviation (actual - line)

Uses walk-forward validation to avoid future data leaks.
Features from team_batting_quality, team_pitching_quality, game_weather,
betting_lines, and game context.

Usage:
    python3 scripts/train_xgb_totals.py
    python3 scripts/train_xgb_totals.py --no-deviation   # skip deviation model
"""

import sys
import pickle
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from xgboost import XGBRegressor, XGBClassifier
from scripts.database import get_connection
from models.xgb_totals_model import (
    build_features, BATTING_FEATURES, PITCHING_FEATURES, FEATURE_NAMES, NUM_FEATURES
)

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_PATH = DATA_DIR / "xgb_totals_v2.pkl"


def load_training_data():
    """Load all completed games with batting+pitching quality data."""
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        SELECT g.id, g.date, g.home_team_id, g.away_team_id,
               g.home_score, g.away_score,
               g.is_conference_game, g.is_neutral_site
        FROM games g
        JOIN team_batting_quality hb ON g.home_team_id = hb.team_id
        JOIN team_batting_quality ab ON g.away_team_id = ab.team_id
        JOIN team_pitching_quality hp ON g.home_team_id = hp.team_id
        JOIN team_pitching_quality ap ON g.away_team_id = ap.team_id
        WHERE g.status = 'final'
          AND g.home_score IS NOT NULL
          AND g.away_score IS NOT NULL
        ORDER BY g.date ASC
    """)
    games = [dict(row) for row in c.fetchall()]

    # Load all batting/pitching quality into dicts
    batting = {}
    c.execute("SELECT * FROM team_batting_quality")
    for row in c.fetchall():
        batting[row['team_id']] = dict(row)

    pitching = {}
    c.execute("SELECT * FROM team_pitching_quality")
    for row in c.fetchall():
        pitching[row['team_id']] = dict(row)

    # Load weather
    weather = {}
    c.execute("SELECT game_id, temp_f, wind_speed_mph, humidity_pct, is_dome FROM game_weather")
    for row in c.fetchall():
        weather[row['game_id']] = dict(row)

    # Load betting lines (most recent per game)
    lines = {}
    c.execute("""
        SELECT game_id, over_under
        FROM betting_lines
        WHERE over_under IS NOT NULL AND over_under > 0
        ORDER BY captured_at ASC
    """)
    for row in c.fetchall():
        if row['game_id']:
            lines[row['game_id']] = float(row['over_under'])

    conn.close()
    return games, batting, pitching, weather, lines


def build_dataset(games, batting, pitching, weather, lines):
    """Build feature matrix and targets."""
    X, y_total, y_ou, y_dev = [], [], [], []
    game_dates = []
    has_line = []
    skipped = 0

    for g in games:
        hid = g['home_team_id']
        aid = g['away_team_id']

        hb = batting.get(hid)
        ab = batting.get(aid)
        hp = pitching.get(hid)
        ap = pitching.get(aid)

        if not all([hb, ab, hp, ap]):
            skipped += 1
            continue

        w = weather.get(g['id'])
        line = lines.get(g['id'])
        actual_total = g['home_score'] + g['away_score']

        features = build_features(
            hb, ab, hp, ap,
            weather=w, line=line, game_date=g['date'],
            is_conference=g.get('is_conference_game', 0) or 0,
            is_neutral=g.get('is_neutral_site', 0) or 0,
        )

        X.append(features)
        y_total.append(float(actual_total))
        game_dates.append(g['date'])

        if line is not None and line > 0:
            y_ou.append(1.0 if actual_total > line else 0.0)
            y_dev.append(float(actual_total - line))
            has_line.append(True)
        else:
            y_ou.append(np.nan)
            y_dev.append(np.nan)
            has_line.append(False)

    if skipped:
        print(f"  Skipped {skipped} games (missing quality data)")

    return (np.array(X, dtype=np.float32),
            np.array(y_total, dtype=np.float32),
            np.array(y_ou, dtype=np.float32),
            np.array(y_dev, dtype=np.float32),
            game_dates,
            np.array(has_line))


def walk_forward_eval(X, y, dates, model_class, model_params, n_folds=5, label=""):
    """Walk-forward validation: train on past, predict future."""
    unique_dates = sorted(set(dates))
    n = len(unique_dates)
    fold_size = n // (n_folds + 1)

    all_preds = []
    all_actuals = []
    all_indices = []

    for fold in range(n_folds):
        train_end_idx = (fold + 1) * fold_size + fold_size
        val_start_idx = train_end_idx
        val_end_idx = min(val_start_idx + fold_size, n)

        if val_start_idx >= n:
            break

        train_cutoff = unique_dates[min(train_end_idx, n - 1)]
        val_start_date = unique_dates[min(val_start_idx, n - 1)]
        val_end_date = unique_dates[min(val_end_idx - 1, n - 1)]

        train_mask = np.array([d <= train_cutoff for d in dates])
        val_mask = np.array([val_start_date <= d <= val_end_date for d in dates])

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        if len(X_train) < 50 or len(X_val) < 10:
            continue

        model = model_class(**model_params)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        all_preds.extend(preds)
        all_actuals.extend(y_val)
        all_indices.extend(np.where(val_mask)[0])

    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)

    mae = np.mean(np.abs(all_preds - all_actuals))
    rmse = np.sqrt(np.mean((all_preds - all_actuals) ** 2))
    print(f"  {label} Walk-forward ({n_folds} folds, {len(all_preds)} predictions):")
    print(f"    MAE: {mae:.3f}")
    print(f"    RMSE: {rmse:.3f}")
    return mae, rmse


def walk_forward_eval_classifier(X, y, dates, model_class, model_params, n_folds=5, label=""):
    """Walk-forward for O/U classifier."""
    unique_dates = sorted(set(dates))
    n = len(unique_dates)
    fold_size = n // (n_folds + 1)

    all_preds = []
    all_actuals = []

    for fold in range(n_folds):
        train_end_idx = (fold + 1) * fold_size + fold_size
        val_start_idx = train_end_idx
        val_end_idx = min(val_start_idx + fold_size, n)

        if val_start_idx >= n:
            break

        train_cutoff = unique_dates[min(train_end_idx, n - 1)]
        val_start_date = unique_dates[min(val_start_idx, n - 1)]
        val_end_date = unique_dates[min(val_end_idx - 1, n - 1)]

        train_mask = np.array([d <= train_cutoff for d in dates])
        val_mask = np.array([val_start_date <= d <= val_end_date for d in dates])

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        if len(X_train) < 30 or len(X_val) < 10:
            continue

        # Need both classes in training
        if len(np.unique(y_train)) < 2:
            continue

        model = model_class(**model_params)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        all_preds.extend(preds)
        all_actuals.extend(y_val)

    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)

    if len(all_preds) > 0:
        accuracy = np.mean(all_preds == all_actuals)
        over_mask = all_actuals == 1
        under_mask = all_actuals == 0
        over_acc = np.mean(all_preds[over_mask] == 1) if over_mask.sum() > 0 else 0
        under_acc = np.mean(all_preds[under_mask] == 0) if under_mask.sum() > 0 else 0
        pred_over_pct = np.mean(all_preds == 1)
        print(f"  {label} Walk-forward classifier ({n_folds} folds, {len(all_preds)} predictions):")
        print(f"    O/U Accuracy: {accuracy:.1%}")
        print(f"    OVER accuracy: {over_acc:.1%} | UNDER accuracy: {under_acc:.1%}")
        print(f"    Predicted OVER rate: {pred_over_pct:.1%}")
        return accuracy
    return 0.0


def train(no_deviation=False):
    print("=" * 60)
    print("XGBoost Totals Model — Training v2")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    print("\nLoading data...")
    games, batting, pitching, weather, lines = load_training_data()
    print(f"  Games: {len(games)}")
    print(f"  Teams with batting quality: {len(batting)}")
    print(f"  Teams with pitching quality: {len(pitching)}")
    print(f"  Games with weather: {len(weather)}")
    print(f"  Games with lines: {len(lines)}")

    print("\nBuilding features...")
    X, y_total, y_ou, y_dev, dates, has_line = build_dataset(
        games, batting, pitching, weather, lines
    )
    print(f"  Feature matrix: {X.shape}")
    print(f"  Target range: {y_total.min():.0f}-{y_total.max():.0f}, mean: {y_total.mean():.1f}")
    print(f"  Games with O/U lines: {has_line.sum():.0f}")

    # ── Task 1: Regressor ──────────────────────────────────
    print("\n" + "=" * 60)
    print("TASK 1: XGBRegressor (total runs)")
    print("=" * 60)

    reg_params = {
        'n_estimators': 300,
        'max_depth': 5,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'objective': 'reg:squarederror',
    }

    print("\nWalk-forward validation...")
    reg_mae, reg_rmse = walk_forward_eval(
        X, y_total, dates, XGBRegressor, reg_params, n_folds=5, label="Regressor"
    )

    print("\nTraining final regressor on all data...")
    regressor = XGBRegressor(**reg_params)
    regressor.fit(X, y_total)

    # Feature importance
    importances = regressor.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    print("\n  Top 10 features:")
    for i in top_idx:
        print(f"    {FEATURE_NAMES[i]:40s} {importances[i]:.4f}")

    # ── Task 2: O/U Classifier ─────────────────────────────
    print("\n" + "=" * 60)
    print("TASK 2: XGBClassifier (OVER/UNDER)")
    print("=" * 60)

    line_mask = has_line
    X_line = X[line_mask]
    y_ou_line = y_ou[line_mask]
    dates_line = [d for d, h in zip(dates, has_line) if h]

    # Remove NaN/push games
    valid_ou = ~np.isnan(y_ou_line)
    X_line = X_line[valid_ou]
    y_ou_line = y_ou_line[valid_ou]
    dates_line = [d for d, v in zip(dates_line, valid_ou) if v]

    print(f"  Games with lines: {len(X_line)}")
    print(f"  OVER rate: {y_ou_line.mean():.1%}")

    clf_params = {
        'n_estimators': 200,
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'scale_pos_weight': 1.0,
    }

    if len(X_line) >= 50:
        ou_acc = walk_forward_eval_classifier(
            X_line, y_ou_line, dates_line, XGBClassifier, clf_params,
            n_folds=3, label="O/U Classifier"
        )

        print("\nTraining final classifier on all line data...")
        classifier = XGBClassifier(**clf_params)
        classifier.fit(X_line, y_ou_line)
    else:
        print("  Not enough line data for classifier")
        classifier = None

    # ── Task 3: Deviation model ────────────────────────────
    deviation_model = None
    if not no_deviation:
        print("\n" + "=" * 60)
        print("TASK 3: Deviation Model (actual - line)")
        print("=" * 60)

        y_dev_line = y_dev[line_mask]
        y_dev_line = y_dev_line[valid_ou]

        print(f"  Games: {len(X_line)}")
        print(f"  Deviation range: {y_dev_line.min():.1f} to {y_dev_line.max():.1f}")
        print(f"  Mean deviation: {y_dev_line.mean():.2f}")
        print(f"  Std deviation: {y_dev_line.std():.2f}")

        dev_params = {
            'n_estimators': 200,
            'max_depth': 4,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,
            'reg_alpha': 0.5,
            'reg_lambda': 2.0,
            'random_state': 42,
            'objective': 'reg:squarederror',
        }

        if len(X_line) >= 50:
            dev_mae, dev_rmse = walk_forward_eval(
                X_line, y_dev_line, dates_line, XGBRegressor, dev_params,
                n_folds=3, label="Deviation"
            )

            print("\nTraining final deviation model...")
            deviation_model = XGBRegressor(**dev_params)
            deviation_model.fit(X_line, y_dev_line)
        else:
            print("  Not enough line data for deviation model")

    # ── Save ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)

    save_data = {
        'regressor': regressor,
        'classifier': classifier,
        'deviation_model': deviation_model,
        'feature_names': FEATURE_NAMES,
        'num_features': NUM_FEATURES,
        'train_size': len(X),
        'line_data_size': len(X_line) if line_mask.any() else 0,
        'reg_mae': reg_mae,
        'reg_rmse': reg_rmse,
        'saved_at': datetime.now().isoformat(),
        'version': 'v2',
    }

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"  Saved to {MODEL_PATH}")

    # ── Summary ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Features: {NUM_FEATURES}")
    print(f"  Training games: {len(X)}")
    print(f"  Regressor MAE: {reg_mae:.3f} | RMSE: {reg_rmse:.3f}")
    if classifier:
        print(f"  O/U Classifier trained on {len(X_line)} games with lines")
    if deviation_model:
        print(f"  Deviation model trained on {len(X_line)} games with lines")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-deviation', action='store_true',
                        help='Skip training deviation model')
    args = parser.parse_args()
    train(no_deviation=args.no_deviation)
