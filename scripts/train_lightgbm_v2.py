#!/usr/bin/env python3
"""
Train LightGBM moneyline model with batting-specialized features.

Uses BattingFeatureComputer (batting-only features) instead of the generic
FeatureComputer. Walk-forward training with recency weighting.

Saves to data/lgb_moneyline.pkl (overwrites existing model).
"""

import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime, date
from math import exp

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.database import get_connection
from models.features_batting import BattingFeatureComputer, HistoricalBattingFeatureComputer

import lightgbm as lgb
from sklearn.metrics import log_loss, accuracy_score

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_PATH = DATA_DIR / "lgb_moneyline.pkl"


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
    return np.array(weights, dtype=np.float32)


def load_data():
    """Load 2026 final games with batting features.

    Uses only 2026 data where team_batting_quality features are available,
    ensuring the model learns from batting-specific signals rather than
    defaulting to generic run production.
    """
    print("Loading 2026 data with batting features...")
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        SELECT id, home_team_id, away_team_id, home_score, away_score,
               date, is_neutral_site
        FROM games
        WHERE status = 'final' AND home_score IS NOT NULL AND away_score IS NOT NULL
        ORDER BY date ASC
    """)
    current_rows = c.fetchall()
    print(f"  2026 final games: {len(current_rows)}")
    conn.close()

    fc_live = BattingFeatureComputer()
    features, labels, dates = [], [], []

    for row in current_rows:
        try:
            feat = fc_live.compute_features(
                row['home_team_id'], row['away_team_id'],
                game_date=row['date'],
                neutral_site=bool(row['is_neutral_site']))
            feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
            label = 1.0 if row['home_score'] > row['away_score'] else 0.0
            features.append(feat)
            labels.append(label)
            dates.append(row['date'])
        except Exception:
            continue

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  Total samples: {len(X)}, features: {X.shape[1]}")
    return X, y, dates


def train():
    print("=" * 60)
    print("LIGHTGBM BATTING-SPECIALIZED TRAINING")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    X, y, dates = load_data()
    weights = compute_recency_weights(dates)

    # Split: last 15% for validation
    n = len(X)
    val_size = max(int(n * 0.15), 50)
    X_train, y_train = X[:-val_size], y[:-val_size]
    X_val, y_val = X[-val_size:], y[-val_size:]
    w_train = weights[:-val_size]

    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}")

    # Normalize features
    feature_mean = X_train.mean(axis=0)
    feature_std = X_train.std(axis=0)
    feature_std[feature_std < 1e-8] = 1.0
    X_train_norm = (X_train - feature_mean) / feature_std
    X_val_norm = (X_val - feature_mean) / feature_std

    params = {
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
        'device': 'gpu',
        'gpu_use_dp': False,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train_norm, y_train,
        sample_weight=w_train,
        eval_set=[(X_val_norm, y_val)],
        eval_metric='logloss',
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
    )

    # Evaluate
    y_pred = model.predict(X_val_norm)
    y_prob = model.predict_proba(X_val_norm)[:, 1]
    acc = accuracy_score(y_val, y_pred)
    ll = log_loss(y_val, y_prob)
    print(f"\nVal accuracy: {acc:.4f}, log_loss: {ll:.4f}")

    # Feature importance
    importance = model.feature_importances_
    fc = BattingFeatureComputer()
    feat_names = fc.get_feature_names()
    if len(feat_names) < len(importance):
        feat_names += [f'f{i}' for i in range(len(feat_names), len(importance))]
    top_idx = np.argsort(importance)[::-1][:10]
    print("\nTop 10 features:")
    for i in top_idx:
        name = feat_names[i] if i < len(feat_names) else f'f{i}'
        print(f"  {name}: {importance[i]:.4f}")

    # Save
    import pickle
    checkpoint = {
        'model': model,
        'feature_mean': feature_mean,
        'feature_std': feature_std,
        'task': 'classification',
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"\nSaved: {MODEL_PATH}")
    print(f"Done: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return acc, ll


if __name__ == "__main__":
    train()
