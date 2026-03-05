#!/usr/bin/env python3
"""
Train LightGBM moneyline model with batting-specialized features.

Uses BattingFeatureComputer (batting-only features) and strict date-based
walk-forward evaluation (train date < D, test date == D).

Saves to data/lgb_moneyline.pkl (overwrites existing model).
"""

import argparse
import json
import pickle
import sys
import numpy as np
from pathlib import Path
from datetime import datetime, date
from math import exp

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.database import get_connection
from models.features_batting import BattingFeatureComputer
from scripts.walkforward_utils import build_strict_date_folds, aggregate_binary_oof

import lightgbm as lgb

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
    rows = c.fetchall()
    print(f"  final games: {len(rows)}")
    conn.close()

    fc = BattingFeatureComputer()
    features, labels, dates = [], [], []

    for row in rows:
        try:
            feat = fc.compute_features(
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
    dates = np.array(dates, dtype=object)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  samples: {len(X)}, features: {X.shape[1] if len(X) else 0}")
    return X, y, dates


def _lgb_params(device='gpu'):
    return {
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
        'device': device,
        'gpu_use_dp': False,
    }


def _fit_lgb(X_train, y_train, w_train):
    model = lgb.LGBMClassifier(**_lgb_params(device='gpu'))
    try:
        model.fit(X_train, y_train, sample_weight=w_train)
        return model
    except Exception:
        model = lgb.LGBMClassifier(**_lgb_params(device='cpu'))
        model.fit(X_train, y_train, sample_weight=w_train)
        return model


def write_report(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# LightGBM Walk-Forward Report",
        "",
        f"- generated_at: {payload['generated_at']}",
        f"- folds: {payload['folds']}",
        f"- oof_n: {payload['oof_metrics']['n']}",
        f"- oof_accuracy: {payload['oof_metrics']['accuracy']}",
        f"- oof_brier: {payload['oof_metrics']['brier']}",
        f"- oof_logloss: {payload['oof_metrics']['logloss']}",
        f"- final_train_size: {payload['final_train_size']}",
        "",
        "```json",
        json.dumps(payload, indent=2),
        "```",
    ]
    path.write_text("\n".join(lines), encoding='utf-8')


def train(min_warmup=200, report_path=None):
    print("=" * 60)
    print("LIGHTGBM BATTING-SPECIALIZED TRAINING")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    X, y, dates = load_data()
    weights = compute_recency_weights(dates)

    folds = build_strict_date_folds(dates, min_warmup=min_warmup)
    oof_probs, oof_true = [], []

    for fold in folds:
        tr_idx = fold.train_idx
        te_idx = fold.test_idx

        X_train, y_train = X[tr_idx], y[tr_idx]
        X_test, y_test = X[te_idx], y[te_idx]
        w_train = weights[tr_idx]

        if len(np.unique(y_train)) < 2:
            continue

        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std < 1e-8] = 1.0

        X_train_norm = (X_train - mean) / std
        X_test_norm = (X_test - mean) / std

        model = _fit_lgb(X_train_norm, y_train, w_train)
        probs = model.predict_proba(X_test_norm)[:, 1]

        oof_probs.extend(probs.tolist())
        oof_true.extend(y_test.tolist())

    metrics = aggregate_binary_oof(np.array(oof_true), np.array(oof_probs))

    # Final fit on all eligible rows
    feature_mean = X.mean(axis=0)
    feature_std = X.std(axis=0)
    feature_std[feature_std < 1e-8] = 1.0
    X_norm = (X - feature_mean) / feature_std

    final_model = _fit_lgb(X_norm, y, weights)

    checkpoint = {
        'model': final_model,
        'feature_mean': feature_mean,
        'feature_std': feature_std,
        'task': 'classification',
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(checkpoint, f)

    print(f"\nfolds: {len(folds)}")
    print(f"OOF metrics: acc={metrics['accuracy']}, brier={metrics['brier']}, logloss={metrics['logloss']}, n={metrics['n']}")
    print(f"final train size: {len(X)}")
    print(f"Saved: {MODEL_PATH}")

    payload = {
        'model': 'lightgbm_v2',
        'generated_at': datetime.now().isoformat(),
        'folds': len(folds),
        'min_warmup': int(min_warmup),
        'oof_metrics': metrics,
        'final_train_size': int(len(X)),
        'model_path': str(MODEL_PATH),
    }
    if report_path:
        write_report(report_path, payload)
        print(f"report: {report_path}")

    return payload


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--min-warmup', type=int, default=200)
    p.add_argument('--report-path', type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(min_warmup=args.min_warmup, report_path=args.report_path)
