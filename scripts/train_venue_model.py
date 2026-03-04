#!/usr/bin/env python3
"""
Train the venue model with walk-forward validation.

Usage:
    python scripts/train_venue_model.py
"""

import sys
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.database import get_connection
from models.venue_model import VenueModel, haversine, MODEL_PATH
from models.venue_model import (DEFAULT_ELEVATION, DEFAULT_LATITUDE,
                                DEFAULT_LONGITUDE, DEFAULT_CAPACITY,
                                DEFAULT_HOME_WIN_PCT, DEFAULT_AVG_TOTAL_RUNS)


def load_training_data():
    """Load completed games with venue info for training."""
    conn = get_connection()
    c = conn.cursor()

    # Get all completed games
    c.execute("""
        SELECT g.id, g.date, g.home_team_id, g.away_team_id,
               g.home_score, g.away_score, g.winner_id
        FROM games g
        WHERE g.status = 'final'
          AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL
        ORDER BY g.date
    """)
    games = [dict(r) for r in c.fetchall()]

    # Load all venues
    c.execute("SELECT * FROM venues")
    venues = {r['team_id']: dict(r) for r in c.fetchall()}

    conn.close()
    return games, venues


def compute_venue_stats_before_date(games, date_str):
    """Compute per-venue home win% and avg total runs from games before date."""
    stats = {}
    for g in games:
        if g['date'] >= date_str:
            break
        hid = g['home_team_id']
        if hid not in stats:
            stats[hid] = {'games': 0, 'home_wins': 0, 'total_runs': 0}
        stats[hid]['games'] += 1
        if g['winner_id'] == hid:
            stats[hid]['home_wins'] += 1
        stats[hid]['total_runs'] += (g['home_score'] + g['away_score'])
    result = {}
    for tid, s in stats.items():
        if s['games'] > 0:
            result[tid] = {
                'home_win_pct': s['home_wins'] / s['games'],
                'avg_total_runs': s['total_runs'] / s['games'],
            }
    return result


def build_features_for_game(game, venues, venue_stats):
    """Build feature vector for a single game."""
    hid = game['home_team_id']
    aid = game['away_team_id']

    hv = venues.get(hid, {})
    av = venues.get(aid, {})

    elevation = hv.get('elevation_ft') or DEFAULT_ELEVATION
    is_dome = hv.get('is_dome') or 0
    capacity = hv.get('capacity') or DEFAULT_CAPACITY
    latitude = hv.get('latitude') or DEFAULT_LATITUDE

    home_lat = hv.get('latitude') or DEFAULT_LATITUDE
    home_lon = hv.get('longitude') or DEFAULT_LONGITUDE
    away_lat = av.get('latitude') or DEFAULT_LATITUDE
    away_lon = av.get('longitude') or DEFAULT_LONGITUDE
    travel = haversine(away_lat, away_lon, home_lat, home_lon)

    vs = venue_stats.get(hid, {})
    home_win_pct = vs.get('home_win_pct', DEFAULT_HOME_WIN_PCT)
    avg_total = vs.get('avg_total_runs', DEFAULT_AVG_TOTAL_RUNS)

    return [elevation, is_dome, capacity, latitude,
            home_win_pct, avg_total, travel]


def main():
    print("=" * 70)
    print("VENUE MODEL TRAINING")
    print("=" * 70)

    games, venues = load_training_data()
    print(f"Loaded {len(games)} completed games, {len(venues)} venues")

    # Build dataset with walk-forward venue stats
    X_all, y_all, dates_all = [], [], []
    # Pre-sort games by date
    games.sort(key=lambda g: g['date'])

    # Need at least 50 games to compute meaningful venue stats
    min_warmup = 50
    for i, game in enumerate(games):
        if i < min_warmup:
            continue
        vs = compute_venue_stats_before_date(games, game['date'])
        feats = build_features_for_game(game, venues, vs)
        X_all.append(feats)
        y_all.append(1 if game['winner_id'] == game['home_team_id'] else 0)
        dates_all.append(game['date'])

    X = np.array(X_all, dtype=np.float64)
    y = np.array(y_all)
    dates = np.array(dates_all)

    print(f"Dataset: {len(X)} games, {y.mean():.1%} home win rate")

    # Walk-forward validation: train on first 70%, test on last 30%
    split = int(len(X) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Standardize
    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0)
    X_train_s = (X_train - means) / (stds + 1e-8)
    X_test_s = (X_test - means) / (stds + 1e-8)

    # Train logistic regression
    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(X_train_s, y_train)

    train_acc = lr.score(X_train_s, y_train)
    test_acc = lr.score(X_test_s, y_test)

    print(f"\nWalk-forward validation:")
    print(f"  Train: {len(X_train)} games, accuracy {train_acc:.1%}")
    print(f"  Test:  {len(X_test)} games, accuracy {test_acc:.1%}")

    # Feature importance (coefficient magnitudes)
    print(f"\nFeature coefficients:")
    for name, coef in zip(VenueModel.FEATURE_NAMES, lr.coef_[0]):
        bar = "+" * int(abs(coef) * 10) if coef > 0 else "-" * int(abs(coef) * 10)
        print(f"  {name:<30} {coef:>8.4f} {bar}")

    # Retrain on full dataset
    means_full = X.mean(axis=0)
    stds_full = X.std(axis=0)
    X_full_s = (X - means_full) / (stds_full + 1e-8)
    lr_full = LogisticRegression(max_iter=1000, C=1.0)
    lr_full.fit(X_full_s, y)

    # Save
    save_data = {
        'model': lr_full,
        'feature_means': means_full,
        'feature_stds': stds_full,
        'feature_names': VenueModel.FEATURE_NAMES,
        'train_size': len(X),
        'test_accuracy': test_acc,
        'saved_at': datetime.now().isoformat(),
        'version': '1.0',
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nModel saved to {MODEL_PATH}")

    # Correlation with existing models
    print("\n" + "=" * 70)
    print("Checking correlation with existing models...")
    _check_correlation(X_test, y_test, lr, means, stds)

    return 0


def _check_correlation(X_test, y_test, model, means, stds):
    """Check correlation of venue model predictions with existing models."""
    try:
        from models.elo_model import EloModel
        elo = EloModel()
    except Exception:
        print("  (Could not load elo model for correlation check)")
        return

    conn = get_connection()
    c = conn.cursor()

    # Get a sample of test predictions
    X_s = (X_test - means) / (stds + 1e-8)
    venue_probs = model.predict_proba(X_s)[:, 1]

    # Get some elo predictions for the same games
    c.execute("""
        SELECT g.home_team_id, g.away_team_id
        FROM games g WHERE status = 'final'
        ORDER BY date DESC LIMIT ?
    """, (len(X_test),))
    sample_games = [dict(r) for r in c.fetchall()]
    conn.close()

    elo_probs = []
    for g in sample_games[:min(200, len(sample_games))]:
        try:
            pred = elo.predict_game(g['home_team_id'], g['away_team_id'])
            elo_probs.append(pred['home_win_probability'])
        except Exception:
            elo_probs.append(0.5)

    if len(elo_probs) >= 50:
        n = min(len(elo_probs), len(venue_probs))
        corr = np.corrcoef(venue_probs[:n], elo_probs[:n])[0, 1]
        print(f"  Correlation with Elo model: r={corr:.3f}")
    else:
        print("  (Insufficient data for correlation check)")


if __name__ == "__main__":
    sys.exit(main())
