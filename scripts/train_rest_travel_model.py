#!/usr/bin/env python3
"""
Train the rest/travel model with walk-forward validation.

Usage:
    python scripts/train_rest_travel_model.py
"""

import sys
import pickle
import math
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.database import get_connection
from models.rest_travel_model import RestTravelModel, haversine, MODEL_PATH

DEFAULT_LATITUDE = 34.0
DEFAULT_LONGITUDE = -86.0


def load_data():
    """Load games and venues."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT id, date, home_team_id, away_team_id,
               home_score, away_score, winner_id
        FROM games
        WHERE status = 'final' AND home_score IS NOT NULL
        ORDER BY date
    """)
    games = [dict(r) for r in c.fetchall()]

    c.execute("SELECT team_id, latitude, longitude FROM venues")
    venues = {}
    for r in c.fetchall():
        venues[r['team_id']] = (r['latitude'] or DEFAULT_LATITUDE,
                                r['longitude'] or DEFAULT_LONGITUDE)
    conn.close()
    return games, venues


def get_team_schedule_features(team_id, game_date, is_home, games_before,
                               venues, game_venue_team):
    """Compute rest/travel features for a team for a specific game."""
    if isinstance(game_date, str):
        game_date = datetime.strptime(game_date, '%Y-%m-%d').date()

    # Recent games for this team (within 14 days)
    cutoff = game_date - timedelta(days=14)
    recent = []
    for g in reversed(games_before):
        gd = datetime.strptime(g['date'], '%Y-%m-%d').date()
        if gd < cutoff:
            break
        if g['home_team_id'] == team_id or g['away_team_id'] == team_id:
            recent.append((g, gd))

    # days_since_last
    if recent:
        days_since = (game_date - recent[0][1]).days
    else:
        days_since = 7
    days_since = min(days_since, 7)

    # games in last 7 / 3 days
    cutoff_7 = game_date - timedelta(days=7)
    cutoff_3 = game_date - timedelta(days=3)
    games_7 = sum(1 for _, gd in recent if gd > cutoff_7)
    games_3 = sum(1 for _, gd in recent if gd > cutoff_3)

    # midweek
    is_midweek = 1 if game_date.weekday() in (1, 2, 3) else 0

    # Travel distance: from last game venue to today's game venue
    today_loc = venues.get(game_venue_team, (DEFAULT_LATITUDE, DEFAULT_LONGITUDE))
    if recent:
        last_venue_team = recent[0][0]['home_team_id']
        last_loc = venues.get(last_venue_team, (DEFAULT_LATITUDE, DEFAULT_LONGITUDE))
    else:
        last_loc = venues.get(team_id, (DEFAULT_LATITUDE, DEFAULT_LONGITUDE))
    travel = haversine(last_loc[0], last_loc[1], today_loc[0], today_loc[1])

    # home_stand / road_trip
    home_stand = 0
    road_trip = 0
    for g, _ in recent:
        g_home = g['home_team_id'] == team_id
        if is_home and g_home:
            home_stand += 1
        elif not is_home and not g_home:
            road_trip += 1
        else:
            break

    crossed_tz = 1 if travel > 500 else 0

    return [days_since, games_7, games_3, is_midweek,
            travel, home_stand, road_trip, crossed_tz]


def build_dataset(games, venues):
    """Build feature matrix from all games."""
    X, y, dates = [], [], []
    # Need some history before computing features
    for i in range(50, len(games)):
        game = games[i]
        hid = game['home_team_id']
        aid = game['away_team_id']

        games_before = games[:i]

        home_feats = get_team_schedule_features(
            hid, game['date'], True, games_before, venues, hid)
        away_feats = get_team_schedule_features(
            aid, game['date'], False, games_before, venues, hid)

        # Differential features
        diff = [h - a for h, a in zip(home_feats, away_feats)]

        X.append(diff)
        y.append(1 if game['winner_id'] == hid else 0)
        dates.append(game['date'])

    return np.array(X, dtype=np.float64), np.array(y), dates


def main():
    print("=" * 70)
    print("REST/TRAVEL MODEL TRAINING")
    print("=" * 70)

    games, venues = load_data()
    print(f"Loaded {len(games)} games, {len(venues)} venues")

    X, y, dates = build_dataset(games, venues)
    print(f"Dataset: {len(X)} games, {y.mean():.1%} home win rate")

    # Walk-forward split
    split = int(len(X) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Standardize
    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0)
    X_train_s = (X_train - means) / (stds + 1e-8)
    X_test_s = (X_test - means) / (stds + 1e-8)

    # Train
    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(X_train_s, y_train)

    train_acc = lr.score(X_train_s, y_train)
    test_acc = lr.score(X_test_s, y_test)

    print(f"\nWalk-forward validation:")
    print(f"  Train: {len(X_train)} games, accuracy {train_acc:.1%}")
    print(f"  Test:  {len(X_test)} games, accuracy {test_acc:.1%}")

    # Feature coefficients
    print(f"\nFeature coefficients (differential: home - away):")
    for name, coef in zip(RestTravelModel.FEATURE_NAMES, lr.coef_[0]):
        bar = "+" * int(abs(coef) * 10) if coef > 0 else "-" * int(abs(coef) * 10)
        print(f"  {name:<25} {coef:>8.4f} {bar}")

    # Retrain on all data
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
        'feature_names': RestTravelModel.FEATURE_NAMES,
        'train_size': len(X),
        'test_accuracy': test_acc,
        'saved_at': datetime.now().isoformat(),
        'version': '1.0',
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nModel saved to {MODEL_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
