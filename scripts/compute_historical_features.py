#!/usr/bin/env python3
"""
Compute Historical Features for Neural Network Training

Processes historical_games chronologically, computes rolling Elo ratings
and team stats, outputs feature vectors ready for training.

CRITICAL: No data leakage. Features for game N use only games 0..N-1.

Usage:
    python3 scripts/compute_historical_features.py [--save-table] [--season 2024]
"""

import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.database import get_connection
from models.nn_features import HistoricalFeatureComputer


def load_historical_games(seasons=None):
    """Load historical games from database, ordered chronologically."""
    conn = get_connection()
    c = conn.cursor()

    query = """
        SELECT id, season, date, home_team, away_team,
               home_score, away_score, neutral_site, postseason
        FROM historical_games
    """
    params = []
    if seasons:
        placeholders = ','.join('?' * len(seasons))
        query += f" WHERE season IN ({placeholders})"
        params = list(seasons)

    query += " ORDER BY date ASC, id ASC"
    c.execute(query, params)

    games = []
    for row in c.fetchall():
        games.append({
            'id': row['id'],
            'season': row['season'],
            'date': row['date'],
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            'home_score': row['home_score'],
            'away_score': row['away_score'],
            'neutral_site': row['neutral_site'],
        })

    conn.close()
    return games


def compute_all_features(games, reset_between_seasons=True):
    """
    Process games chronologically and compute features.

    Args:
        games: List of game dicts, already sorted by date
        reset_between_seasons: Whether to reset Elo/stats at season boundaries

    Returns:
        (X, y, game_ids, seasons) - feature matrix, labels, game IDs, season per game
    """
    computer = HistoricalFeatureComputer()
    current_season = None

    all_features = []
    all_labels = []
    all_ids = []
    all_seasons = []

    for i, game in enumerate(games):
        # Reset at season boundary if desired
        if reset_between_seasons and game['season'] != current_season:
            if current_season is not None:
                print(f"  Season {current_season}: {len([s for s in all_seasons if s == current_season])} games processed")
            current_season = game['season']
            # Don't fully reset Elo - carry over with regression to mean
            _regress_elo_to_mean(computer, factor=0.5)
            # Reset counting stats for new season
            for team in computer.team_stats:
                s = computer.team_stats[team]
                s['wins'] = 0
                s['losses'] = 0
                s['runs_scored'] = 0
                s['runs_allowed'] = 0
                s['games'] = 0
                s['recent_results'] = []
                s['opponents'] = []
                # Keep last_game_date for rest calculation

        # Compute features BEFORE updating state (no leakage)
        features, label = computer.compute_game_features(game)

        # Update state AFTER feature computation
        computer.update_state(game)

        all_features.append(features)
        all_labels.append(label)
        all_ids.append(game['id'])
        all_seasons.append(game['season'])

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(games)} games...")

    if current_season is not None:
        print(f"  Season {current_season}: {len([s for s in all_seasons if s == current_season])} games processed")

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.float32)

    return X, y, all_ids, all_seasons


def _regress_elo_to_mean(computer, factor=0.5):
    """Regress all Elo ratings toward 1500 between seasons."""
    for team in list(computer.elo.keys()):
        computer.elo[team] = 1500 + factor * (computer.elo[team] - 1500)


def save_to_database(X, y, game_ids, seasons, feature_names=None):
    """Save computed features to historical_features table."""
    conn = get_connection()
    c = conn.cursor()

    # Create table
    c.execute("DROP TABLE IF EXISTS historical_features")
    c.execute("""
        CREATE TABLE historical_features (
            game_id INTEGER PRIMARY KEY,
            season INTEGER,
            label REAL,
            features BLOB
        )
    """)

    for i in range(len(game_ids)):
        c.execute("""
            INSERT INTO historical_features (game_id, season, label, features)
            VALUES (?, ?, ?, ?)
        """, (game_ids[i], seasons[i], float(y[i]),
              X[i].tobytes()))

    conn.commit()
    conn.close()
    print(f"Saved {len(game_ids)} feature vectors to historical_features table")


def main():
    parser = argparse.ArgumentParser(
        description='Compute historical features for neural network training'
    )
    parser.add_argument('--season', type=int, nargs='*',
                        help='Specific seasons to process (default: all)')
    parser.add_argument('--save-table', action='store_true',
                        help='Save features to database table')
    parser.add_argument('--no-season-reset', action='store_true',
                        help='Do not reset stats between seasons')
    args = parser.parse_args()

    print("Loading historical games...")
    games = load_historical_games(args.season)
    print(f"Loaded {len(games)} games")

    if not games:
        print("No games found. Make sure historical data has been scraped.")
        return

    print("Computing features...")
    X, y, game_ids, seasons = compute_all_features(
        games, reset_between_seasons=not args.no_season_reset
    )

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Labels: {y.sum():.0f} home wins / {len(y) - y.sum():.0f} away wins "
          f"({y.mean():.1%} home win rate)")

    # Check for NaN/Inf
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"WARNING: {nan_count} NaN values, {inf_count} Inf values in features")
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    if args.save_table:
        save_to_database(X, y, game_ids, seasons)

    # Print season breakdown
    unique_seasons = sorted(set(seasons))
    print("\nSeason breakdown:")
    for s in unique_seasons:
        mask = [1 for ss in seasons if ss == s]
        print(f"  {s}: {len(mask)} games")

    return X, y, game_ids, seasons


if __name__ == '__main__':
    main()
