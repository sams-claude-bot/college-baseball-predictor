#!/usr/bin/env python3
"""
Train the upset model with walk-forward validation.

Trains ONLY on games where one team was a clear favorite (>60% elo-implied).
Uses Random Forest to detect when favorites are vulnerable.

Usage:
    python scripts/train_upset_model.py
"""

import sys
import pickle
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.database import get_connection
from models.upset_model import UpsetModel, elo_expected, ELO_HOME_ADV, MODEL_PATH


def load_data():
    """Load completed games, elo history, and team info."""
    conn = get_connection()
    c = conn.cursor()

    # Games
    c.execute("""
        SELECT g.id, g.date, g.home_team_id, g.away_team_id,
               g.home_score, g.away_score, g.winner_id,
               g.is_conference_game
        FROM games g
        WHERE g.status = 'final' AND g.home_score IS NOT NULL
        ORDER BY g.date
    """)
    games = [dict(r) for r in c.fetchall()]

    # Current elo ratings (used as proxy; ideally we'd have historical)
    c.execute("SELECT team_id, rating FROM elo_ratings")
    elos = {r['team_id']: r['rating'] for r in c.fetchall()}

    # Team conferences
    c.execute("SELECT id, conference FROM teams")
    conferences = {r['id']: r['conference'] for r in c.fetchall()}

    conn.close()
    return games, elos, conferences


def compute_team_stats_before(games, idx, team_id):
    """Compute season and recent stats for team from games before idx."""
    wins, total, margin_sum = 0, 0, 0
    recent_wins, recent_total, recent_margin_sum = 0, 0, 0
    home_losses = 0
    away_games, away_wins = 0, 0

    for i in range(idx - 1, max(idx - 200, -1), -1):
        if i < 0:
            break
        g = games[i]
        if g['home_team_id'] != team_id and g['away_team_id'] != team_id:
            continue

        is_home = g['home_team_id'] == team_id
        won = g['winner_id'] == team_id
        margin = ((g['home_score'] - g['away_score']) if is_home
                  else (g['away_score'] - g['home_score']))

        total += 1
        if won:
            wins += 1
        margin_sum += margin

        if total <= 5:
            recent_total += 1
            if won:
                recent_wins += 1
            recent_margin_sum += margin

        if is_home and not won:
            home_losses += 1
        if not is_home:
            away_games += 1
            if won:
                away_wins += 1

    season_wp = wins / total if total > 0 else 0.5
    recent_wp = recent_wins / recent_total if recent_total > 0 else season_wp
    season_margin = margin_sum / total if total > 0 else 0.0
    recent_margin = recent_margin_sum / recent_total if recent_total > 0 else season_margin
    away_wp = away_wins / away_games if away_games > 0 else 0.5

    return {
        'season_win_pct': season_wp,
        'recent_win_pct': recent_wp,
        'form_vs_avg': recent_wp - season_wp,
        'season_avg_margin': season_margin,
        'recent_avg_margin': recent_margin,
        'margin_trend': recent_margin - season_margin,
        'home_losses': home_losses,
        'away_win_pct': away_wp,
    }


def build_dataset(games, elos, conferences):
    """Build feature matrix. Only include games with clear favorites."""
    X, y, dates, game_indices = [], [], [], []

    # Track upset rates by elo bucket (for walk-forward)
    bucket_upsets = defaultdict(lambda: [0, 0])  # [upsets, total]

    for i in range(50, len(games)):
        g = games[i]
        hid, aid = g['home_team_id'], g['away_team_id']

        h_elo = elos.get(hid, 1500)
        a_elo = elos.get(aid, 1500)
        home_exp = elo_expected(h_elo, a_elo, ELO_HOME_ADV)

        # Only train on games with clear favorite (>60% implied)
        if 0.40 <= home_exp <= 0.60:
            continue

        fav_is_home = home_exp >= 0.5
        if fav_is_home:
            fav_id, dog_id = hid, aid
            fav_elo, dog_elo = h_elo, a_elo
            elo_gap = (fav_elo + ELO_HOME_ADV) - dog_elo
        else:
            fav_id, dog_id = aid, hid
            fav_elo, dog_elo = a_elo, h_elo
            elo_gap = fav_elo - (dog_elo + ELO_HOME_ADV)

        fav_stats = compute_team_stats_before(games, i, fav_id)
        dog_stats = compute_team_stats_before(games, i, dog_id)

        h_conf = conferences.get(hid)
        a_conf = conferences.get(aid)
        is_conf = 1 if (h_conf and a_conf and h_conf == a_conf) else 0

        gd = datetime.strptime(g['date'], '%Y-%m-%d').date()
        is_midweek = 1 if gd.weekday() in (1, 2, 3) else 0

        # Historical upset rate for this elo bucket
        bucket = int(abs(elo_gap) // 50) * 50
        bt = bucket_upsets[bucket]
        hist_rate = bt[0] / bt[1] if bt[1] > 10 else 0.30

        features = [
            elo_gap,
            fav_stats['form_vs_avg'],
            fav_stats['margin_trend'],
            dog_stats['form_vs_avg'],
            is_conf,
            is_midweek,
            dog_stats['away_win_pct'],
            fav_stats['home_losses'],
            hist_rate,
        ]

        # Target: did the underdog win?
        upset = 1 if g['winner_id'] == dog_id else 0

        # Update bucket stats for future games
        bucket_upsets[bucket][0] += upset
        bucket_upsets[bucket][1] += 1

        X.append(features)
        y.append(upset)
        dates.append(g['date'])
        game_indices.append(i)

    return (np.array(X, dtype=np.float64), np.array(y),
            dates, dict(bucket_upsets))


def main():
    print("=" * 70)
    print("UPSET MODEL TRAINING")
    print("=" * 70)

    games, elos, conferences = load_data()
    print(f"Loaded {len(games)} games, {len(elos)} elo ratings")

    X, y, dates, bucket_upsets = build_dataset(games, elos, conferences)
    print(f"Dataset: {len(X)} games with clear favorite (>60%)")
    print(f"Upset rate: {y.mean():.1%}")

    # Walk-forward split
    split = int(len(X) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=20,
        random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    train_acc = rf.score(X_train, y_train)
    test_acc = rf.score(X_test, y_test)

    # Upset-specific accuracy
    test_probs = rf.predict_proba(X_test)[:, 1]
    upset_mask = y_test == 1
    chalk_mask = y_test == 0

    upset_pred_correct = np.sum((test_probs > 0.5) & upset_mask)
    total_upsets = upset_mask.sum()
    chalk_pred_correct = np.sum((test_probs <= 0.5) & chalk_mask)
    total_chalk = chalk_mask.sum()

    print(f"\nWalk-forward validation:")
    print(f"  Train: {len(X_train)} games, accuracy {train_acc:.1%}")
    print(f"  Test:  {len(X_test)} games, accuracy {test_acc:.1%}")
    print(f"  Upsets called correctly: {upset_pred_correct}/{total_upsets} "
          f"({upset_pred_correct/total_upsets:.1%})" if total_upsets > 0 else "")
    print(f"  Chalk called correctly:  {chalk_pred_correct}/{total_chalk} "
          f"({chalk_pred_correct/total_chalk:.1%})" if total_chalk > 0 else "")

    # Feature importance
    print(f"\nFeature importance:")
    for name, imp in sorted(zip(UpsetModel.FEATURE_NAMES,
                                rf.feature_importances_),
                            key=lambda x: -x[1]):
        bar = "#" * int(imp * 100)
        print(f"  {name:<30} {imp:.4f} {bar}")

    # Elo bucket upset rates
    print(f"\nHistorical upset rates by elo gap:")
    elo_gap_rates = {}
    for bucket, (upsets, total) in sorted(bucket_upsets.items()):
        if total > 5:
            rate = upsets / total
            elo_gap_rates[bucket] = rate
            print(f"  Gap {bucket:>3}-{bucket+49}: {upsets}/{total} = {rate:.1%}")

    # Retrain on all data
    rf_full = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=20,
        random_state=42, n_jobs=-1)
    rf_full.fit(X, y)

    # Save
    save_data = {
        'model': rf_full,
        'feature_names': UpsetModel.FEATURE_NAMES,
        'elo_gap_upset_rates': elo_gap_rates,
        'train_size': len(X),
        'upset_rate': float(y.mean()),
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
