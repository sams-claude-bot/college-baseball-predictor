#!/usr/bin/env python3
"""
Train the upset model with strict date-based walk-forward validation.

Trains ONLY on games where one team was a clear favorite (>60% elo-implied).
Uses Random Forest to detect when favorites are vulnerable.

Note: this training still uses a proxy for pregame Elo (current elo table),
not fully reconstructed as-of historical Elo snapshots.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.database import get_connection
from scripts.walkforward_utils import build_strict_date_folds, aggregate_binary_oof
from models.upset_model import UpsetModel, elo_expected, ELO_HOME_ADV, MODEL_PATH


def load_data():
    """Load completed games, elo history, and team info."""
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        SELECT g.id, g.date, g.home_team_id, g.away_team_id,
               g.home_score, g.away_score, g.winner_id,
               g.is_conference_game
        FROM games g
        WHERE g.status = 'final' AND g.home_score IS NOT NULL
        ORDER BY g.date
    """)
    games = [dict(r) for r in c.fetchall()]

    # LIMITATION: Uses current Elo ratings as proxy instead of true as-of values.
    c.execute("SELECT team_id, rating FROM elo_ratings")
    elos = {r['team_id']: r['rating'] for r in c.fetchall()}

    c.execute("SELECT id, conference FROM teams")
    conferences = {r['id']: r['conference'] for r in c.fetchall()}

    conn.close()
    return games, elos, conferences


def compute_team_stats_before(games, idx, team_id):
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
    X, y, dates = [], [], []
    bucket_upsets = defaultdict(lambda: [0, 0])

    for i in range(50, len(games)):
        g = games[i]
        hid, aid = g['home_team_id'], g['away_team_id']

        h_elo = elos.get(hid, 1500)
        a_elo = elos.get(aid, 1500)
        home_exp = elo_expected(h_elo, a_elo, ELO_HOME_ADV)

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

        upset = 1 if g['winner_id'] == dog_id else 0

        bucket_upsets[bucket][0] += upset
        bucket_upsets[bucket][1] += 1

        X.append(features)
        y.append(upset)
        dates.append(g['date'])

    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int32), np.array(dates, dtype=object), dict(bucket_upsets)


def _rf_model():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
    )


def write_report(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Upset Model Walk-Forward Report",
        "",
        f"- generated_at: {payload['generated_at']}",
        f"- folds: {payload['folds']}",
        f"- oof_n: {payload['oof_metrics']['n']}",
        f"- oof_accuracy: {payload['oof_metrics']['accuracy']}",
        f"- oof_brier: {payload['oof_metrics']['brier']}",
        f"- oof_logloss: {payload['oof_metrics']['logloss']}",
        f"- final_train_size: {payload['final_train_size']}",
        "",
        "## Known limitation",
        payload['known_limitation'],
        "",
        "```json",
        json.dumps(payload, indent=2),
        "```",
    ]
    path.write_text("\n".join(lines), encoding='utf-8')


def main(min_warmup=200, report_path=None):
    print("=" * 70)
    print("UPSET MODEL TRAINING")
    print("=" * 70)

    games, elos, conferences = load_data()
    print(f"Loaded {len(games)} games, {len(elos)} elo ratings")

    X, y, dates, bucket_upsets = build_dataset(games, elos, conferences)
    print(f"Dataset: {len(X)} games with clear favorite (>60%)")
    print(f"Upset rate: {y.mean():.1%}")

    folds = build_strict_date_folds(dates, min_warmup=min_warmup)
    oof_probs, oof_true = [], []

    for fold in folds:
        tr_idx = fold.train_idx
        te_idx = fold.test_idx
        X_train, y_train = X[tr_idx], y[tr_idx]
        X_test, y_test = X[te_idx], y[te_idx]

        if len(np.unique(y_train)) < 2:
            continue

        rf = _rf_model()
        rf.fit(X_train, y_train)
        probs = rf.predict_proba(X_test)[:, 1]

        oof_probs.extend(probs.tolist())
        oof_true.extend(y_test.tolist())

    metrics = aggregate_binary_oof(np.array(oof_true), np.array(oof_probs))

    print("\nStrict walk-forward validation:")
    print(f"  folds: {len(folds)}")
    print(f"  OOF metrics: acc={metrics['accuracy']}, brier={metrics['brier']}, logloss={metrics['logloss']}, n={metrics['n']}")

    print("\nFeature importance:")
    rf_full = _rf_model()
    rf_full.fit(X, y)
    for name, imp in sorted(zip(UpsetModel.FEATURE_NAMES, rf_full.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name:<30} {imp:.4f}")

    elo_gap_rates = {}
    for bucket, (upsets, total) in sorted(bucket_upsets.items()):
        if total > 5:
            elo_gap_rates[bucket] = upsets / total

    limitation = (
        "Pregame Elo values are approximated using current elo_ratings table, "
        "not reconstructed true as-of Elo snapshots per historical game date."
    )
    print("\nKnown limitation:")
    print(f"  {limitation}")

    save_data = {
        'model': rf_full,
        'feature_names': UpsetModel.FEATURE_NAMES,
        'elo_gap_upset_rates': elo_gap_rates,
        'train_size': len(X),
        'upset_rate': float(y.mean()),
        'test_accuracy': metrics['accuracy'],
        'saved_at': datetime.now().isoformat(),
        'version': '1.0',
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(save_data, f)

    print(f"\nfinal train size: {len(X)}")
    print(f"Model saved to {MODEL_PATH}")

    payload = {
        'model': 'upset_model',
        'generated_at': datetime.now().isoformat(),
        'folds': len(folds),
        'min_warmup': int(min_warmup),
        'oof_metrics': metrics,
        'final_train_size': int(len(X)),
        'model_path': str(MODEL_PATH),
        'known_limitation': limitation,
    }
    if report_path:
        write_report(report_path, payload)
        print(f"report: {report_path}")

    return 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--min-warmup', type=int, default=200)
    p.add_argument('--report-path', type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(min_warmup=args.min_warmup, report_path=args.report_path))
