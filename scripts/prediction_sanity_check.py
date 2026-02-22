#!/usr/bin/env python3
"""Flag suspicious consensus vs market mismatches."""
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

DB = Path(__file__).resolve().parent.parent / 'data' / 'baseball.db'


def main():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    start = datetime.now().strftime('%Y-%m-%d')
    end = (datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d')

    rows = c.execute('''
        SELECT g.id, g.date, ht.name AS home_name, at.name AS away_name,
               bl.home_ml, bl.away_ml
        FROM games g
        JOIN teams ht ON ht.id = g.home_team_id
        JOIN teams at ON at.id = g.away_team_id
        LEFT JOIN betting_lines bl ON bl.game_id = g.id AND bl.book = 'draftkings'
        WHERE g.date BETWEEN ? AND ?
          AND g.status IN ('scheduled','in-progress')
    ''', (start, end)).fetchall()

    flagged = []
    for g in rows:
        preds = c.execute('''
            SELECT model_name, predicted_home_prob
            FROM model_predictions
            WHERE game_id = ?
        ''', (g['id'],)).fetchall()
        if len(preds) < 8:
            continue

        home_votes = sum(1 for p in preds if p['predicted_home_prob'] >= 0.5)
        away_votes = len(preds) - home_votes
        consensus_side = 'home' if home_votes >= away_votes else 'away'
        consensus_votes = max(home_votes, away_votes)

        home_ml = g['home_ml']
        away_ml = g['away_ml']
        if home_ml is None or away_ml is None:
            continue

        # Flag big disagreement: near-unanimous models but market says big dog
        if consensus_side == 'home' and consensus_votes >= 10 and home_ml >= 180:
            flagged.append((g, consensus_votes, consensus_side))
        elif consensus_side == 'away' and consensus_votes >= 10 and away_ml >= 180:
            flagged.append((g, consensus_votes, consensus_side))

    print('=== Prediction Sanity Check ===')
    print(f'Flagged games: {len(flagged)}')
    for g, votes, side in flagged[:20]:
        print(f"{g['date']} {g['id']} | {g['away_name']} @ {g['home_name']} | consensus={votes}/12 {side} | ML home={g['home_ml']} away={g['away_ml']}")

    conn.close()
    return 1 if flagged else 0


if __name__ == '__main__':
    raise SystemExit(main())
