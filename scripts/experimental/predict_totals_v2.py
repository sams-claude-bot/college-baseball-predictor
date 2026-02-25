#!/usr/bin/env python3
"""
Predict totals using experimental V2 ensemble.

Usage:
    python scripts/experimental/predict_totals_v2.py [DATE]
    
Stores predictions in experimental_totals_predictions table.
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.database import get_connection
from models.experimental.runs_ensemble_v2 import predict


def ensure_table():
    """Create experimental predictions table if needed."""
    conn = get_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS experimental_totals_predictions (
            id INTEGER PRIMARY KEY,
            game_id TEXT NOT NULL,
            over_under_line REAL,
            projected_total REAL NOT NULL,
            prediction TEXT,
            edge_pct REAL,
            confidence REAL,
            predicted_at TEXT DEFAULT CURRENT_TIMESTAMP,
            actual_total INTEGER,
            was_correct INTEGER,
            model_name TEXT DEFAULT 'exp_runs_ensemble_v2',
            neg_bin_over_prob REAL,
            neg_bin_under_prob REAL,
            dispersion_param REAL,
            UNIQUE(game_id, model_name)
        )
    ''')
    conn.commit()
    conn.close()


def get_games_for_date(date: str):
    """Get games with O/U lines for a given date."""
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        SELECT g.id, g.home_team_id, g.away_team_id, g.is_neutral_site,
               g.home_score, g.away_score, g.status,
               bl.over_under as ou_line
        FROM games g
        LEFT JOIN betting_lines bl ON g.id = bl.game_id AND bl.over_under IS NOT NULL
        WHERE g.date = ?
        ORDER BY g.id
    ''', (date,))
    rows = c.fetchall()
    conn.close()
    return rows


def predict_and_store(date: str, verbose: bool = True):
    """Run V2 ensemble on games for a date and store predictions."""
    ensure_table()
    games = get_games_for_date(date)
    
    if not games:
        if verbose:
            print(f"No games found for {date}")
        return 0
    
    conn = get_connection()
    stored = 0
    
    for game in games:
        game_id = game['id']
        ou_line = game['ou_line']
        
        try:
            result = predict(
                game['home_team_id'],
                game['away_team_id'],
                total_line=ou_line,
                neutral_site=bool(game['is_neutral_site']),
                game_id=game_id,
            )
            
            prediction = None
            edge = None
            confidence = None
            nb_over = None
            nb_under = None
            dispersion = None
            
            if 'over_under' in result and ou_line:
                ou = result['over_under']
                prediction = ou['lean']
                edge = ou['edge']
                confidence = ou.get('confidence', 0)
                nb_over = ou.get('negbin_over')
                nb_under = ou.get('negbin_under')
                dispersion = ou.get('dispersion')
            
            # Check if game already has a result
            actual_total = None
            was_correct = None
            if game['status'] == 'final' and game['home_score'] is not None:
                actual_total = game['home_score'] + game['away_score']
                if prediction and ou_line:
                    if actual_total > ou_line and prediction == 'OVER':
                        was_correct = 1
                    elif actual_total < ou_line and prediction == 'UNDER':
                        was_correct = 1
                    elif actual_total == ou_line:
                        was_correct = None  # Push
                    else:
                        was_correct = 0
            
            conn.execute('''
                INSERT OR REPLACE INTO experimental_totals_predictions
                (game_id, over_under_line, projected_total, prediction, edge_pct,
                 confidence, predicted_at, actual_total, was_correct, model_name,
                 neg_bin_over_prob, neg_bin_under_prob, dispersion_param)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'exp_runs_ensemble_v2', ?, ?, ?)
            ''', (
                game_id, ou_line, result['projected_total'],
                prediction, edge, confidence,
                datetime.now().isoformat(),
                actual_total, was_correct,
                nb_over, nb_under, dispersion,
            ))
            stored += 1
            
            if verbose:
                print(f"  {game_id}: proj={result['projected_total']:.1f}"
                      f" line={ou_line or 'N/A'}"
                      f" pred={prediction or 'N/A'}"
                      f" edge={edge or 0:.1f}%"
                      f" actual={actual_total or 'TBD'}"
                      f" correct={was_correct}")
        except Exception as e:
            if verbose:
                print(f"  {game_id}: ERROR - {e}")
    
    conn.commit()
    conn.close()
    
    if verbose:
        print(f"\nStored {stored} predictions for {date}")
    return stored


if __name__ == '__main__':
    date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime('%Y-%m-%d')
    print(f"🧪 Experimental V2 Totals Predictions for {date}")
    print("=" * 60)
    predict_and_store(date)
