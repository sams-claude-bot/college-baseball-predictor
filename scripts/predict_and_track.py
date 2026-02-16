#!/usr/bin/env python3
"""
Predict games and track model performance.

Usage:
    python predict_and_track.py predict [DATE]  # Generate predictions for games on DATE (default: today)
    python predict_and_track.py evaluate [DATE] # Compare predictions to results
    python predict_and_track.py accuracy        # Show overall model accuracy
"""

import sys
import sqlite3
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.predictor_db import Predictor
from models.nn_totals_model import NNTotalsModel
from models.nn_spread_model import NNSpreadModel
from scripts.database import get_connection

MODEL_NAMES = ['pythagorean', 'elo', 'log5', 'advanced', 'pitching', 'conference', 'prior', 'poisson', 'neural', 'ensemble']

def predict_games(date=None):
    """Generate and store predictions for all games on a date"""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Get games without predictions yet
    cur.execute('''
        SELECT g.id, g.home_team_id, g.away_team_id, h.name, a.name
        FROM games g
        JOIN teams h ON g.home_team_id = h.id
        JOIN teams a ON g.away_team_id = a.id
        WHERE g.date = ?
        AND g.id NOT IN (SELECT DISTINCT game_id FROM model_predictions)
    ''', (date,))
    
    games = cur.fetchall()
    print(f"Found {len(games)} games to predict for {date}")
    
    # Initialize predictors for each model
    predictors = {name: Predictor(model=name) for name in MODEL_NAMES}
    
    # Initialize NN totals and spread models
    nn_totals = NNTotalsModel(use_model_predictions=False)
    nn_spread = NNSpreadModel(use_model_predictions=False)
    
    predictions_made = 0
    for game_id, home_id, away_id, home_name, away_name in games:
        print(f"\n{away_name} @ {home_name}:")
        
        for model_name, predictor in predictors.items():
            try:
                result = predictor.predict_game(home_name, away_name)
                home_prob = result.get('home_win_probability', 0.5)
                home_runs = result.get('projected_home_runs', result.get('predicted_home_runs', 0)) or 0
                away_runs = result.get('projected_away_runs', result.get('predicted_away_runs', 0)) or 0
                
                cur.execute('''
                    INSERT INTO model_predictions 
                    (game_id, model_name, predicted_home_prob, predicted_home_runs, predicted_away_runs)
                    VALUES (?, ?, ?, ?, ?)
                ''', (game_id, model_name, home_prob, home_runs, away_runs))
                
                print(f"  {model_name:12}: {home_prob*100:5.1f}% {home_name} | {home_runs:.1f}-{away_runs:.1f}")
                predictions_made += 1
            except Exception as e:
                print(f"  {model_name:12}: ERROR - {e}")
        
        # NN Totals prediction (only when DK line exists)
        if nn_totals.is_trained():
            try:
                cur.execute('''
                    SELECT over_under FROM betting_lines 
                    WHERE home_team_id = ? AND away_team_id = ? AND over_under IS NOT NULL AND over_under > 0
                    ORDER BY captured_at DESC LIMIT 1
                ''', (home_id, away_id))
                dk_row = cur.fetchone()
                if dk_row:
                    dk_line = dk_row[0]
                    t_pred = nn_totals.predict_game(home_id, away_id)
                    proj_total = t_pred.get('projected_total', 0)
                    prediction = 'OVER' if proj_total > dk_line else 'UNDER'
                    edge = abs(proj_total - dk_line) / dk_line * 100
                    cur.execute('''
                        INSERT OR IGNORE INTO totals_predictions 
                        (game_id, over_under_line, projected_total, prediction, edge_pct, model_name)
                        VALUES (?, ?, ?, ?, ?, 'nn_totals')
                    ''', (game_id, dk_line, proj_total, prediction, edge))
                    print(f"  {'nn_totals':12}: projected total {proj_total:.1f} (line {dk_line}) → {prediction}")
            except Exception as e:
                print(f"  {'nn_totals':12}: ERROR - {e}")
        
        # NN Spread prediction  
        if nn_spread.is_trained():
            try:
                s_pred = nn_spread.predict_game(home_id, away_id)
                margin = s_pred.get('projected_margin', 0)
                cover_prob = s_pred.get('cover_prob', 0.5)
                prediction = 'HOME_COVER' if margin > -1.5 else 'AWAY_COVER'
                cur.execute('''
                    INSERT OR IGNORE INTO spread_predictions
                    (game_id, model_name, spread_line, projected_margin, prediction, cover_prob)
                    VALUES (?, 'nn_spread', -1.5, ?, ?, ?)
                ''', (game_id, margin, prediction, cover_prob))
                print(f"  {'nn_spread':12}: margin {margin:+.1f} | cover prob {cover_prob:.1%}")
            except Exception as e:
                print(f"  {'nn_spread':12}: ERROR - {e}")
    
    conn.commit()
    conn.close()
    print(f"\n✅ Stored {predictions_made} predictions for {len(games)} games")

def evaluate_predictions(date=None):
    """Compare predictions to actual results.
    
    If date is None, evaluates ALL pending predictions (was_correct IS NULL)
    where the game has a final score. This ensures we catch up on any missed
    evaluations from past days.
    
    If date is specified, only evaluates predictions for that date.
    """
    conn = get_connection()
    cur = conn.cursor()
    
    # Get games with results and unevaluated predictions
    # Use LEFT JOIN for teams in case team IDs don't match teams table
    if date:
        cur.execute('''
            SELECT 
                g.id, COALESCE(h.name, g.home_team_id), COALESCE(a.name, g.away_team_id), 
                g.home_score, g.away_score,
                mp.model_name, mp.predicted_home_prob, g.date
            FROM games g
            LEFT JOIN teams h ON g.home_team_id = h.id
            LEFT JOIN teams a ON g.away_team_id = a.id
            JOIN model_predictions mp ON mp.game_id = g.id
            WHERE g.date = ?
            AND g.home_score IS NOT NULL
            AND mp.was_correct IS NULL
        ''', (date,))
    else:
        # Evaluate ALL pending predictions with completed games
        cur.execute('''
            SELECT 
                g.id, COALESCE(h.name, g.home_team_id), COALESCE(a.name, g.away_team_id), 
                g.home_score, g.away_score,
                mp.model_name, mp.predicted_home_prob, g.date
            FROM games g
            LEFT JOIN teams h ON g.home_team_id = h.id
            LEFT JOIN teams a ON g.away_team_id = a.id
            JOIN model_predictions mp ON mp.game_id = g.id
            WHERE g.home_score IS NOT NULL
            AND mp.was_correct IS NULL
            ORDER BY g.date
        ''')
    
    rows = cur.fetchall()
    date_str = date if date else "all pending"
    print(f"Evaluating {len(rows)} predictions for {date_str}")
    
    updated = 0
    dates_evaluated = set()
    for game_id, home, away, home_score, away_score, model, home_prob, game_date in rows:
        home_won = home_score > away_score
        predicted_home = home_prob > 0.5
        correct = 1 if (home_won == predicted_home) else 0
        
        cur.execute('''
            UPDATE model_predictions 
            SET was_correct = ?
            WHERE game_id = ? AND model_name = ?
        ''', (correct, game_id, model))
        updated += 1
        dates_evaluated.add(game_date)
    
    conn.commit()
    conn.close()
    
    if dates_evaluated:
        print(f"✅ Updated {updated} predictions across {len(dates_evaluated)} date(s): {', '.join(sorted(dates_evaluated))}")
    else:
        print(f"✅ No predictions needed evaluation")

def show_accuracy():
    """Show model accuracy statistics"""
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute('''
        SELECT 
            model_name,
            COUNT(*) as total,
            SUM(was_correct) as correct,
            ROUND(100.0 * SUM(was_correct) / COUNT(*), 1) as accuracy
        FROM model_predictions
        WHERE was_correct IS NOT NULL
        GROUP BY model_name
        ORDER BY accuracy DESC
    ''')
    
    print("\n📊 MODEL ACCURACY")
    print("=" * 45)
    print(f"{'Model':<15} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("-" * 45)
    
    for model, total, correct, accuracy in cur.fetchall():
        print(f"{model:<15} {correct:>8} {total:>8} {accuracy:>9.1f}%")
    
    # Overall stats
    cur.execute('''
        SELECT COUNT(*), SUM(was_correct)
        FROM model_predictions
        WHERE was_correct IS NOT NULL AND model_name = 'ensemble'
    ''')
    total, correct = cur.fetchone()
    if total and total > 0:
        print("-" * 45)
        print(f"{'ENSEMBLE':<15} {correct:>8} {total:>8} {100*correct/total:>9.1f}%")
    
    conn.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    cmd = sys.argv[1]
    date = sys.argv[2] if len(sys.argv) > 2 else None
    
    if cmd == "predict":
        predict_games(date)
    elif cmd == "evaluate":
        evaluate_predictions(date)
    elif cmd == "accuracy":
        show_accuracy()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
