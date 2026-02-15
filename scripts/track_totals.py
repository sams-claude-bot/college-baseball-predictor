#!/usr/bin/env python3
"""
Track over/under predictions and their accuracy.

Usage:
    python track_totals.py predict     # Record predictions for today's games with lines
    python track_totals.py evaluate    # Evaluate predictions for completed games
    python track_totals.py accuracy    # Show accuracy report
"""

import sqlite3
import sys
from datetime import datetime, timedelta

DB_PATH = '/home/sam/college-baseball-predictor/data/baseball.db'

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def record_predictions():
    """Record over/under predictions for today's games with betting lines."""
    conn = get_connection()
    c = conn.cursor()
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Get games with betting lines that don't have predictions yet
    c.execute('''
        SELECT b.game_id, b.over_under, b.home_team_id, b.away_team_id,
               ht.name as home_name, at.name as away_name
        FROM betting_lines b
        JOIN teams ht ON b.home_team_id = ht.id
        JOIN teams at ON b.away_team_id = at.id
        LEFT JOIN totals_predictions tp ON b.game_id = tp.game_id AND b.over_under = tp.over_under_line
        WHERE b.date = ? AND b.over_under IS NOT NULL AND tp.id IS NULL
    ''', (today,))
    
    games = c.fetchall()
    
    if not games:
        print("No new games to predict")
        return
    
    # Import runs ensemble
    sys.path.insert(0, '/home/sam/college-baseball-predictor')
    import models.runs_ensemble as runs_ens
    
    predicted = 0
    for game in games:
        try:
            result = runs_ens.predict(
                game['home_team_id'], 
                game['away_team_id'],
                total_line=game['over_under']
            )
            
            prediction = result['over_under']['lean']
            edge = result['over_under']['edge']
            proj_total = result['projected_total']
            
            c.execute('''
                INSERT INTO totals_predictions 
                (game_id, over_under_line, projected_total, prediction, edge_pct, predicted_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (game['game_id'], game['over_under'], proj_total, prediction, edge, 
                  datetime.now().isoformat()))
            
            print(f"  {game['away_name']} @ {game['home_name']}: {prediction} {game['over_under']} (proj {proj_total:.1f}, edge {edge:.1f}%)")
            predicted += 1
            
        except Exception as e:
            print(f"  Error for {game['game_id']}: {e}")
    
    conn.commit()
    conn.close()
    print(f"\nRecorded {predicted} predictions")

def evaluate_predictions():
    """Evaluate predictions for completed games."""
    conn = get_connection()
    c = conn.cursor()
    
    # Get predictions for completed games that haven't been evaluated
    c.execute('''
        SELECT tp.id, tp.game_id, tp.over_under_line, tp.prediction, tp.projected_total,
               g.home_score, g.away_score,
               ht.name as home_name, at.name as away_name
        FROM totals_predictions tp
        JOIN games g ON tp.game_id = g.id
        JOIN teams ht ON g.home_team_id = ht.id
        JOIN teams at ON g.away_team_id = at.id
        WHERE g.status = 'final' 
          AND g.home_score IS NOT NULL 
          AND g.away_score IS NOT NULL
          AND tp.was_correct IS NULL
    ''')
    
    predictions = c.fetchall()
    
    if not predictions:
        print("No predictions to evaluate")
        return
    
    correct = 0
    total = 0
    
    for pred in predictions:
        actual_total = pred['home_score'] + pred['away_score']
        line = pred['over_under_line']
        
        # Determine if prediction was correct
        if actual_total == line:
            was_correct = None  # Push
            result = "PUSH"
        elif pred['prediction'] == 'OVER':
            was_correct = 1 if actual_total > line else 0
            result = "✓" if was_correct else "✗"
        else:  # UNDER
            was_correct = 1 if actual_total < line else 0
            result = "✓" if was_correct else "✗"
        
        c.execute('''
            UPDATE totals_predictions 
            SET actual_total = ?, was_correct = ?
            WHERE id = ?
        ''', (actual_total, was_correct, pred['id']))
        
        print(f"  {pred['away_name']} @ {pred['home_name']}: {pred['prediction']} {line} → Actual {actual_total} [{result}]")
        
        if was_correct is not None:
            total += 1
            if was_correct:
                correct += 1
    
    conn.commit()
    conn.close()
    
    if total > 0:
        print(f"\nEvaluated: {correct}/{total} ({correct/total*100:.1f}%)")

def accuracy_report():
    """Show accuracy report for totals predictions."""
    conn = get_connection()
    c = conn.cursor()
    
    # Overall accuracy
    c.execute('''
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct,
            SUM(CASE WHEN was_correct = 0 THEN 1 ELSE 0 END) as incorrect,
            SUM(CASE WHEN was_correct IS NULL AND actual_total IS NOT NULL THEN 1 ELSE 0 END) as pushes
        FROM totals_predictions
        WHERE actual_total IS NOT NULL
    ''')
    overall = c.fetchone()
    
    print("=" * 50)
    print("TOTALS PREDICTION ACCURACY REPORT")
    print("=" * 50)
    
    if overall['total'] == 0:
        print("\nNo evaluated predictions yet")
        return
    
    decided = overall['correct'] + overall['incorrect']
    if decided > 0:
        acc = overall['correct'] / decided * 100
        print(f"\nOverall: {overall['correct']}/{decided} ({acc:.1f}%)")
        print(f"Pushes: {overall['pushes']}")
    
    # By prediction type
    c.execute('''
        SELECT 
            prediction,
            COUNT(*) as total,
            SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct
        FROM totals_predictions
        WHERE actual_total IS NOT NULL AND was_correct IS NOT NULL
        GROUP BY prediction
    ''')
    by_type = c.fetchall()
    
    print("\nBy Type:")
    for row in by_type:
        acc = row['correct'] / row['total'] * 100 if row['total'] > 0 else 0
        print(f"  {row['prediction']}: {row['correct']}/{row['total']} ({acc:.1f}%)")
    
    # By edge bucket
    c.execute('''
        SELECT 
            CASE 
                WHEN edge_pct >= 30 THEN '30%+'
                WHEN edge_pct >= 20 THEN '20-30%'
                WHEN edge_pct >= 10 THEN '10-20%'
                ELSE '<10%'
            END as edge_bucket,
            COUNT(*) as total,
            SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct
        FROM totals_predictions
        WHERE actual_total IS NOT NULL AND was_correct IS NOT NULL
        GROUP BY edge_bucket
        ORDER BY MIN(edge_pct) DESC
    ''')
    by_edge = c.fetchall()
    
    print("\nBy Edge:")
    for row in by_edge:
        acc = row['correct'] / row['total'] * 100 if row['total'] > 0 else 0
        print(f"  {row['edge_bucket']}: {row['correct']}/{row['total']} ({acc:.1f}%)")
    
    # Recent predictions
    c.execute('''
        SELECT tp.prediction, tp.over_under_line, tp.projected_total, tp.edge_pct,
               tp.actual_total, tp.was_correct,
               ht.name as home_name, at.name as away_name, g.date
        FROM totals_predictions tp
        JOIN games g ON tp.game_id = g.id
        JOIN teams ht ON g.home_team_id = ht.id
        JOIN teams at ON g.away_team_id = at.id
        WHERE tp.actual_total IS NOT NULL
        ORDER BY g.date DESC, tp.predicted_at DESC
        LIMIT 10
    ''')
    recent = c.fetchall()
    
    print("\nRecent Predictions:")
    for r in recent:
        result = "✓" if r['was_correct'] == 1 else ("✗" if r['was_correct'] == 0 else "PUSH")
        print(f"  {result} {r['prediction']} {r['over_under_line']} | {r['away_name']} @ {r['home_name']} | Actual: {r['actual_total']}")
    
    conn.close()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python track_totals.py [predict|evaluate|accuracy]")
        sys.exit(1)
    
    action = sys.argv[1]
    
    if action == 'predict':
        print("Recording totals predictions for today...")
        record_predictions()
    elif action == 'evaluate':
        print("Evaluating completed predictions...")
        evaluate_predictions()
    elif action == 'accuracy':
        accuracy_report()
    else:
        print(f"Unknown action: {action}")
        sys.exit(1)
