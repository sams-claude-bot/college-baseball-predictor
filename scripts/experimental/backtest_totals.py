#!/usr/bin/env python3
"""
Backtest V2 ensemble against production on all completed games.

Compares MAE, O/U accuracy, and OVER/UNDER split.
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.database import get_connection
from models.experimental.runs_ensemble_v2 import predict


def get_completed_games_with_lines():
    """Get all final games that have betting lines."""
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        SELECT DISTINCT g.id, g.home_team_id, g.away_team_id, g.is_neutral_site,
               g.home_score, g.away_score, g.date,
               bl.over_under as ou_line
        FROM games g
        JOIN betting_lines bl ON g.id = bl.game_id AND bl.over_under IS NOT NULL
        WHERE g.status = 'final' AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL
        ORDER BY g.date
    ''')
    rows = c.fetchall()
    conn.close()
    return rows


def get_production_metrics():
    """Get production model metrics from totals_predictions."""
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        SELECT 
            COUNT(*) as n,
            AVG(ABS(projected_total - actual_total)) as mae,
            AVG(CASE WHEN was_correct = 1 THEN 1.0 ELSE 0.0 END) as ou_acc,
            SUM(CASE WHEN prediction = 'OVER' AND was_correct = 1 THEN 1 ELSE 0 END) as over_correct,
            SUM(CASE WHEN prediction = 'OVER' THEN 1 ELSE 0 END) as over_total,
            SUM(CASE WHEN prediction = 'UNDER' AND was_correct = 1 THEN 1 ELSE 0 END) as under_correct,
            SUM(CASE WHEN prediction = 'UNDER' THEN 1 ELSE 0 END) as under_total
        FROM totals_predictions
        WHERE model_name = 'runs_ensemble'
        AND actual_total IS NOT NULL
        AND was_correct IS NOT NULL
    ''')
    row = c.fetchone()
    conn.close()
    return row


def run_backtest(limit: int = None, verbose: bool = False):
    """Run V2 ensemble on historical games and compare to production."""
    games = get_completed_games_with_lines()
    if limit:
        games = games[:limit]
    
    print(f"🧪 V2 Ensemble Backtest")
    print(f"   Games with lines: {len(games)}")
    print("=" * 60)
    
    v2_results = []
    errors = 0
    
    for i, game in enumerate(games):
        if verbose and i % 50 == 0:
            print(f"  Processing {i}/{len(games)}...")
        
        actual_total = game['home_score'] + game['away_score']
        ou_line = game['ou_line']
        
        try:
            result = predict(
                game['home_team_id'],
                game['away_team_id'],
                total_line=ou_line,
                neutral_site=bool(game['is_neutral_site']),
                game_id=game['id'],
            )
            
            proj_total = result['projected_total']
            mae_val = abs(proj_total - actual_total)
            
            prediction = None
            was_correct = None
            edge = 0
            
            if 'over_under' in result:
                ou = result['over_under']
                prediction = ou['lean']
                edge = ou['edge']
                
                if actual_total > ou_line and prediction == 'OVER':
                    was_correct = 1
                elif actual_total < ou_line and prediction == 'UNDER':
                    was_correct = 1
                elif actual_total == ou_line:
                    was_correct = None  # Push
                else:
                    was_correct = 0
            
            v2_results.append({
                'game_id': game['id'],
                'projected': proj_total,
                'actual': actual_total,
                'line': ou_line,
                'mae': mae_val,
                'prediction': prediction,
                'was_correct': was_correct,
                'edge': edge,
            })
        except Exception as e:
            errors += 1
            if verbose:
                print(f"  ERROR {game['id']}: {e}")
    
    if not v2_results:
        print("No results generated!")
        return
    
    # V2 metrics
    evaluated = [r for r in v2_results if r['was_correct'] is not None]
    v2_mae = sum(r['mae'] for r in v2_results) / len(v2_results)
    v2_ou_acc = sum(r['was_correct'] for r in evaluated) / len(evaluated) if evaluated else 0
    
    over_preds = [r for r in evaluated if r['prediction'] == 'OVER']
    under_preds = [r for r in evaluated if r['prediction'] == 'UNDER']
    
    over_acc = sum(r['was_correct'] for r in over_preds) / len(over_preds) if over_preds else 0
    under_acc = sum(r['was_correct'] for r in under_preds) / len(under_preds) if under_preds else 0
    
    # Production metrics
    prod = get_production_metrics()
    
    print(f"\n{'METRIC':<25} {'PRODUCTION':>12} {'V2 (EXP)':>12} {'DELTA':>10}")
    print("-" * 60)
    
    prod_mae = prod['mae'] if prod and prod['mae'] else None
    prod_ou = prod['ou_acc'] if prod and prod['ou_acc'] else None
    
    prod_mae_str = f"{prod_mae:.2f}" if prod_mae else "N/A"
    prod_ou_str = f"{prod_ou*100:.1f}%" if prod_ou else "N/A"
    
    mae_delta = f"{v2_mae - prod_mae:+.2f}" if prod_mae else "N/A"
    ou_delta = f"{(v2_ou_acc - prod_ou)*100:+.1f}%" if prod_ou else "N/A"
    
    print(f"{'Games evaluated':<25} {prod['n'] if prod else 'N/A':>12} {len(evaluated):>12}")
    print(f"{'MAE':<25} {prod_mae_str:>12} {v2_mae:>12.2f} {mae_delta:>10}")
    print(f"{'O/U Accuracy':<25} {prod_ou_str:>12} {v2_ou_acc*100:>11.1f}% {ou_delta:>10}")
    
    prod_over_acc_str = "N/A"
    prod_under_acc_str = "N/A"
    if prod and prod['over_total'] and prod['over_total'] > 0:
        prod_over_acc = prod['over_correct'] / prod['over_total']
        prod_over_acc_str = f"{prod_over_acc*100:.1f}%"
    if prod and prod['under_total'] and prod['under_total'] > 0:
        prod_under_acc = prod['under_correct'] / prod['under_total']
        prod_under_acc_str = f"{prod_under_acc*100:.1f}%"
    
    print(f"{'OVER accuracy':<25} {prod_over_acc_str:>12} {over_acc*100:>11.1f}% ({len(over_preds)} preds)")
    print(f"{'UNDER accuracy':<25} {prod_under_acc_str:>12} {under_acc*100:>11.1f}% ({len(under_preds)} preds)")
    print(f"{'Errors/skipped':<25} {'':>12} {errors:>12}")
    
    # Edge bucket analysis
    print(f"\n📊 V2 Accuracy by Edge Bucket:")
    buckets = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 100)]
    for lo, hi in buckets:
        bucket = [r for r in evaluated if lo <= r['edge'] < hi]
        if bucket:
            acc = sum(r['was_correct'] for r in bucket) / len(bucket)
            print(f"   Edge {lo}-{hi}%: {acc*100:.1f}% ({len(bucket)} games)")
    
    # Verdict
    print("\n" + "=" * 60)
    if prod_mae and v2_mae < prod_mae and prod_ou and v2_ou_acc > prod_ou:
        print("🏆 VERDICT: V2 WINS on both MAE and O/U accuracy!")
    elif prod_mae and v2_mae < prod_mae:
        print("📊 VERDICT: V2 better MAE, but O/U accuracy needs work")
    elif prod_ou and v2_ou_acc > prod_ou:
        print("📊 VERDICT: V2 better O/U accuracy, but MAE needs work")
    else:
        print("📊 VERDICT: Production still champion (or insufficient data)")


if __name__ == '__main__':
    limit = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else None
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    run_backtest(limit=limit, verbose=verbose)
