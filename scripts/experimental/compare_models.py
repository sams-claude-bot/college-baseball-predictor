#!/usr/bin/env python3
"""
Side-by-side comparison of production vs experimental totals predictions.

Reads from totals_predictions (production) and experimental_totals_predictions (challenger).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.database import get_connection


def compare():
    conn = get_connection()
    c = conn.cursor()
    
    # Production metrics
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
        AND actual_total IS NOT NULL AND was_correct IS NOT NULL
    ''')
    prod = c.fetchone()
    
    # Experimental metrics
    c.execute('''
        SELECT 
            COUNT(*) as n,
            AVG(ABS(projected_total - actual_total)) as mae,
            AVG(CASE WHEN was_correct = 1 THEN 1.0 ELSE 0.0 END) as ou_acc,
            SUM(CASE WHEN prediction = 'OVER' AND was_correct = 1 THEN 1 ELSE 0 END) as over_correct,
            SUM(CASE WHEN prediction = 'OVER' THEN 1 ELSE 0 END) as over_total,
            SUM(CASE WHEN prediction = 'UNDER' AND was_correct = 1 THEN 1 ELSE 0 END) as under_correct,
            SUM(CASE WHEN prediction = 'UNDER' THEN 1 ELSE 0 END) as under_total
        FROM experimental_totals_predictions
        WHERE actual_total IS NOT NULL AND was_correct IS NOT NULL
    ''')
    exp = c.fetchone()
    
    print("📊 Production vs Experimental Totals Comparison")
    print("=" * 65)
    
    if not prod or not prod['n']:
        print("No production data available.")
        return
    if not exp or not exp['n']:
        print("No experimental data available. Run backtest first.")
        return
    
    def safe_pct(num, denom):
        return num / denom * 100 if denom and denom > 0 else 0
    
    print(f"\n{'METRIC':<25} {'PRODUCTION':>12} {'EXPERIMENTAL':>14} {'DELTA':>10}")
    print("-" * 65)
    print(f"{'Games':<25} {prod['n']:>12} {exp['n']:>14}")
    print(f"{'MAE':<25} {prod['mae']:>12.2f} {exp['mae']:>14.2f} {exp['mae']-prod['mae']:>+10.2f}")
    print(f"{'O/U Accuracy':<25} {prod['ou_acc']*100:>11.1f}% {exp['ou_acc']*100:>13.1f}% {(exp['ou_acc']-prod['ou_acc'])*100:>+9.1f}%")
    
    prod_over = safe_pct(prod['over_correct'], prod['over_total'])
    exp_over = safe_pct(exp['over_correct'], exp['over_total'])
    prod_under = safe_pct(prod['under_correct'], prod['under_total'])
    exp_under = safe_pct(exp['under_correct'], exp['under_total'])
    
    print(f"{'OVER accuracy':<25} {prod_over:>11.1f}% {exp_over:>13.1f}% {exp_over-prod_over:>+9.1f}%")
    print(f"{'UNDER accuracy':<25} {prod_under:>11.1f}% {exp_under:>13.1f}% {exp_under-prod_under:>+9.1f}%")
    print(f"{'OVER predictions':<25} {prod['over_total']:>12} {exp['over_total']:>14}")
    print(f"{'UNDER predictions':<25} {prod['under_total']:>12} {exp['under_total']:>14}")
    
    # Edge bucket analysis for experimental
    print(f"\n📊 Experimental Accuracy by Edge Bucket:")
    c.execute('''
        SELECT 
            CASE 
                WHEN edge_pct < 5 THEN '0-5%'
                WHEN edge_pct < 10 THEN '5-10%'
                WHEN edge_pct < 15 THEN '10-15%'
                WHEN edge_pct < 20 THEN '15-20%'
                ELSE '20%+'
            END as bucket,
            COUNT(*) as n,
            AVG(CASE WHEN was_correct = 1 THEN 1.0 ELSE 0.0 END) as acc
        FROM experimental_totals_predictions
        WHERE was_correct IS NOT NULL
        GROUP BY bucket
        ORDER BY bucket
    ''')
    for row in c.fetchall():
        print(f"   {row['bucket']}: {row['acc']*100:.1f}% ({row['n']} games)")
    
    # Actual total bucket analysis
    print(f"\n📊 Experimental Accuracy by Actual Total:")
    c.execute('''
        SELECT 
            CASE 
                WHEN actual_total < 8 THEN '<8'
                WHEN actual_total < 11 THEN '8-10'
                WHEN actual_total < 14 THEN '11-13'
                WHEN actual_total < 17 THEN '14-16'
                ELSE '17+'
            END as bucket,
            COUNT(*) as n,
            AVG(CASE WHEN was_correct = 1 THEN 1.0 ELSE 0.0 END) as acc
        FROM experimental_totals_predictions
        WHERE was_correct IS NOT NULL
        GROUP BY bucket
        ORDER BY bucket
    ''')
    for row in c.fetchall():
        print(f"   Total {row['bucket']}: {row['acc']*100:.1f}% ({row['n']} games)")
    
    conn.close()
    
    # Verdict
    print("\n" + "=" * 65)
    mae_better = exp['mae'] < prod['mae']
    ou_better = exp['ou_acc'] > prod['ou_acc']
    if mae_better and ou_better:
        print("🏆 VERDICT: EXPERIMENTAL WINS — better MAE and O/U accuracy")
    elif mae_better:
        print("📊 VERDICT: Experimental better MAE, production better O/U")
    elif ou_better:
        print("📊 VERDICT: Experimental better O/U, production better MAE")
    else:
        print("👑 VERDICT: PRODUCTION remains champion")


if __name__ == '__main__':
    compare()
