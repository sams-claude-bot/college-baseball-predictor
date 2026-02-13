#!/usr/bin/env python3
"""
Track and report model accuracy over time

Compares predictions to actual results and generates weekly accuracy reports.
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

_scripts_dir = Path(__file__).parent
_models_dir = _scripts_dir.parent / "models"
_reports_dir = _scripts_dir.parent / "reports"
sys.path.insert(0, str(_scripts_dir))
sys.path.insert(0, str(_models_dir))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fpdf import FPDF

from database import get_connection
from compare_models import MODELS, normalize_team_id

_reports_dir.mkdir(parents=True, exist_ok=True)


def init_accuracy_table():
    """Create accuracy tracking table"""
    conn = get_connection()
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS model_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            predicted_winner_id TEXT,
            home_win_prob REAL,
            projected_home_runs REAL,
            projected_away_runs REAL,
            projected_total REAL,
            run_line_pick TEXT,
            run_line_prob REAL,
            predicted_at TEXT DEFAULT CURRENT_TIMESTAMP,
            -- Results (filled after game)
            actual_winner_id TEXT,
            actual_home_runs INTEGER,
            actual_away_runs INTEGER,
            moneyline_correct INTEGER,
            run_line_correct INTEGER,
            total_over_under TEXT,
            total_correct INTEGER,
            UNIQUE(game_id, model_name)
        )
    ''')
    
    conn.commit()
    conn.close()


def record_prediction(game_id, home_team_id, away_team_id):
    """Record predictions from all models for a game"""
    init_accuracy_table()
    conn = get_connection()
    c = conn.cursor()
    
    for model_name, model in MODELS.items():
        try:
            pred = model.predict_game(home_team_id, away_team_id)
            
            predicted_winner = home_team_id if pred['home_win_probability'] > 0.5 else away_team_id
            run_line = pred.get('run_line', {})
            run_line_pick = f"{home_team_id} -1.5" if run_line.get('home_cover_prob', run_line.get('home_minus_1_5', 0)) > 0.5 else f"{away_team_id} +1.5"
            run_line_prob = max(run_line.get('home_cover_prob', run_line.get('home_minus_1_5', 0)), 
                               run_line.get('away_cover_prob', run_line.get('away_plus_1_5', 0)))
            
            c.execute('''
                INSERT OR REPLACE INTO model_predictions 
                (game_id, model_name, predicted_winner_id, home_win_prob,
                 projected_home_runs, projected_away_runs, projected_total,
                 run_line_pick, run_line_prob)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (game_id, model_name, predicted_winner, pred['home_win_probability'],
                  pred['projected_home_runs'], pred['projected_away_runs'],
                  pred['projected_total'], run_line_pick, run_line_prob))
            
        except Exception as e:
            print(f"Error recording {model_name} prediction: {e}")
    
    conn.commit()
    conn.close()


def update_results(game_id, actual_winner_id, home_runs, away_runs):
    """Update predictions with actual results"""
    conn = get_connection()
    c = conn.cursor()
    
    actual_total = home_runs + away_runs
    home_covered = (home_runs - away_runs) >= 2  # -1.5 spread
    
    c.execute('''
        SELECT id, model_name, predicted_winner_id, projected_total, run_line_pick
        FROM model_predictions
        WHERE game_id = ?
    ''', (game_id,))
    
    for row in c.fetchall():
        pred_id, model_name, pred_winner, proj_total, rl_pick = row
        
        # Check moneyline
        ml_correct = 1 if pred_winner == actual_winner_id else 0
        
        # Check run line
        if "-1.5" in (rl_pick or ""):
            # Picked favorite to cover
            rl_correct = 1 if home_covered else 0
        else:
            # Picked underdog
            rl_correct = 1 if not home_covered else 0
        
        # Check total
        if proj_total:
            total_ou = "over" if actual_total > proj_total else "under" if actual_total < proj_total else "push"
            # For now, we'll consider it "correct" if we were on the right side of our projection
            total_correct = 1 if abs(actual_total - proj_total) <= 2 else 0
        else:
            total_ou = None
            total_correct = None
        
        c.execute('''
            UPDATE model_predictions
            SET actual_winner_id = ?,
                actual_home_runs = ?,
                actual_away_runs = ?,
                moneyline_correct = ?,
                run_line_correct = ?,
                total_over_under = ?,
                total_correct = ?
            WHERE id = ?
        ''', (actual_winner_id, home_runs, away_runs, ml_correct, 
              rl_correct, total_ou, total_correct, pred_id))
    
    conn.commit()
    conn.close()


def get_model_accuracy(days=None, model_name=None):
    """Get accuracy stats for models"""
    conn = get_connection()
    c = conn.cursor()
    
    query = '''
        SELECT model_name,
               COUNT(*) as total_games,
               SUM(moneyline_correct) as ml_wins,
               SUM(run_line_correct) as rl_wins,
               SUM(total_correct) as total_wins,
               AVG(ABS(projected_total - (actual_home_runs + actual_away_runs))) as avg_total_error
        FROM model_predictions
        WHERE actual_winner_id IS NOT NULL
    '''
    
    params = []
    
    if days:
        query += " AND date(predicted_at) >= date('now', ?)"
        params.append(f'-{days} days')
    
    if model_name:
        query += " AND model_name = ?"
        params.append(model_name)
    
    query += " GROUP BY model_name ORDER BY ml_wins DESC"
    
    c.execute(query, params)
    rows = c.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        name, total, ml_w, rl_w, tot_w, avg_err = row
        ml_w = ml_w or 0
        rl_w = rl_w or 0
        tot_w = tot_w or 0
        
        results.append({
            'model': name,
            'games': total,
            'moneyline_wins': ml_w,
            'moneyline_pct': ml_w / total * 100 if total > 0 else 0,
            'run_line_wins': rl_w,
            'run_line_pct': rl_w / total * 100 if total > 0 else 0,
            'total_wins': tot_w,
            'total_pct': tot_w / total * 100 if total > 0 else 0,
            'avg_total_error': avg_err or 0
        })
    
    return results


def generate_accuracy_report(output_path=None):
    """Generate weekly accuracy PDF report"""
    from fpdf import FPDF
    
    if output_path is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
        output_path = _reports_dir / f"model_accuracy_{date_str}.pdf"
    
    # Get accuracy data
    all_time = get_model_accuracy()
    last_7_days = get_model_accuracy(days=7)
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title
    pdf.set_font('Helvetica', 'B', 20)
    pdf.cell(0, 15, 'MODEL ACCURACY REPORT', ln=True, align='C')
    pdf.set_font('Helvetica', '', 12)
    pdf.cell(0, 10, f'Generated: {datetime.now().strftime("%B %d, %Y")}', ln=True, align='C')
    pdf.ln(10)
    
    if not all_time:
        pdf.set_font('Helvetica', 'I', 12)
        pdf.cell(0, 10, 'No completed predictions yet. Check back after games are played!', ln=True)
        pdf.output(str(output_path))
        return output_path
    
    # Create accuracy chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Moneyline accuracy
    models = [r['model'] for r in all_time]
    ml_pcts = [r['moneyline_pct'] for r in all_time]
    
    colors = ['#800000', '#4a4a4a', '#2ecc71', '#3498db', '#9b59b6']
    axes[0].bar(models, ml_pcts, color=colors[:len(models)])
    axes[0].set_ylabel('Accuracy %')
    axes[0].set_title('Moneyline Accuracy (All Time)')
    axes[0].set_ylim(0, 100)
    axes[0].axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Break-even')
    for i, (m, p) in enumerate(zip(models, ml_pcts)):
        axes[0].annotate(f'{p:.1f}%', xy=(i, p), ha='center', va='bottom')
    
    # Run line accuracy
    rl_pcts = [r['run_line_pct'] for r in all_time]
    axes[1].bar(models, rl_pcts, color=colors[:len(models)])
    axes[1].set_ylabel('Accuracy %')
    axes[1].set_title('Run Line Accuracy (All Time)')
    axes[1].set_ylim(0, 100)
    axes[1].axhline(y=52.4, color='red', linestyle='--', alpha=0.5, label='Break-even (-110)')
    for i, (m, p) in enumerate(zip(models, rl_pcts)):
        axes[1].annotate(f'{p:.1f}%', xy=(i, p), ha='center', va='bottom')
    
    plt.tight_layout()
    
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        plt.savefig(tmp.name, dpi=150, bbox_inches='tight')
        plt.close()
        pdf.image(tmp.name, x=10, w=190)
    
    pdf.ln(10)
    
    # All-time stats table
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, 'All-Time Performance', ln=True)
    pdf.set_font('Courier', '', 9)
    
    header = f"{'Model':<12} {'Games':<8} {'ML Win%':<10} {'RL Win%':<10} {'Avg Err':<10}"
    pdf.cell(0, 6, header, ln=True)
    pdf.cell(0, 4, "-" * 55, ln=True)
    
    for r in all_time:
        row = f"{r['model']:<12} {r['games']:<8} {r['moneyline_pct']:>6.1f}%   {r['run_line_pct']:>6.1f}%   {r['avg_total_error']:>6.2f}"
        pdf.cell(0, 6, row, ln=True)
    
    # Last 7 days
    if last_7_days:
        pdf.ln(10)
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, 'Last 7 Days', ln=True)
        pdf.set_font('Courier', '', 9)
        
        pdf.cell(0, 6, header, ln=True)
        pdf.cell(0, 4, "-" * 55, ln=True)
        
        for r in last_7_days:
            row = f"{r['model']:<12} {r['games']:<8} {r['moneyline_pct']:>6.1f}%   {r['run_line_pct']:>6.1f}%   {r['avg_total_error']:>6.2f}"
            pdf.cell(0, 6, row, ln=True)
    
    # Best/Worst
    pdf.ln(10)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, 'Key Insights', ln=True)
    pdf.set_font('Helvetica', '', 11)
    
    if all_time:
        best_ml = max(all_time, key=lambda x: x['moneyline_pct'])
        best_rl = max(all_time, key=lambda x: x['run_line_pct'])
        
        pdf.cell(0, 7, f"Best Moneyline: {best_ml['model']} ({best_ml['moneyline_pct']:.1f}%)", ln=True)
        pdf.cell(0, 7, f"Best Run Line: {best_rl['model']} ({best_rl['run_line_pct']:.1f}%)", ln=True)
        
        # Profitability note
        pdf.ln(5)
        pdf.set_font('Helvetica', 'I', 10)
        pdf.cell(0, 6, "Note: Need 52.4% accuracy on -110 lines to be profitable.", ln=True)
    
    pdf.output(str(output_path))
    print(f"✓ Accuracy report saved to: {output_path}")
    return output_path


def print_accuracy_summary():
    """Print accuracy summary to console"""
    all_time = get_model_accuracy()
    
    if not all_time:
        print("\n📊 No completed predictions yet.")
        print("   Predictions will be tracked once games finish.")
        return
    
    print("\n" + "=" * 60)
    print("📊 MODEL ACCURACY SUMMARY")
    print("=" * 60)
    
    print(f"\n{'Model':<12} {'Games':<8} {'ML Win%':<12} {'RL Win%':<12}")
    print("-" * 50)
    
    for r in all_time:
        ml_str = f"{r['moneyline_pct']:.1f}% ({r['moneyline_wins']}/{r['games']})"
        rl_str = f"{r['run_line_pct']:.1f}% ({r['run_line_wins']}/{r['games']})"
        print(f"{r['model']:<12} {r['games']:<8} {ml_str:<12} {rl_str:<12}")
    
    print("\n" + "=" * 60)


def main():
    if len(sys.argv) < 2:
        print_accuracy_summary()
        print("\nUsage:")
        print("  python model_accuracy.py summary       - Print accuracy summary")
        print("  python model_accuracy.py report        - Generate PDF report")
        print("  python model_accuracy.py record <game_id> <home_id> <away_id>")
        print("  python model_accuracy.py result <game_id> <winner_id> <home_runs> <away_runs>")
        return
    
    cmd = sys.argv[1]
    
    if cmd == "summary":
        print_accuracy_summary()
    
    elif cmd == "report":
        generate_accuracy_report()
    
    elif cmd == "record":
        if len(sys.argv) < 5:
            print("Usage: python model_accuracy.py record <game_id> <home_id> <away_id>")
            return
        record_prediction(sys.argv[2], sys.argv[3], sys.argv[4])
        print(f"✓ Recorded predictions for {sys.argv[2]}")
    
    elif cmd == "result":
        if len(sys.argv) < 6:
            print("Usage: python model_accuracy.py result <game_id> <winner_id> <home_runs> <away_runs>")
            return
        update_results(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
        print(f"✓ Updated results for {sys.argv[2]}")
    
    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
