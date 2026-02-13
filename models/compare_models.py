#!/usr/bin/env python3
"""
Compare predictions across multiple models

Usage:
  python compare_models.py "Mississippi State" "Hofstra"
  python compare_models.py "Mississippi State" "UCLA" --neutral
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from pythagorean_model import PythagoreanModel
from elo_model import EloModel
from log5_model import Log5Model
from ensemble_model import EnsembleModel
from database import get_connection

MODELS = {
    "pythagorean": PythagoreanModel(),
    "elo": EloModel(),
    "log5": Log5Model(),
    "ensemble": EnsembleModel()
}

def normalize_team_id(name):
    """Convert team name to ID"""
    return name.lower().strip().replace(" ", "-")

def compare_predictions(home_team, away_team, neutral_site=False):
    """Run all models and compare predictions"""
    home_id = normalize_team_id(home_team)
    away_id = normalize_team_id(away_team)
    
    site_label = " (Neutral)" if neutral_site else ""
    
    print(f"\n{'='*65}")
    print(f"  MODEL COMPARISON: {away_team} @ {home_team}{site_label}")
    print('='*65)
    
    results = {}
    
    for name, model in MODELS.items():
        try:
            pred = model.predict_game(home_id, away_id, neutral_site)
            results[name] = pred
        except Exception as e:
            print(f"\n⚠️  {name}: Error - {e}")
    
    # Display comparison table
    print(f"\n{'Model':<15} {'Home Win':<10} {'Away Win':<10} {'Home Runs':<10} {'Away Runs':<10} {'Pick'}")
    print("-" * 65)
    
    for name, pred in results.items():
        home_pct = f"{pred['home_win_probability']*100:.1f}%"
        away_pct = f"{pred['away_win_probability']*100:.1f}%"
        home_runs = f"{pred['projected_home_runs']:.1f}"
        away_runs = f"{pred['projected_away_runs']:.1f}"
        pick = home_team if pred['home_win_probability'] > 0.5 else away_team
        
        print(f"{name:<15} {home_pct:<10} {away_pct:<10} {home_runs:<10} {away_runs:<10} {pick}")
    
    # Run line comparison
    print(f"\n{'Model':<15} {'Home -1.5':<12} {'Away +1.5':<12} {'Run Line Pick'}")
    print("-" * 55)
    
    for name, pred in results.items():
        if 'run_line' in pred:
            rl = pred['run_line']
            home_rl = f"{rl.get('home_cover_prob', rl.get('home_minus_1_5', 0))*100:.1f}%"
            away_rl = f"{rl.get('away_cover_prob', rl.get('away_plus_1_5', 0))*100:.1f}%"
            rl_pick = f"{home_team} -1.5" if rl.get('home_cover_prob', rl.get('home_minus_1_5', 0)) > 0.5 else f"{away_team} +1.5"
            print(f"{name:<15} {home_rl:<12} {away_rl:<12} {rl_pick}")
    
    # Consensus
    print(f"\n{'='*65}")
    print("CONSENSUS:")
    
    home_picks = sum(1 for p in results.values() if p['home_win_probability'] > 0.5)
    away_picks = len(results) - home_picks
    
    if home_picks > away_picks:
        consensus_team = home_team
        consensus_count = home_picks
    elif away_picks > home_picks:
        consensus_team = away_team
        consensus_count = away_picks
    else:
        consensus_team = "Split"
        consensus_count = home_picks
    
    avg_home_prob = sum(p['home_win_probability'] for p in results.values()) / len(results)
    avg_home_runs = sum(p['projected_home_runs'] for p in results.values()) / len(results)
    avg_away_runs = sum(p['projected_away_runs'] for p in results.values()) / len(results)
    
    print(f"  Winner: {consensus_team} ({consensus_count}/{len(results)} models agree)")
    print(f"  Avg Home Win Prob: {avg_home_prob*100:.1f}%")
    print(f"  Avg Projected Score: {away_team} {avg_away_runs:.1f} - {home_team} {avg_home_runs:.1f}")
    print()
    
    return results

def track_prediction_accuracy():
    """Show model accuracy based on completed predictions"""
    conn = get_connection()
    c = conn.cursor()
    
    # Check if we have tracked predictions
    c.execute("""
        SELECT model_version, 
               COUNT(*) as total,
               SUM(moneyline_correct) as ml_correct,
               SUM(run_line_correct) as rl_correct
        FROM predictions 
        WHERE actual_winner_id IS NOT NULL
        GROUP BY model_version
    """)
    
    rows = c.fetchall()
    conn.close()
    
    if not rows:
        print("\n📊 No completed predictions to analyze yet")
        print("   Predictions will be tracked once games are played")
        return
    
    print("\n📊 MODEL ACCURACY:")
    print("-" * 50)
    print(f"{'Model':<15} {'Games':<8} {'ML Accuracy':<12} {'RL Accuracy'}")
    print("-" * 50)
    
    for row in rows:
        model = row[0]
        total = row[1]
        ml_correct = row[2] or 0
        rl_correct = row[3] or 0
        
        ml_pct = f"{ml_correct/total*100:.1f}%" if total > 0 else "N/A"
        rl_pct = f"{rl_correct/total*100:.1f}%" if total > 0 else "N/A"
        
        print(f"{model:<15} {total:<8} {ml_pct:<12} {rl_pct}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_models.py <home_team> <away_team> [--neutral]")
        print("\nExamples:")
        print("  python compare_models.py 'Mississippi State' 'Hofstra'")
        print("  python compare_models.py 'Mississippi State' 'UCLA' --neutral")
        print("\nOptions:")
        print("  --neutral    Treat as neutral site game")
        print("  --accuracy   Show model accuracy stats")
        
        if "--accuracy" in sys.argv:
            track_prediction_accuracy()
        return
    
    home = sys.argv[1]
    away = sys.argv[2]
    neutral = "--neutral" in sys.argv
    
    compare_predictions(home, away, neutral)

if __name__ == "__main__":
    main()
