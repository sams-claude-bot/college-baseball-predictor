#!/usr/bin/env python3
"""
Compare predictions across all models (7 models + ensemble)

Usage:
  python compare_models.py "Mississippi State" "Hofstra"
  python compare_models.py "Mississippi State" "UCLA" --neutral
  python compare_models.py --accuracy
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.pythagorean_model import PythagoreanModel
from models.elo_model import EloModel
from models.log5_model import Log5Model
from models.advanced_model import AdvancedModel
from models.pitching_model import PitchingModel
from models.conference_model import ConferenceModel
from models.prior_model import PriorModel
from models.ensemble_model import EnsembleModel, PoissonModelWrapper
from models.neural_model import NeuralModel
# Deprecated: nn_totals, nn_spread, nn_dow_totals — trained on historical data we no longer have
# from models.nn_totals_model import NNTotalsModel
# from models.nn_spread_model import NNSpreadModel
# from models.nn_dow_totals_model import NNDoWTotalsModel
from models.xgboost_model import XGBMoneylineModel, XGBTotalsModel, XGBSpreadModel
from models.lightgbm_model import LGBMoneylineModel, LGBTotalsModel, LGBSpreadModel
from models.momentum_model import get_momentum_score
from models.meta_ensemble import MetaEnsemble
from models.pear_model import PearModel
from models.quality_model import QualityModel
from scripts.database import get_connection

# All available models
MODELS = {
    "pythagorean": PythagoreanModel(),
    "elo": EloModel(),
    "log5": Log5Model(),
    "advanced": AdvancedModel(),
    "pitching": PitchingModel(),
    "conference": ConferenceModel(),
    "prior": PriorModel(),
    "poisson": PoissonModelWrapper(),
    "neural": NeuralModel(use_model_predictions=False),
    # Deprecated: nn_totals, nn_spread, nn_dow_totals removed (no historical data to retrain)
    "xgboost": XGBMoneylineModel(use_model_predictions=False),
    "xgb_totals": XGBTotalsModel(use_model_predictions=False),
    "xgb_spread": XGBSpreadModel(use_model_predictions=False),
    "lightgbm": LGBMoneylineModel(use_model_predictions=False),
    "lgb_totals": LGBTotalsModel(use_model_predictions=False),
    "lgb_spread": LGBSpreadModel(use_model_predictions=False),
    "pear": PearModel(),
    "quality": QualityModel(),
    "ensemble": EnsembleModel(),
    "meta_ensemble": MetaEnsemble(),
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
    
    # Sort by home win probability (ensemble starred)
    sorted_models = sorted(results.items(), 
                          key=lambda x: x[1]['home_win_probability'], 
                          reverse=True)
    
    # Display comparison table
    print(f"\n{'Model':<15} {'Home Win':<10} {'Away Win':<10} {'Home Runs':<10} {'Away Runs':<10} {'Pick'}")
    print("-" * 65)
    
    for name, pred in sorted_models:
        star = " ★" if name == "ensemble" else ""
        home_pct = f"{pred['home_win_probability']*100:.1f}%"
        away_pct = f"{pred['away_win_probability']*100:.1f}%"
        home_runs = f"{pred['projected_home_runs']:.1f}"
        away_runs = f"{pred['projected_away_runs']:.1f}"
        pick = home_team if pred['home_win_probability'] > 0.5 else away_team
        
        print(f"  {name:<13} {home_pct:<10} {away_pct:<10} {home_runs:<10} {away_runs:<10} {pick}{star}")
    
    # Run line comparison
    print(f"\n{'Model':<15} {'Home -1.5':<12} {'Away +1.5':<12} {'Run Line Pick'}")
    print("-" * 55)
    
    for name, pred in sorted_models:
        if 'run_line' in pred:
            rl = pred['run_line']
            home_cover = rl.get('home_cover_prob', rl.get('home_minus_1_5', 0))
            away_cover = rl.get('away_cover_prob', rl.get('away_plus_1_5', 0))
            home_rl = f"{home_cover*100:.1f}%"
            away_rl = f"{away_cover*100:.1f}%"
            rl_pick = f"{home_team} -1.5" if home_cover > 0.5 else f"{away_team} +1.5"
            print(f"  {name:<13} {home_rl:<12} {away_rl:<12} {rl_pick}")
    
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
    
    # Dynamic ensemble weights
    ensemble = MODELS.get('ensemble')
    if ensemble and hasattr(ensemble, 'weights'):
        print(f"\n  Ensemble Weights:")
        for model_name, weight in sorted(ensemble.weights.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(weight * 40)
            print(f"    {model_name:<13} {weight*100:>5.1f}% {bar}")
    
    print()
    return results


def track_prediction_accuracy():
    """Show model accuracy based on completed predictions"""
    conn = get_connection()
    c = conn.cursor()
    
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
    if "--accuracy" in sys.argv:
        track_prediction_accuracy()
        if len(sys.argv) < 3:
            return
    
    if len(sys.argv) < 3:
        print("Usage: python compare_models.py <home_team> <away_team> [--neutral]")
        print("\nExamples:")
        print("  python compare_models.py 'Mississippi State' 'Hofstra'")
        print("  python compare_models.py 'Mississippi State' 'UCLA' --neutral")
        print("\nOptions:")
        print("  --neutral    Treat as neutral site game")
        print("  --accuracy   Show model accuracy stats")
        return
    
    home = sys.argv[1]
    away = sys.argv[2]
    neutral = "--neutral" in sys.argv
    
    compare_predictions(home, away, neutral)


if __name__ == "__main__":
    main()
