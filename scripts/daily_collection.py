#!/usr/bin/env python3
"""
Daily data collection orchestrator

Run this daily to:
1. Collect NCAA stats
2. Update game results
3. Generate predictions for upcoming games using ensemble model
4. Save daily snapshot
5. Report on model performance
"""

import json
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
sys.path.insert(0, str(BASE_DIR / "models"))

from database import get_current_top_25, get_recent_games

# Import predictor with new ensemble
from predictor_db import Predictor


def safe_import(module_name, attr_name=None):
    """Safely import a module/function, return None if not available"""
    try:
        module = __import__(module_name)
        if attr_name:
            return getattr(module, attr_name, None)
        return module
    except ImportError:
        return None


def run_daily_collection():
    """Main daily collection routine"""
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"\n{'='*60}")
    print(f"Daily Collection - {today}")
    print('='*60)
    
    results = {
        "date": today,
        "timestamp": datetime.now().isoformat(),
        "ncaa_stats": False,
        "upcoming_games": [],
        "predictions": [],
        "model_accuracy": {},
        "errors": []
    }
    
    # 1. Collect NCAA stats (if collector available)
    print("\n[1/7] Collecting NCAA stats...")
    fetch_ncaa = safe_import("collect_ncaa_stats", "fetch_ncaa_stats")
    save_stats = safe_import("collect_ncaa_stats", "save_stats")
    
    if fetch_ncaa and save_stats:
        try:
            stats = fetch_ncaa()
            if stats:
                save_stats(stats, f"ncaa_stats_{today}.json")
                results["ncaa_stats"] = True
                print("  ✓ NCAA stats collected")
            else:
                print("  ✗ Failed to collect NCAA stats")
                results["errors"].append("NCAA stats collection failed")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results["errors"].append(str(e))
    else:
        print("  ⊘ NCAA stats collector not available")
    
    # 2. Get upcoming Mississippi State games
    print("\n[2/7] Getting upcoming MS State games...")
    get_upcoming = safe_import("track_mississippi_state", "get_upcoming_games")
    
    if get_upcoming:
        try:
            upcoming = get_upcoming(7)
            results["upcoming_games"] = upcoming
            print(f"  ✓ Found {len(upcoming)} upcoming games")
            for game in upcoming:
                loc = "vs" if game.get("home") else "@"
                print(f"    - {game['date']}: {loc} {game['opponent']}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results["errors"].append(str(e))
    else:
        print("  ⊘ MS State tracker not available")
    
    # 3. Generate predictions using new ensemble
    print("\n[3/7] Generating predictions (Ensemble Model)...")
    try:
        predictor = Predictor(model='ensemble')
        upcoming = results.get("upcoming_games", [])
        
        for game in upcoming[:3]:  # Next 3 games
            try:
                if game.get("home"):
                    pred = predictor.predict_game("Mississippi State", game["opponent"])
                else:
                    pred = predictor.predict_game(game["opponent"], "Mississippi State")
                
                results["predictions"].append({
                    "date": game["date"],
                    "matchup": f"{'Mississippi State' if game.get('home') else game['opponent']} vs {game['opponent'] if game.get('home') else 'Mississippi State'}",
                    "home_win_prob": pred['home_win_probability'],
                    "predicted_winner": pred['predicted_winner'],
                    "confidence": pred['confidence'],
                    "projected_score": f"{pred['projected_away_runs']:.1f}-{pred['projected_home_runs']:.1f}"
                })
                print(f"    - {game['date']} vs {game['opponent']}: {pred['predicted_winner']} ({pred['confidence']*100:.0f}% conf)")
            except Exception as e:
                print(f"    ✗ Error predicting {game['opponent']}: {e}")
                
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results["errors"].append(str(e))
    
    # 4. SEC out-of-conference summary
    print("\n[4/7] SEC out-of-conference tracking...")
    get_ooc = safe_import("track_sec_teams", "get_ooc_games")
    
    if get_ooc:
        try:
            ooc_games = get_ooc()
            results["sec_ooc_games"] = len(ooc_games)
            print(f"  ✓ {len(ooc_games)} SEC OOC games tracked")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results["errors"].append(str(e))
    else:
        print("  ⊘ SEC tracker not available")
    
    # 5. Top 25 tracking
    print("\n[5/7] Top 25 teams...")
    try:
        top_25 = get_current_top_25()
        results["top_25_count"] = len(top_25)
        print(f"  ✓ Tracking {len(top_25)} ranked teams")
        
        # Show recent Top 25 results if any
        for team in top_25[:5]:  # Top 5
            recent = get_recent_games(team['id'], limit=1)
            if recent and recent[0].get('winner_id'):
                g = recent[0]
                won = "W" if g['winner_id'] == team['id'] else "L"
                opp = g['away_team_name'] if g['home_team_id'] == team['id'] else g['home_team_name']
                print(f"    #{team['current_rank']} {team['name']}: {won} vs {opp}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results["errors"].append(str(e))
    
    # 6. Model accuracy report
    print("\n[6/7] Model accuracy report...")
    try:
        from ensemble_model import EnsembleModel
        ensemble = EnsembleModel()
        accuracy = ensemble.get_model_accuracy()
        
        results["model_accuracy"] = accuracy
        
        print("  Current model weights:")
        for name, stats in sorted(accuracy.items(), 
                                 key=lambda x: x[1].get('current_weight', 0),
                                 reverse=True):
            weight = stats.get('current_weight', 0) * 100
            recent = stats.get('recent_accuracy')
            recent_str = f"{recent*100:.1f}%" if recent else "N/A"
            print(f"    {name:<12}: w={weight:>5.1f}%, acc={recent_str}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results["errors"].append(str(e))
    
    # 7. Save daily snapshot
    print("\n[7/7] Saving daily snapshot...")
    snapshot_dir = BASE_DIR / "data" / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_file = snapshot_dir / f"daily_{today}.json"
    
    with open(snapshot_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  ✓ Saved to {snapshot_file}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"Stats collected: {'Yes' if results['ncaa_stats'] else 'No'}")
    print(f"Upcoming games: {len(results['upcoming_games'])}")
    print(f"Predictions made: {len(results['predictions'])}")
    print(f"Errors: {len(results['errors'])}")
    
    if results["errors"]:
        print("\nErrors encountered:")
        for err in results["errors"]:
            print(f"  - {err}")
    
    return results


def show_model_report():
    """Show detailed model accuracy report"""
    try:
        from ensemble_model import EnsembleModel
        ensemble = EnsembleModel()
        print(ensemble.get_weights_report())
    except Exception as e:
        print(f"Error: {e}")


def compare_prediction(home_team, away_team, neutral=False):
    """Show prediction comparison from all models"""
    predictor = Predictor()
    comparison = predictor.compare_models(home_team, away_team, neutral)
    
    from predictor_db import print_comparison
    print_comparison(comparison)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        
        if cmd == "models":
            show_model_report()
        elif cmd == "compare" and len(sys.argv) > 3:
            neutral = "--neutral" in sys.argv
            compare_prediction(sys.argv[2], sys.argv[3], neutral)
        else:
            print("Usage:")
            print("  python daily_collection.py         # Run daily collection")
            print("  python daily_collection.py models  # Show model accuracy")
            print("  python daily_collection.py compare <home> <away> [--neutral]")
    else:
        run_daily_collection()
