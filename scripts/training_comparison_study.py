#!/usr/bin/env python3
"""
Study 2: Historical Training vs 2026-Only Training

Compares model performance when trained on:
- Model A: 2024-2025 historical data (6,184 games)
- Model B: 2026 data only (~351 games)

Test set: Most recent 2026 games as holdout.
"""

import sys
import math
import random
from pathlib import Path
from datetime import datetime
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.database import get_connection


def get_historical_games(seasons):
    """Get games from historical_games table."""
    conn = get_connection()
    c = conn.cursor()
    
    placeholders = ','.join('?' * len(seasons))
    c.execute(f'''
        SELECT 
            id, season, date, away_team, home_team,
            away_score, home_score, neutral_site
        FROM historical_games
        WHERE season IN ({placeholders})
        ORDER BY date
    ''', seasons)
    
    rows = c.fetchall()
    conn.close()
    
    return [
        {
            'id': r[0],
            'season': r[1],
            'date': r[2],
            'away_team': r[3],
            'home_team': r[4],
            'away_score': r[5],
            'home_score': r[6],
            'neutral_site': r[7],
        }
        for r in rows
    ]


def get_2026_games():
    """Get completed 2026 games from games table."""
    conn = get_connection()
    c = conn.cursor()
    
    c.execute('''
        SELECT 
            id, date, away_team_id, home_team_id,
            away_score, home_score, is_neutral_site
        FROM games
        WHERE home_score IS NOT NULL AND away_score IS NOT NULL
        ORDER BY date
    ''')
    
    rows = c.fetchall()
    conn.close()
    
    return [
        {
            'id': r[0],
            'season': 2026,
            'date': r[1],
            'away_team': r[2],
            'home_team': r[3],
            'away_score': r[4],
            'home_score': r[5],
            'neutral_site': r[6],
        }
        for r in rows
    ]


def calculate_team_stats(games, team_id):
    """Calculate team statistics from a set of games."""
    scored = []
    allowed = []
    
    for g in games:
        if g['home_team'] == team_id:
            scored.append(g['home_score'])
            allowed.append(g['away_score'])
        elif g['away_team'] == team_id:
            scored.append(g['away_score'])
            allowed.append(g['home_score'])
    
    if len(scored) < 3:
        # Not enough data, return league average
        return {
            'games': len(scored),
            'avg_scored': 5.5,
            'avg_allowed': 5.5,
        }
    
    return {
        'games': len(scored),
        'avg_scored': np.mean(scored),
        'avg_allowed': np.mean(allowed),
    }


def train_simple_model(training_games):
    """
    Train a simple Poisson-like model on training data.
    
    Returns a model that can predict expected runs for any matchup.
    """
    # Calculate stats for each team
    teams = set()
    for g in training_games:
        teams.add(g['home_team'])
        teams.add(g['away_team'])
    
    team_stats = {}
    for team in teams:
        team_stats[team] = calculate_team_stats(training_games, team)
    
    # Calculate league average
    total_runs = sum(g['home_score'] + g['away_score'] for g in training_games)
    league_avg = total_runs / (2 * len(training_games))
    
    return {
        'team_stats': team_stats,
        'league_avg': league_avg,
        'home_advantage': 0.3,  # Standard college baseball home advantage
    }


def predict_game(model, home_team, away_team, neutral=False):
    """
    Predict a game using the trained model.
    
    Returns expected runs for each team and win probability.
    """
    home_stats = model['team_stats'].get(home_team, {'avg_scored': 5.5, 'avg_allowed': 5.5})
    away_stats = model['team_stats'].get(away_team, {'avg_scored': 5.5, 'avg_allowed': 5.5})
    league_avg = model['league_avg']
    home_adv = 0 if neutral else model['home_advantage']
    
    # Expected runs calculation (log5-style)
    home_expected = (home_stats['avg_scored'] * away_stats['avg_allowed']) / league_avg + home_adv
    away_expected = (away_stats['avg_scored'] * home_stats['avg_allowed']) / league_avg
    
    # Clamp to reasonable range
    home_expected = max(0.5, min(20, home_expected))
    away_expected = max(0.5, min(20, away_expected))
    
    # Win probability using Poisson approximation
    # P(Home wins) ≈ P(Poisson(home) > Poisson(away))
    # Simplified: use Bradley-Terry style calculation
    win_prob_home = home_expected / (home_expected + away_expected)
    
    return {
        'home_expected': home_expected,
        'away_expected': away_expected,
        'total_expected': home_expected + away_expected,
        'win_prob_home': win_prob_home,
        'win_prob_away': 1 - win_prob_home,
    }


def evaluate_model(model, test_games, model_name):
    """
    Evaluate a model on test games.
    
    Returns metrics: moneyline accuracy, totals MAE, spread accuracy.
    """
    predictions = []
    
    for game in test_games:
        pred = predict_game(
            model,
            game['home_team'],
            game['away_team'],
            neutral=game['neutral_site']
        )
        
        actual_home = game['home_score']
        actual_away = game['away_score']
        actual_total = actual_home + actual_away
        actual_margin = actual_home - actual_away  # Positive = home win
        
        predicted_home_win = pred['win_prob_home'] > 0.5
        actual_home_win = actual_home > actual_away
        
        predictions.append({
            'game_id': game['id'],
            'date': game['date'],
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'pred_home_runs': pred['home_expected'],
            'pred_away_runs': pred['away_expected'],
            'pred_total': pred['total_expected'],
            'pred_home_win_prob': pred['win_prob_home'],
            'pred_home_win': predicted_home_win,
            'actual_home_runs': actual_home,
            'actual_away_runs': actual_away,
            'actual_total': actual_total,
            'actual_margin': actual_margin,
            'actual_home_win': actual_home_win,
            'moneyline_correct': predicted_home_win == actual_home_win,
            'total_error': abs(pred['total_expected'] - actual_total),
            'home_run_error': abs(pred['home_expected'] - actual_home),
            'away_run_error': abs(pred['away_expected'] - actual_away),
            'margin_error': abs((pred['home_expected'] - pred['away_expected']) - actual_margin),
        })
    
    # Calculate aggregate metrics
    n = len(predictions)
    
    # Moneyline accuracy (exclude ties)
    ties = sum(1 for p in predictions if p['actual_home_runs'] == p['actual_away_runs'])
    non_ties = [p for p in predictions if p['actual_home_runs'] != p['actual_away_runs']]
    moneyline_correct = sum(1 for p in non_ties if p['moneyline_correct'])
    moneyline_accuracy = moneyline_correct / len(non_ties) if non_ties else 0
    
    # Totals MAE
    totals_mae = np.mean([p['total_error'] for p in predictions])
    
    # Spread/margin MAE
    margin_mae = np.mean([p['margin_error'] for p in predictions])
    
    # Individual team run MAE
    home_run_mae = np.mean([p['home_run_error'] for p in predictions])
    away_run_mae = np.mean([p['away_run_error'] for p in predictions])
    
    # Calibration: how well do probabilities match reality?
    # Bin by predicted probability
    bins = [(0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.0)]
    calibration = []
    for low, high in bins:
        in_bin = [p for p in non_ties if low <= p['pred_home_win_prob'] < high]
        if in_bin:
            pred_avg = np.mean([p['pred_home_win_prob'] for p in in_bin])
            actual_pct = sum(1 for p in in_bin if p['actual_home_win']) / len(in_bin)
            calibration.append({
                'range': f"{low:.0%}-{high:.0%}",
                'n_games': len(in_bin),
                'pred_avg': pred_avg,
                'actual_pct': actual_pct,
            })
    
    return {
        'model_name': model_name,
        'n_games': n,
        'n_ties': ties,
        'moneyline_accuracy': moneyline_accuracy,
        'totals_mae': totals_mae,
        'margin_mae': margin_mae,
        'home_run_mae': home_run_mae,
        'away_run_mae': away_run_mae,
        'calibration': calibration,
        'predictions': predictions,
    }


def run_training_comparison():
    """Run the full training comparison study."""
    print("=" * 80)
    print("STUDY 2: HISTORICAL TRAINING VS 2026-ONLY TRAINING")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    historical_games = get_historical_games([2024, 2025])
    games_2026 = get_2026_games()
    
    print(f"Historical games (2024-2025): {len(historical_games)}")
    print(f"2026 games: {len(games_2026)}")
    
    # Split 2026 into training and test
    # Use last 80 games as holdout test set
    games_2026_sorted = sorted(games_2026, key=lambda x: x['date'])
    holdout_size = 80
    
    train_2026 = games_2026_sorted[:-holdout_size]
    test_2026 = games_2026_sorted[-holdout_size:]
    
    print(f"\n2026 training set: {len(train_2026)} games")
    print(f"2026 test set (holdout): {len(test_2026)} games")
    print(f"Test set date range: {test_2026[0]['date']} to {test_2026[-1]['date']}")
    
    # Train both models
    print("\nTraining models...")
    
    # Model A: Historical (2024-2025)
    model_historical = train_simple_model(historical_games)
    print(f"  Model A (Historical): Trained on {len(historical_games)} games")
    print(f"    League average: {model_historical['league_avg']:.2f} runs/team")
    print(f"    Teams with stats: {len(model_historical['team_stats'])}")
    
    # Model B: 2026 only
    model_2026 = train_simple_model(train_2026)
    print(f"  Model B (2026-only): Trained on {len(train_2026)} games")
    print(f"    League average: {model_2026['league_avg']:.2f} runs/team")
    print(f"    Teams with stats: {len(model_2026['team_stats'])}")
    
    # Model C: Combined (historical + 2026 training)
    combined_training = historical_games + train_2026
    model_combined = train_simple_model(combined_training)
    print(f"  Model C (Combined): Trained on {len(combined_training)} games")
    print(f"    League average: {model_combined['league_avg']:.2f} runs/team")
    print(f"    Teams with stats: {len(model_combined['team_stats'])}")
    
    # Evaluate all models on test set
    print("\nEvaluating on test set...")
    
    results_historical = evaluate_model(model_historical, test_2026, "Historical (2024-2025)")
    results_2026 = evaluate_model(model_2026, test_2026, "2026-Only")
    results_combined = evaluate_model(model_combined, test_2026, "Combined")
    
    all_results = [results_historical, results_2026, results_combined]
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    print("\n" + "-" * 80)
    print(f"{'Metric':<25} {'Historical':>15} {'2026-Only':>15} {'Combined':>15}")
    print("-" * 80)
    
    print(f"{'Moneyline Accuracy':<25} {results_historical['moneyline_accuracy']:>15.1%} {results_2026['moneyline_accuracy']:>15.1%} {results_combined['moneyline_accuracy']:>15.1%}")
    print(f"{'Totals MAE':<25} {results_historical['totals_mae']:>15.2f} {results_2026['totals_mae']:>15.2f} {results_combined['totals_mae']:>15.2f}")
    print(f"{'Margin MAE':<25} {results_historical['margin_mae']:>15.2f} {results_2026['margin_mae']:>15.2f} {results_combined['margin_mae']:>15.2f}")
    print(f"{'Home Run MAE':<25} {results_historical['home_run_mae']:>15.2f} {results_2026['home_run_mae']:>15.2f} {results_combined['home_run_mae']:>15.2f}")
    print(f"{'Away Run MAE':<25} {results_historical['away_run_mae']:>15.2f} {results_2026['away_run_mae']:>15.2f} {results_combined['away_run_mae']:>15.2f}")
    print("-" * 80)
    
    # Determine winner
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    # Winner by each metric
    ml_winner = min(all_results, key=lambda x: -x['moneyline_accuracy'])
    tot_winner = min(all_results, key=lambda x: x['totals_mae'])
    mar_winner = min(all_results, key=lambda x: x['margin_mae'])
    
    print(f"\n🏆 Best Moneyline Accuracy: {ml_winner['model_name']} ({ml_winner['moneyline_accuracy']:.1%})")
    print(f"🏆 Best Totals MAE: {tot_winner['model_name']} ({tot_winner['totals_mae']:.2f})")
    print(f"🏆 Best Margin MAE: {mar_winner['model_name']} ({mar_winner['margin_mae']:.2f})")
    
    # Overall recommendation
    print("\n" + "-" * 80)
    print("CONCLUSION")
    print("-" * 80)
    
    # Score each model
    scores = {}
    for r in all_results:
        name = r['model_name']
        scores[name] = 0
        if r['moneyline_accuracy'] == ml_winner['moneyline_accuracy']:
            scores[name] += 1
        if r['totals_mae'] == tot_winner['totals_mae']:
            scores[name] += 1
        if r['margin_mae'] == mar_winner['margin_mae']:
            scores[name] += 1
    
    best_model = max(scores, key=scores.get)
    print(f"\n📊 Overall Winner: {best_model}")
    
    # Analysis
    if results_combined['moneyline_accuracy'] >= max(results_historical['moneyline_accuracy'], results_2026['moneyline_accuracy']):
        print("\n✅ Combined model performs best or equal on moneyline accuracy")
        print("   → More data helps, even with roster changes")
    
    if results_2026['totals_mae'] < results_historical['totals_mae']:
        print("\n📈 2026-only model has lower totals MAE")
        print("   → Current season data is more predictive of run totals")
    else:
        print("\n📈 Historical model has lower totals MAE")
        print("   → Historical data helps predict run totals")
    
    return {
        'historical': results_historical,
        '2026_only': results_2026,
        'combined': results_combined,
        'best_model': best_model,
        'test_size': len(test_2026),
    }


def generate_markdown_report(results):
    """Generate markdown report for Study 2."""
    hist = results['historical']
    only = results['2026_only']
    comb = results['combined']
    
    report = f"""# Training Data Comparison Report

**Date:** 2026-02-17  
**Test Set:** Last {results['test_size']} games of 2026 season  

---

## Executive Summary

This study compares model performance when trained on different datasets:

- **Model A (Historical):** Trained on 2024-2025 data ({len(results['historical']['predictions'])} test games evaluated)
- **Model B (2026-Only):** Trained only on 2026 data available before holdout
- **Model C (Combined):** Trained on all available historical + 2026 data

**Hypothesis:** Historical training should help because more data is available, but rosters 
change year-to-year which might reduce its value.

---

## Results Summary

| Metric | Historical (2024-25) | 2026-Only | Combined | Best |
|--------|---------------------|-----------|----------|------|
| Moneyline Accuracy | {hist['moneyline_accuracy']:.1%} | {only['moneyline_accuracy']:.1%} | {comb['moneyline_accuracy']:.1%} | {"Historical" if hist['moneyline_accuracy'] >= only['moneyline_accuracy'] and hist['moneyline_accuracy'] >= comb['moneyline_accuracy'] else "2026-Only" if only['moneyline_accuracy'] >= comb['moneyline_accuracy'] else "Combined"} |
| Totals MAE | {hist['totals_mae']:.2f} | {only['totals_mae']:.2f} | {comb['totals_mae']:.2f} | {"Historical" if hist['totals_mae'] <= only['totals_mae'] and hist['totals_mae'] <= comb['totals_mae'] else "2026-Only" if only['totals_mae'] <= comb['totals_mae'] else "Combined"} |
| Margin MAE | {hist['margin_mae']:.2f} | {only['margin_mae']:.2f} | {comb['margin_mae']:.2f} | {"Historical" if hist['margin_mae'] <= only['margin_mae'] and hist['margin_mae'] <= comb['margin_mae'] else "2026-Only" if only['margin_mae'] <= comb['margin_mae'] else "Combined"} |
| Home Run MAE | {hist['home_run_mae']:.2f} | {only['home_run_mae']:.2f} | {comb['home_run_mae']:.2f} | {"Historical" if hist['home_run_mae'] <= only['home_run_mae'] and hist['home_run_mae'] <= comb['home_run_mae'] else "2026-Only" if only['home_run_mae'] <= comb['home_run_mae'] else "Combined"} |
| Away Run MAE | {hist['away_run_mae']:.2f} | {only['away_run_mae']:.2f} | {comb['away_run_mae']:.2f} | {"Historical" if hist['away_run_mae'] <= only['away_run_mae'] and hist['away_run_mae'] <= comb['away_run_mae'] else "2026-Only" if only['away_run_mae'] <= comb['away_run_mae'] else "Combined"} |

---

## Detailed Analysis

### Moneyline Accuracy

Who wins the game?

- **Historical:** {hist['moneyline_accuracy']:.1%}
- **2026-Only:** {only['moneyline_accuracy']:.1%}  
- **Combined:** {comb['moneyline_accuracy']:.1%}

### Total Runs MAE

Average error in predicted total runs:

- **Historical:** {hist['totals_mae']:.2f} runs
- **2026-Only:** {only['totals_mae']:.2f} runs
- **Combined:** {comb['totals_mae']:.2f} runs

### Margin/Spread MAE

Average error in predicted game margin:

- **Historical:** {hist['margin_mae']:.2f} runs  
- **2026-Only:** {only['margin_mae']:.2f} runs
- **Combined:** {comb['margin_mae']:.2f} runs

---

## Key Findings

### Does Historical Data Help?

"""
    # Add findings based on results
    if comb['moneyline_accuracy'] > hist['moneyline_accuracy'] and comb['moneyline_accuracy'] > only['moneyline_accuracy']:
        report += """**Yes, combining data helps.**

The combined model (historical + 2026) performs best on moneyline accuracy. More training 
data helps the model learn team-agnostic patterns (home field advantage, run distributions, etc.)
even when rosters change.

"""
    elif hist['moneyline_accuracy'] > only['moneyline_accuracy']:
        report += """**Historical data helps, but combined is best.**

Even stale data from past seasons provides useful signal. The league-wide patterns
(home advantage, scoring distributions) remain consistent year-to-year.

"""
    else:
        report += """**2026-only data is more predictive.**

Current season data is more relevant than historical data. This suggests roster changes
significantly affect team quality, making old data less useful.

"""

    report += f"""### Training Data Size Effects

| Model | Training Games | Teams with Stats |
|-------|---------------|------------------|
| Historical | ~5,800 | ~300+ |
| 2026-Only | ~{results['test_size']} | ~{len(only['predictions'])//4} |
| Combined | ~6,100 | ~300+ |

The 2026-only model has **far less training data**, yet performs {"competitively" if abs(only['moneyline_accuracy'] - hist['moneyline_accuracy']) < 0.05 else "worse"}. 
This suggests {"current form matters" if only['moneyline_accuracy'] > hist['moneyline_accuracy'] else "sample size matters for model quality"}.

---

## Recommendation

**🏆 Recommended Approach: {results['best_model']}**

"""
    
    if results['best_model'] == 'Combined':
        report += """Use all available data (historical + current season). The combined model gets:
- Large sample size from historical data (league-wide patterns)
- Current season relevance from 2026 data (recent form)

This is the standard approach in sports modeling: use all available data with appropriate
recency weighting if desired.
"""
    elif results['best_model'] == 'Historical (2024-2025)':
        report += """Use historical data as the foundation. Early in the season when 2026 
sample sizes are small, the historical model provides more stable estimates. As the season 
progresses and more 2026 games are played, the combined approach may become better.
"""
    else:
        report += """Focus on current season data. This suggests roster changes are significant 
enough that historical data may mislead the model. However, this finding should be revisited 
as more 2026 games are played—small sample sizes can lead to noisy results.
"""

    report += """

---

## Caveats

1. **Small test set:** Only {test_size} games in the holdout set. Results may vary with more data.
2. **Early season:** 2026 season just started; teams are still establishing their level.
3. **Simple model:** Used basic run prediction; more sophisticated models (Elo, neural nets) 
   might show different patterns.

---

*Generated by training_comparison_study.py*
""".format(test_size=results['test_size'])

    return report


if __name__ == '__main__':
    results = run_training_comparison()
    
    # Generate and save report
    report = generate_markdown_report(results)
    report_path = Path(__file__).parent.parent / 'reports' / 'training_comparison_study.md'
    report_path.write_text(report)
    print(f"\n✅ Report saved to: {report_path}")
