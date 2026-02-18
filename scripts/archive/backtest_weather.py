#!/usr/bin/env python3
"""
Weather Features Backtest Study

Compare model accuracy WITH weather vs WITHOUT weather adjustments.
Train on 2024-2025, test on 2026 games.

Reports:
- Accuracy comparison (with/without weather)
- MAE for total runs prediction
- Which weather features matter most
- Statistical significance of improvement
"""

import sys
import math
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.database import get_connection
from models.weather_model import (
    train_weather_model, 
    calculate_weather_adjustment, 
    load_coefficients,
    compute_wind_out_component
)


def get_2026_games_with_weather():
    """Get all completed 2026 games with weather data."""
    conn = get_connection()
    c = conn.cursor()
    
    # Get games from main games table with weather
    c.execute('''
        SELECT 
            g.id as game_id,
            g.home_team_id,
            g.away_team_id,
            g.home_score,
            g.away_score,
            g.date,
            gw.temp_f,
            gw.wind_speed_mph,
            gw.wind_direction_deg,
            gw.humidity_pct,
            gw.is_dome
        FROM games g
        JOIN game_weather gw ON g.id = gw.game_id
        WHERE g.status = 'final'
        AND g.home_score IS NOT NULL
        AND g.away_score IS NOT NULL
        AND gw.temp_f IS NOT NULL
    ''')
    
    rows = c.fetchall()
    conn.close()
    
    games = []
    for row in rows:
        games.append({
            'game_id': row[0],
            'home_team': row[1],
            'away_team': row[2],
            'home_score': row[3],
            'away_score': row[4],
            'total_runs': row[3] + row[4],
            'date': row[5],
            'weather': {
                'temp_f': row[6],
                'wind_speed_mph': row[7],
                'wind_direction_deg': row[8],
                'humidity_pct': row[9],
                'is_dome': row[10]
            }
        })
    
    return games


def get_baseline_total_prediction(game, season_avg=13.0):
    """
    Simple baseline prediction: league average for total runs.
    In a real model, this would use team stats, but for this test
    we're isolating the weather effect.
    """
    return season_avg


def calculate_mae(predictions, actuals):
    """Calculate Mean Absolute Error."""
    errors = [abs(p - a) for p, a in zip(predictions, actuals)]
    return sum(errors) / len(errors) if errors else 0


def calculate_rmse(predictions, actuals):
    """Calculate Root Mean Square Error."""
    sq_errors = [(p - a) ** 2 for p, a in zip(predictions, actuals)]
    return math.sqrt(sum(sq_errors) / len(sq_errors)) if sq_errors else 0


def calculate_over_under_accuracy(predictions, actuals, lines):
    """Calculate accuracy of over/under predictions."""
    correct = 0
    total = 0
    for pred, actual, line in zip(predictions, actuals, lines):
        if actual == line:  # Push
            continue
        total += 1
        pred_over = pred > line
        actual_over = actual > line
        if pred_over == actual_over:
            correct += 1
    return correct / total if total > 0 else 0.5


def run_backtest():
    """Run the weather backtest study."""
    print("=" * 70)
    print("WEATHER FEATURES BACKTEST STUDY")
    print("=" * 70)
    print()
    
    # Step 1: Train weather model on 2024-2025
    print("Step 1: Training weather model on 2024-2025 data...")
    coefficients = train_weather_model(seasons=[2024, 2025], save=True)
    print(f"  Trained on {coefficients['n_games']} games")
    print(f"  R² = {coefficients['r_squared']:.4f}")
    print()
    
    # Step 2: Get 2026 test data
    print("Step 2: Loading 2026 test data...")
    games = get_2026_games_with_weather()
    print(f"  Found {len(games)} games with weather data")
    print()
    
    if len(games) < 50:
        print("ERROR: Not enough test games for reliable analysis")
        return
    
    # Step 3: Calculate predictions with and without weather
    print("Step 3: Generating predictions...")
    
    # Calculate 2026 average for baseline
    actual_totals = [g['total_runs'] for g in games]
    season_avg = np.mean(actual_totals)
    print(f"  2026 average total: {season_avg:.2f} runs")
    
    predictions_no_weather = []
    predictions_with_weather = []
    weather_adjustments = []
    
    for game in games:
        # Baseline prediction (no weather)
        baseline = season_avg
        predictions_no_weather.append(baseline)
        
        # Weather-adjusted prediction
        adj, _ = calculate_weather_adjustment(game['weather'], coefficients)
        predictions_with_weather.append(baseline + adj)
        weather_adjustments.append(adj)
    
    print(f"  Weather adjustments: min={min(weather_adjustments):.2f}, max={max(weather_adjustments):.2f}")
    print()
    
    # Step 4: Calculate accuracy metrics
    print("Step 4: Calculating accuracy metrics...")
    print()
    
    mae_no_weather = calculate_mae(predictions_no_weather, actual_totals)
    mae_with_weather = calculate_mae(predictions_with_weather, actual_totals)
    
    rmse_no_weather = calculate_rmse(predictions_no_weather, actual_totals)
    rmse_with_weather = calculate_rmse(predictions_with_weather, actual_totals)
    
    # Over/under accuracy using season average as line
    lines = [season_avg] * len(games)
    ou_acc_no_weather = calculate_over_under_accuracy(predictions_no_weather, actual_totals, lines)
    ou_acc_with_weather = calculate_over_under_accuracy(predictions_with_weather, actual_totals, lines)
    
    print("-" * 70)
    print("RESULTS: Model Comparison")
    print("-" * 70)
    print(f"{'Metric':<30} {'No Weather':<15} {'With Weather':<15} {'Improvement':<15}")
    print("-" * 70)
    print(f"{'MAE (runs)':<30} {mae_no_weather:<15.3f} {mae_with_weather:<15.3f} {mae_no_weather - mae_with_weather:+.3f}")
    print(f"{'RMSE (runs)':<30} {rmse_no_weather:<15.3f} {rmse_with_weather:<15.3f} {rmse_no_weather - rmse_with_weather:+.3f}")
    print(f"{'Over/Under Accuracy':<30} {ou_acc_no_weather:<15.1%} {ou_acc_with_weather:<15.1%} {(ou_acc_with_weather - ou_acc_no_weather)*100:+.1f}%")
    print("-" * 70)
    print()
    
    # Step 5: Analyze which weather features matter
    print("-" * 70)
    print("FEATURE IMPORTANCE (from learned coefficients)")
    print("-" * 70)
    
    t_stats = coefficients.get('t_statistics', {})
    features = [
        ('Temperature', 'temp', coefficients.get('temp_coef', 0)),
        ('Wind (out)', 'wind_out', coefficients.get('wind_out_coef', 0)),
        ('Humidity', 'humidity', coefficients.get('humidity_coef', 0)),
        ('Dome', 'dome', coefficients.get('dome_coef', 0)),
    ]
    
    print(f"{'Feature':<20} {'Coefficient':<15} {'t-stat':<10} {'Significant?':<12}")
    print("-" * 70)
    for name, key, coef in features:
        t = t_stats.get(key, 0)
        sig = "YES *" if abs(t) > 1.96 else "no"
        print(f"{name:<20} {coef:<15.4f} {t:<10.2f} {sig:<12}")
    print()
    
    # Step 6: Statistical significance of improvement
    print("-" * 70)
    print("STATISTICAL SIGNIFICANCE")
    print("-" * 70)
    
    # Paired t-test on absolute errors
    errors_no_weather = [abs(p - a) for p, a in zip(predictions_no_weather, actual_totals)]
    errors_with_weather = [abs(p - a) for p, a in zip(predictions_with_weather, actual_totals)]
    
    # Calculate paired differences
    diffs = [e1 - e2 for e1, e2 in zip(errors_no_weather, errors_with_weather)]
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    n = len(diffs)
    
    t_stat = mean_diff / (std_diff / math.sqrt(n)) if std_diff > 0 else 0
    
    # Two-tailed p-value approximation
    from scipy import stats
    try:
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
    except:
        # Fallback if scipy not available
        p_value = 1.0 if abs(t_stat) < 1.96 else 0.05
    
    print(f"Paired t-test on absolute errors:")
    print(f"  Mean error reduction: {mean_diff:.4f} runs")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Significant at α=0.05? {'YES' if p_value < 0.05 else 'NO'}")
    print()
    
    # Step 7: Example games with large weather effects
    print("-" * 70)
    print("EXAMPLE GAMES WITH LARGE WEATHER EFFECTS")
    print("-" * 70)
    
    # Sort games by absolute weather adjustment
    games_with_adj = list(zip(games, weather_adjustments))
    games_with_adj.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print(f"{'Date':<12} {'Matchup':<35} {'Actual':<8} {'Adj':<8} {'Conditions'}")
    print("-" * 70)
    for game, adj in games_with_adj[:10]:
        w = game['weather']
        wind_out = compute_wind_out_component(w['wind_speed_mph'], w['wind_direction_deg'])
        direction = 'out' if wind_out > 3 else 'in' if wind_out < -3 else ''
        conditions = f"{w['temp_f']:.0f}°F, {w['wind_speed_mph']:.0f}mph {direction}, {w['humidity_pct']:.0f}%"
        matchup = f"{game['away_team'][:15]} @ {game['home_team'][:15]}"
        print(f"{game['date']:<12} {matchup:<35} {game['total_runs']:<8} {adj:+.2f}    {conditions}")
    print()
    
    # Step 8: Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    improvement_pct = ((mae_no_weather - mae_with_weather) / mae_no_weather) * 100 if mae_no_weather > 0 else 0
    
    if p_value < 0.05 and mae_with_weather < mae_no_weather:
        print(f"✅ Weather features IMPROVE predictions significantly")
        print(f"   MAE reduced by {improvement_pct:.1f}% ({mae_no_weather - mae_with_weather:.3f} runs)")
    elif mae_with_weather < mae_no_weather:
        print(f"⚠️  Weather features improve predictions but NOT significantly (p={p_value:.3f})")
        print(f"   MAE reduced by {improvement_pct:.1f}% ({mae_no_weather - mae_with_weather:.3f} runs)")
    else:
        print(f"❌ Weather features do NOT improve predictions")
        print(f"   MAE changed by {improvement_pct:.1f}%")
    
    print()
    print("Key findings:")
    for name, key, coef in features:
        t = t_stats.get(key, 0)
        if abs(t) > 1.96:
            direction = "increases" if coef > 0 else "decreases"
            print(f"  • {name}: {direction} scoring (coef={coef:.4f}, t={t:.2f})")
    
    print("=" * 70)
    
    return {
        'mae_no_weather': mae_no_weather,
        'mae_with_weather': mae_with_weather,
        'improvement_pct': improvement_pct,
        'p_value': p_value,
        'n_games': len(games),
        'coefficients': coefficients
    }


if __name__ == '__main__':
    run_backtest()
