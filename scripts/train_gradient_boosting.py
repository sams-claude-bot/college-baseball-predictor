#!/usr/bin/env python3
"""
Train XGBoost and LightGBM Gradient Boosting Models

Trains models for:
- Moneyline prediction (2026 season data - classification)
- Totals prediction (historical data - regression)
- Spread prediction (historical data - regression)

Uses GPU acceleration when available.
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.database import get_connection
from models.nn_features import HistoricalFeatureComputer, FeatureComputer

# Import trainers
try:
    from models.xgboost_model import XGBTrainer, MONEYLINE_PATH as XGB_ML_PATH
    from models.xgboost_model import TOTALS_PATH as XGB_TOT_PATH, SPREAD_PATH as XGB_SPR_PATH
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("Warning: XGBoost not available")

try:
    from models.lightgbm_model import LGBTrainer, MONEYLINE_PATH as LGB_ML_PATH
    from models.lightgbm_model import TOTALS_PATH as LGB_TOT_PATH, SPREAD_PATH as LGB_SPR_PATH
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("Warning: LightGBM not available")


def check_gpu():
    """Check if GPU is available for training."""
    gpu_available = False
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"GPU detected: {gpu_name}")
            gpu_available = True
    except ImportError:
        pass
    
    if not gpu_available:
        print("No GPU detected, using CPU training")
    
    return gpu_available


def load_historical_data(min_games=20):
    """
    Load historical games for totals/spread training.
    Returns features, totals targets, and spread targets.
    """
    print("\n" + "="*60)
    print("Loading historical game data...")
    print("="*60)
    
    conn = get_connection()
    c = conn.cursor()
    
    # Get all historical games with weather
    c.execute("""
        SELECT h.id, h.home_team, h.away_team, h.home_score, h.away_score,
               h.date, h.season, h.neutral_site,
               w.temp_f, w.humidity_pct, w.wind_speed_mph, w.wind_direction_deg,
               w.precip_prob_pct, w.is_dome
        FROM historical_games h
        LEFT JOIN historical_game_weather w ON h.id = w.game_id
        WHERE h.home_score IS NOT NULL AND h.away_score IS NOT NULL
        ORDER BY h.date ASC
    """)
    
    rows = c.fetchall()
    conn.close()
    
    print(f"Found {len(rows)} historical games")
    
    # Compute features using HistoricalFeatureComputer (no data leakage)
    fc = HistoricalFeatureComputer()
    
    features = []
    totals_targets = []
    spread_targets = []
    
    for row in rows:
        game_row = {
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            'home_score': row['home_score'],
            'away_score': row['away_score'],
            'date': row['date'],
            'season': row['season'],
            'neutral_site': row['neutral_site'] or 0,
        }
        
        weather_row = None
        if row['temp_f'] is not None:
            weather_row = {
                'temp_f': row['temp_f'],
                'humidity_pct': row['humidity_pct'],
                'wind_speed_mph': row['wind_speed_mph'],
                'wind_direction_deg': row['wind_direction_deg'],
                'precip_prob_pct': row['precip_prob_pct'],
                'is_dome': row['is_dome'],
            }
        
        # Compute features BEFORE seeing this game
        feat, _ = fc.compute_game_features(game_row, weather_row)
        
        # Update state AFTER computing features
        fc.update_state(game_row)
        
        # Only include games where both teams have enough history
        home_games = fc.team_stats[row['home_team']]['games']
        away_games = fc.team_stats[row['away_team']]['games']
        
        if home_games >= min_games and away_games >= min_games:
            features.append(feat)
            total = row['home_score'] + row['away_score']
            margin = row['home_score'] - row['away_score']
            totals_targets.append(total)
            spread_targets.append(margin)
    
    X = np.array(features, dtype=np.float32)
    y_totals = np.array(totals_targets, dtype=np.float32)
    y_spread = np.array(spread_targets, dtype=np.float32)
    
    # Handle NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"Training samples (after min_games filter): {len(X)}")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"Totals range: {y_totals.min():.0f} - {y_totals.max():.0f} (mean: {y_totals.mean():.1f})")
    print(f"Spread range: {y_spread.min():.0f} - {y_spread.max():.0f} (mean: {y_spread.mean():.1f})")
    
    return X, y_totals, y_spread


def load_moneyline_data():
    """
    Load 2026 season games for moneyline training.
    Uses games table with team IDs and FeatureComputer.
    """
    print("\n" + "="*60)
    print("Loading 2026 season data for moneyline...")
    print("="*60)
    
    conn = get_connection()
    c = conn.cursor()
    
    # Get completed 2026 games
    c.execute("""
        SELECT g.id, g.home_team_id, g.away_team_id, g.home_score, g.away_score,
               g.date, g.is_neutral_site as neutral_site
        FROM games g
        WHERE g.status = 'final'
        AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL
        ORDER BY g.date ASC
    """)
    
    rows = c.fetchall()
    conn.close()
    
    print(f"Found {len(rows)} completed 2026 games")
    
    # Use FeatureComputer (with model predictions disabled for training)
    fc = FeatureComputer(use_model_predictions=False)
    
    features = []
    labels = []
    
    for i, row in enumerate(rows):
        try:
            feat = fc.compute_features(
                row['home_team_id'],
                row['away_team_id'],
                game_date=row['date'],
                neutral_site=bool(row['neutral_site']),
                game_id=row['id']
            )
            
            home_won = row['home_score'] > row['away_score']
            
            features.append(feat)
            labels.append(1 if home_won else 0)
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(rows)} games...")
                
        except Exception as e:
            print(f"  Warning: Skipping game {row['id']}: {e}")
            continue
    
    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    
    # Handle NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"Training samples: {len(X)}")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"Home win rate: {y.mean()*100:.1f}%")
    
    return X, y


def split_data(X, y, test_ratio=0.15, val_ratio=0.15):
    """Split data into train/val/test sets (chronological for time series)."""
    n = len(X)
    test_size = int(n * test_ratio)
    val_size = int(n * val_ratio)
    
    X_train = X[:-(test_size + val_size)]
    y_train = y[:-(test_size + val_size)]
    X_val = X[-(test_size + val_size):-test_size]
    y_val = y[-(test_size + val_size):-test_size]
    X_test = X[-test_size:]
    y_test = y[-test_size:]
    
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_totals_models(X, y, use_gpu=True):
    """Train XGBoost and LightGBM totals models."""
    print("\n" + "="*60)
    print("Training TOTALS models (regression)")
    print("="*60)
    
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)
    
    results = {}
    
    # XGBoost
    if XGB_AVAILABLE:
        print("\n--- XGBoost Totals ---")
        start = time.time()
        trainer = XGBTrainer(task='regression', use_gpu=use_gpu,
                            n_estimators=800, max_depth=8, learning_rate=0.05)
        trainer.train(X_train, y_train, X_val, y_val)
        trainer.save(XGB_TOT_PATH)
        
        eval_result = trainer.evaluate(X_test, y_test)
        elapsed = time.time() - start
        print(f"XGBoost Totals - MAE: {eval_result['mae']:.3f}, RMSE: {eval_result['rmse']:.3f}")
        print(f"Training time: {elapsed:.1f}s")
        
        results['xgb_totals'] = eval_result
        results['xgb_totals']['time'] = elapsed
        results['xgb_totals']['features'] = trainer.get_feature_importance()
    
    # LightGBM
    if LGB_AVAILABLE:
        print("\n--- LightGBM Totals ---")
        start = time.time()
        trainer = LGBTrainer(task='regression', use_gpu=use_gpu,
                            n_estimators=800, max_depth=8, learning_rate=0.05)
        trainer.train(X_train, y_train, X_val, y_val)
        trainer.save(LGB_TOT_PATH)
        
        eval_result = trainer.evaluate(X_test, y_test)
        elapsed = time.time() - start
        print(f"LightGBM Totals - MAE: {eval_result['mae']:.3f}, RMSE: {eval_result['rmse']:.3f}")
        print(f"Training time: {elapsed:.1f}s")
        
        results['lgb_totals'] = eval_result
        results['lgb_totals']['time'] = elapsed
        results['lgb_totals']['features'] = trainer.get_feature_importance()
    
    return results


def train_spread_models(X, y, use_gpu=True):
    """Train XGBoost and LightGBM spread models."""
    print("\n" + "="*60)
    print("Training SPREAD models (regression)")
    print("="*60)
    
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)
    
    results = {}
    
    # XGBoost
    if XGB_AVAILABLE:
        print("\n--- XGBoost Spread ---")
        start = time.time()
        trainer = XGBTrainer(task='regression', use_gpu=use_gpu,
                            n_estimators=800, max_depth=8, learning_rate=0.05)
        trainer.train(X_train, y_train, X_val, y_val)
        trainer.save(XGB_SPR_PATH)
        
        eval_result = trainer.evaluate(X_test, y_test)
        elapsed = time.time() - start
        print(f"XGBoost Spread - MAE: {eval_result['mae']:.3f}, RMSE: {eval_result['rmse']:.3f}")
        print(f"Training time: {elapsed:.1f}s")
        
        results['xgb_spread'] = eval_result
        results['xgb_spread']['time'] = elapsed
        results['xgb_spread']['features'] = trainer.get_feature_importance()
    
    # LightGBM
    if LGB_AVAILABLE:
        print("\n--- LightGBM Spread ---")
        start = time.time()
        trainer = LGBTrainer(task='regression', use_gpu=use_gpu,
                            n_estimators=800, max_depth=8, learning_rate=0.05)
        trainer.train(X_train, y_train, X_val, y_val)
        trainer.save(LGB_SPR_PATH)
        
        eval_result = trainer.evaluate(X_test, y_test)
        elapsed = time.time() - start
        print(f"LightGBM Spread - MAE: {eval_result['mae']:.3f}, RMSE: {eval_result['rmse']:.3f}")
        print(f"Training time: {elapsed:.1f}s")
        
        results['lgb_spread'] = eval_result
        results['lgb_spread']['time'] = elapsed
        results['lgb_spread']['features'] = trainer.get_feature_importance()
    
    return results


def train_moneyline_models(X, y, use_gpu=True):
    """Train XGBoost and LightGBM moneyline models."""
    print("\n" + "="*60)
    print("Training MONEYLINE models (classification)")
    print("="*60)
    
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)
    
    results = {}
    
    # XGBoost
    if XGB_AVAILABLE:
        print("\n--- XGBoost Moneyline ---")
        start = time.time()
        trainer = XGBTrainer(task='classification', use_gpu=use_gpu,
                            n_estimators=500, max_depth=6, learning_rate=0.05)
        trainer.train(X_train, y_train, X_val, y_val)
        trainer.save(XGB_ML_PATH)
        
        eval_result = trainer.evaluate(X_test, y_test)
        elapsed = time.time() - start
        print(f"XGBoost Moneyline - Accuracy: {eval_result['accuracy']*100:.1f}%, Log Loss: {eval_result['log_loss']:.4f}")
        print(f"Training time: {elapsed:.1f}s")
        
        results['xgb_moneyline'] = eval_result
        results['xgb_moneyline']['time'] = elapsed
        results['xgb_moneyline']['features'] = trainer.get_feature_importance()
    
    # LightGBM
    if LGB_AVAILABLE:
        print("\n--- LightGBM Moneyline ---")
        start = time.time()
        trainer = LGBTrainer(task='classification', use_gpu=use_gpu,
                            n_estimators=500, max_depth=6, learning_rate=0.05)
        trainer.train(X_train, y_train, X_val, y_val)
        trainer.save(LGB_ML_PATH)
        
        eval_result = trainer.evaluate(X_test, y_test)
        elapsed = time.time() - start
        print(f"LightGBM Moneyline - Accuracy: {eval_result['accuracy']*100:.1f}%, Log Loss: {eval_result['log_loss']:.4f}")
        print(f"Training time: {elapsed:.1f}s")
        
        results['lgb_moneyline'] = eval_result
        results['lgb_moneyline']['time'] = elapsed
        results['lgb_moneyline']['features'] = trainer.get_feature_importance()
    
    return results


def print_comparison(all_results):
    """Print comparison of XGBoost vs LightGBM."""
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    
    # Moneyline comparison
    print("\n📊 MONEYLINE (Classification)")
    print("-" * 40)
    if 'xgb_moneyline' in all_results and 'lgb_moneyline' in all_results:
        xgb = all_results['xgb_moneyline']
        lgb = all_results['lgb_moneyline']
        print(f"{'Metric':<15} {'XGBoost':>12} {'LightGBM':>12} {'Winner':>12}")
        print("-" * 40)
        
        acc_winner = "XGBoost" if xgb['accuracy'] > lgb['accuracy'] else "LightGBM"
        print(f"{'Accuracy':<15} {xgb['accuracy']*100:>11.1f}% {lgb['accuracy']*100:>11.1f}% {acc_winner:>12}")
        
        ll_winner = "XGBoost" if xgb['log_loss'] < lgb['log_loss'] else "LightGBM"
        print(f"{'Log Loss':<15} {xgb['log_loss']:>12.4f} {lgb['log_loss']:>12.4f} {ll_winner:>12}")
        
        time_winner = "XGBoost" if xgb['time'] < lgb['time'] else "LightGBM"
        print(f"{'Train Time':<15} {xgb['time']:>11.1f}s {lgb['time']:>11.1f}s {time_winner:>12}")
    
    # Totals comparison
    print("\n📊 TOTALS (Regression)")
    print("-" * 40)
    if 'xgb_totals' in all_results and 'lgb_totals' in all_results:
        xgb = all_results['xgb_totals']
        lgb = all_results['lgb_totals']
        print(f"{'Metric':<15} {'XGBoost':>12} {'LightGBM':>12} {'Winner':>12}")
        print("-" * 40)
        
        mae_winner = "XGBoost" if xgb['mae'] < lgb['mae'] else "LightGBM"
        print(f"{'MAE':<15} {xgb['mae']:>12.3f} {lgb['mae']:>12.3f} {mae_winner:>12}")
        
        rmse_winner = "XGBoost" if xgb['rmse'] < lgb['rmse'] else "LightGBM"
        print(f"{'RMSE':<15} {xgb['rmse']:>12.3f} {lgb['rmse']:>12.3f} {rmse_winner:>12}")
        
        time_winner = "XGBoost" if xgb['time'] < lgb['time'] else "LightGBM"
        print(f"{'Train Time':<15} {xgb['time']:>11.1f}s {lgb['time']:>11.1f}s {time_winner:>12}")
    
    # Spread comparison
    print("\n📊 SPREAD (Regression)")
    print("-" * 40)
    if 'xgb_spread' in all_results and 'lgb_spread' in all_results:
        xgb = all_results['xgb_spread']
        lgb = all_results['lgb_spread']
        print(f"{'Metric':<15} {'XGBoost':>12} {'LightGBM':>12} {'Winner':>12}")
        print("-" * 40)
        
        mae_winner = "XGBoost" if xgb['mae'] < lgb['mae'] else "LightGBM"
        print(f"{'MAE':<15} {xgb['mae']:>12.3f} {lgb['mae']:>12.3f} {mae_winner:>12}")
        
        rmse_winner = "XGBoost" if xgb['rmse'] < lgb['rmse'] else "LightGBM"
        print(f"{'RMSE':<15} {xgb['rmse']:>12.3f} {lgb['rmse']:>12.3f} {rmse_winner:>12}")
    
    # Top features
    print("\n📊 TOP FEATURES (by importance)")
    print("-" * 40)
    
    # Get feature names
    fc = FeatureComputer(use_model_predictions=False)
    feature_names = fc.get_feature_names()
    
    for model_name in ['xgb_totals', 'lgb_totals']:
        if model_name in all_results and 'features' in all_results[model_name]:
            print(f"\n{model_name.upper()} Top 10 Features:")
            features = all_results[model_name]['features']
            for i, (fname, score) in enumerate(list(features.items())[:10], 1):
                # Map feature index to name if numeric
                if fname.startswith('f') and fname[1:].isdigit():
                    idx = int(fname[1:])
                    if idx < len(feature_names):
                        fname = feature_names[idx]
                print(f"  {i:2}. {fname}: {score:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train XGBoost and LightGBM models')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
    parser.add_argument('--moneyline-only', action='store_true', help='Only train moneyline')
    parser.add_argument('--totals-only', action='store_true', help='Only train totals')
    parser.add_argument('--spread-only', action='store_true', help='Only train spread')
    args = parser.parse_args()
    
    print("="*60)
    print("GRADIENT BOOSTING MODEL TRAINING")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    use_gpu = not args.no_gpu and check_gpu()
    
    all_results = {}
    
    # Determine which models to train
    train_all = not (args.moneyline_only or args.totals_only or args.spread_only)
    
    # Load data and train
    if train_all or args.totals_only or args.spread_only:
        X_hist, y_totals, y_spread = load_historical_data()
        
        if train_all or args.totals_only:
            results = train_totals_models(X_hist, y_totals, use_gpu)
            all_results.update(results)
        
        if train_all or args.spread_only:
            results = train_spread_models(X_hist, y_spread, use_gpu)
            all_results.update(results)
    
    if train_all or args.moneyline_only:
        X_ml, y_ml = load_moneyline_data()
        if len(X_ml) >= 50:  # Need minimum data
            results = train_moneyline_models(X_ml, y_ml, use_gpu)
            all_results.update(results)
        else:
            print(f"Warning: Only {len(X_ml)} moneyline samples, skipping training")
    
    # Print comparison
    if len(all_results) > 0:
        print_comparison(all_results)
    
    print("\n" + "="*60)
    print(f"Training complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)


if __name__ == "__main__":
    main()
