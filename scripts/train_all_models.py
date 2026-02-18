#!/usr/bin/env python3
"""
Unified Weekly Model Training Pipeline

Train/fine-tune ALL trainable models with a consistent data split:
  - Training: All data up to 7 days before today
  - Validation: Last 7 days of completed games

Models trained:
  1. Neural Networks (4): win, totals, spread, dow_totals (fine-tune from base weights)
  2. XGBoost (3): moneyline, totals, spread
  3. LightGBM (3): moneyline, totals, spread

Usage:
    python3 scripts/train_all_models.py                  # Train all models
    python3 scripts/train_all_models.py --nn-only        # Just neural networks
    python3 scripts/train_all_models.py --gb-only        # Just XGBoost + LightGBM
    python3 scripts/train_all_models.py --dry-run        # Show data splits, don't train
    python3 scripts/train_all_models.py --val-days 7     # Custom validation window (default 7)
"""

import sys
import argparse
import time
import pickle
import math
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from scripts.database import get_connection
from models.nn_features import FeatureComputer, HistoricalFeatureComputer

# ============================================================
# Paths
# ============================================================
DATA_DIR = Path(__file__).parent.parent / "data"

NN_CONFIGS = {
    'win': {
        'base_path': DATA_DIR / "nn_model.pt",
        'finetuned_path': DATA_DIR / "nn_model_finetuned.pt",
        'type': 'win',
    },
    'totals': {
        'base_path': DATA_DIR / "nn_totals_model.pt",
        'finetuned_path': DATA_DIR / "nn_totals_model_finetuned.pt",
        'type': 'totals',
    },
    'spread': {
        'base_path': DATA_DIR / "nn_spread_model.pt",
        'finetuned_path': DATA_DIR / "nn_spread_model_finetuned.pt",
        'type': 'spread',
    },
    'dow_totals': {
        'base_path': DATA_DIR / "nn_dow_totals_model.pt",
        'finetuned_path': DATA_DIR / "nn_dow_totals_model_finetuned.pt",
        'type': 'dow_totals',
    },
}

GB_PATHS = {
    'xgb_moneyline': DATA_DIR / "xgb_moneyline.pkl",
    'lgb_moneyline': DATA_DIR / "lgb_moneyline.pkl",
    'xgb_totals': DATA_DIR / "xgb_totals.pkl",
    'lgb_totals': DATA_DIR / "lgb_totals.pkl",
    'xgb_spread': DATA_DIR / "xgb_spread.pkl",
    'lgb_spread': DATA_DIR / "lgb_spread.pkl",
}

# Minimum data requirements
MIN_TRAIN_GAMES = 50
MIN_VAL_GAMES = 20


# ============================================================
# Data Loading — Shared split for all models
# ============================================================

def get_2026_games():
    """Get all completed 2026 games."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT id, date, home_team_id, away_team_id, home_score, away_score,
               is_neutral_site, is_conference_game
        FROM games
        WHERE status = 'final'
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
          AND date >= '2026-01-01'
        ORDER BY date ASC
    """)
    games = [dict(row) for row in c.fetchall()]
    conn.close()
    return games


def get_historical_games():
    """Get pre-2026 historical games for base training."""
    conn = get_connection()
    c = conn.cursor()
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
    rows = [dict(row) for row in c.fetchall()]
    conn.close()
    return rows


def split_2026_games(games, val_days=7, today=None):
    """
    Split 2026 games into train and validation.
    
    Train: games completed more than val_days ago
    Validation: games from the last val_days
    """
    if today is None:
        today = datetime.now()
    
    cutoff = (today - timedelta(days=val_days)).strftime('%Y-%m-%d')
    
    train = [g for g in games if g['date'] < cutoff]
    val = [g for g in games if g['date'] >= cutoff]
    
    return train, val, cutoff


def compute_2026_features(games, feature_computer, model_type):
    """Build feature matrix and targets for 2026 games using live FeatureComputer."""
    X_list, y_list, dow_list = [], [], []
    skipped = 0
    
    for game in games:
        try:
            features = feature_computer.compute_features(
                home_team_id=game['home_team_id'],
                away_team_id=game['away_team_id'],
                game_date=game['date'],
                neutral_site=bool(game.get('is_neutral_site', 0)),
                is_conference=bool(game.get('is_conference_game', 0)),
            )
            features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
            
            if model_type == 'win' or model_type == 'moneyline':
                target = 1.0 if game['home_score'] > game['away_score'] else 0.0
            elif model_type in ('totals', 'dow_totals'):
                target = float(game['home_score'] + game['away_score'])
            elif model_type == 'spread':
                target = float(game['home_score'] - game['away_score'])
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            X_list.append(features)
            y_list.append(target)
            
            try:
                dt = datetime.strptime(game['date'], '%Y-%m-%d')
                dow_list.append(int(dt.strftime('%w')))
            except:
                dow_list.append(5)
                
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"  Warning: skipped {game.get('id', '?')}: {e}")
    
    if skipped > 3:
        print(f"  ({skipped} total skipped)")
    
    if not X_list:
        return None, None, None
    
    return (np.array(X_list, dtype=np.float32), 
            np.array(y_list, dtype=np.float32),
            np.array(dow_list, dtype=np.int64))


def compute_historical_features(rows, model_type):
    """Build feature matrix and targets for historical games using HistoricalFeatureComputer."""
    hfc = HistoricalFeatureComputer()
    X_list, y_list, dow_list = [], [], []
    skipped = 0
    
    for row in rows:
        try:
            game_row = {
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                'home_score': row['home_score'],
                'away_score': row['away_score'],
                'date': row['date'],
                'season': row['season'],
                'neutral_site': row.get('neutral_site') or 0,
            }
            weather_row = None
            if row.get('temp_f') is not None:
                weather_row = {k: row[k] for k in ['temp_f', 'humidity_pct', 'wind_speed_mph',
                                                      'wind_direction_deg', 'precip_prob_pct', 'is_dome']}
            
            features, label = hfc.compute_game_features(game_row, weather_row)
            features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
            
            if model_type == 'win' or model_type == 'moneyline':
                target = label
            elif model_type in ('totals', 'dow_totals'):
                target = float(row['home_score'] + row['away_score'])
            elif model_type == 'spread':
                target = float(row['home_score'] - row['away_score'])
            
            X_list.append(features)
            y_list.append(target)
            
            try:
                dt = datetime.strptime(row['date'], '%Y-%m-%d')
                dow_list.append(int(dt.strftime('%w')))
            except:
                dow_list.append(5)
            
            hfc.update_state(game_row)
            
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"  Warning: skipped historical game: {e}")
    
    if skipped > 3:
        print(f"  ({skipped} total skipped)")
    
    if not X_list:
        return None, None, None
    
    return (np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.float32),
            np.array(dow_list, dtype=np.int64))


# ============================================================
# Neural Network Training
# ============================================================

def create_nn_model(checkpoint, model_type):
    """Create a PyTorch model from checkpoint."""
    input_size = checkpoint.get('input_size')
    
    if model_type == 'win':
        from models.neural_model import NeuralModel
        model = NeuralModel.__new__(NeuralModel)
        # Build the same architecture
        hidden = checkpoint.get('hidden_sizes', [256, 128, 64])
        layers = []
        prev = input_size
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.3)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Sigmoid())
        model.net = nn.Sequential(*layers)
    elif model_type in ('totals', 'dow_totals'):
        from models.nn_totals_model import NNTotalsModel
        hidden = checkpoint.get('hidden_sizes', [256, 128, 64])
        actual_input = input_size + (7 if model_type == 'dow_totals' else 0)
        layers = []
        prev = actual_input
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.3)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        model = nn.Sequential(*layers)
    elif model_type == 'spread':
        from models.nn_spread_model import NNSpreadModel
        hidden = checkpoint.get('hidden_sizes', [256, 128, 64])
        layers = []
        prev = input_size
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.3)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        model = nn.Sequential(*layers)
    
    return model


def finetune_nn_model(config, X_train, y_train, X_val, y_val, dow_train=None, dow_val=None):
    """Fine-tune a single NN model. Returns (improved, base_metric, new_metric)."""
    model_type = config['type']
    base_path = config['base_path']
    finetuned_path = config['finetuned_path']
    
    if not base_path.exists():
        print(f"  ⚠️  Base model not found: {base_path}")
        return False, None, None
    
    checkpoint = torch.load(base_path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
        print(f"  ⚠️  Invalid checkpoint: {base_path}")
        return False, None, None
    
    input_size = checkpoint.get('input_size')
    
    # Handle feature size mismatch
    actual_features = X_train.shape[1]
    if input_size != actual_features:
        print(f"  ⚠️  Feature size mismatch: model expects {input_size}, got {actual_features}")
        print(f"  Adapting model to new feature size...")
        checkpoint['input_size'] = actual_features
        input_size = actual_features
    
    model = create_nn_model(checkpoint, model_type)
    
    # Load weights (handle size mismatch on first layer)
    state_dict = checkpoint['model_state_dict']
    try:
        if hasattr(model, 'net'):
            model.net.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
    except RuntimeError:
        print(f"  ⚠️  Weight shape mismatch, training from scratch on 2026 data")
        # Re-create with correct input size
        model = create_nn_model(checkpoint, model_type)
    
    # Prepare tensors
    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train)
    X_v = torch.FloatTensor(X_val)
    y_v = torch.FloatTensor(y_val)
    
    if model_type == 'dow_totals' and dow_train is not None:
        dow_onehot_t = torch.zeros(len(dow_train), 7)
        for i, d in enumerate(dow_train):
            dow_onehot_t[i, d] = 1.0
        X_t = torch.cat([X_t, dow_onehot_t], dim=1)
        
        dow_onehot_v = torch.zeros(len(dow_val), 7)
        for i, d in enumerate(dow_val):
            dow_onehot_v[i, d] = 1.0
        X_v = torch.cat([X_v, dow_onehot_v], dim=1)
    
    # Evaluate base model
    net = model.net if hasattr(model, 'net') else model
    net.eval()
    with torch.no_grad():
        base_preds = net(X_v).squeeze()
        if model_type == 'win':
            base_correct = ((base_preds > 0.5) == (y_v > 0.5)).float().mean().item()
            base_metric = base_correct
            metric_name = "accuracy"
        else:
            base_metric = torch.sqrt(nn.MSELoss()(base_preds, y_v)).item()
            metric_name = "RMSE"
    
    print(f"  Base {metric_name}: {base_metric:.4f}")
    
    # Fine-tune
    net.train()
    if model_type == 'win':
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()
    
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_metric = base_metric
    best_state = None
    patience_counter = 0
    max_patience = 15
    
    for epoch in range(100):
        net.train()
        optimizer.zero_grad()
        preds = net(X_t).squeeze()
        loss = criterion(preds, y_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        
        net.eval()
        with torch.no_grad():
            val_preds = net(X_v).squeeze()
            if model_type == 'win':
                val_metric = ((val_preds > 0.5) == (y_v > 0.5)).float().mean().item()
                improved = val_metric > best_metric
            else:
                val_metric = torch.sqrt(nn.MSELoss()(val_preds, y_v)).item()
                improved = val_metric < best_metric
        
        scheduler.step(val_metric if model_type != 'win' else -val_metric)
        
        if improved:
            best_metric = val_metric
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                break
    
    # Save if improved
    if best_state is not None and best_metric != base_metric:
        new_checkpoint = dict(checkpoint)
        new_checkpoint['model_state_dict'] = best_state
        new_checkpoint['finetuned_at'] = datetime.now().isoformat()
        new_checkpoint['finetuned_val_metric'] = best_metric
        new_checkpoint['input_size'] = input_size
        torch.save(new_checkpoint, finetuned_path)
        
        if model_type == 'win':
            print(f"  ✅ Improved: {base_metric:.4f} → {best_metric:.4f} ({metric_name})")
        else:
            print(f"  ✅ Improved: {base_metric:.4f} → {best_metric:.4f} ({metric_name})")
        return True, base_metric, best_metric
    else:
        print(f"  ⏭️  No improvement over base ({metric_name}: {base_metric:.4f})")
        return False, base_metric, base_metric


# ============================================================
# Gradient Boosting Training
# ============================================================

def train_gb_models(X_train, y_train, X_val, y_val, task, model_prefix, use_gpu=True):
    """Train XGBoost + LightGBM for a single task. Returns results dict."""
    results = {}
    
    # Normalize features
    feature_mean = X_train.mean(axis=0)
    feature_std = X_train.std(axis=0)
    
    X_train_norm = (X_train - feature_mean) / (feature_std + 1e-8)
    X_val_norm = (X_val - feature_mean) / (feature_std + 1e-8)
    X_train_norm = np.clip(X_train_norm, -5.0, 5.0)
    X_val_norm = np.clip(X_val_norm, -5.0, 5.0)
    
    # XGBoost
    try:
        import xgboost as xgb
        print(f"\n  --- XGBoost {model_prefix} ---")
        start = time.time()
        
        if task == 'classification':
            model = xgb.XGBClassifier(
                n_estimators=800, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                tree_method='hist', device='cuda' if use_gpu else 'cpu',
                eval_metric='logloss', verbosity=0
            )
            model.fit(X_train_norm, y_train, eval_set=[(X_val_norm, y_val)], verbose=False)
            
            val_preds = model.predict(X_val_norm)
            accuracy = (val_preds == y_val).mean()
            val_probs = model.predict_proba(X_val_norm)[:, 1]
            from sklearn.metrics import log_loss
            ll = log_loss(y_val, val_probs)
            
            elapsed = time.time() - start
            print(f"  Accuracy: {accuracy*100:.1f}%, Log Loss: {ll:.4f} ({elapsed:.1f}s)")
            results[f'xgb_{model_prefix}'] = {'accuracy': accuracy, 'log_loss': ll}
        else:
            model = xgb.XGBRegressor(
                n_estimators=800, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                tree_method='hist', device='cuda' if use_gpu else 'cpu',
                eval_metric='rmse', verbosity=0
            )
            model.fit(X_train_norm, y_train, eval_set=[(X_val_norm, y_val)], verbose=False)
            
            val_preds = model.predict(X_val_norm)
            mae = np.mean(np.abs(val_preds - y_val))
            rmse = np.sqrt(np.mean((val_preds - y_val) ** 2))
            
            elapsed = time.time() - start
            print(f"  MAE: {mae:.3f}, RMSE: {rmse:.3f} ({elapsed:.1f}s)")
            results[f'xgb_{model_prefix}'] = {'mae': mae, 'rmse': rmse}
        
        # Save
        path = GB_PATHS[f'xgb_{model_prefix}']
        with open(path, 'wb') as f:
            pickle.dump({
                'model': model,
                'feature_mean': feature_mean,
                'feature_std': feature_std,
                'trained_at': datetime.now().isoformat(),
            }, f)
        print(f"  Saved: {path}")
        
    except ImportError:
        print("  ⚠️  XGBoost not installed")
    except Exception as e:
        print(f"  ❌ XGBoost error: {e}")
    
    # LightGBM
    try:
        import lightgbm as lgb
        print(f"\n  --- LightGBM {model_prefix} ---")
        start = time.time()
        
        if task == 'classification':
            model = lgb.LGBMClassifier(
                n_estimators=500, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                device='gpu' if use_gpu else 'cpu',
                verbosity=-1
            )
            model.fit(X_train_norm, y_train,
                     eval_set=[(X_val_norm, y_val)],
                     callbacks=[lgb.early_stopping(50, verbose=False)])
            
            val_preds = model.predict(X_val_norm)
            accuracy = (val_preds == y_val).mean()
            val_probs = model.predict_proba(X_val_norm)[:, 1]
            from sklearn.metrics import log_loss
            ll = log_loss(y_val, val_probs)
            
            elapsed = time.time() - start
            print(f"  Accuracy: {accuracy*100:.1f}%, Log Loss: {ll:.4f} ({elapsed:.1f}s)")
            results[f'lgb_{model_prefix}'] = {'accuracy': accuracy, 'log_loss': ll}
        else:
            model = lgb.LGBMRegressor(
                n_estimators=500, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                device='gpu' if use_gpu else 'cpu',
                verbosity=-1
            )
            model.fit(X_train_norm, y_train,
                     eval_set=[(X_val_norm, y_val)],
                     callbacks=[lgb.early_stopping(50, verbose=False)])
            
            val_preds = model.predict(X_val_norm)
            mae = np.mean(np.abs(val_preds - y_val))
            rmse = np.sqrt(np.mean((val_preds - y_val) ** 2))
            
            elapsed = time.time() - start
            print(f"  MAE: {mae:.3f}, RMSE: {rmse:.3f} ({elapsed:.1f}s)")
            results[f'lgb_{model_prefix}'] = {'mae': mae, 'rmse': rmse}
        
        # Save
        path = GB_PATHS[f'lgb_{model_prefix}']
        with open(path, 'wb') as f:
            pickle.dump({
                'model': model,
                'feature_mean': feature_mean,
                'feature_std': feature_std,
                'trained_at': datetime.now().isoformat(),
            }, f)
        print(f"  Saved: {path}")
        
    except ImportError:
        print("  ⚠️  LightGBM not installed")
    except Exception as e:
        print(f"  ❌ LightGBM error: {e}")
    
    return results


# ============================================================
# Main Pipeline
# ============================================================

def run_training(val_days=7, nn_only=False, gb_only=False, dry_run=False, use_gpu=True):
    today = datetime.now()
    print("=" * 60)
    print("UNIFIED MODEL TRAINING PIPELINE")
    print(f"Date: {today.strftime('%Y-%m-%d %H:%M')}")
    print(f"Validation window: last {val_days} days")
    print("=" * 60)
    
    # ---- Load 2026 games and split ----
    games_2026 = get_2026_games()
    train_2026, val_2026, cutoff = split_2026_games(games_2026, val_days=val_days, today=today)
    
    print(f"\n2026 Games: {len(games_2026)} total")
    print(f"  Train (before {cutoff}): {len(train_2026)}")
    print(f"  Validation (last {val_days} days): {len(val_2026)}")
    
    if len(val_2026) < MIN_VAL_GAMES:
        print(f"\n⚠️  Only {len(val_2026)} validation games (need {MIN_VAL_GAMES}). Waiting for more data.")
        return
    
    # ---- Load historical games ----
    historical = get_historical_games()
    print(f"  Historical (pre-2026): {len(historical)}")
    print(f"  Combined training set: {len(historical)} + {len(train_2026)} = {len(historical) + len(train_2026)}")
    
    if dry_run:
        print("\n🔍 DRY RUN — no models will be trained")
        if val_2026:
            dates = sorted(set(g['date'] for g in val_2026))
            print(f"  Validation dates: {dates[0]} to {dates[-1]}")
        return
    
    # ---- Feature computers ----
    fc_with_meta = FeatureComputer(use_model_predictions=True)
    fc_no_meta = FeatureComputer(use_model_predictions=False)
    
    results_summary = {}
    
    # ============================================================
    # NEURAL NETWORKS
    # ============================================================
    if not gb_only:
        print("\n" + "=" * 60)
        print("NEURAL NETWORK FINE-TUNING")
        print("=" * 60)
        
        for name, config in NN_CONFIGS.items():
            model_type = config['type']
            print(f"\n📊 {name.upper()} ({model_type})")
            
            # NN uses FeatureComputer with meta predictions
            X_train, y_train, dow_train = compute_2026_features(
                train_2026, fc_with_meta, model_type)
            X_val, y_val, dow_val = compute_2026_features(
                val_2026, fc_with_meta, model_type)
            
            if X_train is None or X_val is None:
                print(f"  ⚠️  Insufficient data, skipping")
                continue
            
            if len(X_train) < MIN_TRAIN_GAMES:
                print(f"  ⚠️  Only {len(X_train)} train games, need {MIN_TRAIN_GAMES}")
                continue
            
            print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Features: {X_train.shape[1]}")
            
            improved, base, new = finetune_nn_model(
                config, X_train, y_train, X_val, y_val,
                dow_train=dow_train, dow_val=dow_val)
            
            results_summary[f'nn_{name}'] = {
                'improved': improved, 'base': base, 'new': new
            }
    
    # ============================================================
    # GRADIENT BOOSTING (XGBoost + LightGBM)
    # ============================================================
    if not nn_only:
        print("\n" + "=" * 60)
        print("GRADIENT BOOSTING TRAINING (XGBoost + LightGBM)")
        print("=" * 60)
        
        # GB uses historical + 2026 train, no meta features
        print("\nBuilding historical features...")
        X_hist, y_hist_ml, dow_hist = compute_historical_features(historical, 'moneyline')
        
        if X_hist is not None:
            print(f"  Historical: {len(X_hist)} games, {X_hist.shape[1]} features")
        
        for task_name, task_type, target_type in [
            ('moneyline', 'classification', 'moneyline'),
            ('totals', 'regression', 'totals'),
            ('spread', 'regression', 'spread'),
        ]:
            print(f"\n📊 {task_name.upper()}")
            
            # 2026 features (no meta for GB)
            X_2026_train, y_2026_train, _ = compute_2026_features(
                train_2026, fc_no_meta, target_type)
            X_2026_val, y_2026_val, _ = compute_2026_features(
                val_2026, fc_no_meta, target_type)
            
            # Historical targets for this task
            if target_type == 'moneyline':
                y_hist = y_hist_ml
            elif target_type == 'totals':
                _, y_hist, _ = compute_historical_features(historical, 'totals')
            elif target_type == 'spread':
                _, y_hist, _ = compute_historical_features(historical, 'spread')
            
            # Combine historical + 2026 train
            if X_hist is not None and X_2026_train is not None:
                X_train_combined = np.vstack([X_hist, X_2026_train])
                y_train_combined = np.concatenate([y_hist, y_2026_train])
            elif X_2026_train is not None:
                X_train_combined = X_2026_train
                y_train_combined = y_2026_train
            elif X_hist is not None:
                X_train_combined = X_hist
                y_train_combined = y_hist
            else:
                print(f"  ⚠️  No training data, skipping")
                continue
            
            if X_2026_val is None or len(X_2026_val) < MIN_VAL_GAMES:
                print(f"  ⚠️  Insufficient validation data, skipping")
                continue
            
            print(f"  Train: {len(X_train_combined)} ({len(X_hist or [])} hist + {len(X_2026_train or [])} 2026)")
            print(f"  Val: {len(X_2026_val)} (2026 only)")
            print(f"  Features: {X_train_combined.shape[1]}")
            
            gb_results = train_gb_models(
                X_train_combined, y_train_combined,
                X_2026_val, y_2026_val,
                task=task_type, model_prefix=task_name, use_gpu=use_gpu)
            
            results_summary.update(gb_results)
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Date: {today.strftime('%Y-%m-%d')}")
    print(f"Split: train < {cutoff} | val >= {cutoff}")
    print(f"Train: {len(historical)} historical + {len(train_2026)} 2026 = {len(historical) + len(train_2026)}")
    print(f"Val: {len(val_2026)} games")
    
    for name, res in results_summary.items():
        if 'improved' in res:
            status = "✅ improved" if res['improved'] else "⏭️  no change"
            print(f"  {name}: {status} ({res.get('base', '?')} → {res.get('new', '?')})")
        elif 'accuracy' in res:
            print(f"  {name}: {res['accuracy']*100:.1f}% accuracy")
        elif 'rmse' in res:
            print(f"  {name}: MAE {res['mae']:.3f}, RMSE {res['rmse']:.3f}")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified model training pipeline")
    parser.add_argument('--val-days', type=int, default=7, help='Validation window in days (default: 7)')
    parser.add_argument('--nn-only', action='store_true', help='Only train neural networks')
    parser.add_argument('--gb-only', action='store_true', help='Only train gradient boosting')
    parser.add_argument('--dry-run', action='store_true', help='Show data splits without training')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU')
    args = parser.parse_args()
    
    run_training(
        val_days=args.val_days,
        nn_only=args.nn_only,
        gb_only=args.gb_only,
        dry_run=args.dry_run,
        use_gpu=not args.no_gpu,
    )
