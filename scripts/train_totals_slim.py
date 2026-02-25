#!/usr/bin/env python3
"""
Train slim neural network for total runs regression — v3.

Same 58 v3 features as the win model. Target = home_score + away_score.
v3: GELU + residual, season-weighted training, SWA, LR finder.

Usage:
    python3 scripts/train_totals_slim.py
    python3 scripts/train_totals_slim.py --val-days 7
"""

import sys
import math
import argparse
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

from scripts.database import get_connection
from models.nn_features_slim import (
    SlimFeatureComputer, SlimHistoricalFeatureComputer, NUM_FEATURES
)
from models.nn_totals_slim import TotalsNet

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_PATH = DATA_DIR / "nn_slim_totals.pt"
FINETUNED_PATH = DATA_DIR / "nn_slim_totals_finetuned.pt"

BASE_LR = 0.001
FINETUNE_LR = 0.0001
BATCH_SIZE = 64
BASE_EPOCHS = 200
FINETUNE_EPOCHS = 50
BASE_PATIENCE = 20
FINETUNE_PATIENCE = 15
MIN_VAL = 20


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def load_historical():
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


def load_2026_games():
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT id, date, home_team_id, away_team_id, home_score, away_score,
               is_neutral_site, is_conference_game
        FROM games
        WHERE status = 'final' AND home_score IS NOT NULL AND date >= '2026-01-01'
        ORDER BY date ASC
    """)
    games = [dict(row) for row in c.fetchall()]
    conn.close()
    return games


def build_historical_features(rows):
    """Returns (X, y_totals, seasons) where y = home_score + away_score."""
    hfc = SlimHistoricalFeatureComputer()
    X, y, seasons = [], [], []
    skipped = 0
    for row in rows:
        try:
            game_row = {
                'home_team': row['home_team'], 'away_team': row['away_team'],
                'home_score': row['home_score'], 'away_score': row['away_score'],
                'date': row['date'], 'season': row['season'],
                'neutral_site': row.get('neutral_site') or 0,
            }
            weather = None
            if row.get('temp_f') is not None:
                weather = {k: row[k] for k in ['temp_f', 'humidity_pct', 'wind_speed_mph',
                                                 'wind_direction_deg', 'precip_prob_pct', 'is_dome']}
            features, _ = hfc.compute_game_features(game_row, weather)
            features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
            total_runs = float(row['home_score'] + row['away_score'])
            X.append(features)
            y.append(total_runs)
            seasons.append(row['season'])
            hfc.update_state(game_row)
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"  Skip: {e}")
    if skipped > 3:
        print(f"  ({skipped} total skipped)")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(seasons)


def build_2026_features(games):
    fc = SlimFeatureComputer()
    X, y = [], []
    skipped = 0
    for g in games:
        try:
            features = fc.compute_features(
                g['home_team_id'], g['away_team_id'],
                game_date=g['date'],
                neutral_site=bool(g.get('is_neutral_site', 0)),
                is_conference=bool(g.get('is_conference_game', 0)),
            )
            features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
            total_runs = float(g['home_score'] + g['away_score'])
            X.append(features)
            y.append(total_runs)
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"  Skip {g.get('id', '?')}: {e}")
    if skipped > 3:
        print(f"  ({skipped} total skipped)")
    if not X:
        return None, None
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def compute_season_weights(seasons, decay=0.85):
    """Weight recent seasons more heavily."""
    if len(seasons) == 0:
        return np.ones(0)
    max_season = seasons.max()
    weights = np.array([decay ** (max_season - s) for s in seasons], dtype=np.float32)
    weights = weights / weights.mean()
    return weights


def train_model(model, X_train, y_train, X_val, y_val, lr, epochs, patience,
                batch_size, device, phase_name="Training", season_weights=None,
                use_swa=True):
    model = model.to(device)

    # Normalize features
    feature_mean = X_train.mean(axis=0)
    feature_std = X_train.std(axis=0)
    feature_std[feature_std < 1e-8] = 1.0
    X_train_n = np.clip((X_train - feature_mean) / feature_std, -5, 5)
    X_val_n = np.clip((X_val - feature_mean) / feature_std, -5, 5)

    # Normalize targets
    target_mean = y_train.mean()
    target_std = y_train.std()
    if target_std < 1e-8:
        target_std = 1.0
    y_train_n = (y_train - target_mean) / target_std
    y_val_n = (y_val - target_mean) / target_std

    X_train_t = torch.FloatTensor(X_train_n).to(device)
    y_train_t = torch.FloatTensor(y_train_n).to(device)

    if season_weights is not None:
        sampler = WeightedRandomSampler(
            weights=torch.FloatTensor(season_weights),
            num_samples=len(season_weights),
            replacement=True,
        )
        train_ds = TensorDataset(X_train_t, y_train_t)
        loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    else:
        train_ds = TensorDataset(X_train_t, y_train_t)
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_val_t = torch.FloatTensor(X_val_n).to(device)
    y_val_t = torch.FloatTensor(y_val_n).to(device)
    y_val_raw = torch.FloatTensor(y_val).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    warmup_epochs = min(5, epochs // 10)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # SWA
    swa_start = int(epochs * 0.8) if use_swa else epochs + 1
    swa_model = torch.optim.swa_utils.AveragedModel(model) if use_swa else None
    swa_scheduler = torch.optim.swa_utils.SWALR(
        optimizer, swa_lr=lr * 0.5, anneal_epochs=5
    ) if use_swa else None

    best_val_loss = float('inf')
    best_val_mae = float('inf')
    best_state = None
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"{phase_name}: {NUM_FEATURES} features, lr={lr}")
    print(f"  Train: {len(X_train)} | Val: {len(X_val)}")
    print(f"  Target — train mean: {target_mean:.1f}, std: {target_std:.1f}")
    print(f"  Val actual mean: {y_val.mean():.1f}")
    print(f"{'='*60}")

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        if epoch >= swa_start and use_swa:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        model.eval()
        with torch.no_grad():
            val_preds_n = model(X_val_t)
            val_loss = criterion(val_preds_n, y_val_t).item()
            val_preds_raw = val_preds_n * target_std + target_mean
            val_mae = torch.abs(val_preds_raw - y_val_raw).mean().item()

        avg_train = sum(train_losses) / len(train_losses)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            cur_lr = optimizer.param_groups[0]['lr']
            swa_tag = " [SWA]" if epoch >= swa_start and use_swa else ""
            print(f"  Epoch {epoch+1:3d} | Train MSE: {avg_train:.4f} | "
                  f"Val MSE: {val_loss:.4f} | MAE: {val_mae:.2f} runs | LR: {cur_lr:.6f}{swa_tag}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_mae = val_mae
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Check SWA model
    if use_swa and swa_model is not None and epoch >= swa_start:
        torch.optim.swa_utils.update_bn(loader, swa_model, device=device)
        swa_model.eval()
        with torch.no_grad():
            swa_preds_n = swa_model(X_val_t)
            swa_loss = criterion(swa_preds_n, y_val_t).item()
            swa_preds_raw = swa_preds_n * target_std + target_mean
            swa_mae = torch.abs(swa_preds_raw - y_val_raw).mean().item()
        print(f"  SWA model — Val MSE: {swa_loss:.4f} | MAE: {swa_mae:.2f}")
        if swa_loss < best_val_loss:
            best_val_loss = swa_loss
            best_val_mae = swa_mae
            best_state = {k: v.clone().cpu() for k, v in swa_model.module.state_dict().items()}
            print(f"  SWA model is better, using it.")

    print(f"  Best: MSE={best_val_loss:.4f}, MAE={best_val_mae:.2f} runs")
    return best_state, best_val_mae, best_val_loss, feature_mean, feature_std, target_mean, target_std


def save_checkpoint(path, state_dict, feature_mean, feature_std, target_mean, target_std, meta=None):
    checkpoint = {
        'model_state_dict': state_dict,
        'feature_mean': feature_mean,
        'feature_std': feature_std,
        'target_mean': target_mean,
        'target_std': target_std,
        'input_size': NUM_FEATURES,
        'saved_at': datetime.now().isoformat(),
        'version': 'v3',
    }
    if meta:
        checkpoint.update(meta)
    torch.save(checkpoint, path)
    print(f"  Saved: {path}")


def freeze_early_layers(model, freeze_fraction=0.5):
    """Freeze early layers for fine-tuning."""
    params = list(model.net.parameters())
    n_freeze = int(len(params) * freeze_fraction)
    for i, p in enumerate(params):
        if i < n_freeze:
            p.requires_grad = False
    frozen = sum(1 for p in params if not p.requires_grad)
    print(f"  Frozen {frozen}/{len(params)} parameters")


def run(val_days=7, dry_run=False, full_train=False):
    today = datetime.now()
    cutoff = (today - timedelta(days=val_days)).strftime('%Y-%m-%d')
    device = get_device()

    mode_label = "FULL TRAIN" if full_train else f"val last {val_days} days"
    print("=" * 60)
    print(f"SLIM TOTALS REGRESSION NN v3 ({NUM_FEATURES} features)")
    print(f"Date: {today.strftime('%Y-%m-%d %H:%M')}")
    print(f"Mode: {mode_label}")
    print("=" * 60)

    print("\nLoading historical games...")
    historical = load_historical()
    print(f"  Historical: {len(historical)} games")

    print("Loading 2026 games...")
    games_2026 = load_2026_games()

    if full_train:
        train_2026 = []
        val_2026 = games_2026
        print(f"  2026 (finetune+val): {len(val_2026)}")
    else:
        train_2026 = [g for g in games_2026 if g['date'] < cutoff]
        val_2026 = [g for g in games_2026 if g['date'] >= cutoff]
        print(f"  2026 train: {len(train_2026)} | val: {len(val_2026)}")

    if dry_run:
        print("\nDRY RUN")
        return

    print("\nComputing historical features...")
    X_hist, y_hist, seasons_hist = build_historical_features(historical)
    print(f"  Shape: {X_hist.shape}, target range: {y_hist.min():.0f}-{y_hist.max():.0f}, mean: {y_hist.mean():.1f}")

    season_weights = compute_season_weights(seasons_hist, decay=0.85)

    if not full_train and train_2026:
        print("Computing 2026 train features...")
        X_2026_train, y_2026_train = build_2026_features(train_2026)
    else:
        X_2026_train, y_2026_train = None, None

    print("Computing 2026 val features...")
    X_2026_val, y_2026_val = build_2026_features(val_2026)

    if X_2026_val is None or len(X_2026_val) < MIN_VAL:
        print(f"  Only {len(X_2026_val) if X_2026_val is not None else 0} val games")
        return

    if full_train:
        X_all = X_hist
        y_all = y_hist
        sw_all = season_weights
    else:
        if X_2026_train is not None and len(X_2026_train) > 0:
            X_all = np.vstack([X_hist, X_2026_train])
            y_all = np.concatenate([y_hist, y_2026_train])
            sw_2026 = np.ones(len(X_2026_train), dtype=np.float32) * season_weights.max()
            sw_all = np.concatenate([season_weights, sw_2026])
        else:
            X_all = X_hist
            y_all = y_hist
            sw_all = season_weights

    # Phase 1: Base
    print("\n" + "=" * 60)
    print("PHASE 1: BASE TRAINING")
    print("=" * 60)

    model = TotalsNet()
    best_state, base_mae, base_loss, feat_mean, feat_std, tgt_mean, tgt_std = train_model(
        model, X_all, y_all, X_2026_val, y_2026_val,
        lr=BASE_LR, epochs=BASE_EPOCHS, patience=BASE_PATIENCE,
        batch_size=BATCH_SIZE, device=device, phase_name="Phase 1: Base",
        season_weights=sw_all, use_swa=True,
    )

    if not best_state:
        print("Training failed")
        return

    save_checkpoint(MODEL_PATH, best_state, feat_mean, feat_std, tgt_mean, tgt_std,
                    meta={'phase': 'base', 'base_mae': base_mae,
                          'train_size': len(X_all), 'val_size': len(X_2026_val),
                          'full_train': full_train})

    # Phase 2: Fine-tune with layer freezing
    print("\n" + "=" * 60)
    if full_train:
        print("PHASE 2: FINE-TUNE ON ALL 2026 (validate on same)")
    else:
        print("PHASE 2: FINE-TUNE ON RECENT 2026")
    print("=" * 60)

    if full_train:
        model = TotalsNet()
        model.load_state_dict(best_state)
        freeze_early_layers(model, freeze_fraction=0.5)

        ft_state, ft_mae, ft_loss, ft_feat_mean, ft_feat_std, ft_tgt_mean, ft_tgt_std = train_model(
            model, X_2026_val, y_2026_val, X_2026_val, y_2026_val,
            lr=FINETUNE_LR, epochs=FINETUNE_EPOCHS, patience=FINETUNE_PATIENCE,
            batch_size=min(BATCH_SIZE, len(X_2026_val)),
            device=device, phase_name="Phase 2: Fine-tune (all 2026)",
            use_swa=False,
        )

        if ft_state and ft_mae < base_mae:
            save_checkpoint(FINETUNED_PATH, ft_state, ft_feat_mean, ft_feat_std,
                            ft_tgt_mean, ft_tgt_std,
                            meta={'phase': 'finetuned', 'base_mae': base_mae,
                                  'finetuned_mae': ft_mae, 'full_train': True})
            print(f"\n  Fine-tuned: MAE {base_mae:.2f} -> {ft_mae:.2f}")
        else:
            print(f"\n  Fine-tune didn't beat base ({ft_mae:.2f} vs {base_mae:.2f})")
    else:
        n_val = len(X_2026_val)
        split_idx = int(n_val * 0.7)
        if split_idx < 10 or (n_val - split_idx) < 10:
            print(f"  Val too small ({n_val}). Keeping base.")
        else:
            X_ft_train, y_ft_train = X_2026_val[:split_idx], y_2026_val[:split_idx]
            X_ft_val, y_ft_val = X_2026_val[split_idx:], y_2026_val[split_idx:]

            model = TotalsNet()
            model.load_state_dict(best_state)
            freeze_early_layers(model, freeze_fraction=0.5)

            ft_state, ft_mae, ft_loss, ft_feat_mean, ft_feat_std, ft_tgt_mean, ft_tgt_std = train_model(
                model, X_ft_train, y_ft_train, X_ft_val, y_ft_val,
                lr=FINETUNE_LR, epochs=FINETUNE_EPOCHS, patience=FINETUNE_PATIENCE,
                batch_size=min(BATCH_SIZE, len(X_ft_train)),
                device=device, phase_name="Phase 2: Fine-tune",
                use_swa=False,
            )

            if ft_state and ft_mae < base_mae:
                save_checkpoint(FINETUNED_PATH, ft_state, ft_feat_mean, ft_feat_std,
                                ft_tgt_mean, ft_tgt_std,
                                meta={'phase': 'finetuned', 'base_mae': base_mae,
                                      'finetuned_mae': ft_mae})
                print(f"\n  Fine-tuned: MAE {base_mae:.2f} -> {ft_mae:.2f}")
            else:
                print(f"\n  Fine-tune didn't beat base ({ft_mae:.2f} vs {base_mae:.2f})")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Features: {NUM_FEATURES} (v3 w/ NCAA stats)")
    print(f"  Mode: {'full-train' if full_train else 'standard'}")
    print(f"  Base: {MODEL_PATH} (MAE: {base_mae:.2f} runs)")
    if FINETUNED_PATH.exists():
        cp = torch.load(FINETUNED_PATH, map_location='cpu', weights_only=False)
        print(f"  Fine-tuned: {FINETUNED_PATH} (MAE: {cp.get('finetuned_mae', '?'):.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--val-days', type=int, default=7)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--full-train', action='store_true',
                        help='Train base on all historical, finetune+validate on all 2026')
    args = parser.parse_args()
    run(val_days=args.val_days, dry_run=args.dry_run, full_train=args.full_train)
