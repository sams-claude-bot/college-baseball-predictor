#!/usr/bin/env python3
"""
Neural Network Training Pipeline v2

Two-phase training matching the unified model policy:
  Phase 1 (Base): Train on historical (2024-2025) + 2026 games up to val_days ago
  Phase 2 (Fine-tune): Fine-tune on last val_days with lower learning rate

Usage:
    python3 scripts/train_neural_v2.py                    # Full pipeline
    python3 scripts/train_neural_v2.py --base-only        # Phase 1 only
    python3 scripts/train_neural_v2.py --finetune-only    # Phase 2 only (requires base)
    python3 scripts/train_neural_v2.py --val-days 7       # Custom val window
    python3 scripts/train_neural_v2.py --dry-run          # Show splits only
"""

import sys
import argparse
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from scripts.database import get_connection
from models.nn_features import FeatureComputer, HistoricalFeatureComputer
from models.neural_model import BaseballNet

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_PATH = DATA_DIR / "nn_model.pt"
FINETUNED_PATH = DATA_DIR / "nn_model_finetuned.pt"

# Training defaults
BASE_LR = 0.001
FINETUNE_LR = 0.0001  # 10x lower for fine-tuning
BATCH_SIZE = 64
BASE_EPOCHS = 100
FINETUNE_EPOCHS = 50
BASE_PATIENCE = 10
FINETUNE_PATIENCE = 15
MIN_TRAIN = 50
MIN_VAL = 20


def get_device():
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return 'cuda'
    print("Using CPU")
    return 'cpu'


# ============================================================
# Data Loading
# ============================================================

def load_historical():
    """Load pre-2026 historical games."""
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
    """Load completed 2026 games."""
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
    """Compute features for historical games using HistoricalFeatureComputer."""
    hfc = HistoricalFeatureComputer()
    X, y = [], []
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
            weather = None
            if row.get('temp_f') is not None:
                weather = {k: row[k] for k in ['temp_f', 'humidity_pct', 'wind_speed_mph',
                                                 'wind_direction_deg', 'precip_prob_pct', 'is_dome']}

            features, label = hfc.compute_game_features(game_row, weather)
            features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
            X.append(features)
            y.append(label)
            hfc.update_state(game_row)
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"  Skip: {e}")

    if skipped > 3:
        print(f"  ({skipped} total skipped)")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_2026_features(games, fc):
    """Compute features for 2026 games using live FeatureComputer."""
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
            label = 1.0 if g['home_score'] > g['away_score'] else 0.0
            X.append(features)
            y.append(label)
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"  Skip {g.get('id', '?')}: {e}")

    if skipped > 3:
        print(f"  ({skipped} total skipped)")

    if not X:
        return None, None
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ============================================================
# Training
# ============================================================

def train_model(model, X_train, y_train, X_val, y_val, lr, epochs, patience,
                batch_size, device, phase_name="Training"):
    """Train/fine-tune the model. Returns (best_state, best_val_acc, history)."""
    model = model.to(device)

    # Normalize
    feature_mean = X_train.mean(axis=0)
    feature_std = X_train.std(axis=0)
    feature_std[feature_std < 1e-8] = 1.0

    X_train_n = (X_train - feature_mean) / feature_std
    X_val_n = (X_val - feature_mean) / feature_std

    # Clip outliers
    X_train_n = np.clip(X_train_n, -5.0, 5.0)
    X_val_n = np.clip(X_val_n, -5.0, 5.0)

    # Tensors
    train_ds = TensorDataset(
        torch.FloatTensor(X_train_n).to(device),
        torch.FloatTensor(y_train).to(device)
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    X_val_t = torch.FloatTensor(X_val_n).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    history = []

    print(f"\n{'='*60}")
    print(f"{phase_name}: lr={lr}, epochs={epochs}, patience={patience}")
    print(f"{'='*60}")

    for epoch in range(epochs):
        # Train
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

        # Validate
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = criterion(val_preds, y_val_t).item()
            val_acc = ((val_preds > 0.5).float() == y_val_t).float().mean().item()

        scheduler.step(val_loss)
        avg_train = sum(train_losses) / len(train_losses)
        cur_lr = optimizer.param_groups[0]['lr']

        history.append({'train_loss': avg_train, 'val_loss': val_loss, 'val_acc': val_acc})

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d} | Train: {avg_train:.4f} | "
                  f"Val: {val_loss:.4f} | Acc: {val_acc:.4f} | LR: {cur_lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print(f"  Best: loss={best_val_loss:.4f}, acc={best_val_acc:.4f}")
    return best_state, best_val_acc, feature_mean, feature_std, history


def save_checkpoint(path, state_dict, feature_mean, feature_std, input_size, meta=None):
    """Save model checkpoint."""
    checkpoint = {
        'model_state_dict': state_dict,
        'feature_mean': feature_mean,
        'feature_std': feature_std,
        'input_size': input_size,
        'saved_at': datetime.now().isoformat(),
    }
    if meta:
        checkpoint.update(meta)
    torch.save(checkpoint, path)
    print(f"  Saved: {path}")


# ============================================================
# Main Pipeline
# ============================================================

def run(val_days=7, base_only=False, finetune_only=False, dry_run=False):
    today = datetime.now()
    cutoff = (today - timedelta(days=val_days)).strftime('%Y-%m-%d')
    device = get_device()

    print("=" * 60)
    print("NEURAL NETWORK TRAINING PIPELINE v2")
    print(f"Date: {today.strftime('%Y-%m-%d %H:%M')}")
    print(f"Validation: last {val_days} days (>= {cutoff})")
    print("=" * 60)

    # ---- Load data ----
    print("\nLoading historical games...")
    historical = load_historical()
    print(f"  Historical: {len(historical)} games")

    print("Loading 2026 games...")
    games_2026 = load_2026_games()
    train_2026 = [g for g in games_2026 if g['date'] < cutoff]
    val_2026 = [g for g in games_2026 if g['date'] >= cutoff]
    print(f"  2026 total: {len(games_2026)}")
    print(f"  2026 train (< {cutoff}): {len(train_2026)}")
    print(f"  2026 val (>= {cutoff}): {len(val_2026)}")

    if dry_run:
        print("\n🔍 DRY RUN — no training")
        return

    # ---- Build features ----
    # Use FeatureComputer WITHOUT meta predictions (same as GB models — 81 features)
    # This keeps the NN on the same feature set as XGB/LGB
    fc = FeatureComputer(use_model_predictions=False)

    print("\nComputing historical features...")
    X_hist, y_hist = build_historical_features(historical)
    print(f"  Shape: {X_hist.shape}")

    print("Computing 2026 train features...")
    X_2026_train, y_2026_train = build_2026_features(train_2026, fc)

    print("Computing 2026 val features...")
    X_2026_val, y_2026_val = build_2026_features(val_2026, fc)

    if X_2026_val is None or len(X_2026_val) < MIN_VAL:
        print(f"\n⚠️  Only {len(X_2026_val) if X_2026_val is not None else 0} "
              f"val games (need {MIN_VAL}). Try --val-days 3")
        return

    # Verify feature dimensions match
    hist_dim = X_hist.shape[1]
    val_dim = X_2026_val.shape[1]
    if hist_dim != val_dim:
        print(f"\n❌ Feature dimension mismatch: historical={hist_dim}, 2026={val_dim}")
        return
    
    input_size = hist_dim
    print(f"\nFeature dimension: {input_size}")

    # ============================================================
    # PHASE 1: Base training on historical + 2026 train
    # ============================================================
    if not finetune_only:
        print("\n" + "=" * 60)
        print("PHASE 1: BASE TRAINING")
        print("=" * 60)

        # Combine historical + 2026 train
        if X_2026_train is not None and len(X_2026_train) > 0:
            X_base_train = np.vstack([X_hist, X_2026_train])
            y_base_train = np.concatenate([y_hist, y_2026_train])
            print(f"  Training: {len(X_hist)} hist + {len(X_2026_train)} 2026 = {len(X_base_train)}")
        else:
            X_base_train = X_hist
            y_base_train = y_hist
            print(f"  Training: {len(X_hist)} historical only (no 2026 train data)")

        # Use 2026 val as validation for phase 1 too
        print(f"  Validation: {len(X_2026_val)} 2026 games")
        print(f"  Home win rate — train: {y_base_train.mean():.1%}, val: {y_2026_val.mean():.1%}")

        model = BaseballNet(input_size)

        best_state, best_acc, feat_mean, feat_std, _ = train_model(
            model, X_base_train, y_base_train, X_2026_val, y_2026_val,
            lr=BASE_LR, epochs=BASE_EPOCHS, patience=BASE_PATIENCE,
            batch_size=BATCH_SIZE, device=device,
            phase_name="Phase 1: Base Training"
        )

        if best_state:
            save_checkpoint(MODEL_PATH, best_state, feat_mean, feat_std, input_size,
                           meta={'phase': 'base', 'base_acc': best_acc,
                                 'train_size': len(X_base_train),
                                 'val_size': len(X_2026_val)})
            base_acc = best_acc
        else:
            print("  ❌ Base training failed")
            return
    else:
        # Load existing base model
        if not MODEL_PATH.exists():
            print("❌ No base model found. Run without --finetune-only first.")
            return
        cp = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
        base_acc = cp.get('base_acc', 0)
        print(f"\nUsing existing base model (acc: {base_acc:.4f})")

    # ============================================================
    # PHASE 2: Fine-tune on 2026 validation window
    # ============================================================
    if not base_only:
        print("\n" + "=" * 60)
        print("PHASE 2: FINE-TUNING ON RECENT DATA")
        print("=" * 60)

        if X_2026_val is None or len(X_2026_val) < MIN_VAL:
            print(f"  ⚠️  Not enough validation data ({len(X_2026_val) if X_2026_val is not None else 0})")
            return

        # For fine-tuning: split the val window into finetune/test
        # Use first 70% for fine-tuning, last 30% for evaluation
        n_val = len(X_2026_val)
        split_idx = int(n_val * 0.7)
        
        if split_idx < 10 or (n_val - split_idx) < 10:
            print(f"  ⚠️  Val window too small to split ({n_val} games). Skipping fine-tune.")
            print(f"  Base model accuracy: {base_acc:.4f}")
            return

        X_ft_train = X_2026_val[:split_idx]
        y_ft_train = y_2026_val[:split_idx]
        X_ft_val = X_2026_val[split_idx:]
        y_ft_val = y_2026_val[split_idx:]

        print(f"  Fine-tune train: {len(X_ft_train)} games")
        print(f"  Fine-tune val: {len(X_ft_val)} games")

        # Load base model
        checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
        model = BaseballNet(input_size)
        model.load_state_dict(checkpoint['model_state_dict'])

        ft_state, ft_acc, ft_mean, ft_std, _ = train_model(
            model, X_ft_train, y_ft_train, X_ft_val, y_ft_val,
            lr=FINETUNE_LR, epochs=FINETUNE_EPOCHS, patience=FINETUNE_PATIENCE,
            batch_size=min(BATCH_SIZE, len(X_ft_train)),
            device=device,
            phase_name=f"Phase 2: Fine-tune (lr={FINETUNE_LR})"
        )

        if ft_state and ft_acc > base_acc:
            save_checkpoint(FINETUNED_PATH, ft_state, ft_mean, ft_std, input_size,
                           meta={'phase': 'finetuned', 'base_acc': base_acc,
                                 'finetuned_acc': ft_acc})
            print(f"\n  ✅ Fine-tuned model saved: {base_acc:.4f} → {ft_acc:.4f}")
        elif ft_state:
            print(f"\n  ⏭️  Fine-tuning didn't beat base ({ft_acc:.4f} vs {base_acc:.4f})")
            print(f"  Keeping base model only")
        else:
            print(f"\n  ⏭️  Fine-tuning produced no improvement")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Input features: {input_size}")
    print(f"  Base model: {MODEL_PATH} (acc: {base_acc:.4f})")
    if FINETUNED_PATH.exists():
        cp = torch.load(FINETUNED_PATH, map_location='cpu', weights_only=False)
        print(f"  Fine-tuned: {FINETUNED_PATH} (acc: {cp.get('finetuned_acc', '?')})")
    print(f"  Done: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neural network training v2')
    parser.add_argument('--val-days', type=int, default=7)
    parser.add_argument('--base-only', action='store_true')
    parser.add_argument('--finetune-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    run(val_days=args.val_days, base_only=args.base_only,
        finetune_only=args.finetune_only, dry_run=args.dry_run)
