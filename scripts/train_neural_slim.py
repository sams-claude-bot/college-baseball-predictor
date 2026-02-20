#!/usr/bin/env python3
"""
Train slim neural network on historical data with real features only.

Improvements over v1:
  - K-fold cross-validation for robust accuracy estimate
  - Cosine annealing LR schedule with warmup
  - Hyperparameter search (architecture + regularization)
  - Two-phase: base on historical, fine-tune on recent 2026

Usage:
    python3 scripts/train_neural_slim.py
    python3 scripts/train_neural_slim.py --val-days 7
    python3 scripts/train_neural_slim.py --dry-run
    python3 scripts/train_neural_slim.py --no-search    # Skip hyperparameter search
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
from models.nn_features_slim import (
    SlimFeatureComputer, SlimHistoricalFeatureComputer, SlimBaseballNet, NUM_FEATURES
)

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_PATH = DATA_DIR / "nn_slim_model.pt"
FINETUNED_PATH = DATA_DIR / "nn_slim_model_finetuned.pt"

MIN_VAL = 20
K_FOLDS = 5


# ============================================================
# Hyperparameter configs to search
# ============================================================

CONFIGS = {
    'default': {
        'hidden': [64, 32, 16], 'dropout': [0.3, 0.2, 0.1],
        'lr': 0.001, 'batch_size': 64, 'epochs': 150, 'patience': 15,
        'weight_decay': 1e-5,
    },
    'wider': {
        'hidden': [96, 48, 24], 'dropout': [0.3, 0.2, 0.1],
        'lr': 0.001, 'batch_size': 64, 'epochs': 150, 'patience': 15,
        'weight_decay': 1e-5,
    },
    'deeper': {
        'hidden': [64, 48, 32, 16], 'dropout': [0.3, 0.25, 0.2, 0.1],
        'lr': 0.0008, 'batch_size': 64, 'epochs': 150, 'patience': 15,
        'weight_decay': 1e-4,
    },
    'regularized': {
        'hidden': [64, 32, 16], 'dropout': [0.4, 0.3, 0.2],
        'lr': 0.001, 'batch_size': 128, 'epochs': 150, 'patience': 15,
        'weight_decay': 1e-4,
    },
}

FINETUNE_LR = 0.0001
FINETUNE_EPOCHS = 50
FINETUNE_PATIENCE = 15


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def build_model(config):
    """Build a SlimBaseballNet with custom architecture."""
    hidden = config['hidden']
    dropout = config['dropout']

    layers = []
    in_size = NUM_FEATURES
    for i, (h, d) in enumerate(zip(hidden, dropout)):
        layers.append(nn.Linear(in_size, h))
        layers.append(nn.BatchNorm1d(h))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(d))
        in_size = h
    layers.append(nn.Linear(in_size, 1))
    layers.append(nn.Sigmoid())

    model = SlimBaseballNet(NUM_FEATURES)
    model.net = nn.Sequential(*layers)
    return model


# ============================================================
# Data Loading
# ============================================================

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
    hfc = SlimHistoricalFeatureComputer()
    X, y = [], []
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
# Training with cosine annealing + warmup
# ============================================================

def train_model(model, X_train, y_train, X_val, y_val, config, device,
                phase_name="Training"):
    lr = config['lr']
    epochs = config['epochs']
    patience = config['patience']
    batch_size = config['batch_size']
    weight_decay = config.get('weight_decay', 1e-5)

    model = model.to(device)

    feature_mean = X_train.mean(axis=0)
    feature_std = X_train.std(axis=0)
    feature_std[feature_std < 1e-8] = 1.0

    X_train_n = np.clip((X_train - feature_mean) / feature_std, -5, 5)
    X_val_n = np.clip((X_val - feature_mean) / feature_std, -5, 5)

    train_ds = TensorDataset(
        torch.FloatTensor(X_train_n).to(device),
        torch.FloatTensor(y_train).to(device)
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    X_val_t = torch.FloatTensor(X_val_n).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Cosine annealing with warmup
    warmup_epochs = min(5, epochs // 10)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # Linear warmup
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))  # Cosine decay

    import math
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

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

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = criterion(val_preds, y_val_t).item()
            val_acc = ((val_preds > 0.5).float() == y_val_t).float().mean().item()

        avg_train = sum(train_losses) / len(train_losses)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            cur_lr = optimizer.param_groups[0]['lr']
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

    return best_state, best_val_acc, best_val_loss, feature_mean, feature_std


# ============================================================
# K-Fold Cross-Validation
# ============================================================

def kfold_evaluate(X, y, config, device, k=K_FOLDS):
    """Run k-fold CV and return mean accuracy."""
    n = len(X)
    fold_size = n // k
    accs = []

    for fold in range(k):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < k - 1 else n

        X_val = X[val_start:val_end]
        y_val = y[val_start:val_end]
        X_train = np.concatenate([X[:val_start], X[val_end:]])
        y_train = np.concatenate([y[:val_start], y[val_end:]])

        model = build_model(config)
        _, acc, _, _, _ = train_model(
            model, X_train, y_train, X_val, y_val, config, device,
            phase_name=f"  Fold {fold+1}/{k}"
        )
        accs.append(acc)
        print(f"    Fold {fold+1}: {acc:.4f}")

    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    print(f"  CV Mean: {mean_acc:.4f} ± {std_acc:.4f}")
    return mean_acc, std_acc


def save_checkpoint(path, state_dict, feature_mean, feature_std, meta=None):
    checkpoint = {
        'model_state_dict': state_dict,
        'feature_mean': feature_mean,
        'feature_std': feature_std,
        'input_size': NUM_FEATURES,
        'saved_at': datetime.now().isoformat(),
    }
    if meta:
        checkpoint.update(meta)
    torch.save(checkpoint, path)
    print(f"  Saved: {path}")


# ============================================================
# Main
# ============================================================

def run(val_days=7, dry_run=False, no_search=False, full_train=False):
    today = datetime.now()
    cutoff = (today - timedelta(days=val_days)).strftime('%Y-%m-%d')
    device = get_device()

    mode_label = "FULL TRAIN" if full_train else f"val last {val_days} days"
    print("=" * 60)
    print(f"SLIM NEURAL NETWORK v2 ({NUM_FEATURES} features)")
    print(f"Date: {today.strftime('%Y-%m-%d %H:%M')}")
    print(f"Mode: {mode_label}")
    print("=" * 60)

    # Load data
    print("\nLoading historical games...")
    historical = load_historical()
    print(f"  Historical: {len(historical)} games")

    print("Loading 2026 games...")
    games_2026 = load_2026_games()

    if full_train:
        # Full train: base on historical, finetune+validate on ALL 2026
        train_2026 = []  # Not used for base — historical only
        val_2026 = games_2026
        print(f"  2026 (finetune+val): {len(val_2026)}")
    else:
        train_2026 = [g for g in games_2026 if g['date'] < cutoff]
        val_2026 = [g for g in games_2026 if g['date'] >= cutoff]
        print(f"  2026 train: {len(train_2026)} | val: {len(val_2026)}")

    if dry_run:
        print("\n🔍 DRY RUN")
        return

    # Build features
    print("\nComputing historical features...")
    X_hist, y_hist = build_historical_features(historical)
    print(f"  Shape: {X_hist.shape}")

    if not full_train and train_2026:
        print("Computing 2026 train features...")
        X_2026_train, y_2026_train = build_2026_features(train_2026)
    else:
        X_2026_train, y_2026_train = None, None

    print("Computing 2026 val features...")
    X_2026_val, y_2026_val = build_2026_features(val_2026)

    if X_2026_val is None or len(X_2026_val) < MIN_VAL:
        print(f"⚠️  Only {len(X_2026_val) if X_2026_val is not None else 0} val games")
        return

    assert X_hist.shape[1] == NUM_FEATURES, f"Dim mismatch: {X_hist.shape[1]} != {NUM_FEATURES}"

    if full_train:
        # Full train: base uses ONLY historical
        X_all_train = X_hist
        y_all_train = y_hist
    else:
        # Normal: historical + pre-cutoff 2026
        if X_2026_train is not None and len(X_2026_train) > 0:
            X_all_train = np.vstack([X_hist, X_2026_train])
            y_all_train = np.concatenate([y_hist, y_2026_train])
        else:
            X_all_train = X_hist
            y_all_train = y_hist

    # ============================================================
    # Hyperparameter search with k-fold CV
    # ============================================================
    if not no_search:
        print("\n" + "=" * 60)
        print(f"HYPERPARAMETER SEARCH ({K_FOLDS}-fold CV on historical)")
        print("=" * 60)

        best_config_name = None
        best_cv_acc = 0.0

        for name, config in CONFIGS.items():
            print(f"\n📐 Config: {name}")
            print(f"   hidden={config['hidden']}, dropout={config['dropout']}, "
                  f"lr={config['lr']}, wd={config['weight_decay']}")
            cv_acc, cv_std = kfold_evaluate(X_hist, y_hist, config, device)
            if cv_acc > best_cv_acc:
                best_cv_acc = cv_acc
                best_config_name = name

        print(f"\n🏆 Best config: {best_config_name} (CV acc: {best_cv_acc:.4f})")
        best_config = CONFIGS[best_config_name]
    else:
        best_config_name = 'default'
        best_config = CONFIGS['default']
        print(f"\nUsing default config (search skipped)")

    # ============================================================
    # Phase 1: Train on historical, validate on 2026
    # ============================================================
    print("\n" + "=" * 60)
    print(f"PHASE 1: BASE TRAINING ({best_config_name})")
    print(f"  {len(X_all_train)} train + {len(X_2026_val)} val")
    print("=" * 60)

    model = build_model(best_config)
    best_state, base_acc, base_loss, feat_mean, feat_std = train_model(
        model, X_all_train, y_all_train, X_2026_val, y_2026_val,
        best_config, device, phase_name="Phase 1: Base"
    )

    if not best_state:
        print("❌ Base training failed")
        return

    save_checkpoint(MODEL_PATH, best_state, feat_mean, feat_std,
                    meta={'phase': 'base', 'base_acc': base_acc,
                          'config': best_config_name,
                          'train_size': len(X_all_train),
                          'val_size': len(X_2026_val),
                          'full_train': full_train})

    # ============================================================
    # Phase 2: Fine-tune on 2026, validate on 2026
    # ============================================================
    print("\n" + "=" * 60)
    if full_train:
        print("PHASE 2: FINE-TUNE ON ALL 2026 (validate on same)")
    else:
        print("PHASE 2: FINE-TUNE ON RECENT 2026")
    print("=" * 60)

    if full_train:
        # Full train: finetune on ALL 2026, validate on ALL 2026
        ft_config = dict(best_config)
        ft_config['lr'] = FINETUNE_LR
        ft_config['epochs'] = FINETUNE_EPOCHS
        ft_config['patience'] = FINETUNE_PATIENCE

        model = build_model(best_config)
        model.load_state_dict(best_state)

        ft_state, ft_acc, ft_loss, ft_mean, ft_std = train_model(
            model, X_2026_val, y_2026_val, X_2026_val, y_2026_val,
            ft_config, device, phase_name="Phase 2: Fine-tune (all 2026)"
        )

        if ft_state and ft_acc > base_acc:
            save_checkpoint(FINETUNED_PATH, ft_state, ft_mean, ft_std,
                            meta={'phase': 'finetuned', 'base_acc': base_acc,
                                  'finetuned_acc': ft_acc, 'config': best_config_name,
                                  'full_train': True})
            print(f"\n  ✅ Fine-tuned: {base_acc:.4f} → {ft_acc:.4f}")
        else:
            print(f"\n  ⏭️  Fine-tune didn't beat base ({ft_acc:.4f} vs {base_acc:.4f})")
    else:
        n_val = len(X_2026_val)
        split_idx = int(n_val * 0.7)
        if split_idx < 10 or (n_val - split_idx) < 10:
            print(f"  Val too small to split ({n_val}). Keeping base.")
        else:
            X_ft_train, y_ft_train = X_2026_val[:split_idx], y_2026_val[:split_idx]
            X_ft_val, y_ft_val = X_2026_val[split_idx:], y_2026_val[split_idx:]

            ft_config = dict(best_config)
            ft_config['lr'] = FINETUNE_LR
            ft_config['epochs'] = FINETUNE_EPOCHS
            ft_config['patience'] = FINETUNE_PATIENCE

            model = build_model(best_config)
            model.load_state_dict(best_state)

            ft_state, ft_acc, ft_loss, ft_mean, ft_std = train_model(
                model, X_ft_train, y_ft_train, X_ft_val, y_ft_val,
                ft_config, device, phase_name="Phase 2: Fine-tune"
            )

            if ft_state and ft_acc > base_acc:
                save_checkpoint(FINETUNED_PATH, ft_state, ft_mean, ft_std,
                                meta={'phase': 'finetuned', 'base_acc': base_acc,
                                      'finetuned_acc': ft_acc, 'config': best_config_name})
                print(f"\n  ✅ Fine-tuned: {base_acc:.4f} → {ft_acc:.4f}")
            else:
                print(f"\n  ⏭️  Fine-tune didn't beat base ({ft_acc:.4f} vs {base_acc:.4f})")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Features: {NUM_FEATURES}")
    print(f"  Config: {best_config_name}")
    print(f"  Mode: {'full-train' if full_train else 'standard'}")
    print(f"  Base: {MODEL_PATH} (acc: {base_acc:.4f})")
    if FINETUNED_PATH.exists():
        cp = torch.load(FINETUNED_PATH, map_location='cpu', weights_only=False)
        print(f"  Fine-tuned: {FINETUNED_PATH} (acc: {cp.get('finetuned_acc', '?')})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--val-days', type=int, default=7)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--no-search', action='store_true', help='Skip hyperparameter search')
    parser.add_argument('--full-train', action='store_true',
                        help='Train base on all historical, finetune+validate on all 2026')
    args = parser.parse_args()
    run(val_days=args.val_days, dry_run=args.dry_run, no_search=args.no_search,
        full_train=args.full_train)
