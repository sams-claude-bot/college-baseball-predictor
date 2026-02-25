#!/usr/bin/env python3
"""
Train slim neural network on historical data — v3.

v3 improvements:
  - 58 features (40 base + 18 NCAA team stats)
  - Residual connections with GELU activation
  - Label smoothing + mixup augmentation
  - Focal loss option for class imbalance
  - Learning rate finder
  - Stochastic Weight Averaging (SWA) in final 20% of epochs
  - Season-weighted training (recent seasons weighted more)
  - Better fine-tuning: freeze early layers
  - Feature importance via gradient
  - Expanded config search (v3_standard, v3_deep, v3_wide + legacy)

Usage:
    python3 scripts/train_neural_slim.py
    python3 scripts/train_neural_slim.py --val-days 7
    python3 scripts/train_neural_slim.py --dry-run
    python3 scripts/train_neural_slim.py --no-search
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
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

from scripts.database import get_connection
from models.nn_features_slim import (
    SlimFeatureComputer, SlimHistoricalFeatureComputer, SlimBaseballNet,
    ResidualBlock, NUM_FEATURES, FEATURE_NAMES,
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
    # --- Legacy configs (still in search for comparison) ---
    'default': {
        'hidden': [64, 32, 16], 'dropout': [0.3, 0.2, 0.1],
        'lr': 0.001, 'batch_size': 64, 'epochs': 150, 'patience': 15,
        'weight_decay': 1e-5, 'residual': False, 'label_smoothing': 0.0,
    },
    'wider': {
        'hidden': [96, 48, 24], 'dropout': [0.3, 0.2, 0.1],
        'lr': 0.001, 'batch_size': 64, 'epochs': 150, 'patience': 15,
        'weight_decay': 1e-5, 'residual': False, 'label_smoothing': 0.0,
    },
    'deeper': {
        'hidden': [64, 48, 32, 16], 'dropout': [0.3, 0.25, 0.2, 0.1],
        'lr': 0.0008, 'batch_size': 64, 'epochs': 150, 'patience': 15,
        'weight_decay': 1e-4, 'residual': False, 'label_smoothing': 0.0,
    },
    'regularized': {
        'hidden': [64, 32, 16], 'dropout': [0.4, 0.3, 0.2],
        'lr': 0.001, 'batch_size': 128, 'epochs': 150, 'patience': 15,
        'weight_decay': 1e-4, 'residual': False, 'label_smoothing': 0.0,
    },
    # --- v3 configs: wider/deeper to leverage 58 features ---
    'v3_standard': {
        'hidden': [128, 64, 32], 'dropout': [0.3, 0.2, 0.1],
        'lr': 0.001, 'batch_size': 64, 'epochs': 200, 'patience': 20,
        'weight_decay': 1e-4, 'residual': True, 'label_smoothing': 0.05,
    },
    'v3_deep': {
        'hidden': [96, 64, 48, 32, 16], 'dropout': [0.3, 0.25, 0.2, 0.15, 0.1],
        'lr': 0.0008, 'batch_size': 64, 'epochs': 200, 'patience': 20,
        'weight_decay': 1e-4, 'residual': True, 'label_smoothing': 0.05,
    },
    'v3_wide': {
        'hidden': [192, 96, 48], 'dropout': [0.35, 0.25, 0.15],
        'lr': 0.0008, 'batch_size': 128, 'epochs': 200, 'patience': 20,
        'weight_decay': 2e-4, 'residual': True, 'label_smoothing': 0.1,
    },
}

FINETUNE_LR = 0.0001
FINETUNE_EPOCHS = 50
FINETUNE_PATIENCE = 15


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================
# Model building with optional residual connections
# ============================================================

def build_model(config, input_size=None):
    """Build a SlimBaseballNet with custom architecture.

    If config['residual'] is True, pairs of same-dimension hidden layers
    are wrapped in ResidualBlocks. Otherwise, uses standard feedforward.
    
    Args:
        config: model architecture config dict
        input_size: override input feature count (for loading old checkpoints).
                    Defaults to NUM_FEATURES if not specified.
    """
    hidden = config['hidden']
    dropout = config['dropout']
    use_residual = config.get('residual', False)
    feat_size = input_size or NUM_FEATURES

    layers = []
    in_size = feat_size

    for i, (h, d) in enumerate(zip(hidden, dropout)):
        layers.append(nn.Linear(in_size, h))
        layers.append(nn.BatchNorm1d(h))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(d))

        # Add residual block after each layer (except last) if enabled
        if use_residual and i < len(hidden) - 1:
            layers.append(ResidualBlock(h, dropout=d))

        in_size = h

    layers.append(nn.Linear(in_size, 1))
    layers.append(nn.Sigmoid())

    model = SlimBaseballNet(feat_size)
    model.net = nn.Sequential(*layers)
    return model


# ============================================================
# Loss functions
# ============================================================

class FocalBCELoss(nn.Module):
    """Focal loss for binary classification — downweights easy examples."""
    def __init__(self, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce = F.binary_cross_entropy(inputs, targets, reduction='none')
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class LabelSmoothingBCELoss(nn.Module):
    """BCE loss with label smoothing."""
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy(inputs, targets)


# ============================================================
# Mixup augmentation
# ============================================================

def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation: convex combination of random pairs."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y


# ============================================================
# Learning rate finder
# ============================================================

def find_lr(model, X_train, y_train, device, min_lr=1e-7, max_lr=1.0, steps=100):
    """Log-sweep LR finder. Returns suggested learning rate."""
    model = deepcopy(model).to(device)
    train_ds = TensorDataset(
        torch.FloatTensor(X_train).to(device),
        torch.FloatTensor(y_train).to(device)
    )
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=min_lr, weight_decay=1e-5)
    criterion = nn.BCELoss()

    lr_mult = (max_lr / min_lr) ** (1 / steps)
    lrs, losses = [], []
    best_loss = float('inf')
    current_lr = min_lr

    model.train()
    step = 0
    for X_batch, y_batch in loader:
        if step >= steps:
            break
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()

        lrs.append(current_lr)
        losses.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
        if loss.item() > best_loss * 4:
            break

        current_lr *= lr_mult
        for pg in optimizer.param_groups:
            pg['lr'] = current_lr
        step += 1

    # Find LR where loss is decreasing fastest (steepest negative gradient)
    if len(losses) < 10:
        return 1e-3  # fallback

    smoothed = np.convolve(losses, np.ones(5) / 5, mode='valid')
    gradients = np.gradient(smoothed)
    min_idx = np.argmin(gradients)
    # Map back to original index range
    suggested_lr = lrs[min(min_idx + 2, len(lrs) - 1)]
    # Don't go too aggressive
    suggested_lr = min(suggested_lr, 0.01)
    suggested_lr = max(suggested_lr, 1e-5)
    return suggested_lr


# ============================================================
# Feature importance via gradient
# ============================================================

def compute_feature_importance(model, X, y, device, feature_names=None):
    """Compute feature importance via average |gradient| w.r.t. inputs."""
    model = model.to(device)
    model.eval()

    X_t = torch.FloatTensor(X).to(device)
    X_t.requires_grad_(True)
    y_t = torch.FloatTensor(y).to(device)

    preds = model(X_t)
    loss = F.binary_cross_entropy(preds, y_t)
    loss.backward()

    importance = X_t.grad.abs().mean(dim=0).detach().cpu().numpy()

    if feature_names is None:
        feature_names = FEATURE_NAMES

    # Sort by importance
    indices = np.argsort(importance)[::-1]
    print("\n  Feature Importance (top 20):")
    for i, idx in enumerate(indices[:20]):
        name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
        print(f"    {i+1:2d}. {name:30s} {importance[idx]:.6f}")

    return importance


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
            features, label = hfc.compute_game_features(game_row, weather)
            features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
            X.append(features)
            y.append(label)
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
# Season weights: exponential decay for older seasons
# ============================================================

def compute_season_weights(seasons, decay=0.85):
    """Weight recent seasons more heavily. decay=0.85 means each older season
    is worth 85% of the next more recent one."""
    if len(seasons) == 0:
        return np.ones(0)
    max_season = seasons.max()
    weights = np.array([decay ** (max_season - s) for s in seasons], dtype=np.float32)
    # Normalize so mean weight = 1.0
    weights = weights / weights.mean()
    return weights


# ============================================================
# Training with all v3 improvements
# ============================================================

def train_model(model, X_train, y_train, X_val, y_val, config, device,
                phase_name="Training", season_weights=None, use_mixup=True,
                use_focal=False, use_swa=True):
    lr = config['lr']
    epochs = config['epochs']
    patience = config['patience']
    batch_size = config['batch_size']
    weight_decay = config.get('weight_decay', 1e-5)
    label_smoothing = config.get('label_smoothing', 0.0)

    model = model.to(device)

    feature_mean = X_train.mean(axis=0)
    feature_std = X_train.std(axis=0)
    feature_std[feature_std < 1e-8] = 1.0

    X_train_n = np.clip((X_train - feature_mean) / feature_std, -5, 5)
    X_val_n = np.clip((X_val - feature_mean) / feature_std, -5, 5)

    X_train_t = torch.FloatTensor(X_train_n).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)

    # Build sampler with season weights if provided
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
    y_val_t = torch.FloatTensor(y_val).to(device)

    # Loss function
    if use_focal:
        criterion = FocalBCELoss(gamma=2.0, label_smoothing=label_smoothing)
    elif label_smoothing > 0:
        criterion = LabelSmoothingBCELoss(smoothing=label_smoothing)
    else:
        criterion = nn.BCELoss()

    val_criterion = nn.BCELoss()  # always use plain BCE for val

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Cosine annealing with warmup
    warmup_epochs = min(5, epochs // 10)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # SWA setup
    swa_start = int(epochs * 0.8) if use_swa else epochs + 1
    swa_model = torch.optim.swa_utils.AveragedModel(model) if use_swa else None
    swa_scheduler = torch.optim.swa_utils.SWALR(
        optimizer, swa_lr=lr * 0.5, anneal_epochs=5
    ) if use_swa else None

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in loader:
            # Mixup augmentation
            if use_mixup and epoch < swa_start:
                X_batch, y_batch = mixup_data(X_batch, y_batch, alpha=0.2)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Schedule
        if epoch >= swa_start and use_swa:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = val_criterion(val_preds, y_val_t).item()
            val_acc = ((val_preds > 0.5).float() == y_val_t).float().mean().item()

        avg_train = sum(train_losses) / len(train_losses)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            cur_lr = optimizer.param_groups[0]['lr']
            swa_tag = " [SWA]" if epoch >= swa_start and use_swa else ""
            print(f"  Epoch {epoch+1:3d} | Train: {avg_train:.4f} | "
                  f"Val: {val_loss:.4f} | Acc: {val_acc:.4f} | LR: {cur_lr:.6f}{swa_tag}")

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

    # If SWA was used, also evaluate the SWA model
    if use_swa and swa_model is not None and epoch >= swa_start:
        # Update BN statistics for SWA model
        torch.optim.swa_utils.update_bn(loader, swa_model, device=device)
        swa_model.eval()
        with torch.no_grad():
            swa_preds = swa_model(X_val_t)
            swa_loss = val_criterion(swa_preds, y_val_t).item()
            swa_acc = ((swa_preds > 0.5).float() == y_val_t).float().mean().item()
        print(f"  SWA model — Val: {swa_loss:.4f} | Acc: {swa_acc:.4f}")
        if swa_loss < best_val_loss:
            best_val_loss = swa_loss
            best_val_acc = swa_acc
            best_state = {k: v.clone().cpu() for k, v in swa_model.module.state_dict().items()}
            print(f"  SWA model is better, using it.")

    return best_state, best_val_acc, best_val_loss, feature_mean, feature_std


# ============================================================
# K-Fold Cross-Validation
# ============================================================

def kfold_evaluate(X, y, config, device, k=K_FOLDS, seasons=None):
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

        # Season weights for training fold
        sw = None
        if seasons is not None:
            s_train = np.concatenate([seasons[:val_start], seasons[val_end:]])
            sw = compute_season_weights(s_train)

        model = build_model(config)
        _, acc, _, _, _ = train_model(
            model, X_train, y_train, X_val, y_val, config, device,
            phase_name=f"  Fold {fold+1}/{k}",
            season_weights=sw, use_mixup=True, use_focal=False, use_swa=False,
        )
        accs.append(acc)
        print(f"    Fold {fold+1}: {acc:.4f}")

    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    print(f"  CV Mean: {mean_acc:.4f} +/- {std_acc:.4f}")
    return mean_acc, std_acc


def save_checkpoint(path, state_dict, feature_mean, feature_std, meta=None):
    checkpoint = {
        'model_state_dict': state_dict,
        'feature_mean': feature_mean,
        'feature_std': feature_std,
        'input_size': NUM_FEATURES,
        'saved_at': datetime.now().isoformat(),
        'version': 'v3',
    }
    if meta:
        checkpoint.update(meta)
    torch.save(checkpoint, path)
    print(f"  Saved: {path}")


# ============================================================
# Fine-tuning with layer freezing
# ============================================================

def freeze_early_layers(model, freeze_fraction=0.5):
    """Freeze the first fraction of layers for fine-tuning."""
    params = list(model.net.parameters())
    n_freeze = int(len(params) * freeze_fraction)
    for i, p in enumerate(params):
        if i < n_freeze:
            p.requires_grad = False
    frozen = sum(1 for p in params if not p.requires_grad)
    print(f"  Frozen {frozen}/{len(params)} parameters for fine-tuning")


# ============================================================
# Main
# ============================================================

def run(val_days=7, dry_run=False, no_search=False, full_train=False):
    today = datetime.now()
    cutoff = (today - timedelta(days=val_days)).strftime('%Y-%m-%d')
    device = get_device()

    mode_label = "FULL TRAIN" if full_train else f"val last {val_days} days"
    print("=" * 60)
    print(f"SLIM NEURAL NETWORK v3 ({NUM_FEATURES} features)")
    print(f"Date: {today.strftime('%Y-%m-%d %H:%M')}")
    print(f"Mode: {mode_label}")
    print(f"Features: {NUM_FEATURES} (15 base x2 + 9 NCAA x2 + 3 game + 7 weather)")
    print("=" * 60)

    # Load data
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
        print("\nDRY RUN — checking feature dimensions...")
        hfc = SlimHistoricalFeatureComputer()
        if historical:
            row = historical[0]
            game_row = {
                'home_team': row['home_team'], 'away_team': row['away_team'],
                'home_score': row['home_score'], 'away_score': row['away_score'],
                'date': row['date'], 'season': row['season'],
                'neutral_site': row.get('neutral_site') or 0,
            }
            features, label = hfc.compute_game_features(game_row)
            print(f"  Feature vector length: {len(features)} (expected {NUM_FEATURES})")
            assert len(features) == NUM_FEATURES, f"MISMATCH: {len(features)} != {NUM_FEATURES}"
            print("  OK")
        return

    # Build features
    print("\nComputing historical features...")
    X_hist, y_hist, seasons_hist = build_historical_features(historical)
    print(f"  Shape: {X_hist.shape}")

    # Season weights for historical data
    season_weights_hist = compute_season_weights(seasons_hist, decay=0.85)
    print(f"  Season weight range: {season_weights_hist.min():.3f} - {season_weights_hist.max():.3f}")

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

    assert X_hist.shape[1] == NUM_FEATURES, f"Dim mismatch: {X_hist.shape[1]} != {NUM_FEATURES}"

    if full_train:
        X_all_train = X_hist
        y_all_train = y_hist
        sw_all = season_weights_hist
    else:
        if X_2026_train is not None and len(X_2026_train) > 0:
            X_all_train = np.vstack([X_hist, X_2026_train])
            y_all_train = np.concatenate([y_hist, y_2026_train])
            # 2026 games get weight 1.0 (max season weight)
            sw_2026 = np.ones(len(X_2026_train), dtype=np.float32) * season_weights_hist.max()
            sw_all = np.concatenate([season_weights_hist, sw_2026])
        else:
            X_all_train = X_hist
            y_all_train = y_hist
            sw_all = season_weights_hist

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
            print(f"\n  Config: {name}")
            res_tag = " +residual" if config.get('residual') else ""
            ls_tag = f" ls={config.get('label_smoothing', 0)}" if config.get('label_smoothing') else ""
            print(f"   hidden={config['hidden']}, dropout={config['dropout']}, "
                  f"lr={config['lr']}, wd={config['weight_decay']}{res_tag}{ls_tag}")
            cv_acc, cv_std = kfold_evaluate(X_hist, y_hist, config, device,
                                            seasons=seasons_hist)
            if cv_acc > best_cv_acc:
                best_cv_acc = cv_acc
                best_config_name = name

        print(f"\n  Best config: {best_config_name} (CV acc: {best_cv_acc:.4f})")
        best_config = CONFIGS[best_config_name]
    else:
        best_config_name = 'v3_standard'
        best_config = CONFIGS['v3_standard']
        print(f"\nUsing {best_config_name} config (search skipped)")

    # ============================================================
    # LR finder
    # ============================================================
    print("\n  Running LR finder...")
    # Normalize for LR finder
    fm = X_all_train.mean(axis=0)
    fs = X_all_train.std(axis=0)
    fs[fs < 1e-8] = 1.0
    X_norm_for_lr = np.clip((X_all_train - fm) / fs, -5, 5)
    tmp_model = build_model(best_config)
    suggested_lr = find_lr(tmp_model, X_norm_for_lr, y_all_train, device)
    print(f"  Suggested LR: {suggested_lr:.6f} (config LR: {best_config['lr']})")
    # Use suggested LR if it's in reasonable range
    if 0.5 * best_config['lr'] <= suggested_lr <= 5 * best_config['lr']:
        best_config = dict(best_config)
        best_config['lr'] = suggested_lr
        print(f"  Using suggested LR: {suggested_lr:.6f}")
    else:
        print(f"  Keeping config LR: {best_config['lr']}")

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
        best_config, device, phase_name="Phase 1: Base",
        season_weights=sw_all, use_mixup=True,
        use_focal=True, use_swa=True,
    )

    if not best_state:
        print("Base training failed")
        return

    save_checkpoint(MODEL_PATH, best_state, feat_mean, feat_std,
                    meta={'phase': 'base', 'base_acc': base_acc,
                          'config': best_config_name,
                          'train_size': len(X_all_train),
                          'val_size': len(X_2026_val),
                          'full_train': full_train})

    # Feature importance
    print("\n  Computing feature importance...")
    model.load_state_dict(best_state)
    X_val_n = np.clip((X_2026_val - feat_mean) / (feat_std + 1e-8), -5, 5)
    compute_feature_importance(model, X_val_n, y_2026_val, device)

    # ============================================================
    # Phase 2: Fine-tune on 2026 with layer freezing
    # ============================================================
    print("\n" + "=" * 60)
    if full_train:
        print("PHASE 2: FINE-TUNE ON ALL 2026 (validate on same)")
    else:
        print("PHASE 2: FINE-TUNE ON RECENT 2026 (early layers frozen)")
    print("=" * 60)

    if full_train:
        ft_config = dict(best_config)
        ft_config['lr'] = FINETUNE_LR
        ft_config['epochs'] = FINETUNE_EPOCHS
        ft_config['patience'] = FINETUNE_PATIENCE

        model = build_model(best_config)
        model.load_state_dict(best_state)
        freeze_early_layers(model, freeze_fraction=0.5)

        ft_state, ft_acc, ft_loss, ft_mean, ft_std = train_model(
            model, X_2026_val, y_2026_val, X_2026_val, y_2026_val,
            ft_config, device, phase_name="Phase 2: Fine-tune (all 2026)",
            use_mixup=False, use_focal=False, use_swa=False,
        )

        if ft_state and ft_acc > base_acc:
            save_checkpoint(FINETUNED_PATH, ft_state, ft_mean, ft_std,
                            meta={'phase': 'finetuned', 'base_acc': base_acc,
                                  'finetuned_acc': ft_acc, 'config': best_config_name,
                                  'full_train': True})
            print(f"\n  Fine-tuned: {base_acc:.4f} -> {ft_acc:.4f}")
        else:
            print(f"\n  Fine-tune didn't beat base ({ft_acc:.4f} vs {base_acc:.4f})")
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
            freeze_early_layers(model, freeze_fraction=0.5)

            ft_state, ft_acc, ft_loss, ft_mean, ft_std = train_model(
                model, X_ft_train, y_ft_train, X_ft_val, y_ft_val,
                ft_config, device, phase_name="Phase 2: Fine-tune",
                use_mixup=False, use_focal=False, use_swa=False,
            )

            if ft_state and ft_acc > base_acc:
                save_checkpoint(FINETUNED_PATH, ft_state, ft_mean, ft_std,
                                meta={'phase': 'finetuned', 'base_acc': base_acc,
                                      'finetuned_acc': ft_acc, 'config': best_config_name})
                print(f"\n  Fine-tuned: {base_acc:.4f} -> {ft_acc:.4f}")
            else:
                print(f"\n  Fine-tune didn't beat base ({ft_acc:.4f} vs {base_acc:.4f})")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Features: {NUM_FEATURES} (v3 w/ NCAA stats)")
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
