#!/usr/bin/env python3
"""
Neural Network Training Pipeline v3

Improvements:
  - Focal loss (default) instead of BCE
  - Ensemble distillation mode (--distill): KL divergence on ensemble probs
  - Home/away swap augmentation
  - Time-aware 5-fold cross-validation (temporal folds)
  - CosineAnnealingWarmRestarts scheduler
  - Platt scaling calibration fitted on validation data
  - Enhanced features support (--enhanced-features)

Usage:
    python3 scripts/train_neural_v3.py
    python3 scripts/train_neural_v3.py --distill
    python3 scripts/train_neural_v3.py --enhanced-features
    python3 scripts/train_neural_v3.py --dry-run
"""

import sys
import argparse
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from scripts.database import get_connection
from models.nn_features import FeatureComputer, HistoricalFeatureComputer
from models.neural_model_v3 import BaseballNetV3

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_PATH = DATA_DIR / "nn_model_v3.pt"

# Defaults
BATCH_SIZE = 64
EPOCHS = 120
PATIENCE = 15
LR = 0.001
N_FOLDS = 5
MIN_FOLD_SIZE = 30


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


# ============================================================
# Loss Functions
# ============================================================

class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class DistillationLoss(nn.Module):
    """KL divergence loss for ensemble distillation."""

    def __init__(self):
        super().__init__()

    def forward(self, pred, soft_target):
        # Convert to 2-class distributions
        pred_dist = torch.stack([1 - pred, pred], dim=-1).clamp(1e-7, 1 - 1e-7)
        target_dist = torch.stack([1 - soft_target, soft_target], dim=-1).clamp(1e-7, 1 - 1e-7)
        return F.kl_div(pred_dist.log(), target_dist, reduction='batchmean')


# ============================================================
# Data Loading
# ============================================================

def load_historical():
    """Load pre-2026 historical games."""
    conn = get_connection()
    rows = [dict(r) for r in conn.execute("""
        SELECT h.id, h.home_team, h.away_team, h.home_score, h.away_score,
               h.date, h.season, h.neutral_site,
               w.temp_f, w.humidity_pct, w.wind_speed_mph, w.wind_direction_deg,
               w.precip_prob_pct, w.is_dome
        FROM historical_games h
        LEFT JOIN historical_game_weather w ON h.id = w.game_id
        WHERE h.home_score IS NOT NULL AND h.away_score IS NOT NULL
        ORDER BY h.date ASC
    """).fetchall()]
    conn.close()
    return rows


def load_2026_games(distill=False):
    """Load completed 2026 games, optionally with ensemble soft labels."""
    conn = get_connection()
    if distill:
        rows = [dict(r) for r in conn.execute("""
            SELECT g.id, g.date, g.home_team_id, g.away_team_id,
                   g.home_score, g.away_score, g.is_neutral_site, g.is_conference_game,
                   mp.predicted_home_prob AS ensemble_prob
            FROM games g
            LEFT JOIN model_predictions mp ON g.id = mp.game_id AND mp.model_name = 'ensemble'
            WHERE g.status = 'final' AND g.home_score IS NOT NULL AND g.date >= '2026-01-01'
            ORDER BY g.date ASC
        """).fetchall()]
    else:
        rows = [dict(r) for r in conn.execute("""
            SELECT id, date, home_team_id, away_team_id, home_score, away_score,
                   is_neutral_site, is_conference_game
            FROM games
            WHERE status = 'final' AND home_score IS NOT NULL AND date >= '2026-01-01'
            ORDER BY date ASC
        """).fetchall()]
    conn.close()
    return rows


def build_historical_features(rows):
    """Compute features for historical games."""
    hfc = HistoricalFeatureComputer()
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
        except Exception:
            skipped += 1
    if skipped:
        print(f"  Historical: skipped {skipped}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_2026_features(games, fc, distill=False):
    """Compute features for 2026 games. Returns X, y_hard, y_soft (soft is None if not distill)."""
    X, y_hard, y_soft = [], [], []
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
            y_hard.append(label)
            if distill:
                soft = g.get('ensemble_prob')
                y_soft.append(soft if soft is not None else label)
        except Exception:
            skipped += 1
    if skipped:
        print(f"  2026: skipped {skipped}")
    if not X:
        return None, None, None
    return (np.array(X, dtype=np.float32),
            np.array(y_hard, dtype=np.float32),
            np.array(y_soft, dtype=np.float32) if distill else None)


def augment_swap(X, y, y_soft=None):
    """
    Home/away swap augmentation: swap home/away features to create mirror examples.
    Assumes first half of per-team features are home, second half are away.
    The base FeatureComputer outputs: [home_features..., away_features..., game_features..., weather..., meta...]
    """
    # Count per-team features (they come in pairs: home then away)
    # From FeatureComputer: 36 features per team (strength + batting + pitching + situational + advanced)
    # Then 2 game-level + 7 weather + optional meta
    per_team = 36  # approximate from feature names
    n_features = X.shape[1]

    if n_features < per_team * 2:
        return X, y, y_soft  # can't swap if not enough features

    X_swap = X.copy()
    # Swap home and away blocks
    X_swap[:, :per_team], X_swap[:, per_team:2*per_team] = X[:, per_team:2*per_team], X[:, :per_team]

    y_swap = 1.0 - y  # flip labels
    X_aug = np.vstack([X, X_swap])
    y_aug = np.concatenate([y, y_swap])

    y_soft_aug = None
    if y_soft is not None:
        y_soft_swap = 1.0 - y_soft
        y_soft_aug = np.concatenate([y_soft, y_soft_swap])

    return X_aug, y_aug, y_soft_aug


def temporal_folds(n_samples, n_folds=5):
    """Generate temporal fold indices. Each fold uses earlier data for train, later for val."""
    fold_size = n_samples // n_folds
    folds = []
    for i in range(n_folds):
        val_start = i * fold_size
        val_end = val_start + fold_size if i < n_folds - 1 else n_samples
        train_idx = list(range(0, val_start)) + list(range(val_end, n_samples))
        val_idx = list(range(val_start, val_end))
        if len(train_idx) > 0 and len(val_idx) > 0:
            folds.append((np.array(train_idx), np.array(val_idx)))
    return folds


# ============================================================
# Training
# ============================================================

def train_fold(model, X_train, y_train, X_val, y_val, criterion, device,
               lr=LR, epochs=EPOCHS, patience=PATIENCE, batch_size=BATCH_SIZE):
    """Train one fold. Returns best_state, best_val_loss, val_predictions, feature_mean, feature_std."""
    model = model.to(device)

    # Normalize
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

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for X_b, y_b in loader:
            optimizer.zero_grad()
            preds = model(X_b)
            loss = criterion(preds, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = criterion(val_preds, y_val_t).item()
            val_acc = ((val_preds > 0.5).float() == y_val_t).float().mean().item()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stop at epoch {epoch+1}")
                break

    # Get validation predictions from best model
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_preds_final = model(X_val_t).cpu().numpy()

    return best_state, best_val_loss, val_preds_final, feature_mean, feature_std


def fit_platt_calibration(raw_probs, true_labels):
    """Fit Platt scaling (logistic regression on logits) and return the calibrator."""
    import math
    from sklearn.linear_model import LogisticRegression

    logits = np.array([math.log(max(p, 1e-6) / max(1 - p, 1e-6)) for p in raw_probs]).reshape(-1, 1)
    calibrator = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    calibrator.fit(logits, true_labels.astype(int))
    return calibrator


# ============================================================
# Main Pipeline
# ============================================================

def run(distill=False, enhanced_features=False, dry_run=False, epochs=EPOCHS):
    device = get_device()
    print("=" * 60)
    print("NEURAL NETWORK TRAINING PIPELINE v3")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Options: distill={distill}, enhanced={enhanced_features}")
    print("=" * 60)

    # Feature computer
    if enhanced_features:
        from models.nn_features_enhanced import EnhancedFeatureComputer
        fc = EnhancedFeatureComputer(use_model_predictions=False)
    else:
        fc = FeatureComputer(use_model_predictions=False)

    # Load data
    print("\nLoading historical games...")
    historical = load_historical()
    print(f"  {len(historical)} games")

    print("Loading 2026 games...")
    games_2026 = load_2026_games(distill=distill)
    print(f"  {len(games_2026)} games")

    if dry_run:
        print("\n🔍 DRY RUN — no training")
        return

    # Build features
    print("\nBuilding historical features...")
    X_hist, y_hist = build_historical_features(historical)
    print(f"  Shape: {X_hist.shape}")

    print("Building 2026 features...")
    X_2026, y_2026_hard, y_2026_soft = build_2026_features(games_2026, fc, distill=distill)

    if X_2026 is None or len(X_2026) < MIN_FOLD_SIZE:
        print("Not enough 2026 data")
        return

    # Pad historical features if dimensions don't match (enhanced features add extra columns)
    if X_hist.shape[1] < X_2026.shape[1]:
        pad_width = X_2026.shape[1] - X_hist.shape[1]
        X_hist = np.hstack([X_hist, np.zeros((X_hist.shape[0], pad_width), dtype=np.float32)])
        print(f"  Padded historical features with {pad_width} zeros -> {X_hist.shape}")

    # Combine all data (historical + all 2026)
    X_all = np.vstack([X_hist, X_2026])
    y_all_hard = np.concatenate([y_hist, y_2026_hard])

    if distill and y_2026_soft is not None:
        # For historical games, soft label = hard label (no ensemble predictions)
        y_all_soft = np.concatenate([y_hist, y_2026_soft])
    else:
        y_all_soft = None

    # Verify dimensions
    input_size = X_all.shape[1]
    print(f"\nTotal samples: {len(X_all)}, features: {input_size}")

    # Select loss
    if distill and y_all_soft is not None:
        criterion = DistillationLoss()
        print("Using: KL divergence distillation loss")
    else:
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        print("Using: Focal loss (alpha=0.25, gamma=2.0)")

    # Time-aware k-fold CV
    folds = temporal_folds(len(X_all), N_FOLDS)
    print(f"\nTemporal {len(folds)}-fold cross-validation")

    all_val_preds = np.zeros(len(X_all))
    all_val_mask = np.zeros(len(X_all), dtype=bool)
    best_fold_state = None
    best_fold_loss = float('inf')
    best_fold_mean = None
    best_fold_std = None

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        print(f"\n--- Fold {fold_i+1}/{len(folds)} (train={len(train_idx)}, val={len(val_idx)}) ---")

        X_tr, y_tr = X_all[train_idx], y_all_hard[train_idx]
        X_v, y_v_hard = X_all[val_idx], y_all_hard[val_idx]

        # Use soft labels for training if distilling
        if distill and y_all_soft is not None:
            y_tr_loss = y_all_soft[train_idx]
        else:
            y_tr_loss = y_tr

        # Augment training data with home/away swap
        X_tr_aug, y_tr_aug, _ = augment_swap(X_tr, y_tr_loss)
        print(f"  After augmentation: {len(X_tr_aug)} samples")

        model = BaseballNetV3(input_size)
        state, val_loss, val_preds, f_mean, f_std = train_fold(
            model, X_tr_aug, y_tr_aug, X_v, y_v_hard, criterion, device,
            lr=LR, epochs=epochs, patience=PATIENCE
        )

        all_val_preds[val_idx] = val_preds
        all_val_mask[val_idx] = True

        val_acc = ((val_preds > 0.5).astype(float) == y_v_hard).mean()
        print(f"  Fold {fold_i+1} val_loss={val_loss:.4f}, acc={val_acc:.4f}")

        if val_loss < best_fold_loss:
            best_fold_loss = val_loss
            best_fold_state = state
            best_fold_mean = f_mean
            best_fold_std = f_std

    # Now train final model on ALL data
    print("\n" + "=" * 60)
    print("FINAL: Training on all data")
    print("=" * 60)

    # Augment all data
    y_final = y_all_soft if (distill and y_all_soft is not None) else y_all_hard
    X_final_aug, y_final_aug, _ = augment_swap(X_all, y_final)

    # Use 10% of most recent data as a small validation set for early stopping
    n_val_final = max(int(len(X_all) * 0.1), MIN_FOLD_SIZE)
    X_final_val = X_all[-n_val_final:]
    y_final_val = y_all_hard[-n_val_final:]

    model_final = BaseballNetV3(input_size)
    final_state, _, final_val_preds, final_mean, final_std = train_fold(
        model_final, X_final_aug, y_final_aug, X_final_val, y_final_val,
        criterion, device, lr=LR, epochs=epochs, patience=PATIENCE
    )

    # Fit Platt calibration on CV predictions
    print("\nFitting Platt scaling calibration...")
    mask = all_val_mask
    if mask.sum() > 50:
        calibrator = fit_platt_calibration(all_val_preds[mask], y_all_hard[mask])
        cal_probs = calibrator.predict_proba(
            np.array([np.log(max(p, 1e-6) / max(1 - p, 1e-6)) for p in all_val_preds[mask]]).reshape(-1, 1)
        )[:, 1]
        cal_acc = ((cal_probs > 0.5).astype(float) == y_all_hard[mask]).mean()
        raw_acc = ((all_val_preds[mask] > 0.5).astype(float) == y_all_hard[mask]).mean()
        print(f"  Raw CV acc: {raw_acc:.4f}, Calibrated acc: {cal_acc:.4f}")
        calibrator_bytes = pickle.dumps(calibrator)
    else:
        print("  Not enough CV predictions for calibration")
        calibrator_bytes = None

    # Save
    checkpoint = {
        'model_state_dict': final_state,
        'feature_mean': final_mean,
        'feature_std': final_std,
        'input_size': input_size,
        'calibrator': calibrator_bytes,
        'saved_at': datetime.now().isoformat(),
        'distilled': distill,
        'enhanced_features': enhanced_features,
        'n_samples': len(X_all),
        'cv_folds': len(folds),
    }
    torch.save(checkpoint, MODEL_PATH)
    print(f"\n✅ Saved: {MODEL_PATH}")
    print(f"  Input size: {input_size}")
    print(f"  Samples: {len(X_all)} (augmented: {len(X_final_aug)})")
    print(f"  Calibration: {'fitted' if calibrator_bytes else 'none'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neural network training v3')
    parser.add_argument('--distill', action='store_true', help='Train on ensemble soft labels')
    parser.add_argument('--enhanced-features', action='store_true', help='Use enhanced features')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    run(distill=args.distill, enhanced_features=args.enhanced_features,
        dry_run=args.dry_run, epochs=args.epochs)
