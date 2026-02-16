#!/usr/bin/env python3
"""
Weekly Fine-Tuning Pipeline for Neural Network Models

Fine-tunes all four NN models using 2026 season data with a 2-week delay
to prevent data leakage into validation.

Data split:
  - Fine-tune set: 2026 games completed MORE than 14 days ago
  - Validation set: 2026 games from 14 to 7 days ago
  - Live (unseen): Games from the last 7 days + upcoming
  
The 2-week delay ensures we validate on recent data the model hasn't
seen during fine-tuning, mimicking real prediction conditions.

Usage:
    python3 scripts/finetune_weekly.py              # Fine-tune all 4 models
    python3 scripts/finetune_weekly.py --model win   # Just the win model
    python3 scripts/finetune_weekly.py --dry-run     # Show stats without saving
    python3 scripts/finetune_weekly.py --force       # Fine-tune even if <2 weeks of data
"""

import sys
import argparse
import math
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from scripts.database import get_connection
from models.nn_features import FeatureComputer

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"

# Model configs: (name, base_weights, finetuned_weights, model_type)
MODEL_CONFIGS = {
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

MIN_FINETUNE_GAMES = 50
MIN_VALIDATION_GAMES = 30


def get_2026_completed_games():
    """Get all completed 2026 games with scores from the games table."""
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


def split_games_by_date(games, today=None):
    """
    Split games into fine-tune and validation sets using 2-week delay.
    
    Fine-tune: games completed > 14 days ago
    Validation: games completed 14 to 7 days ago
    Live/unseen: games from last 7 days (not used here)
    """
    if today is None:
        today = datetime.now()
    elif isinstance(today, str):
        today = datetime.strptime(today, '%Y-%m-%d')

    cutoff_finetune = (today - timedelta(days=14)).strftime('%Y-%m-%d')
    cutoff_validation = (today - timedelta(days=7)).strftime('%Y-%m-%d')

    finetune_games = [g for g in games if g['date'] < cutoff_finetune]
    validation_games = [g for g in games if cutoff_finetune <= g['date'] < cutoff_validation]

    return finetune_games, validation_games, cutoff_finetune, cutoff_validation


def compute_game_features_live(game, feature_computer):
    """
    Compute features for a 2026 game using the live FeatureComputer.
    
    The FeatureComputer queries the DB for team stats using only data
    available before the game date (no leakage built-in via date filter).
    """
    features = feature_computer.compute_features(
        home_team_id=game['home_team_id'],
        away_team_id=game['away_team_id'],
        game_date=game['date'],
        neutral_site=bool(game.get('is_neutral_site', 0)),
        is_conference=bool(game.get('is_conference_game', 0)),
    )
    return features


def build_features_and_targets(games, feature_computer, model_type):
    """
    Build feature matrix and target vector for a set of games.
    
    Returns (X, y, dow) where dow is day-of-week array (only for dow_totals).
    """
    X_list = []
    y_list = []
    dow_list = []
    skipped = 0

    for game in games:
        try:
            features = compute_game_features_live(game, feature_computer)
            if np.isnan(features).any() or np.isinf(features).any():
                features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

            # Target depends on model type
            if model_type == 'win':
                target = 1.0 if game['home_score'] > game['away_score'] else 0.0
            elif model_type == 'totals' or model_type == 'dow_totals':
                target = float(game['home_score'] + game['away_score'])
            elif model_type == 'spread':
                target = float(game['home_score'] - game['away_score'])
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            X_list.append(features)
            y_list.append(target)

            # Day of week for dow_totals
            try:
                dt = datetime.strptime(game['date'], '%Y-%m-%d')
                dow_list.append(int(dt.strftime('%w')))  # Sunday=0..Saturday=6
            except (ValueError, TypeError):
                dow_list.append(5)  # default Friday

        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"  Warning: skipped game {game.get('id', '?')}: {e}")

    if skipped > 3:
        print(f"  ({skipped} total games skipped due to errors)")

    if not X_list:
        return None, None, None

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    dow = np.array(dow_list, dtype=np.int64)
    return X, y, dow


def load_base_model(config):
    """Load the base pretrained model and return (model, checkpoint)."""
    base_path = config['base_path']
    if not base_path.exists():
        print(f"  ERROR: Base model not found: {base_path}")
        return None, None

    checkpoint = torch.load(base_path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
        print(f"  ERROR: Invalid checkpoint format in {base_path}")
        return None, None

    return checkpoint, checkpoint.get('input_size')


def create_model_from_checkpoint(checkpoint, model_type):
    """Create a PyTorch model from a checkpoint."""
    input_size = checkpoint.get('input_size')

    if model_type == 'win':
        from models.neural_model import BaseballNet
        model = BaseballNet(input_size)
    elif model_type == 'totals':
        from models.nn_totals_model import TotalsNet
        model = TotalsNet(input_size)
    elif model_type == 'spread':
        from models.nn_spread_model import SpreadNet
        model = SpreadNet(input_size)
    elif model_type == 'dow_totals':
        from models.nn_dow_totals_model import DoWTotalsNet
        dow_embed_dim = checkpoint.get('dow_embed_dim', 4)
        model = DoWTotalsNet(input_size, dow_embed_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def evaluate_base_model(model, checkpoint, X_val, y_val, model_type, dow_val=None):
    """Evaluate the base model on validation data. Returns metric value."""
    model.eval()

    # Apply normalization from the base checkpoint
    feature_mean = checkpoint.get('feature_mean')
    feature_std = checkpoint.get('feature_std')
    X = X_val.copy()
    if feature_mean is not None and feature_std is not None:
        X = (X - feature_mean) / (feature_std + 1e-8)

    with torch.no_grad():
        x_t = torch.tensor(X, dtype=torch.float32)

        if model_type == 'win':
            preds = model(x_t).numpy()
            acc = ((preds > 0.5).astype(float) == y_val).mean()
            return acc, 'accuracy'
        elif model_type == 'totals':
            mean, _ = model(x_t)
            preds = mean.numpy()
            mae = np.abs(preds - y_val).mean()
            return mae, 'mae'
        elif model_type == 'spread':
            mean, _ = model(x_t)
            preds = mean.numpy()
            mae = np.abs(preds - y_val).mean()
            return mae, 'mae'
        elif model_type == 'dow_totals':
            dow_t = torch.tensor(dow_val, dtype=torch.long)
            mean, _ = model(x_t, dow_t)
            preds = mean.numpy()
            mae = np.abs(preds - y_val).mean()
            return mae, 'mae'


def finetune_model(model, checkpoint, X_ft, y_ft, X_val, y_val, model_type,
                   dow_ft=None, dow_val=None, epochs=8, lr=0.0001):
    """
    Fine-tune a model on 2026 data.
    
    Uses the base model's normalization stats (not recomputed) to stay
    consistent with the pretrained weights.
    """
    # Apply base normalization
    feature_mean = checkpoint.get('feature_mean')
    feature_std = checkpoint.get('feature_std')

    X_train = X_ft.copy()
    X_v = X_val.copy()
    if feature_mean is not None and feature_std is not None:
        X_train = (X_train - feature_mean) / (feature_std + 1e-8)
        X_v = (X_v - feature_mean) / (feature_std + 1e-8)

    # Set up training
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_ft, dtype=torch.float32)
    X_val_t = torch.tensor(X_v, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    if model_type == 'dow_totals':
        dow_train_t = torch.tensor(dow_ft, dtype=torch.long)
        dow_val_t = torch.tensor(dow_val, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(X_train_t, dow_train_t, y_train_t)

    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Loss function
    if model_type == 'win':
        criterion = nn.BCELoss()
    else:
        def gaussian_nll(mean, logvar, target):
            var = torch.exp(logvar) + 1e-6
            return 0.5 * torch.mean(torch.log(var) + (target - mean) ** 2 / var)
        criterion = gaussian_nll

    best_val_metric = None
    best_state = None

    for epoch in range(epochs):
        model.train()
        for batch in loader:
            optimizer.zero_grad()

            if model_type == 'win':
                xb, yb = batch
                preds = model(xb)
                loss = criterion(preds, yb)
            elif model_type == 'dow_totals':
                xb, dow_b, yb = batch
                mean, logvar = model(xb, dow_b)
                loss = criterion(mean, logvar, yb)
            else:
                xb, yb = batch
                mean, logvar = model(xb)
                loss = criterion(mean, logvar, yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Evaluate on validation
        model.eval()
        with torch.no_grad():
            if model_type == 'win':
                val_preds = model(X_val_t).numpy()
                metric = ((val_preds > 0.5).astype(float) == y_val).mean()
                improved = best_val_metric is None or metric > best_val_metric
            elif model_type == 'dow_totals':
                val_mean, _ = model(X_val_t, dow_val_t)
                metric = np.abs(val_mean.numpy() - y_val).mean()
                improved = best_val_metric is None or metric < best_val_metric
            else:
                val_mean, _ = model(X_val_t)
                metric = np.abs(val_mean.numpy() - y_val).mean()
                improved = best_val_metric is None or metric < best_val_metric

            if improved:
                best_val_metric = metric
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Restore best epoch
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_metric


def finetune_one_model(name, config, feature_computer, ft_games, val_games,
                       dry_run=False):
    """Fine-tune a single model. Returns report string."""
    model_type = config['type']
    print(f"\n{'='*50}")
    print(f"  {name.upper()} Model")
    print(f"{'='*50}")

    # Load base model
    checkpoint, input_size = load_base_model(config)
    if checkpoint is None:
        return f"  {name}: SKIPPED (base model not found)"

    # Build features
    print(f"  Computing features for {len(ft_games)} fine-tune games...")
    X_ft, y_ft, dow_ft = build_features_and_targets(ft_games, feature_computer, model_type)
    if X_ft is None or len(X_ft) == 0:
        return f"  {name}: SKIPPED (no valid features for fine-tune set)"

    print(f"  Computing features for {len(val_games)} validation games...")
    X_val, y_val, dow_val = build_features_and_targets(val_games, feature_computer, model_type)
    if X_val is None or len(X_val) == 0:
        return f"  {name}: SKIPPED (no valid features for validation set)"

    # Handle feature size mismatch: base model may have different input_size
    # than what FeatureComputer produces (e.g. with/without meta features)
    if X_ft.shape[1] != input_size:
        print(f"  Feature size mismatch: model expects {input_size}, got {X_ft.shape[1]}")
        if X_ft.shape[1] < input_size:
            # Pad with zeros
            pad_ft = np.zeros((X_ft.shape[0], input_size), dtype=np.float32)
            pad_ft[:, :X_ft.shape[1]] = X_ft
            X_ft = pad_ft
            pad_val = np.zeros((X_val.shape[0], input_size), dtype=np.float32)
            pad_val[:, :X_val.shape[1]] = X_val
            X_val = pad_val
        else:
            # Truncate
            X_ft = X_ft[:, :input_size]
            X_val = X_val[:, :input_size]
        print(f"  Adjusted to {input_size} features")

    # Create and evaluate base model
    base_model = create_model_from_checkpoint(checkpoint, model_type)
    base_metric, metric_name = evaluate_base_model(
        base_model, checkpoint, X_val, y_val, model_type, dow_val)

    # Create fresh copy for fine-tuning (don't modify base)
    ft_model = create_model_from_checkpoint(checkpoint, model_type)

    # Fine-tune
    print(f"  Fine-tuning for 8 epochs (lr=0.0001)...")
    ft_model, ft_metric = finetune_model(
        ft_model, checkpoint, X_ft, y_ft, X_val, y_val, model_type,
        dow_ft, dow_val)

    # Compare
    if metric_name == 'accuracy':
        improved = ft_metric > base_metric
        diff = ft_metric - base_metric
        base_str = f"{base_metric:.1%}"
        ft_str = f"{ft_metric:.1%}"
        diff_str = f"+{diff:.1%}" if diff > 0 else f"{diff:.1%}"
    else:  # mae
        improved = ft_metric < base_metric
        diff = ft_metric - base_metric
        base_str = f"{base_metric:.3f}"
        ft_str = f"{ft_metric:.3f}"
        diff_str = f"{diff:+.3f}"

    status = "✅ IMPROVED" if improved else "❌ WORSE"

    report = f"""  {name.replace('_', ' ').title()} Model:
    Base val {metric_name}: {base_str}
    Finetuned val {metric_name}: {ft_str}  {status} ({diff_str})"""

    if improved and not dry_run:
        # Save finetuned weights (preserve base normalization stats)
        save_checkpoint = {
            'model_state_dict': ft_model.state_dict(),
            'feature_mean': checkpoint.get('feature_mean'),
            'feature_std': checkpoint.get('feature_std'),
            'input_size': input_size,
        }
        if model_type == 'dow_totals':
            save_checkpoint['dow_embed_dim'] = checkpoint.get('dow_embed_dim', 4)
        torch.save(save_checkpoint, config['finetuned_path'])
        report += f"\n    → Saved finetuned weights to {config['finetuned_path'].name}"
    elif improved and dry_run:
        report += f"\n    → Would save finetuned weights (dry run)"
    else:
        report += f"\n    → Keeping base weights"

    return report


def main():
    parser = argparse.ArgumentParser(description='Weekly fine-tuning of NN models')
    parser.add_argument('--model', choices=['win', 'totals', 'spread', 'dow_totals'],
                        help='Fine-tune only this model (default: all)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would happen without saving')
    parser.add_argument('--force', action='store_true',
                        help='Fine-tune even if insufficient data')
    args = parser.parse_args()

    today = datetime.now()
    print(f"=== Weekly Fine-Tuning Report ===")
    print(f"Date: {today.strftime('%Y-%m-%d')}")

    # Get 2026 games
    games = get_2026_completed_games()
    print(f"2026 completed games: {len(games)}")

    if not games:
        print("No completed 2026 games found. Nothing to fine-tune.")
        sys.exit(0)

    # Split by date
    ft_games, val_games, ft_cutoff, val_cutoff = split_games_by_date(games, today)
    print(f"Fine-tune set: {len(ft_games)} games (before {ft_cutoff})")
    print(f"Validation set: {len(val_games)} games ({ft_cutoff} to {val_cutoff})")

    # Check minimum data requirements
    if len(ft_games) < MIN_FINETUNE_GAMES and not args.force:
        print(f"\nInsufficient fine-tune data ({len(ft_games)} < {MIN_FINETUNE_GAMES} minimum).")
        print("The season needs ~2 weeks of games before fine-tuning is useful.")
        print("Use --force to override.")
        sys.exit(0)

    if len(val_games) < MIN_VALIDATION_GAMES and not args.force:
        print(f"\nInsufficient validation data ({len(val_games)} < {MIN_VALIDATION_GAMES} minimum).")
        print("Use --force to override.")
        sys.exit(0)

    # Create feature computer (without model predictions to avoid circular deps)
    feature_computer = FeatureComputer(use_model_predictions=False)

    # Determine which models to fine-tune
    if args.model:
        models_to_run = {args.model: MODEL_CONFIGS[args.model]}
    else:
        models_to_run = MODEL_CONFIGS

    # Fine-tune each model
    reports = []
    for name, config in models_to_run.items():
        report = finetune_one_model(name, config, feature_computer,
                                     ft_games, val_games, dry_run=args.dry_run)
        reports.append(report)

    # Final summary
    print(f"\n{'='*50}")
    print("=== Summary ===")
    print(f"{'='*50}")
    for r in reports:
        print(r)

    # Print recommended cron configuration
    print(f"\n# Suggested cron (Sunday 9:30 PM CT, before accuracy report at 10 PM):")
    print(f"# 30 21 * * 0 cd /home/sam/college-baseball-predictor && python3 scripts/finetune_weekly.py >> logs/finetune.log 2>&1")


if __name__ == '__main__':
    main()
