#!/usr/bin/env python3
"""
Train the Day-of-Week Aware Neural Network Totals Model

Compares against the regular totals NN to validate improvement.
Temporal split: 2024 train, 2025 validate.

Usage:
    python3 scripts/train_dow_totals.py [--epochs 100] [--lr 0.001]
"""

import sys
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.compute_historical_features import load_historical_games, compute_all_features
from models.nn_dow_totals_model import DoWTotalsTrainer
from models.nn_totals_model import TotalsTrainer, MODEL_PATH as REGULAR_MODEL_PATH


def extract_dow(games, game_ids):
    """Extract day-of-week for each game (strftime %w: Sunday=0..Saturday=6)."""
    game_by_id = {g['id']: g for g in games}
    dow = []
    for gid in game_ids:
        date_str = game_by_id[gid]['date']
        try:
            dt = datetime.strptime(date_str, '%Y-%m-%d')
            dow.append(int(dt.strftime('%w')))
        except (ValueError, TypeError):
            dow.append(5)  # default Friday
    return np.array(dow, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description='Train DoW-aware totals NN')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--dow-embed-dim', type=int, default=4)
    parser.add_argument('--train-seasons', type=int, nargs='*', default=[2024])
    parser.add_argument('--val-seasons', type=int, nargs='*', default=[2025])
    args = parser.parse_args()

    all_seasons = sorted(set(args.train_seasons + args.val_seasons))

    print("=" * 70)
    print("  TRAINING DAY-OF-WEEK AWARE TOTALS MODEL")
    print("=" * 70)
    print(f"Train: {args.train_seasons}, Val: {args.val_seasons}")
    print(f"DoW embedding dim: {args.dow_embed_dim}")

    games = load_historical_games(all_seasons)
    if not games:
        print("ERROR: No historical games found.")
        sys.exit(1)

    print(f"Loaded {len(games)} games")
    print("Computing features...")

    X, _, game_ids, seasons = compute_all_features(games)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    # Build targets and day-of-week
    game_by_id = {g['id']: g for g in games}
    y_totals = np.array([
        game_by_id[gid]['home_score'] + game_by_id[gid]['away_score']
        for gid in game_ids
    ], dtype=np.float32)

    dow = extract_dow(games, game_ids)

    # Temporal split
    seasons_arr = np.array(seasons)
    train_mask = np.isin(seasons_arr, args.train_seasons)
    val_mask = np.isin(seasons_arr, args.val_seasons)

    X_train, y_train, dow_train = X[train_mask], y_totals[train_mask], dow[train_mask]
    X_val, y_val, dow_val = X[val_mask], y_totals[val_mask], dow[val_mask]

    print(f"\nSplit sizes:")
    print(f"  Train: {len(X_train)} games (avg total: {y_train.mean():.1f} runs)")
    print(f"  Val:   {len(X_val)} games (avg total: {y_val.mean():.1f} runs)")

    # Show DoW distribution
    dow_names = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    print(f"\nDoW distribution (validation):")
    for d in range(7):
        n = (dow_val == d).sum()
        if n > 0:
            avg = y_val[dow_val == d].mean()
            print(f"  {dow_names[d]}: {n:4d} games, avg total: {avg:.2f}")

    if len(X_train) == 0:
        print("ERROR: No training data.")
        sys.exit(1)

    input_size = X_train.shape[1]
    print(f"\nBase input features: {input_size}")
    print(f"Total input (with embedding): {input_size + args.dow_embed_dim}")

    # ========================================
    # Train DoW model
    # ========================================
    print("\n" + "=" * 70)
    print("  TRAINING DoW TOTALS MODEL")
    print("=" * 70)

    dow_trainer = DoWTotalsTrainer(
        input_size=input_size,
        dow_embed_dim=args.dow_embed_dim,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
    )

    dow_history = dow_trainer.train(X_train, dow_train, y_train, X_val, dow_val, y_val)

    # ========================================
    # Train regular model for comparison
    # ========================================
    print("\n" + "=" * 70)
    print("  TRAINING REGULAR TOTALS MODEL (for comparison)")
    print("=" * 70)

    reg_trainer = TotalsTrainer(
        input_size=input_size,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
    )

    reg_history = reg_trainer.train(X_train, y_train, X_val, y_val)

    # ========================================
    # Compare results
    # ========================================
    print("\n" + "=" * 70)
    print("  COMPARISON: REGULAR vs DoW TOTALS MODEL")
    print("=" * 70)

    # Overall metrics
    median_line = np.median(y_val)
    lines = np.full(len(y_val), median_line)

    dow_results = dow_trainer.evaluate(X_val, dow_val, y_val, over_under_lines=lines)
    reg_results = reg_trainer.evaluate(X_val, y_val, over_under_lines=lines)

    print(f"\nOverall Validation (line={median_line:.1f}):")
    print(f"  {'Metric':<20} {'Regular':>10} {'DoW':>10} {'Diff':>10}")
    print(f"  {'-'*50}")
    print(f"  {'MAE':<20} {reg_results['mae']:>10.3f} {dow_results['mae']:>10.3f} {dow_results['mae'] - reg_results['mae']:>+10.3f}")
    print(f"  {'RMSE':<20} {reg_results['rmse']:>10.3f} {dow_results['rmse']:>10.3f} {dow_results['rmse'] - reg_results['rmse']:>+10.3f}")
    if 'ou_accuracy' in reg_results and 'ou_accuracy' in dow_results:
        print(f"  {'O/U Accuracy':<20} {reg_results['ou_accuracy']:>10.1%} {dow_results['ou_accuracy']:>10.1%} {dow_results['ou_accuracy'] - reg_results['ou_accuracy']:>+10.1%}")

    # Per-day breakdown
    print(f"\n  DAY-OF-WEEK BREAKDOWN:")
    print(f"  {'Day':<12} {'N':>5} {'Reg MAE':>9} {'DoW MAE':>9} {'Improve':>9} {'Avg Actual':>11} {'DoW Pred':>9}")
    print(f"  {'-'*65}")

    dow_by_day = dow_trainer.evaluate_by_dow(X_val, dow_val, y_val, over_under_lines=lines)

    # Get regular model predictions for per-day comparison
    X_val_norm_reg = reg_trainer.apply_normalization(X_val)
    import torch
    with torch.no_grad():
        reg_preds_t, _ = reg_trainer.model(
            torch.tensor(X_val_norm_reg, dtype=torch.float32).to(reg_trainer.device)
        )
        reg_preds = reg_preds_t.cpu().numpy()

    dow_full_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday',
                      'Thursday', 'Friday', 'Saturday']

    for d in range(7):
        name = dow_full_names[d]
        if name not in dow_by_day:
            continue
        mask = dow_val == d
        reg_mae_d = float(np.abs(reg_preds[mask] - y_val[mask]).mean())
        dow_entry = dow_by_day[name]

        improve = reg_mae_d - dow_entry['mae']
        marker = "✓" if improve > 0 else "✗"
        print(f"  {name:<12} {dow_entry['n']:>5} {reg_mae_d:>9.3f} {dow_entry['mae']:>9.3f} {improve:>+9.3f} {marker} {dow_entry['mean_actual']:>11.2f} {dow_entry['mean_pred']:>9.2f}")

    # O/U accuracy by day
    print(f"\n  O/U ACCURACY BY DAY (line={median_line:.1f}):")
    print(f"  {'Day':<12} {'N':>5} {'Reg O/U':>9} {'DoW O/U':>9} {'Improve':>9}")
    print(f"  {'-'*45}")

    for d in range(7):
        name = dow_full_names[d]
        if name not in dow_by_day:
            continue
        mask = dow_val == d
        n_d = mask.sum()
        if n_d == 0:
            continue

        # Regular O/U accuracy for this day
        reg_correct = sum(1 for i in np.where(mask)[0]
                         if (reg_preds[i] > median_line) == (y_val[i] > median_line))
        reg_ou_d = reg_correct / n_d

        dow_entry = dow_by_day[name]
        dow_ou_d = dow_entry.get('ou_accuracy', 0)

        improve = dow_ou_d - reg_ou_d
        marker = "✓" if improve > 0 else "✗"
        print(f"  {name:<12} {n_d:>5} {reg_ou_d:>9.1%} {dow_ou_d:>9.1%} {improve:>+9.1%} {marker}")

    # Embedding analysis
    print(f"\n  LEARNED DoW EMBEDDINGS:")
    emb_weights = dow_trainer.model.dow_embedding.weight.detach().cpu().numpy()
    print(f"  {'Day':<12} {'Embed (4-dim)':>40}")
    print(f"  {'-'*52}")
    for d in range(7):
        vec = emb_weights[d]
        vec_str = ', '.join(f'{v:+.3f}' for v in vec)
        print(f"  {dow_full_names[d]:<12} [{vec_str}]")

    # Compute embedding distances to show learned similarity
    print(f"\n  DoW EMBEDDING DISTANCES (closer = model thinks similar):")
    from itertools import combinations
    pairs = []
    for i, j in combinations(range(7), 2):
        dist = np.linalg.norm(emb_weights[i] - emb_weights[j])
        pairs.append((dow_full_names[i], dow_full_names[j], dist))
    pairs.sort(key=lambda x: x[2])
    for name1, name2, dist in pairs[:5]:
        print(f"  {name1:<10} ↔ {name2:<10}: {dist:.3f}")
    print(f"  ...")
    for name1, name2, dist in pairs[-3:]:
        print(f"  {name1:<10} ↔ {name2:<10}: {dist:.3f}")

    # Final verdict
    print(f"\n" + "=" * 70)
    dow_better = dow_results['mae'] < reg_results['mae']
    if dow_better:
        print(f"  ✓ DoW model WINS: MAE {dow_results['mae']:.3f} vs {reg_results['mae']:.3f} (improvement: {reg_results['mae'] - dow_results['mae']:.3f})")
    else:
        print(f"  ✗ Regular model wins: MAE {reg_results['mae']:.3f} vs {dow_results['mae']:.3f}")
        print(f"    DoW model may still add value on specific days (check per-day breakdown)")

    print(f"\n  DoW model saved to: data/nn_dow_totals_model.pt")
    print(f"  Training complete! ({len(dow_history['train_loss'])} epochs)")


if __name__ == '__main__':
    main()
