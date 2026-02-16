#!/usr/bin/env python3
"""
Train the Neural Network Spread Model

Temporal split: 2024 train, 2025 validate, 2026 test
Target: home_score - away_score (margin)

Usage:
    python3 scripts/train_spread_nn.py [--epochs 100] [--lr 0.001]
"""

import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.compute_historical_features import load_historical_games, compute_all_features
from models.nn_spread_model import SpreadTrainer


def main():
    parser = argparse.ArgumentParser(description='Train neural network spread model')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--train-seasons', type=int, nargs='*', default=[2024])
    parser.add_argument('--val-seasons', type=int, nargs='*', default=[2025])
    parser.add_argument('--test-seasons', type=int, nargs='*', default=[2026])
    args = parser.parse_args()

    all_seasons = sorted(set(args.train_seasons + args.val_seasons + args.test_seasons))

    print(f"=== TRAINING NN SPREAD MODEL ===")
    print(f"Train: {args.train_seasons}, Val: {args.val_seasons}, Test: {args.test_seasons}")

    games = load_historical_games(all_seasons)
    if not games:
        print("ERROR: No historical games found.")
        sys.exit(1)

    print(f"Loaded {len(games)} games")
    print("Computing features...")

    X, _, game_ids, seasons = compute_all_features(games)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    # Build margin target: home_score - away_score
    game_by_id = {g['id']: g for g in games}
    y_margin = np.array([
        game_by_id[gid]['home_score'] - game_by_id[gid]['away_score']
        for gid in game_ids
    ], dtype=np.float32)

    # Temporal split
    seasons_arr = np.array(seasons)
    train_mask = np.isin(seasons_arr, args.train_seasons)
    val_mask = np.isin(seasons_arr, args.val_seasons)
    test_mask = np.isin(seasons_arr, args.test_seasons)

    X_train, y_train = X[train_mask], y_margin[train_mask]
    X_val, y_val = X[val_mask], y_margin[val_mask]
    X_test, y_test = X[test_mask], y_margin[test_mask]

    print(f"\nSplit sizes:")
    print(f"  Train: {len(X_train)} games (avg margin: {y_train.mean():+.1f})")
    print(f"  Val:   {len(X_val)} games (avg margin: {y_val.mean():+.1f})")
    print(f"  Test:  {len(X_test)} games (avg margin: {y_test.mean():+.1f})")

    if len(X_train) == 0:
        print("ERROR: No training data.")
        sys.exit(1)

    if len(X_val) == 0:
        print("WARNING: No validation data. Using last 20% of training.")
        split = int(len(X_train) * 0.8)
        X_val, y_val = X_train[split:], y_train[split:]
        X_train, y_train = X_train[:split], y_train[:split]

    input_size = X_train.shape[1]
    print(f"\nInput features: {input_size}")
    print("=" * 60)

    trainer = SpreadTrainer(
        input_size=input_size,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
    )

    history = trainer.train(X_train, y_train, X_val, y_val)

    # Evaluate
    print("\n" + "=" * 60)
    print("VALIDATION SET RESULTS:")
    val_results = trainer.evaluate(X_val, y_val)
    print(f"  MAE:  {val_results['mae']:.3f} runs")
    print(f"  RMSE: {val_results['rmse']:.3f} runs")
    print(f"  Mean predicted: {val_results['mean_predicted']:+.2f}")
    print(f"  Mean actual:    {val_results['mean_actual']:+.2f}")
    print(f"  Run line (-1.5) accuracy: {val_results['rl_accuracy']:.1%} ({val_results['rl_correct']}/{val_results['rl_total']})")

    if len(X_test) > 0:
        print("\nTEST SET RESULTS:")
        test_results = trainer.evaluate(X_test, y_test)
        print(f"  MAE:  {test_results['mae']:.3f} runs")
        print(f"  RMSE: {test_results['rmse']:.3f} runs")
        print(f"  Run line accuracy: {test_results['rl_accuracy']:.1%}")
    else:
        print("\nNo test data available yet.")

    print(f"\nModel saved to data/nn_spread_model.pt")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Training complete! ({len(history['train_loss'])} epochs)")


if __name__ == '__main__':
    main()
