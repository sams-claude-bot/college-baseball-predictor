#!/usr/bin/env python3
"""
Train the Neural Network Model

Loads historical game data, computes features chronologically (no leakage),
trains the PyTorch model with early stopping, and saves weights.

Usage:
    python3 scripts/train_neural.py [--epochs 100] [--lr 0.001] [--batch-size 64]

Temporal split:
    - 2024: Training
    - 2025: Validation (early stopping)
    - 2026: Test (final evaluation)
"""

import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.compute_historical_features import load_historical_games, compute_all_features
from models.neural_model import Trainer


def main():
    parser = argparse.ArgumentParser(description='Train neural network model')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (default: 10)')
    parser.add_argument('--train-seasons', type=int, nargs='*', default=[2024],
                        help='Seasons for training (default: 2024)')
    parser.add_argument('--val-seasons', type=int, nargs='*', default=[2025],
                        help='Seasons for validation (default: 2025)')
    parser.add_argument('--test-seasons', type=int, nargs='*', default=[2026],
                        help='Seasons for testing (default: 2026)')
    args = parser.parse_args()

    all_seasons = sorted(set(args.train_seasons + args.val_seasons + args.test_seasons))

    print(f"Train: {args.train_seasons}, Val: {args.val_seasons}, Test: {args.test_seasons}")
    print(f"Loading historical games for seasons {all_seasons}...")

    games = load_historical_games(all_seasons)
    if not games:
        print("ERROR: No historical games found. Run the scraper first.")
        sys.exit(1)

    print(f"Loaded {len(games)} games total")
    print("Computing features (chronological, no leakage)...")

    X, y, game_ids, seasons = compute_all_features(games)

    # Clean features
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    # Temporal split
    seasons_arr = np.array(seasons)

    train_mask = np.isin(seasons_arr, args.train_seasons)
    val_mask = np.isin(seasons_arr, args.val_seasons)
    test_mask = np.isin(seasons_arr, args.test_seasons)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"\nSplit sizes:")
    print(f"  Train: {len(X_train)} games (home win rate: {y_train.mean():.1%})")
    print(f"  Val:   {len(X_val)} games (home win rate: {y_val.mean():.1%})")
    print(f"  Test:  {len(X_test)} games (home win rate: {y_test.mean():.1%})")

    if len(X_train) == 0:
        print("ERROR: No training data. Check that historical games exist for "
              f"seasons {args.train_seasons}")
        sys.exit(1)

    if len(X_val) == 0:
        print("WARNING: No validation data. Using last 20% of training data.")
        split = int(len(X_train) * 0.8)
        X_val, y_val = X_train[split:], y_train[split:]
        X_train, y_train = X_train[:split], y_train[:split]

    input_size = X_train.shape[1]
    print(f"\nInput features: {input_size}")
    print(f"Training with lr={args.lr}, batch_size={args.batch_size}, "
          f"max_epochs={args.epochs}, patience={args.patience}")
    print("=" * 60)

    trainer = Trainer(
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
    print(f"  Accuracy: {val_results['accuracy']:.4f}")
    print(f"  Log Loss: {val_results['log_loss']:.4f}")
    print(f"  Samples:  {val_results['n_samples']}")
    print("\n  Calibration:")
    for bucket in val_results['calibration']:
        print(f"    {bucket['bin']}: predicted={bucket['predicted']:.3f}, "
              f"actual={bucket['actual']:.3f} (n={bucket['count']})")

    if len(X_test) > 0:
        print("\nTEST SET RESULTS:")
        test_results = trainer.evaluate(X_test, y_test)
        print(f"  Accuracy: {test_results['accuracy']:.4f}")
        print(f"  Log Loss: {test_results['log_loss']:.4f}")
        print(f"  Samples:  {test_results['n_samples']}")
        print("\n  Calibration:")
        for bucket in test_results['calibration']:
            print(f"    {bucket['bin']}: predicted={bucket['predicted']:.3f}, "
                  f"actual={bucket['actual']:.3f} (n={bucket['count']})")
    else:
        print("\nNo test data available yet.")

    print(f"\nModel saved to data/nn_model.pt")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Training complete! ({len(history['train_loss'])} epochs)")


if __name__ == '__main__':
    main()
