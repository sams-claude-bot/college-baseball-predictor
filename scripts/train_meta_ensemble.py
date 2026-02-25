#!/usr/bin/env python3
"""
Train the meta-ensemble model and report walk-forward validation results.

Usage:
    python scripts/train_meta_ensemble.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.meta_ensemble import MetaEnsemble


def main():
    print("=" * 70)
    print("META-ENSEMBLE TRAINING")
    print("=" * 70)

    meta = MetaEnsemble()
    results = meta.train(retrain=True)

    if results is None:
        print("Training failed or insufficient data.")
        return 1

    print(f"\nWalk-forward validation on {results['n_games']} games "
          f"(out of {results['n_total']} total graded):\n")

    # Comparison table
    print(f"{'Model':<20} {'Accuracy':>10}")
    print("-" * 32)
    print(f"{'META XGBoost':<20} {results['xgb_accuracy']:>9.1f}%")
    print(f"{'META LogReg':<20} {results['lr_accuracy']:>9.1f}%")
    print("-" * 32)
    for m in ['prior', 'neural', 'elo', 'ensemble']:
        if m in results['model_accuracies']:
            print(f"{m:<20} {results['model_accuracies'][m]:>9.1f}%")
    print("-" * 32)
    for m in sorted(results['model_accuracies'].keys()):
        if m not in ['prior', 'neural', 'elo', 'ensemble']:
            print(f"{m:<20} {results['model_accuracies'][m]:>9.1f}%")

    # Per-date accuracy
    print(f"\n{'Date':<14} {'Games':>6} {'XGB Correct':>12} {'XGB Acc':>9}")
    print("-" * 45)
    for date, info in sorted(results['per_date'].items()):
        print(f"{date:<14} {info['total']:>6} {info['xgb_correct']:>12} {info['xgb_acc']:>8.1f}%")

    # Feature importance
    print("\nFeature Importance (top 10):")
    print("-" * 40)
    fi = results['feature_importance']
    for name, imp in sorted(fi.items(), key=lambda x: -x[1])[:10]:
        bar = "█" * int(imp * 100)
        print(f"  {name:<25} {imp:.4f} {bar}")

    print(f"\nModel saved to data/meta_ensemble_xgb.pkl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
