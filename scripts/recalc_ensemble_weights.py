#!/usr/bin/env python3
"""Cron-safe weekly ensemble hard rebalance."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.ensemble_model import EnsembleModel


def _sorted_weights(weights):
    return dict(sorted(weights.items(), key=lambda kv: kv[1], reverse=True))


def _print_weight_table(before, after):
    names = sorted(set(before) | set(after), key=lambda n: after.get(n, 0), reverse=True)
    print(f"{'model':<14} {'before':>9} {'after':>9} {'delta':>9}")
    print("-" * 44)
    for name in names:
        b = float(before.get(name, 0.0))
        a = float(after.get(name, 0.0))
        print(f"{name:<14} {b:>8.4f} {a:>8.4f} {a-b:>+8.4f}")


def main():
    try:
        model = EnsembleModel()
        before = model.weights.copy()
        changed = model.hard_rebalance_weights(save_accuracy_history=False)
        after = model.weights.copy()

        print("ensemble hard rebalance complete")
        print(f"changed={int(bool(changed))}")
        print("\nBefore (sorted):")
        print(json.dumps(_sorted_weights(before), indent=2, sort_keys=False))
        print("\nAfter (sorted):")
        print(json.dumps(_sorted_weights(after), indent=2, sort_keys=False))
        print("\nDelta:")
        _print_weight_table(before, after)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
