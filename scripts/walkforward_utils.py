#!/usr/bin/env python3
"""Shared strict date-based walk-forward helpers for binary classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date
from typing import Iterable, List

import numpy as np
from sklearn.metrics import accuracy_score, log_loss


@dataclass(frozen=True)
class WalkForwardFold:
    test_date: str
    train_idx: np.ndarray
    test_idx: np.ndarray


def _normalize_date(d) -> str:
    if isinstance(d, str):
        return d
    if isinstance(d, datetime):
        return d.strftime("%Y-%m-%d")
    if isinstance(d, date):
        return d.strftime("%Y-%m-%d")
    raise TypeError(f"Unsupported date type: {type(d)}")


def build_strict_date_folds(dates: Iterable, min_warmup: int = 200) -> List[WalkForwardFold]:
    """Build strict non-leaky folds.

    For each test date D, train uses rows with date < D and test uses rows with date == D,
    beginning only once train size reaches min_warmup.
    """
    dates_norm = np.array([_normalize_date(d) for d in dates], dtype=object)
    unique_dates = sorted(set(dates_norm.tolist()))

    folds: List[WalkForwardFold] = []
    for d in unique_dates:
        train_idx = np.where(dates_norm < d)[0]
        test_idx = np.where(dates_norm == d)[0]
        if len(test_idx) == 0:
            continue
        if len(train_idx) < min_warmup:
            continue
        folds.append(WalkForwardFold(test_date=d, train_idx=train_idx, test_idx=test_idx))

    return folds


def aggregate_binary_oof(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    """Aggregate deterministic OOF metrics for binary classification."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    if len(y_true) == 0:
        return {
            "n": 0,
            "accuracy": None,
            "brier": None,
            "logloss": None,
        }

    y_prob_clip = np.clip(y_prob, 1e-15, 1 - 1e-15)
    y_pred = (y_prob >= threshold).astype(np.int32)

    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "brier": float(np.mean((y_prob - y_true) ** 2)),
        "logloss": float(log_loss(y_true, y_prob_clip, labels=[0, 1])),
    }
