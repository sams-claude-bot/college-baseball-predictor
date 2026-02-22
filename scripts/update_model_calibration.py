#!/usr/bin/env python3
"""Fit per-model Platt calibration from recent evaluated predictions."""

import math
import sqlite3
from pathlib import Path
from datetime import datetime

import numpy as np
from sklearn.linear_model import LogisticRegression

DB = Path(__file__).resolve().parent.parent / "data" / "baseball.db"
MIN_SAMPLES = 120
MAX_SAMPLES = 1200


def logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def ensure_table(cur):
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS model_calibration (
            model_name TEXT PRIMARY KEY,
            a REAL NOT NULL,
            b REAL NOT NULL,
            n_samples INTEGER NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )


def fit_for_model(cur, model_name: str):
    rows = cur.execute(
        """
        SELECT predicted_home_prob, was_correct
        FROM model_predictions
        WHERE model_name = ?
          AND was_correct IS NOT NULL
          AND predicted_home_prob IS NOT NULL
        ORDER BY predicted_at DESC
        LIMIT ?
        """,
        (model_name, MAX_SAMPLES),
    ).fetchall()

    if len(rows) < MIN_SAMPLES:
        return None

    x = np.array([[logit(r[0])] for r in rows], dtype=float)
    y = np.array([int(r[1]) for r in rows], dtype=int)

    # If degenerate labels, skip
    if y.min() == y.max():
        return None

    clf = LogisticRegression(solver="lbfgs")
    clf.fit(x, y)

    a = float(clf.coef_[0][0])
    b = float(clf.intercept_[0])
    return a, b, len(rows)


def main():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    ensure_table(cur)

    models = [
        r[0]
        for r in cur.execute(
            "SELECT DISTINCT model_name FROM model_predictions ORDER BY model_name"
        ).fetchall()
    ]

    updated = 0
    skipped = 0
    for m in models:
        result = fit_for_model(cur, m)
        if not result:
            skipped += 1
            continue
        a, b, n = result
        cur.execute(
            """
            INSERT INTO model_calibration (model_name, a, b, n_samples, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(model_name) DO UPDATE SET
                a=excluded.a,
                b=excluded.b,
                n_samples=excluded.n_samples,
                updated_at=excluded.updated_at
            """,
            (m, a, b, n, datetime.now().isoformat(timespec="seconds")),
        )
        updated += 1
        print(f"{m:12} a={a:+.4f} b={b:+.4f} n={n}")

    conn.commit()
    conn.close()
    print(f"\nupdated={updated} skipped={skipped}")


if __name__ == "__main__":
    main()
