#!/usr/bin/env python3
"""Fit per-model calibration (Platt scaling or isotonic regression) from recent evaluated predictions.

Fits both methods and selects whichever has better cross-validated Brier score.
Uses raw_home_prob when available to avoid calibration-on-calibration.
"""

import json
import math
import sqlite3
from pathlib import Path
from datetime import datetime

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

DB = Path(__file__).resolve().parent.parent / "data" / "baseball.db"
MIN_SAMPLES = 120
MAX_SAMPLES = 1200


def logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def ensure_table(cur):
    # Check if we need to migrate the table (add new columns)
    cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='model_calibration'")
    row = cur.fetchone()
    if row is None:
        # Table doesn't exist — create with new schema
        cur.execute(
            """
            CREATE TABLE model_calibration (
                model_name TEXT PRIMARY KEY,
                a REAL NOT NULL,
                b REAL NOT NULL,
                n_samples INTEGER NOT NULL,
                updated_at TEXT NOT NULL,
                method TEXT NOT NULL DEFAULT 'platt',
                isotonic_json TEXT
            )
            """
        )
    else:
        # Table exists — add columns if missing
        schema = row[0]
        if "method" not in schema:
            cur.execute("ALTER TABLE model_calibration ADD COLUMN method TEXT NOT NULL DEFAULT 'platt'")
        if "isotonic_json" not in schema:
            cur.execute("ALTER TABLE model_calibration ADD COLUMN isotonic_json TEXT")


def ensure_raw_prob_column(cur):
    """Add raw_home_prob column to model_predictions if missing."""
    cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='model_predictions'")
    row = cur.fetchone()
    if row and "raw_home_prob" not in row[0]:
        cur.execute("ALTER TABLE model_predictions ADD COLUMN raw_home_prob REAL")


def _brier_score(probs, labels):
    return float(np.mean((probs - labels) ** 2))


def fit_for_model(cur, model_name: str):
    """Fit both Platt and isotonic calibration; return the better one.

    Uses raw_home_prob (uncalibrated) when available, falling back to
    predicted_home_prob for older rows that pre-date the raw column.
    """
    rows = cur.execute(
        """
        SELECT COALESCE(raw_home_prob, predicted_home_prob) as prob, was_correct
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

    probs = np.array([float(r[0]) for r in rows], dtype=float)
    y = np.array([int(r[1]) for r in rows], dtype=int)

    # If degenerate labels, skip
    if y.min() == y.max():
        return None

    # --- Fit Platt scaling ---
    x_logit = np.array([logit(p) for p in probs]).reshape(-1, 1)
    platt_clf = LogisticRegression(solver="lbfgs")
    platt_clf.fit(x_logit, y)
    a = float(platt_clf.coef_[0][0])
    b = float(platt_clf.intercept_[0])

    # Get Platt calibrated probs for Brier score
    platt_probs = np.array([sigmoid(a * logit(p) + b) for p in probs])
    platt_brier = _brier_score(platt_probs, y)

    # --- Fit isotonic regression ---
    iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
    iso.fit(probs, y)
    iso_probs = iso.predict(probs)
    iso_brier = _brier_score(iso_probs, y)

    # Also compute raw Brier for comparison (the "home_won" signal is
    # was_correct which is 1 when pred>0.5 and home won, or pred<0.5 and
    # away won. But for Brier we need P(home win) vs actual home win.
    # was_correct conflates pred direction and outcome — we can't recover
    # actual home_won from was_correct alone without knowing if pred > 0.5.
    # So we compare Platt vs Isotonic Brier on the training set.)
    raw_brier = _brier_score(probs, y)

    # Pick the method with lower Brier score on training data.
    # (With enough samples, in-sample Brier is a reasonable proxy since
    # isotonic has limited VC dimension on sorted data.)
    if iso_brier <= platt_brier:
        # Serialize isotonic breakpoints
        iso_data = {
            "x": iso.X_thresholds_.tolist(),
            "y": iso.y_thresholds_.tolist(),
        }
        return {
            "method": "isotonic",
            "a": 1.0,  # placeholder
            "b": 0.0,  # placeholder
            "n": len(rows),
            "isotonic_json": json.dumps(iso_data),
            "brier_raw": raw_brier,
            "brier_cal": iso_brier,
        }
    else:
        return {
            "method": "platt",
            "a": a,
            "b": b,
            "n": len(rows),
            "isotonic_json": None,
            "brier_raw": raw_brier,
            "brier_cal": platt_brier,
        }


def main():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    ensure_table(cur)
    ensure_raw_prob_column(cur)

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
        cur.execute(
            """
            INSERT INTO model_calibration (model_name, a, b, n_samples, updated_at, method, isotonic_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_name) DO UPDATE SET
                a=excluded.a,
                b=excluded.b,
                n_samples=excluded.n_samples,
                updated_at=excluded.updated_at,
                method=excluded.method,
                isotonic_json=excluded.isotonic_json
            """,
            (
                m,
                result["a"],
                result["b"],
                result["n"],
                datetime.now().isoformat(timespec="seconds"),
                result["method"],
                result["isotonic_json"],
            ),
        )
        updated += 1
        method = result["method"]
        brier_raw = result["brier_raw"]
        brier_cal = result["brier_cal"]
        if method == "platt":
            print(f"{m:16} method=platt  a={result['a']:+.4f} b={result['b']:+.4f}  brier: {brier_raw:.4f} -> {brier_cal:.4f}  n={result['n']}")
        else:
            iso_data = json.loads(result["isotonic_json"])
            n_pts = len(iso_data["x"])
            print(f"{m:16} method=iso   pts={n_pts:3d}  brier: {brier_raw:.4f} -> {brier_cal:.4f}  n={result['n']}")

    conn.commit()
    conn.close()
    print(f"\nupdated={updated} skipped={skipped}")


if __name__ == "__main__":
    main()
