#!/usr/bin/env python3
"""
Train logistic regression for the pitching model.

Builds feature vectors from team pitching quality + opponent batting quality
for every finalized game, then trains a LogisticRegression on matchup
differentials to predict home wins.

Usage:
    python3 scripts/train_pitching_model.py
    python3 scripts/train_pitching_model.py --cv-only   # cross-val report only
"""

import argparse
import pickle
import sqlite3
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

DATA_DIR = PROJECT_DIR / "data"
MODEL_PATH = DATA_DIR / "pitching_logreg.pkl"

KWFIP_CONSTANT = 6.434

# Feature names in the order the model expects
FEATURE_NAMES = [
    # Pitching differentials (home - away)
    "d_rotation_era",
    "d_bullpen_era",
    "d_rotation_whip",
    "d_bullpen_whip",
    "d_rotation_k9",
    "d_bullpen_k9",
    "d_rotation_bb9",
    "d_bullpen_bb9",
    "d_rotation_fip",
    "d_bullpen_fip",
    "d_quality_arms",
    "d_shutdown_arms",
    "d_liability_arms",
    "d_innings_hhi",
    # Hitting differentials (home - away)
    "d_lineup_ops",
    "d_lineup_woba",
    "d_lineup_wrc_plus",
    "d_lineup_k_pct",
    "d_lineup_bb_pct",
    "d_runs_per_game",
    # Cross-matchup: home pitching vs away hitting and vice versa
    "home_rot_era_vs_away_ops",  # home rotation ERA * away lineup OPS
    "away_rot_era_vs_home_ops",
    # Starter-level features (0 when unknown)
    "d_starter_era",
    "d_starter_whip",
    "d_starter_kwfip",
    "starter_known",  # 1 if both starters known, 0 otherwise
]


def get_connection():
    db = sqlite3.connect(str(DATA_DIR / "baseball.db"), timeout=30)
    db.row_factory = sqlite3.Row
    return db


def load_pitching_quality(db):
    """Load all team pitching quality into a dict keyed by team_id."""
    rows = db.execute("SELECT * FROM team_pitching_quality").fetchall()
    return {r["team_id"]: dict(r) for r in rows}


def load_batting_quality(db):
    """Load all team batting quality into a dict keyed by team_id."""
    rows = db.execute("SELECT * FROM team_batting_quality").fetchall()
    return {r["team_id"]: dict(r) for r in rows}


def load_starter_stats(db):
    """Load starter stats keyed by player_stats ID."""
    rows = db.execute("""
        SELECT id, era, whip, k_per_9, bb_per_9, innings_pitched,
               strikeouts_pitched, walks_allowed
        FROM player_stats
        WHERE innings_pitched > 3
    """).fetchall()
    out = {}
    for r in rows:
        d = dict(r)
        ip = d["innings_pitched"] or 0
        k = d.get("strikeouts_pitched")
        bb = d.get("walks_allowed")
        if ip > 0 and k is not None and bb is not None:
            d["kwfip"] = (3 * bb - 2 * k) / ip + KWFIP_CONSTANT
        else:
            d["kwfip"] = d["era"] or 4.50
        out[d["id"]] = d
    return out


def load_matchups(db):
    """Load pitching matchups keyed by game_id."""
    rows = db.execute("""
        SELECT game_id, home_starter_id, away_starter_id
        FROM pitching_matchups
    """).fetchall()
    return {r["game_id"]: dict(r) for r in rows}


def _safe(val, default):
    return val if val is not None else default


def build_feature_vector(home_pitch, away_pitch, home_hit, away_hit,
                         home_starter=None, away_starter=None):
    """Build the feature vector for a single game matchup.

    Returns numpy array of shape (len(FEATURE_NAMES),) or None if insufficient data.
    """
    if home_pitch is None or away_pitch is None:
        return None

    # Defaults for missing batting
    hh = home_hit or {}
    ah = away_hit or {}

    def gp(d, k, default):
        v = d.get(k)
        return v if v is not None else default

    # Pitching differentials (negative = home better for ERA-type stats)
    d_rot_era = gp(home_pitch, "rotation_era", 4.50) - gp(away_pitch, "rotation_era", 4.50)
    d_bp_era = gp(home_pitch, "bullpen_era", 4.50) - gp(away_pitch, "bullpen_era", 4.50)
    d_rot_whip = gp(home_pitch, "rotation_whip", 1.35) - gp(away_pitch, "rotation_whip", 1.35)
    d_bp_whip = gp(home_pitch, "bullpen_whip", 1.35) - gp(away_pitch, "bullpen_whip", 1.35)
    d_rot_k9 = gp(home_pitch, "rotation_k_per_9", 7.5) - gp(away_pitch, "rotation_k_per_9", 7.5)
    d_bp_k9 = gp(home_pitch, "bullpen_k_per_9", 7.5) - gp(away_pitch, "bullpen_k_per_9", 7.5)
    d_rot_bb9 = gp(home_pitch, "rotation_bb_per_9", 3.5) - gp(away_pitch, "rotation_bb_per_9", 3.5)
    d_bp_bb9 = gp(home_pitch, "bullpen_bb_per_9", 3.5) - gp(away_pitch, "bullpen_bb_per_9", 3.5)
    d_rot_fip = gp(home_pitch, "rotation_fip", 4.50) - gp(away_pitch, "rotation_fip", 4.50)
    d_bp_fip = gp(home_pitch, "bullpen_fip", 4.50) - gp(away_pitch, "bullpen_fip", 4.50)
    d_quality = gp(home_pitch, "quality_arms", 0) - gp(away_pitch, "quality_arms", 0)
    d_shutdown = gp(home_pitch, "shutdown_arms", 0) - gp(away_pitch, "shutdown_arms", 0)
    d_liability = gp(home_pitch, "liability_arms", 0) - gp(away_pitch, "liability_arms", 0)
    d_hhi = gp(home_pitch, "innings_hhi", 0.15) - gp(away_pitch, "innings_hhi", 0.15)

    # Hitting differentials
    d_ops = gp(hh, "lineup_ops", 0.740) - gp(ah, "lineup_ops", 0.740)
    d_woba = gp(hh, "lineup_woba", 0.320) - gp(ah, "lineup_woba", 0.320)
    d_wrc = gp(hh, "lineup_wrc_plus", 100) - gp(ah, "lineup_wrc_plus", 100)
    d_kpct = gp(hh, "lineup_k_pct", 0.20) - gp(ah, "lineup_k_pct", 0.20)
    d_bbpct = gp(hh, "lineup_bb_pct", 0.10) - gp(ah, "lineup_bb_pct", 0.10)
    d_rpg = gp(hh, "runs_per_game", 5.5) - gp(ah, "runs_per_game", 5.5)

    # Cross-matchup interactions
    home_rot_vs_away_ops = gp(home_pitch, "rotation_era", 4.50) * gp(ah, "lineup_ops", 0.740)
    away_rot_vs_home_ops = gp(away_pitch, "rotation_era", 4.50) * gp(hh, "lineup_ops", 0.740)

    # Starter features
    starter_known = 0
    d_starter_era = 0.0
    d_starter_whip = 0.0
    d_starter_kwfip = 0.0

    if home_starter and away_starter:
        starter_known = 1
        d_starter_era = _safe(home_starter.get("era"), 4.50) - _safe(away_starter.get("era"), 4.50)
        d_starter_whip = _safe(home_starter.get("whip"), 1.35) - _safe(away_starter.get("whip"), 1.35)
        d_starter_kwfip = _safe(home_starter.get("kwfip"), 4.50) - _safe(away_starter.get("kwfip"), 4.50)

    return np.array([
        d_rot_era, d_bp_era, d_rot_whip, d_bp_whip,
        d_rot_k9, d_bp_k9, d_rot_bb9, d_bp_bb9,
        d_rot_fip, d_bp_fip,
        d_quality, d_shutdown, d_liability, d_hhi,
        d_ops, d_woba, d_wrc, d_kpct, d_bbpct, d_rpg,
        home_rot_vs_away_ops, away_rot_vs_home_ops,
        d_starter_era, d_starter_whip, d_starter_kwfip, starter_known,
    ], dtype=np.float64)


def build_training_data(db):
    """Build X, y arrays from all finalized games."""
    pitching = load_pitching_quality(db)
    batting = load_batting_quality(db)
    starters = load_starter_stats(db)
    matchups = load_matchups(db)

    games = db.execute("""
        SELECT id, date, home_team_id, away_team_id, home_score, away_score,
               COALESCE(is_neutral_site, 0) as neutral
        FROM games
        WHERE status = 'final' AND home_score IS NOT NULL AND away_score IS NOT NULL
          AND home_score != away_score
        ORDER BY date
    """).fetchall()

    X_list = []
    y_list = []
    skipped = 0

    for g in games:
        hp = pitching.get(g["home_team_id"])
        ap = pitching.get(g["away_team_id"])
        hh = batting.get(g["home_team_id"])
        ah = batting.get(g["away_team_id"])

        # Starter data
        home_starter = None
        away_starter = None
        matchup = matchups.get(g["id"])
        if matchup:
            hs_id = matchup.get("home_starter_id")
            as_id = matchup.get("away_starter_id")
            if hs_id and hs_id in starters:
                home_starter = starters[hs_id]
            if as_id and as_id in starters:
                away_starter = starters[as_id]

        fv = build_feature_vector(hp, ap, hh, ah, home_starter, away_starter)
        if fv is None:
            skipped += 1
            continue

        X_list.append(fv)
        y_list.append(1 if g["home_score"] > g["away_score"] else 0)

    print(f"Built {len(X_list)} training samples ({skipped} skipped for missing data)")
    return np.array(X_list), np.array(y_list)


def train_model(X, y, cv_only=False):
    """Train logistic regression with cross-validation."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")),
    ])

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    acc_scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    ll_scores = cross_val_score(pipe, X, y, cv=cv, scoring="neg_log_loss")
    brier_scores = cross_val_score(pipe, X, y, cv=cv, scoring="neg_brier_score")

    print(f"\n=== 5-Fold Cross-Validation ===")
    print(f"Accuracy: {acc_scores.mean():.4f} (+/- {acc_scores.std():.4f})")
    print(f"Log Loss: {-ll_scores.mean():.4f} (+/- {ll_scores.std():.4f})")
    print(f"Brier:    {-brier_scores.mean():.4f} (+/- {brier_scores.std():.4f})")
    print(f"Folds:    {acc_scores}")

    if cv_only:
        return None

    # Train on all data
    pipe.fit(X, y)

    # Report feature importances
    coefs = pipe.named_steps["lr"].coef_[0]
    scale = pipe.named_steps["scaler"].scale_
    # Scaled coefficients show actual impact
    scaled_coefs = coefs * scale
    print(f"\n=== Feature Importance (|scaled coef|) ===")
    importance = sorted(zip(FEATURE_NAMES, coefs, scaled_coefs),
                        key=lambda x: abs(x[2]), reverse=True)
    for name, raw_coef, sc in importance:
        print(f"  {name:<30s} coef={raw_coef:+.4f}  scaled={sc:+.4f}")

    # Calibration check
    probs = pipe.predict_proba(X)[:, 1]
    bins = [(0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.0)]
    print(f"\n=== Calibration (training set) ===")
    for lo, hi in bins:
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() > 0:
            actual = y[mask].mean()
            predicted = probs[mask].mean()
            print(f"  P({lo:.1f}-{hi:.1f}): n={mask.sum():4d}  pred={predicted:.3f}  actual={actual:.3f}")

    return pipe


def main():
    parser = argparse.ArgumentParser(description="Train pitching model logistic regression")
    parser.add_argument("--cv-only", action="store_true", help="Only run cross-validation, don't save model")
    args = parser.parse_args()

    db = get_connection()
    X, y = build_training_data(db)
    db.close()

    if len(X) == 0:
        print("ERROR: No training data available")
        sys.exit(1)

    print(f"\nTraining data: {X.shape[0]} games, {X.shape[1]} features")
    print(f"Home win rate: {y.mean():.3f}")

    model = train_model(X, y, cv_only=args.cv_only)

    if model is not None:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"pipeline": model, "feature_names": FEATURE_NAMES}, f)
        print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
