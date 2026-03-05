# Model Improvement Plan — 2026-02-22

> **Refreshed: 2026-03-05** (this file now tracks the current leak-safe strategy and post-P1.3 status)

## Executive Summary

Model development moved from ad-hoc metric chasing to a **gated reliability strategy**:

1. **P0-1 Provenance & leakage controls**
2. **P0-2 As-of feature hygiene**
3. **P0-3 Canonical benchmark**
4. **P1.1 Strict walk-forward retraining**
5. **P1.3 Trusted replay uplift validation**

This sequence gave us a credible, reproducible way to validate improvements without lookahead contamination.

---

## Current Active Stack

### Base win-probability models used by meta (12)
`elo, pythagorean, lightgbm, poisson, xgboost, pitching, pear, quality, neural, venue, rest_travel, upset`

### Stacker
`meta_ensemble` (XGBoost inference path)

### Meta feature schema (P0-2)
- 12 base model probabilities
- 3 agreement features:
  - `models_predicting_home`
  - `avg_home_prob`
  - `prob_spread`

Total: **15 features** (leak-safe hardened schema).

---

## What Was Completed

## P0-1 — Prediction provenance + leak guard
- Added `prediction_source` and `prediction_context` to `model_predictions`
- Excluded backfill + post-cutoff rows from meta training/eval cohorts
- Added cohort integrity reporting

## P0-2 — As-of feature hygiene
- Removed leak-prone context features from meta training/prediction
- Removed dependency on current-state rank/rating tables for meta context
- Ensured train/predict feature schema consistency

## P0-3 — Canonical benchmark
- Added `scripts/evaluate_meta_stack.py`
- Standardized metrics:
  - accuracy
  - brier
  - log loss
  - ECE
- Added strict apples-to-apples cohort, correlation tables, disagreement analysis

## P1.1 — Strict walk-forward submodel retraining
- Upgraded trainers:
  - `train_lightgbm_v2.py`
  - `train_xgboost_v2.py`
  - `train_upset_model.py`
- Added strict date-based fold utility (`scripts/walkforward_utils.py`)
- Reported OOF metrics from chronological folds

## P1.3 — Trusted replay uplift
- Added `scripts/replay_uplift_benchmark.py`
- Baseline: stored historical pregame meta predictions
- Candidate: current meta replay on stored base probabilities (no base recompute)
- Demonstrated large directional uplift in replay artifact

---

## Benchmark / Replay Artifacts (Current)

Primary references:
- `artifacts/model_benchmark_post_p12_2026-03-05.md`
- `artifacts/replay_uplift_2026-03-05.md`

Trusted replay (2026-02-20 .. 2026-03-05):
- n = 634
- accuracy: 0.6483 → 0.7776
- brier: 0.2309 → 0.1545
- log loss: 0.6762 → 0.4742
- ece: 0.1119 → 0.0582
- side flips: 152
- net correct change from flips: +82

Interpretation: strong meta-layer uplift signal; continue validating with forward live results.

---

## Remaining Plan

## P1.4 Coverage Reliability (next)
- Ensure complete pregame prediction coverage for all active models
- Add pre-first-pitch missing-model guardrail/alert
- Reduce strict-cohort starvation caused by incomplete rows

## P1.5 Upset As-of Elo correction
- Replace current Elo proxy in upset model training with true as-of historical Elo snapshots
- Re-run strict walk-forward + replay comparison

## P2 Training/Ops alignment
- Align cron training pipeline to strict walk-forward trainers
- Automate benchmark artifact generation after retraining
- Improve browser/gateway reliability for odds scrapes (FD/DK timeout hardening)

## P2 Betting strategy instrumentation
- Daily CLV summary automation
- Calibrated Kelly deployment checks
- Strategy-level accountability (EV/confident/parlay buckets)

---

## Decision Gates (must pass before “improved” claims)

Any model/stack update must produce:
1. Canonical benchmark artifact
2. Trusted replay artifact (or forward live window if replay not applicable)
3. Explicit cohort counts and leakage exclusions
4. Metrics on same game IDs for baseline vs candidate
5. Written interpretation + risk notes

No gate pass = no production quality claim.
