# College Baseball Predictor — TODO

*Updated: 2026-03-05*

## Current Phase

We completed P0 + P1.1/P1.3 foundation work:
- leak-safe prediction provenance
- meta feature hygiene (15-feature schema)
- canonical benchmark tooling
- strict walk-forward retraining for LGB/XGB/Upset
- trusted replay uplift harness
- CLV infrastructure + closing-line capture

Next work focuses on coverage reliability and as-of feature quality.

---

## P1 — In Progress

### P1.4 Coverage Reliability (high priority)
- [ ] Ensure every scheduled game has full active stack pregame:
  - 12 base models + meta
- [ ] Add coverage guardrail check before first pitch (alert/report missing models by game)
- [ ] Add cron-safe auto-refresh for games missing model rows
- [ ] Add benchmark note when strict cohort is small due to coverage, not model quality

### P1.5 Upset Model As-of Elo (high priority)
- [ ] Replace current Elo proxy with true as-of historical Elo snapshots in upset training
- [ ] Re-run strict walk-forward OOF for upset model
- [ ] Re-run trusted replay uplift and compare delta

### P1.6 Meta Calibration / Confidence
- [ ] Evaluate post-stack calibration (isotonic vs Platt) on strict cohort only
- [ ] Add confidence gating policy for low-quality consensus zones

---

## P2 — Planned

### Training + Ops Hardening
- [ ] Align cron training pipeline with strict walk-forward scripts
  - `train_lightgbm_v2.py`
  - `train_xgboost_v2.py`
  - `train_upset_model.py`
  - `train_meta_ensemble.py`
- [ ] Add daily/weekly benchmark artifact generation and retention
- [ ] Stabilize OpenClaw browser/gateway reliability for FD/DK scrape jobs (recurring timeout incidents)

### Betting Quality
- [ ] Add CLV trend card/summary message automation (daily)
- [ ] Add Kelly sizing that explicitly uses calibrated probabilities
- [ ] Add strategy-level reporting (EV bets vs confident bets vs parlays)

### Model/Data
- [ ] NCAA stat expansion validation for historical depth and missing season coverage
- [ ] Early-season Bayesian dampening review for run models
- [ ] Revisit totals OVER bias handling with updated calibration + CLV feedback loop

---

## Completed (Recent)

### 2026-03-05
- [x] **P0-1** Prediction provenance + leak guard
  - `prediction_source`, `prediction_context`
  - backfill/late-row exclusions in training cohort
- [x] **P0-2** Meta as-of feature hygiene
  - removed leak-prone context features from meta
  - moved to 15-feature leak-safe schema
- [x] **P0-3** Canonical benchmark script + artifact
  - `scripts/evaluate_meta_stack.py`
- [x] **P1.1** Strict walk-forward retraining
  - `train_lightgbm_v2.py`, `train_xgboost_v2.py`, `train_upset_model.py`
- [x] **P1.2** Meta retrain + benchmark diff workflow
- [x] **P1.3** Trusted replay uplift harness
  - `scripts/replay_uplift_benchmark.py`
- [x] CLV tracking implementation
  - line history, closing capture, tracker summary

### 2026-03-04
- [x] Added new base models to active stack: `venue`, `rest_travel`, `upset`
- [x] Specialized feature builders:
  - LightGBM batting-focused
  - XGBoost pitching-focused
- [x] Updated models dashboard and betting agreement counts
- [x] Parlay strategy tightened; default kept at 4 legs with 3-leg fallback

### Prior
- [x] Schedule gateway single-write-path and core cron pipeline cleanup
- [x] Model pages/trend updates and general testing hardening
