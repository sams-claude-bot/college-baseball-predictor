# MANIFEST.md — Code Classification

> Canonical inventory of active code paths and runtime-critical files.  
> **Last updated: 2026-03-05** (post P0/P1.3 strategy refresh)

## Legend
- 🔴 **CRITICAL** — Required for core app/runtime
- 🟡 **CRON** — Scheduled pipeline script
- 🔵 **LIBRARY** — Shared/imported module
- 🟢 **UTILITY** — Manual/analysis/report tooling
- 📦 **LEGACY** — Kept for reference/history, not active strategy core

---

## Runtime-Critical

### 🔴 Web App
- `web/app.py`
- `web/blueprints/*`
- `web/templates/*`

### 🔴 Live Scoring Daemons (systemd)
- `scripts/statbroadcast_poller.py` — SB live data, 20s interval
- `scripts/sidearm_poller.py` — SIDEARM live data, 30s interval
- `scripts/espn_fastcast_listener.py` — ESPN WebSocket listener
- `models/win_probability.py` — live WP from sb_situation + sa_situation events
- See `docs/live-scoring-architecture.md` for full details

### 🔴 Core Prediction Path
- `scripts/predict_and_track.py`
- `models/predictor_db.py`
- `models/meta_ensemble.py`
- `scripts/database.py`

### 🔴 Active base models feeding meta (12)
- `models/elo_model.py`
- `models/pythagorean_model.py`
- `models/lightgbm_model.py`
- `models/poisson_model.py`
- `models/xgboost_model.py`
- `models/pitching_model.py`
- `models/pear_model.py`
- `models/quality_model.py`
- `models/neural_model.py`
- `models/venue_model.py`
- `models/rest_travel_model.py`
- `models/upset_model.py`

### 🔴 Active feature builders
- `models/features_batting.py` (used by LightGBM moneyline)
- `models/features_pitching.py` (used by XGBoost moneyline)

---

## Benchmark / Evaluation (new strategy)

### 🔵 Leak-safe evaluation stack
- `scripts/evaluate_meta_stack.py` (canonical benchmark)
- `scripts/replay_uplift_benchmark.py` (trusted replay uplift)
- `scripts/diff_meta_benchmark.py` (baseline vs post-change benchmark deltas)

### 🔵 Walk-forward training utility
- `scripts/walkforward_utils.py`

### 🟢 Generated artifacts
- `artifacts/model_benchmark_*.md`
- `artifacts/replay_uplift_*.md`
- `artifacts/*walkforward_latest.md`

---

## Scheduled Cron Pipeline

### 🟡 Daily/Recurring pipeline scripts (`cron/`)
- `01_schedule_and_finalize.sh`
- `02_stats_scrape.sh`
- `03_derived_stats.sh`
- `04_nightly_eval.sh`
- `05_morning_pipeline.sh`
- `full_train.sh`
- `weekly_training.sh`
- `weekly_accuracy.sh`
- `weekly_power_rankings.sh`
- `record_bets.sh`
- `parlay_upgrade.sh`
- `dk_odds_scrape.sh`
- `pregame_discovery.sh`
- `d1b_game_times.sh`
- `pear_ratings.sh`
- `ncaa_stats_scrape.sh`

### 🟡 CLV / closing line capture (system cron target)
- `scripts/capture_closing_lines.py`

---

## Shared Libraries / Infrastructure

### 🔵 DB + utility
- `scripts/database.py`
- `scripts/run_utils.py`
- `scripts/team_resolver.py`
- `scripts/schedule_gateway.py`

### 🔵 Data ingestion/scoring
- `scripts/d1b_scraper.py`
- `scripts/d1b_schedule.py`
- `scripts/d1b_team_sync.py`
- `scripts/finalize_games.py`
- `scripts/record_daily_bets.py`
- `scripts/bet_selection_v2.py`
- `scripts/update_elo.py`
- `scripts/weather.py`
- `scripts/dk_odds_scraper.py`
- `scripts/fd_odds_scraper.py`

---

## Legacy / De-emphasized Components

### 📦 Legacy win models still present in DB/runtime registry
- `models/conference_model.py`
- `models/advanced_model.py`
- `models/log5_model.py`
- `models/prior_model.py`
- `models/ensemble_model.py`

These may appear in historical tables/reports but are not part of the active 12-model stack strategy.

### 📦 Older/experimental training code
- `models/*_v2.py`, `models/neural_model_v3.py`, `models/nn_features_enhanced.py`
- `scripts/train_all_models.py` (legacy orchestration path still exists)

---

## Notes

1. **Source of truth for current strategy:**
   - `README.md`
   - `docs/MODEL_IMPROVEMENT_PLAN_2026-02-22.md` (refreshed 2026-03-05)

2. **Before deleting/archiving files:**
   - Verify they are not referenced by cron scripts, `predict_and_track.py`, or dashboard routes.

3. **Benchmark-first policy:**
   - Any model/stack change should produce updated benchmark/replay artifacts before claiming performance improvement.
