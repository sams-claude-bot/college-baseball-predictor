# College Baseball Predictor

NCAA D1 college baseball prediction + betting analytics platform with automated pipelines, model stacking, and a live dashboard.

**Live:** [baseball.mcdevitt.page](https://baseball.mcdevitt.page)  
**Season:** 2026 (Feb–Jun)  
**Live data:** StatBroadcast (~65%) + SIDEARM (~15%) + ESPN (100%) — runners, count, plays, win probability

---

## Current Strategy (as of 2026-03-05)

The project moved to a **leak-safe model strategy**:

1. **P0-1 Provenance Guard**
   - `model_predictions` now tracks:
     - `prediction_source` (`live|refresh|backfill|manual`)
     - `prediction_context`
   - Training/eval filters exclude backfilled and late rows.

2. **P0-2 As-of Feature Hygiene**
   - Meta-ensemble temporarily removed leak-prone context features from current-state tables.
   - Meta now uses a **15-feature schema**:
     - 12 base model probabilities
     - 3 agreement features (`models_predicting_home`, `avg_home_prob`, `prob_spread`)

3. **P0-3 Canonical Benchmarking**
   - `scripts/evaluate_meta_stack.py` provides one reproducible benchmark format:
     - accuracy, brier, log loss, ECE
     - strict apples-to-apples cohort
     - correlation + disagreement analysis

4. **P1.1 Strict Walk-Forward Retraining**
   - LightGBM/XGBoost/Upset trainers now run chronological OOF evaluation (no random holdout shortcuts).

5. **P1.3 Trusted Replay Uplift**
   - `scripts/replay_uplift_benchmark.py` compares:
     - baseline stored pregame meta predictions
     - candidate meta replay on the **same stored base probabilities**
   - This isolates meta-layer uplift without recomputing base models from current DB state.

---

## Active Win-Probability Stack

### Base models used by meta-ensemble (12)
- `elo`
- `pythagorean`
- `lightgbm` (batting-focused features)
- `poisson`
- `xgboost` (pitching-focused features)
- `pitching`
- `pear`
- `quality`
- `neural`
- `venue`
- `rest_travel`
- `upset`

### Stacker
- `meta_ensemble` (XGBoost inference path)

Legacy models (`conference`, `advanced`, `log5`, `prior`, legacy `ensemble`) remain in DB history but are not part of the active voting stack.

---

## Benchmark Commands

### Canonical leak-safe benchmark
```bash
python3 scripts/evaluate_meta_stack.py \
  --start-date 2026-01-01 \
  --end-date 2026-12-31
# optional: --out artifacts/model_benchmark_custom.md
```

### Trusted replay uplift benchmark
```bash
python3 scripts/replay_uplift_benchmark.py \
  --start-date 2026-02-20 \
  --end-date 2026-03-05
```

Optional exploratory (non-trusted) shadow section:
```bash
python3 scripts/replay_uplift_benchmark.py \
  --start-date 2026-02-20 \
  --end-date 2026-03-05 \
  --exploratory-shadow-submodels
```

---

## Recent Findings Snapshot

From `artifacts/replay_uplift_2026-03-05.md` (trusted replay):

- Cohort games: **634**
- Meta accuracy: **0.6483 → 0.7776**
- Brier: **0.2309 → 0.1545**
- Log loss: **0.6762 → 0.4742**
- ECE: **0.1119 → 0.0582**
- Side flips: **152**, net correct change: **+82**

Interpret this as strong directional uplift at the meta layer; continue validating with forward live performance.

---

## Betting / CLV

CLV tracking is now live:
- Line history table: `betting_line_history`
- Closing line capture script: `scripts/capture_closing_lines.py`
- CLV fields on tracked bets: `closing_ml`, `clv_implied`, `clv_cents`
- Tracker page includes CLV summary card.

Suggested system cron:
```bash
*/15 10-22 * * * cd /home/sam/college-baseball-predictor && python3 scripts/capture_closing_lines.py >> logs/cron/$(date +\%Y-\%m-\%d)_closing_lines.log 2>&1
```

Parlay strategy:
- Default target: 4 legs
- Fallback allowed: 3 legs if only 3 qualifying legs exist
- Uses tighter filters (higher probability/edge thresholds, ML-first selection)

---

## Dashboard

Core routes include:
- `/` Dashboard
- `/scores`
- `/teams`, `/team/<id>`
- `/models`, `/models/trends`
- `/betting`
- `/tracker` (includes CLV)
- `/game/<id>`

See `docs/DASHBOARD.md` for route/data details.

---

## Project Layout

```text
models/      # model implementations + feature builders
scripts/     # data pipeline, training, prediction, benchmarking, betting
web/         # Flask app + blueprints + templates
docs/        # operational/model documentation
artifacts/   # generated benchmark/replay reports
data/        # local DB and model weights (not committed)
```

---

## Documentation Map

- `README.md` — overview + current strategy + run commands
- `TODO.md` — active backlog/priorities
- `MANIFEST.md` — code classification + cron/runtime inventory
- `docs/MODEL_IMPROVEMENT_PLAN_2026-02-22.md` — model strategy + execution plan (refreshed)
- `docs/DASHBOARD.md` — dashboard route/data dependency reference

If docs conflict, treat benchmark artifacts + current scripts as source of truth and update docs accordingly.
