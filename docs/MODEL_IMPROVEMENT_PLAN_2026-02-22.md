# Model Improvement Plan — 2026-02-22

Based on current evaluated performance in `model_predictions` and `totals_predictions`.

## Current Snapshot

### Win model accuracy (evaluated)

- `prior`: **68.67%**
- `neural`: **68.12%**
- `ensemble`: **68.00%**
- `elo`: **67.71%**
- `pythagorean`: **66.67%**
- `poisson`: **66.00%**
- `lightgbm`: **65.55%**
- `conference`: **64.67%**
- `advanced`: **64.22%**
- `pitching`: **64.00%**
- `log5`: **63.78%**
- `xgboost`: **63.31%**

### Totals model O/U accuracy (evaluated)

- `runs_ensemble`: **35.62%**
- `runs_poisson`: **35.62%**
- `runs_advanced`: **34.91%**
- `runs_pitching`: **26.18%**
- `nn_slim_totals`: **24.38%**

## Priority Plan (in order)

## 1) Win Ensemble Stabilization (High ROI, low risk)

1. Make ensemble recency-aware (e.g., 14/30/60-game weighted windows).
2. Temporarily downweight persistently weak models (`xgboost`, `log5`, `pitching`) until recovery.
3. Add calibration checks (Brier score + reliability bins) for top models.
4. Add champion/challenger gating: no weight increase unless challenger beats baseline on rolling holdout.

**Target:** ensemble outperforms best base model by ~1–2 points over rolling windows.

---

## 2) Totals Pipeline Rebuild (Critical)

1. Run data integrity audit first:
   - line timestamp vs prediction timestamp alignment
   - push/final grading logic
   - game ID/join correctness (no swap/duplicate issues)
2. Reframe totals target as residual vs market total (delta-to-line), not only raw total runs.
3. Segment totals by regime:
   - park/venue effects
   - weather-sensitive spots
   - conference/game-context buckets
4. Tighten confidence/selection policy:
   - no-bet band around zero edge
   - higher minimum edge for action
5. Reduce/disable weak components until validation improves (`runs_pitching`, `nn_slim_totals`).

**Target:** first recover to >50% hit rate, then iterate toward 53–55%.

---

## 3) Feature Quality Upgrades

1. Add starter certainty + bullpen fatigue proxies.
2. Add rest/travel + series-game context features (Fri/Sat/Sun effects).
3. Add park factor × weather interaction features.
4. Reduce stale/default-heavy features; keep validated signal features.

---

## 4) Training + Validation Discipline

1. Use strict walk-forward time splits (no random CV leakage for production metrics).
2. Report performance by segment:
   - conference
   - favorite/underdog buckets
   - odds bands
3. Promote models only when challenger beats champion on:
   - accuracy
   - calibration
   - value proxy metrics

---

## 5) Betting Logic Hardening

1. Treat outputs primarily as ranking signal first, not automatic bet trigger.
2. Add line-movement sanity checks.
3. Require agreement + calibration gate before recording bets.
4. Track CLV-like process metric where closing line unavailable.

---

## 6) Execution Plan (2 weeks)

### Week 1

- Totals integrity audit + fixes
- Ensemble recency weighting + challenger gate
- Calibration reporting additions

### Week 2

- Totals v2 (delta-to-line approach)
- Downweight/disable weakest totals components
- Segment-level validation and controlled rollout

---

## Success Criteria

- Ensemble consistently exceeds top base model by ≥1 point on rolling evaluation.
- Totals models recover above 50% before re-expanding bet volume.
- Promotion/deployment decisions are benchmark-gated and reproducible.
