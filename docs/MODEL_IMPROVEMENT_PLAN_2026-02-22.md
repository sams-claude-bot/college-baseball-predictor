# Model Improvement Plan — 2026-02-22

> **Updated: 2026-02-26** — Corrected totals accuracy (was measurement bug), added pear/quality models, retrained meta-ensemble.

## Current Snapshot (Feb 26)

### Win model accuracy (evaluated on graded predictions)

| Model | Predictions | Correct | Accuracy | Notes |
|-------|-------------|---------|----------|-------|
| pear | 1,059 | 806 | **76.1%** | NEW — Power/Experience/Adjusted Rating |
| quality | 1,124 | 810 | **72.1%** | NEW — Pitching+batting quality matchup |
| neural | 552 | 372 | 67.4% | |
| elo | 587 | 395 | 67.3% | |
| prior | 588 | 395 | 67.2% | |
| ensemble | 588 | 392 | 66.7% | Dynamic weighted blend |
| lightgbm | 585 | 385 | 65.8% | |
| pythagorean | 588 | 387 | 65.8% | |
| conference | 588 | 380 | 64.6% | |
| poisson | 588 | 378 | 64.3% | |
| xgboost | 585 | 376 | 64.3% | |
| pitching | 588 | 375 | 63.8% | |
| advanced | 588 | 374 | 63.6% | |
| log5 | 588 | 374 | 63.6% | |
| **meta_ensemble** | — | — | **76.7%** | Walk-forward; LogReg over all 14 models |

### Totals model O/U accuracy (evaluated — CORRECTED)

> ⚠️ **The original 35% figures were a measurement bug** — the query divided by ALL rows
> (including ~37,000 ungraded) instead of only graded rows. See `docs/totals_audit_report.md`.

| Model | Graded | Correct | Accuracy |
|-------|--------|---------|----------|
| runs_ensemble | 131 | 89 | **67.9%** |
| runs_poisson | 131 | 88 | 67.2% |
| runs_advanced | 130 | 87 | 66.9% |
| nn_slim_totals | 85 | 55 | 64.7% |
| runs_pitching | 131 | 67 | 51.1% |

**Direction breakdown (runs_ensemble):** UNDER 76.0% (73/96), OVER 45.7% (16/35) — strong UNDER bias.

---

## Completed Items ✅

### Meta-Ensemble Retrained (Feb 26)
- Added pear + quality as features (14 models total)
- Switched from XGBoost to LogReg primary (better on small-data walk-forward)
- Walk-forward accuracy: 76.7% on 549 games
- pear_prob is 2nd most important feature after elo_diff
- All 7 meta_ensemble tests fixed and passing

### Schedule Gateway (Feb 26)
- Single write path to games table (`scripts/schedule_gateway.py`)
- 4 scripts rewired (d1bb_team_sync, finalize_games, espn_live_scores, d1bb_schedule)
- backfill_missing_games removed from cron (redundant)
- 24 tests including W/L verification against D1Baseball
- Eliminates ghost games, duplicate IDs, and score conflicts

### Totals Accuracy Bug Fixed (Feb 24)
- 35% was a measurement bug, not a model bug
- Actual accuracy is 67.9% (runs_ensemble)
- Documented in `docs/totals_audit_report.md`

### Previous (Feb 24 Overhaul)
- Betting v3 quality gates (no underdogs, margin requirements, Vegas disagreement cap)
- Runs Ensemble v2 (NegBin, OVER gate, context adjustment)
- MAE + O/U accuracy metrics
- Real NegBin CDF for parlays
- Models page overhaul + trend chart markers

---

## Remaining Priority Plan

### P1 — This Week

1. **Probability calibration** (Platt/isotonic) for top models — betting is losing money on overconfident probs
2. **Weekly meta-ensemble retraining cron** — retrain as more games accumulate
3. **Totals OVER accuracy** — currently 45.7%; consider asymmetric threshold or higher edge requirement for OVER calls
4. **runs_pitching at 51%** — effectively random; consider disabling or zero-weighting in ensemble

### P2 — Next 1-2 Weeks

5. **CLV tracking** — compare model prob vs closing line to measure edge quality
6. **Kelly sizing with calibrated probabilities** — current Kelly uses uncalibrated probs
7. **NCAA stats scraper — expand to all 9 stat types** (currently only ERA + OBP)
8. **Model confidence intervals** — track prediction uncertainty, not just point estimates
9. **Re-evaluate XGBoost vs LogReg for meta-ensemble** at ~1,500 graded games

### P3 — Ongoing

10. Feature quality upgrades (starter certainty, bullpen fatigue, rest/travel, park×weather)
11. Segment-level validation (conference, favorite/underdog, odds bands)
12. nn_slim retraining with historical seasons (2021-2025)

---

## Success Criteria

- Meta-ensemble consistently exceeds best base model (pear 76.1%) on rolling evaluation ✅ (76.7%)
- Totals OVER accuracy > 50% (currently 45.7%)
- Betting P&L turns positive after calibration
- Promotion/deployment decisions are benchmark-gated and reproducible
