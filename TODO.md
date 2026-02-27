# College Baseball Predictor — TODO

*Updated Feb 26, 2026*

## Pipeline & Data
- [x] **6. Schedule Gateway** — *(done 2026-02-26: single write path, 4 scripts rewired, backfill removed from cron, 24 tests)*
- [x] **9. Doubleheader game 2 tracking bug** — *(resolved: 114 gm2 games scored correctly, 0 missing)*
- [ ] **11. Early season regression for all run models** — Bayesian dampening like Poisson fix
- [ ] **15. NCAA stats scraper — expand to all 9 stat types**
  - Currently only collecting ERA + OBP (2 of 9)
  - Missing: batting_avg, fielding_pct, scoring, slugging, k_per_9, whip, k_bb_ratio
  - Need historical seasons (2021-2025) for nn_slim retraining

## Models
- [x] **1. Runs ensemble auto-weights** — *(done 2026-02-24: MAE-based + O/U accuracy blended weighting)*
- [ ] **5. Review Poisson model** once more data accumulates
- [x] **12. Parlay totals** — *(done 2026-02-24: real NegBin+Poisson CDF probabilities replace fake formulas)*
- [x] **13. Auto-update ensemble weights** — *(done: meta-ensemble retraining, rolling accuracy)*
- [x] **14. Trained meta-ensemble** — *(done 2026-02-26: LogReg stacking, 76.7% walk-forward, 20 features from 14 models)*
- [ ] **16. Weekly meta-ensemble retraining cron** — retrain as more games accumulate
- [ ] **17. Probability calibration** — Platt/isotonic calibration for all models (betting P&L audit showed probabilities are overconfident)
- [ ] **18. Model confidence intervals** — track prediction uncertainty, not just point estimates
- [ ] **21. Totals OVER accuracy** — currently 45.7%; asymmetric threshold or higher edge requirement
- [ ] **22. runs_pitching at 51%** — consider disabling or zero-weighting in totals ensemble
- [ ] **23. Re-evaluate XGBoost vs LogReg** for meta-ensemble at ~1,500 graded games

## Betting
- [x] **Betting quality gates (v3)** — *(done 2026-02-24: no underdogs, margin requirements, Vegas disagreement cap, team cooldowns)*
- [ ] **19. Track CLV (Closing Line Value)** — compare model prob vs closing line to measure edge quality
- [ ] **20. Kelly sizing with calibrated probabilities** — current Kelly uses uncalibrated probs

## Verification & Cleanup
- [ ] **2. Verify nn_slim_totals is wired into runs/totals projections**
- [ ] **8. Audit model outputs & clean up display** — separate win prob vs run-scoring models
- [ ] **23. Clean noisy rosters** — remove players with no stats from team rosters (leftover from previous scrape attempts)

## Dashboard & UI
- [ ] **3. Power rankings top 25** — downloadable card
- [ ] **4. Parlay of the day** — live scores on parlay card or highlighted on scores tab
- [ ] **7. Exact score consensus indicator** — when models converge, show ✓ in table
- [x] **10. Add SOS sort option to teams page** *(done 2026-02-22)*

## Completed
### Feb 26
- Schedule Gateway: single write path to games table (24 tests, W/L verification vs D1BB)
- Rewired 4 scripts (d1bb_team_sync, finalize_games, espn_live_scores, d1bb_schedule)
- Removed backfill_missing_games from cron (redundant)
- Meta-ensemble retrained with pear+quality (76.7% walk-forward, LogReg primary)
- Fixed all 4 broken meta_ensemble tests → 350/350 tests green
- PEAR model (76.1%) and Quality model (72.1%) added
- Committed 3,721 lines of outstanding work
- Fixed phantom Georgia game (Oakland series ghost from ESPN import)

### Feb 24 (Overhaul)
- Fixed 10 failing tests (90→114 passing)
- Totals accuracy audit (measurement bug: was 68% not 35%)
- MAE + O/U accuracy metrics (CLI + dashboard)
- Runs Ensemble v2 (NegBin, no pitching, OVER gate, context adjustment)
- DOW + temperature + volatility + MAE auto-weights for totals
- Real NegBin CDF for parlays/betting page
- Models page overhaul + trend chart markers
- Betting v3 quality gates (simulated +$381 improvement)
