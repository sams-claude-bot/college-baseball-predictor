# College Baseball Predictor — TODO

*Updated Feb 24, 2026 (post-overhaul)*

## Pipeline & Data
- [ ] **6. Rebuild schedule sync as record verification job** — D1Baseball conference pages, 3 AM, push pipeline back
- [x] **9. Doubleheader game 2 tracking bug** — *(resolved: 114 gm2 games scored correctly, 0 missing)*
- [ ] **11. Early season regression for all run models** — Bayesian dampening like Poisson fix
- [ ] **15. NCAA stats scraper — expand to all 9 stat types** *(Tue Feb 25)*
  - Currently only collecting ERA + OBP (2 of 9)
  - Missing: batting_avg, fielding_pct, scoring, slugging, k_per_9, whip, k_bb_ratio
  - Need historical seasons (2021-2025) for nn_slim retraining

## Models
- [x] **1. Runs ensemble auto-weights** — *(done 2026-02-24: MAE-based + O/U accuracy blended weighting)*
- [ ] **5. Review Poisson model** once more data accumulates
- [x] **12. Parlay totals** — *(done 2026-02-24: real NegBin+Poisson CDF probabilities replace fake formulas)*
- [ ] **13. Auto-update ensemble weights** — rolling accuracy, weekly recalc *(partially done: meta-ensemble retraining script exists)*
- [x] **14. Trained meta-ensemble** — *(done 2026-02-24: XGBoost+LogReg stacking, 77.5% walk-forward accuracy, 18 features, integrated into pipeline)*
- [ ] **16. Weekly meta-ensemble retraining cron** — retrain as more games accumulate
- [ ] **17. Probability calibration** — Platt/isotonic calibration for all models (betting P&L audit showed probabilities are overconfident)
- [ ] **18. Model confidence intervals** — track prediction uncertainty, not just point estimates

## Betting
- [x] **Betting quality gates (v3)** — *(done 2026-02-24: no underdogs, margin requirements, Vegas disagreement cap, team cooldowns)*
- [ ] **19. Track CLV (Closing Line Value)** — compare model prob vs closing line to measure edge quality
- [ ] **20. Kelly sizing with calibrated probabilities** — current Kelly uses uncalibrated probs

## Verification & Cleanup
- [ ] **2. Verify nn_slim_totals is wired into runs/totals projections**
- [ ] **8. Audit model outputs & clean up display** — separate win prob vs run-scoring models

## Dashboard & UI
- [ ] **3. Power rankings top 25** — downloadable card
- [ ] **4. Parlay of the day** — live scores on parlay card or highlighted on scores tab
- [ ] **7. Exact score consensus indicator** — when models converge, show ✓ in table
- [x] **10. Add SOS sort option to teams page** *(done 2026-02-22)*

## Completed (Feb 24, 2026 Overhaul)
- Fixed 10 failing tests (90→114 passing)
- Totals accuracy audit (measurement bug: was 68% not 35%)
- MAE + O/U accuracy metrics (CLI + dashboard)
- Runs Ensemble v2 (NegBin, no pitching, OVER gate, context adjustment)
- DOW + temperature + volatility + MAE auto-weights for totals
- Real NegBin CDF for parlays/betting page
- Meta-ensemble (77.5% walk-forward)
- Models page overhaul + trend chart markers
- Betting v3 quality gates (simulated +$381 improvement)
