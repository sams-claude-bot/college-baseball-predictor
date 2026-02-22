# College Baseball Predictor — TODO

*Execute after Tue 8 PM (Feb 25, 2026)*

## Pipeline & Data
- [ ] **6. Rebuild schedule sync as record verification job** — D1Baseball conference pages, 3 AM, push pipeline back
- [ ] **9. Doubleheader game 2 tracking bug** — scores not updating for gm2 *(partially fixed: finalize_games.py handles it, but root cause in d1bb_schedule.py remains)*
- [ ] **11. Early season regression for all run models** — Bayesian dampening like Poisson fix

## Models
- [ ] **1. Runs ensemble auto-weights** — use accuracy from games WITHOUT DK lines to adjust component weights
- [ ] **5. Review Poisson model** once more data accumulates
- [ ] **12. Parlay totals** — use real Poisson CDF instead of fake formula
- [ ] **13. Auto-update ensemble weights** — rolling accuracy, weekly recalc
- [ ] **14. Trained meta-ensemble** (when ~500+ games tracked, ~early March):
  - XGBoost/logistic regression stacking model
  - Inputs: all model win probs, run projections, model agreement stats, game context, rolling model accuracy
  - **Immediate**: audit what we're storing per game in `model_predictions` — make sure we're capturing everything needed to train it (individual model confidence, run projections, context features). Add columns/table now so we don't lose weeks of data waiting.

## Verification & Cleanup
- [ ] **2. Verify nn_slim_totals is wired into runs/totals projections**
- [ ] **8. Audit model outputs & clean up display** — separate win prob vs run-scoring models

## Dashboard & UI
- [ ] **3. Power rankings top 25** — downloadable card
- [ ] **4. Parlay of the day** — live scores on parlay card or highlighted on scores tab
- [ ] **7. Exact score consensus indicator** — when models converge, show ✓ in table
- [ ] **10. Add SOS sort option to teams page**
