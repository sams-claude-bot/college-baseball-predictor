# College Baseball Predictor — TODO

*Execute after Tue 8 PM (Feb 25, 2026)*

## Pipeline & Data
- [ ] **6. Rebuild schedule sync as record verification job** — D1Baseball conference pages, 3 AM, push pipeline back
- [x] **9. Doubleheader game 2 tracking bug** — *(resolved: 114 gm2 games scored correctly, 0 missing)*
- [ ] **11. Early season regression for all run models** — Bayesian dampening like Poisson fix

## Models
- [ ] **1. Runs ensemble auto-weights** — use accuracy from games WITHOUT DK lines to adjust component weights
- [ ] **5. Review Poisson model** once more data accumulates
- [ ] **12. Parlay totals** — use real Poisson CDF instead of fake formula
- [ ] **13. Auto-update ensemble weights** — rolling accuracy, weekly recalc
- [ ] **14. Trained meta-ensemble** (when ~500+ games tracked, ~early March):
  - XGBoost/logistic regression stacking model
  - Inputs: all model win probs, run projections, model agreement stats, game context, rolling model accuracy
  - **Audit complete (2026-02-22)**: All needed features available via joins — no schema changes required. model_predictions has all 12 model probs + runs. Context (Elo, conference, neutral) joins from games/elo_ratings. Agreement/spread computed at training time.

## Verification & Cleanup
- [ ] **2. Verify nn_slim_totals is wired into runs/totals projections**
- [ ] **8. Audit model outputs & clean up display** — separate win prob vs run-scoring models

## Dashboard & UI
- [ ] **3. Power rankings top 25** — downloadable card
- [ ] **4. Parlay of the day** — live scores on parlay card or highlighted on scores tab
- [ ] **7. Exact score consensus indicator** — when models converge, show ✓ in table
- [x] **10. Add SOS sort option to teams page** *(done 2026-02-22)*
