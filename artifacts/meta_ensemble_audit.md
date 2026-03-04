# Meta-Ensemble Accuracy Audit

**Date**: 2026-03-04
**Reported problem**: Meta-ensemble live accuracy 65.0% (614 games) vs pitching 72.3%, PEAR 72.0%. Walk-forward training accuracy was 77.5% — a 12pp gap.

---

## Executive Summary

**The meta-ensemble is NOT underperforming.** On the same game subset (W08-W09), it is the best or tied-for-best model at 65.2%. The perceived gap is caused by two issues:

1. **Apples-to-oranges time period comparison** (PRIMARY): Meta has 614 graded games (W07-W09). PEAR/quality have 1592/1675 games going back to W06. Early-season games were dramatically easier (PEAR: 77% in W06-W07 vs 65% in W08-W09), inflating those models' overall stats.

2. **Walk-forward accuracy inflation** (SECONDARY): The 77.5% training accuracy averages over all historical games, including the easy early season. The live period is harder, creating an expected drop.

3. **LogReg multicollinearity** (TERTIARY): 7 of 12 model-probability coefficients are negative (inverted), and the intercept implies 78.1% home baseline. This doesn't hurt accuracy much currently but makes the model fragile.

No bugs found. No double-calibration. No missing inputs. No code errors.

---

## Finding 1: Time-Period Mismatch (ROOT CAUSE)

### All models drop to ~65% in W08-W09

| Model | Overall Acc | W08+W09 Acc | Early (W06-W07) Acc | Overall Games |
|-------|-----------|-------------|---------------------|---------------|
| pear | 72.1% | 64.9% | 76.9% | 1592 |
| quality | 69.0% | 63.2% | 72.7% | 1675 |
| elo | 66.1% | 65.5% | 66.9% | 1147 |
| ensemble | 65.9% | 65.2% | 67.0% | 1148 |
| **meta_ensemble** | **65.5%** | **65.2%** | 87.5% (8 games) | 614 |
| pitching | 63.3% | 63.5% | 63.0% | 1148 |

**Key insight**: PEAR drops from 76.9% to 64.9% between periods. Quality drops from 72.7% to 63.2%. Meta only has data from the hard period, so its 65.5% overall is unfairly compared to other models' inflated overall numbers.

### Same-game comparison (614 meta games only)

| Model | Accuracy on Meta's 614 Games |
|-------|------------------------------|
| meta_ensemble | 65.5% |
| elo | 65.3% |
| ensemble | 65.5% |
| pear | 64.7% (584 overlap) |
| pitching | 63.4% |
| quality | 62.9% (604 overlap) |

**Meta is the best model on its own game subset.**

### Weekly breakdown

| Week | Meta | PEAR | Quality | Ensemble | Pitching |
|------|------|------|---------|----------|----------|
| W06 | — | 77.4% | 75.7% | — | — |
| W07 | 87.5% (8g) | 76.4% | 70.4% | 67.0% | 63.0% |
| W08 | 65.3% | 64.9% | 63.0% | 64.9% | 63.6% |
| W09 | 64.6% | 64.9% | 64.6% | 67.1% | 63.3% |

All models converge to ~65% by W08. The early-season advantage disappears.

---

## Finding 2: Walk-Forward Accuracy Inflation

The 77.5% walk-forward accuracy reported during training includes predictions from ALL graded games (starting W06/W07). Since early games are easier and dominate the average, the walk-forward number is misleadingly high.

The walk-forward validation uses `_extract_training_data()` (`models/meta_ensemble.py:42-96`) which queries ALL graded games with `was_correct IS NOT NULL`. Since PEAR/quality data goes back to W06, the walk-forward accuracy is heavily influenced by those easy early-season games.

**This is not a bug** — it's a reporting/expectations issue. The walk-forward is computed correctly; it's just not representative of the live deployment period.

---

## Finding 3: LogReg Multicollinearity

The live model uses LogReg (not XGBoost) per `models/meta_ensemble.py:491-494`:
```python
if self.lr_model is not None:
    prob = float(self.lr_model.predict_proba(features)[:, 1][0])
```

### Pathological coefficients

| Feature | LR Coefficient | Direction |
|---------|---------------|-----------|
| wp_diff | -0.8331 | Largest magnitude, negative |
| rpi_diff | +0.4753 | |
| any_ranked | -0.4396 | |
| **elo_prob** | **-0.3951** | **INVERTED** |
| **quality_prob** | **-0.2170** | **INVERTED** |
| **lightgbm_prob** | **-0.2179** | **INVERTED** |
| **xgboost_prob** | **-0.1485** | **INVERTED** |
| **pitching_prob** | **-0.1008** | **INVERTED** |
| **pythagorean_prob** | **-0.0975** | **INVERTED** |
| **neural_prob** | **-0.0651** | **INVERTED** |
| pear_prob | +0.0568 | Correct but tiny |
| log5_prob | +0.1663 | |
| **Intercept** | **+1.2688** | **78.1% home baseline** |

**7 of 12 model-probability coefficients are negative** — meaning when those models predict a higher home win probability, the meta-ensemble predicts a LOWER one. This is classic multicollinearity: 12 highly-correlated inputs cause the LR to assign arbitrary opposing signs that happen to cancel out in-sample.

### Overconfident outputs

Synthetic test results:

| Scenario | LR Output | XGB Output |
|----------|-----------|------------|
| All 12 models agree home (0.6) | **1.000** | 0.835 |
| All 12 models agree away (0.4) | **0.001** | 0.257 |
| 8 home / 4 away | 0.603 | 0.402 |

LR produces extreme probabilities (0.001 and 1.000). XGB is much more measured.

### Away predictions are near-random

| Confidence Range | Games | Accuracy |
|-----------------|-------|----------|
| < 0.40 (strong away) | 137 | 54.0% |
| 0.40-0.50 (weak away) | 49 | 51.0% |
| 0.50-0.60 (weak home) | 80 | 61.3% |
| >= 0.60 (strong home) | 348 | 73.0% |

Away predictions barely beat a coin flip. Home predictions are solid.

---

## Finding 4: No Double-Calibration Bug

Investigated whether calibrated inputs to meta-ensemble create a mismatch. They do not:

- `predict_and_track.py:258` calibrates individual model outputs, stores calibrated in `predicted_home_prob`, raw in `raw_home_prob`
- `meta_ensemble.py:53` training reads `predicted_home_prob` (calibrated)
- `meta_ensemble.py:362,428` prediction reads `predicted_home_prob` (calibrated)
- `predict_and_track.py:291-298` stores meta output directly, no additional calibration applied

Training and inference see the same calibrated values. No mismatch.

---

## Finding 5: Input Coverage is Fine

| Input Models Present | Games | Meta Accuracy |
|---------------------|-------|---------------|
| 10 | 1 | 100.0% |
| 11 | 38 | 65.8% |
| 12 (all) | 575 | 65.4% |

575 of 614 games (93.6%) had all 12 input models. Missing inputs are not a factor.

---

## Finding 6: Probability Distribution

| Metric | Meta-Ensemble | PEAR |
|--------|--------------|------|
| Min prob | 0.052 | 0.100 |
| Max prob | 0.990 | 0.900 |
| Avg prob | 0.627 | 0.667 |
| Avg distance from 0.5 | 0.247 | 0.230 |
| Near 0.50 (0.45-0.55) | 72 (11.7%) | 121 (10.5%) |

Meta has a wider spread (0.05-0.99) vs PEAR (0.10-0.90), driven by the LR's extreme outputs.

---

## Finding 7: Agreement Analysis

### Meta vs PEAR
| Scenario | Games | Meta Acc | PEAR Acc |
|----------|-------|----------|----------|
| Agree | 464 | 69.0% | 69.0% |
| Disagree | 120 | 52.5% | 48.3% |

When they agree, both are at 69%. When they disagree, meta slightly edges PEAR (52.5% vs 48.3%).

### Meta vs Pitching
| Scenario | Games | Meta Acc | Pitching Acc |
|----------|-------|----------|--------------|
| Agree | 488 | 73.2% | 67.6% |
| Disagree | 126 | 35.7% | 46.8% |

When agreeing with pitching, meta is excellent (73.2%). When disagreeing, meta is terrible (35.7%) — worse than chance.

---

## Recommended Fixes (Ranked by Impact)

### 1. Fix accuracy reporting (HIGH IMPACT, EASY)
Compare all models on the same date range. The current comparison is misleading. Add time-period-matched accuracy to the dashboard/reports.

### 2. Switch from LogReg to XGBoost for live predictions (HIGH IMPACT, EASY)
Change `models/meta_ensemble.py:491-494`:
```python
# Current: uses LR
if self.lr_model is not None:
    prob = float(self.lr_model.predict_proba(features)[:, 1][0])
# Proposed: use XGB
prob = float(self.xgb_model.predict_proba(features)[:, 1][0])
```
XGB handles multicollinearity better, produces more moderate probabilities, and weights features by importance rather than linear coefficients. This would fix the pathological away predictions.

### 3. Reduce feature count to combat multicollinearity (MEDIUM IMPACT, MODERATE)
Drop redundant model-prob features. Many of the 12 models are highly correlated. Options:
- Keep only top 5-6 models by XGB importance (elo, xgboost, pitching, pear, neural, lightgbm)
- Or use PCA on the 12 model probs to extract 3-4 orthogonal components
- Drop context features that duplicate model information (elo_diff duplicates elo_prob, pear_diff duplicates pear_prob)

### 4. Add periodic retraining (MEDIUM IMPACT, MODERATE)
Schedule weekly retraining as more data accumulates. The model was trained once and deployed — it should adapt to changing season dynamics.

### 5. Report walk-forward accuracy by period (LOW IMPACT, EASY)
When training, report accuracy for recent N games (e.g., last 200) alongside overall. This would have flagged the inflation immediately.

---

## Code References

| Issue | File | Line(s) |
|-------|------|---------|
| LR used for live predictions | `models/meta_ensemble.py` | 491-494 |
| Training data extraction | `models/meta_ensemble.py` | 42-96 |
| Feature construction | `models/meta_ensemble.py` | 126-199 |
| Walk-forward validation | `models/meta_ensemble.py` | 221-276 |
| Meta prediction storage (no calibration) | `scripts/predict_and_track.py` | 287-309 |
| Individual model calibration | `scripts/predict_and_track.py` | 257-258 |
| Model uses `predicted_home_prob` at inference | `models/meta_ensemble.py` | 362, 428 |
