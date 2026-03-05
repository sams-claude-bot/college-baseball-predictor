# Model Benchmark (Leak-Safe Meta Stack)

- Generated: `2026-03-05T09:10:32`
- Date window: `2026-01-01` to `2026-12-31`

## Data/Cohort definition

Final games with known winner. Prediction rows are leak-safe only:
- exclude `prediction_source='backfill'`
- include only `prediction_source IN ('live','refresh')` or NULL/empty (legacy-live)
- require `predicted_at <= (game_datetime - 5 minutes)`; fallback cutoff = `game_date 23:54:59` when game time missing

### Filter counts

- Total candidate prediction rows: **23495**
- Excluded backfill rows: **4944**
- Excluded disallowed source rows: **0**
- Excluded late/invalid timestamp rows: **5316**
- Kept leak-safe rows: **13235**

## Leak-safe per-model leaderboard

| Model | n predictions | Win accuracy | Brier | Log loss | ECE |
|---|---:|---:|---:|---:|---:|
| meta_ensemble | 634 | 0.6483 | 0.2309 | 0.6762 | 0.1119 |
| elo | 946 | 0.6459 | 0.2158 | 0.6219 | 0.0248 |
| pythagorean | 946 | 0.6342 | 0.2288 | 0.6556 | 0.0528 |
| lightgbm | 944 | 0.6356 | 0.2363 | 0.6840 | 0.0701 |
| poisson | 946 | 0.6247 | 0.2380 | 0.7257 | 0.0852 |
| xgboost | 944 | 0.6186 | 0.2417 | 0.7043 | 0.0903 |
| pitching | 946 | 0.7230 | 0.1842 | 0.5458 | 0.0322 |
| pear | 565 | 0.6460 | 0.2196 | 0.6348 | 0.0860 |
| quality | 586 | 0.6195 | 0.2303 | 0.6655 | 0.0823 |
| neural | 943 | 0.6320 | 0.2299 | 0.6573 | 0.0552 |
| venue | 35 | 0.6286 | 0.2227 | 0.6354 | 0.1094 |
| rest_travel | 35 | 0.6857 | 0.1897 | 0.5601 | 0.1125 |
| upset | 35 | 0.6571 | 0.2344 | 0.6881 | 0.1461 |

### Legacy models present (leak-safe cohort)

| Model | n predictions | Win accuracy | Brier | Log loss | ECE |
|---|---:|---:|---:|---:|---:|
| advanced | 946 | 0.6184 | 0.2486 | 0.7520 | 0.0972 |
| conference | 946 | 0.6131 | 0.2374 | 0.6928 | 0.0922 |
| ensemble | 946 | 0.6406 | 0.2247 | 0.6527 | 0.0559 |
| log5 | 946 | 0.6184 | 0.2477 | 0.7636 | 0.0953 |
| prior | 946 | 0.6374 | 0.2198 | 0.6324 | 0.0394 |

## Strict cohort leaderboard

- Strict cohort size (games with predictions from all active models + meta): **26**
- Strict cohort date range: **2026-03-04 to 2026-03-04**

| Model | n predictions | Win accuracy | Brier | Log loss | ECE |
|---|---:|---:|---:|---:|---:|
| meta_ensemble | 26 | 0.7308 | 0.2167 | 0.6749 | 0.2277 |
| elo | 26 | 0.6923 | 0.2151 | 0.6273 | 0.1890 |
| pythagorean | 26 | 0.6154 | 0.2106 | 0.6072 | 0.0615 |
| lightgbm | 26 | 0.6154 | 0.2044 | 0.5951 | 0.1302 |
| poisson | 26 | 0.6154 | 0.2312 | 0.6563 | 0.1473 |
| xgboost | 26 | 0.6154 | 0.2032 | 0.5949 | 0.1909 |
| pitching | 26 | 0.5769 | 0.2414 | 0.6697 | 0.2577 |
| pear | 26 | 0.6538 | 0.2077 | 0.6022 | 0.1585 |
| quality | 26 | 0.4615 | 0.2475 | 0.7103 | 0.2989 |
| neural | 26 | 0.6154 | 0.2184 | 0.6244 | 0.0549 |
| venue | 26 | 0.5000 | 0.2576 | 0.7082 | 0.1820 |
| rest_travel | 26 | 0.6154 | 0.2209 | 0.6321 | 0.1142 |
| upset | 26 | 0.6538 | 0.2509 | 0.7414 | 0.1880 |

## Calibration table

Reliability bins on strict cohort (10 bins):

| Model | Bin range | n | Avg predicted home win | Actual home win | Gap |
|---|---|---:|---:|---:|---:|
| meta_ensemble | [0.0, 0.1) | 2 | 0.0766 | 1.0000 | +0.9234 |
|  | [0.1, 0.2) | 1 | 0.1323 | 0.0000 | -0.1323 |
|  | [0.2, 0.3) | 2 | 0.2616 | 0.0000 | -0.2616 |
|  | [0.3, 0.4) | 1 | 0.3693 | 0.0000 | -0.3693 |
|  | [0.4, 0.5) | 3 | 0.4506 | 0.3333 | -0.1172 |
|  | [0.5, 0.6) | 3 | 0.5552 | 0.6667 | +0.1114 |
|  | [0.6, 0.7) | 4 | 0.6693 | 0.5000 | -0.1693 |
|  | [0.7, 0.8) | 1 | 0.7005 | 1.0000 | +0.2995 |
|  | [0.8, 0.9) | 4 | 0.8341 | 1.0000 | +0.1659 |
|  | [0.9, 1.0] | 5 | 0.9444 | 0.8000 | -0.1444 |
| elo | [0.3, 0.4) | 2 | 0.3939 | 1.0000 | +0.6061 |
|  | [0.4, 0.5) | 10 | 0.4801 | 0.3000 | -0.1801 |
|  | [0.6, 0.7) | 7 | 0.6759 | 0.7143 | +0.0384 |
|  | [0.7, 0.8) | 3 | 0.7645 | 1.0000 | +0.2355 |
|  | [0.8, 0.9) | 1 | 0.8000 | 1.0000 | +0.2000 |
|  | [0.9, 1.0] | 3 | 0.9088 | 0.6667 | -0.2421 |
| pythagorean | [0.5, 0.6) | 13 | 0.5275 | 0.4615 | -0.0660 |
|  | [0.6, 0.7) | 3 | 0.6294 | 0.6667 | +0.0373 |
|  | [0.7, 0.8) | 10 | 0.7372 | 0.8000 | +0.0628 |
| lightgbm | [0.5, 0.6) | 8 | 0.5334 | 0.2500 | -0.2834 |
|  | [0.6, 0.7) | 11 | 0.6464 | 0.7273 | +0.0809 |
|  | [0.8, 0.9) | 7 | 0.8246 | 0.8571 | +0.0326 |
| poisson | [0.5, 0.6) | 9 | 0.5251 | 0.4444 | -0.0807 |
|  | [0.6, 0.7) | 4 | 0.6442 | 1.0000 | +0.3558 |
|  | [0.7, 0.8) | 9 | 0.7175 | 0.5556 | -0.1620 |
|  | [0.8, 0.9) | 4 | 0.8054 | 0.7500 | -0.0554 |
| xgboost | [0.5, 0.6) | 9 | 0.5347 | 0.2222 | -0.3125 |
|  | [0.6, 0.7) | 8 | 0.6460 | 0.8750 | +0.2290 |
|  | [0.7, 0.8) | 8 | 0.7654 | 0.7500 | -0.0154 |
|  | [0.8, 0.9) | 1 | 0.8044 | 1.0000 | +0.1956 |
| pitching | [0.2, 0.3) | 2 | 0.2445 | 0.5000 | +0.2555 |
|  | [0.3, 0.4) | 3 | 0.3653 | 1.0000 | +0.6347 |
|  | [0.4, 0.5) | 4 | 0.4235 | 0.2500 | -0.1735 |
|  | [0.5, 0.6) | 3 | 0.5467 | 0.3333 | -0.2133 |
|  | [0.6, 0.7) | 3 | 0.6523 | 0.3333 | -0.3190 |
|  | [0.7, 0.8) | 3 | 0.7610 | 1.0000 | +0.2390 |
|  | [0.8, 0.9) | 6 | 0.8463 | 0.6667 | -0.1797 |
|  | [0.9, 1.0] | 2 | 0.9000 | 1.0000 | +0.1000 |
| pear | [0.2, 0.3) | 3 | 0.2500 | 0.6667 | +0.4167 |
|  | [0.3, 0.4) | 2 | 0.3387 | 0.0000 | -0.3387 |
|  | [0.4, 0.5) | 6 | 0.4359 | 0.5000 | +0.0641 |
|  | [0.5, 0.6) | 4 | 0.5741 | 0.5000 | -0.0741 |
|  | [0.6, 0.7) | 1 | 0.6961 | 0.0000 | -0.6961 |
|  | [0.7, 0.8) | 1 | 0.7914 | 1.0000 | +0.2086 |
|  | [0.8, 0.9) | 1 | 0.8306 | 1.0000 | +0.1694 |
|  | [0.9, 1.0] | 8 | 0.9299 | 0.8750 | -0.0549 |
| quality | [0.4, 0.5) | 6 | 0.4230 | 0.8333 | +0.4104 |
|  | [0.5, 0.6) | 4 | 0.5283 | 0.0000 | -0.5283 |
|  | [0.6, 0.7) | 6 | 0.6154 | 0.5000 | -0.1154 |
|  | [0.7, 0.8) | 2 | 0.7486 | 1.0000 | +0.2514 |
|  | [0.8, 0.9) | 3 | 0.8625 | 1.0000 | +0.1375 |
|  | [0.9, 1.0] | 5 | 0.9175 | 0.6000 | -0.3175 |
| neural | [0.5, 0.6) | 16 | 0.5552 | 0.5000 | -0.0552 |
|  | [0.7, 0.8) | 10 | 0.7454 | 0.8000 | +0.0546 |
| venue | [0.3, 0.4) | 1 | 0.3860 | 1.0000 | +0.6140 |
|  | [0.4, 0.5) | 2 | 0.4705 | 1.0000 | +0.5295 |
|  | [0.6, 0.7) | 15 | 0.6568 | 0.4667 | -0.1901 |
|  | [0.7, 0.8) | 8 | 0.7241 | 0.7500 | +0.0259 |
| rest_travel | [0.4, 0.5) | 2 | 0.4825 | 0.5000 | +0.0175 |
|  | [0.5, 0.6) | 4 | 0.5707 | 0.2500 | -0.3207 |
|  | [0.6, 0.7) | 12 | 0.6422 | 0.5833 | -0.0588 |
|  | [0.7, 0.8) | 8 | 0.7570 | 0.8750 | +0.1180 |
| upset | [0.0, 0.1) | 1 | 0.0710 | 1.0000 | +0.9290 |
|  | [0.1, 0.2) | 1 | 0.1380 | 1.0000 | +0.8620 |
|  | [0.2, 0.3) | 5 | 0.2710 | 0.2000 | -0.0710 |
|  | [0.7, 0.8) | 8 | 0.7299 | 0.5000 | -0.2299 |
|  | [0.8, 0.9) | 4 | 0.8535 | 0.7500 | -0.1035 |
|  | [0.9, 1.0] | 7 | 0.9269 | 0.8571 | -0.0697 |

## Correlation findings

Top 10 highest absolute pairwise correlations (strict cohort):

| Model A | Model B | Correlation | abs(corr) |
|---|---|---:|---:|
| meta_ensemble | upset | 0.9339 | 0.9339 |
| meta_ensemble | elo | 0.9211 | 0.9211 |
| meta_ensemble | pear | 0.9070 | 0.9070 |
| pitching | quality | 0.8965 | 0.8965 |
| lightgbm | xgboost | 0.8900 | 0.8900 |
| elo | quality | 0.8811 | 0.8811 |
| pythagorean | pitching | 0.8763 | 0.8763 |
| pythagorean | quality | 0.8699 | 0.8699 |
| elo | pear | 0.8645 | 0.8645 |
| elo | pitching | 0.8599 | 0.8599 |

Meta disagreement analysis (strict cohort):

| Meta vs model | Agreement rate | Meta accuracy when agree | Meta accuracy when disagree |
|---|---:|---:|---:|
| upset | 0.9231 | 0.7083 | 1.0000 |
| pitching | 0.8462 | 0.6818 | 1.0000 |
| pear | 0.8462 | 0.7273 | 0.7500 |
| venue | 0.6154 | 0.6875 | 0.8000 |
| rest_travel | 0.6538 | 0.7647 | 0.6667 |

## Risks/interpretation notes

- Strict cohort can be much smaller than leak-safe per-model cohort; this may change rank ordering.
- High correlation means less independent signal and limits stacking upside.
- ECE is sample-size sensitive; sparse bins can look noisy on small cohorts.
- Accuracy alone can hide confidence miscalibration, so Brier/log loss/ECE should be considered together.

## Recommended next experiments

- Re-rank meta base-feature set using strict-cohort incremental gain vs high-correlation redundancy.
- Compare meta calibration before/after isotonic/Platt post-calibration on strict cohort only.
- Stress-test upset/pitching/venue contributions on disagreement-only slices.
- Add time-slice stability (weekly rolling strict-cohort metrics) before any hyperparameter tuning.
- Audit games dropped from strict cohort to identify model coverage gaps by source/date.
