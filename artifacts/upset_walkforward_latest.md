# Upset Model Walk-Forward Report

- generated_at: 2026-03-05T09:06:15.233884
- folds: 18
- oof_n: 964
- oof_accuracy: 0.8423236514522822
- oof_brier: 0.12624909652934338
- oof_logloss: 0.40996260880409924
- final_train_size: 1183

## Known limitation
Pregame Elo values are approximated using current elo_ratings table, not reconstructed true as-of Elo snapshots per historical game date.

```json
{
  "model": "upset_model",
  "generated_at": "2026-03-05T09:06:15.233884",
  "folds": 18,
  "min_warmup": 200,
  "oof_metrics": {
    "n": 964,
    "accuracy": 0.8423236514522822,
    "brier": 0.12624909652934338,
    "logloss": 0.40996260880409924
  },
  "final_train_size": 1183,
  "model_path": "/home/sam/college-baseball-predictor/data/upset_model.pkl",
  "known_limitation": "Pregame Elo values are approximated using current elo_ratings table, not reconstructed true as-of Elo snapshots per historical game date."
}
```