# Unified Model Evaluation

- Date window: `2026-01-01` to `2026-12-31`
- Finalized games evaluated: `1050`
- Models: `poisson, ensemble, elo`
- Protocol: same game set, same date window, read-only evaluation

| Model | Win Accuracy | Brier | Log Loss | Totals MAE | n_games |
|---|---:|---:|---:|---:|---:|
| poisson | 0.7876 | 0.1470 | 0.4491 | 4.192 | 1050 |
| ensemble | 0.7962 | 0.1400 | 0.4296 | 4.287 | 1050 |
| elo | 0.7676 | 0.1634 | 0.5023 | 4.920 | 1050 |
