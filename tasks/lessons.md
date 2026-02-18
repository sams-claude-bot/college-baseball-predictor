# Lessons Learned

Rules to prevent repeated mistakes. Review at session start.

## Data Integrity
- **Always run `predict_and_track.py evaluate` before `accuracy`** — accuracy only displays, evaluate actually grades predictions (Feb 17: 792 predictions sat unevaluated for days)
- **Feature dimensions must match between training and prediction** — HistoricalFeatureComputer and FeatureComputer must produce the same number of features (Feb 17: XGB/LGB trained on 77 features but predicted with 81)
- **Don't backfill predictions on completed games** — creates data leakage. Models "know" outcomes. Only forward predictions are honest (Feb 17: backfilled XGB/LGB on finished games, had to wipe)
- **`g.id NOT IN (SELECT DISTINCT game_id FROM model_predictions)` skips partially-predicted games** — if 10/12 models predicted a game, the other 2 never get a chance. Check per-model, not per-game (Feb 17)

## Cron Jobs
- **Sub-agents report "ok" when tables are empty** — always include SQL verification queries with exact row counts (Feb 17)
- **Silent failures are the worst failures** — a job that runs successfully but inserts 0 rows looks fine in monitoring. Always check output counts.

## Model Training
- **All trainable models use the same data split** — train on data > 7 days old, validate on last 7 days
- **Feature dimension: 81** (FeatureComputer with use_model_predictions=False). Historical and live must match.
- **Neural net uses 2-phase training** — base at lr=0.001, fine-tune at lr=0.0001. Only save fine-tuned if it beats base.
