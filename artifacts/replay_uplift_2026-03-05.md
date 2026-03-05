# P1.3 Trusted Replay Uplift Benchmark

- Generated: `2026-03-05T10:31:02`
- Date range: `2026-02-20` .. `2026-03-05`

## Cohort definition

Leak-safe cohort filters (P0 semantics):
- final games only with known winner and scores
- exclude `prediction_source='backfill'`
- enforce pregame timestamp cutoff (`predicted_at <= game_start - 5m`, fallback end-of-date)
- baseline and candidate evaluated on the same game IDs (stored `meta_ensemble` rows)

| Cohort stat | Value |
|---|---:|
| raw rows in date range | 14512 |
| leak-safe kept rows | 8505 |
| excluded backfill | 3225 |
| excluded disallowed source | 0 |
| excluded late/non-parseable timestamp | 2782 |
| trusted benchmark games (`n`) | 634 |

## Trusted uplift

Baseline = stored pregame `meta_ensemble`; candidate = current meta model replayed on stored base probabilities only.

| Metric | Baseline stored meta | Candidate replayed meta | Delta (candidate-baseline) |
|---|---:|---:|---:|
| n | 634 | 634 | +0 |
| accuracy | 0.6483 | 0.7776 | +0.1293 |
| brier | 0.2309 | 0.1545 | -0.0764 |
| log_loss | 0.6762 | 0.4742 | -0.2020 |
| ece | 0.1119 | 0.0582 | -0.0536 |

## Flip analysis

| Metric | Value |
|---|---:|
| side flips count | 152 |
| net correct change from flips | 82 |

## Interpretation / confidence limits

- This is a replay-style observational benchmark, not a randomized experiment.
- Trusted uplift isolates only meta-layer changes because base probabilities are held fixed from stored pregame rows.
- Sample size and date-window composition can dominate deltas; confidence is limited for small `n` or low flip counts.
- Exploratory shadow results (if present) are intentionally non-trusted and directional only.
