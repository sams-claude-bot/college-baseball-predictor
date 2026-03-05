# P1.2 Benchmark Diff (Baseline vs Post-P1.2)

- Generated: `2026-03-05T09:11:47`
- Baseline: `artifacts/model_benchmark_2026-03-05.md`
- Current: `artifacts/model_benchmark_post_p12_2026-03-05.md`

## Meta-ensemble delta (leak-safe cohort)

| Metric | Baseline | Current | Delta (current-baseline) |
|---|---:|---:|---:|
| accuracy | 0.6483 | 0.6483 | +0.0000 |
| brier | 0.2309 | 0.2309 | +0.0000 |
| log_loss | 0.6762 | 0.6762 | +0.0000 |
| ece | 0.1119 | 0.1119 | +0.0000 |

## Meta-ensemble delta (strict cohort)

| Metric | Baseline | Current | Delta (current-baseline) |
|---|---:|---:|---:|
| accuracy | 0.7308 | 0.7308 | +0.0000 |
| brier | 0.2167 | 0.2167 | +0.0000 |
| log_loss | 0.6749 | 0.6749 | +0.0000 |
| ece | 0.2277 | 0.2277 | +0.0000 |

## Top-5 ranking changes (leak-safe, by win accuracy)

| Rank | Baseline | Current | Changed? |
|---:|---|---|---|
| 1 | pitching (0.7230) | pitching (0.7230) | no |
| 2 | rest_travel (0.6857) | rest_travel (0.6857) | no |
| 3 | upset (0.6571) | upset (0.6571) | no |
| 4 | meta_ensemble (0.6483) | meta_ensemble (0.6483) | no |
| 5 | pear (0.6460) | pear (0.6460) | no |

## Correlation shifts relevant to stack quality

Meta-to-base pair shifts from strict-cohort top-correlation tables:

| Pair | Baseline corr | Current corr | Delta |
|---|---:|---:|---:|
| elo vs meta_ensemble | 0.9211 | 0.9211 | +0.0000 |
| meta_ensemble vs pear | 0.9070 | 0.9070 | +0.0000 |
| meta_ensemble vs upset | 0.9339 | 0.9339 | +0.0000 |

## Upset-model diagnostics (quick extension)

Proxy definition used for this diagnostic: upset event = home team loss (away win).
Predicted upset at threshold 0.5 when `upset predicted_home_prob < 0.5`.

| Cohort | n | Base upset rate | Majority-class baseline acc | Upset precision@0.5 | Upset recall@0.5 |
|---|---:|---:|---:|---:|---:|
| Leak-safe upset rows | 35 | 0.2857 | 0.7143 | 0.4000 | 0.4000 |
| Strict upset rows | 26 | 0.3846 | 0.6154 | 0.5714 | 0.4000 |
