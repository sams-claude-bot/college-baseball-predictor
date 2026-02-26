# College Baseball Predictor

NCAA D1 college baseball prediction system with 14 win-probability models, 5 totals models, automated data pipelines, betting analytics, and a live web dashboard.

**Live:** [baseball.mcdevitt.page](https://baseball.mcdevitt.page)  
**Season:** 2026 (Feb 14 – June 22, CWS in Omaha)

## Models (14 Win Probability + Meta-Ensemble)

| Model | Type | Accuracy | Description |
|-------|------|----------|-------------|
| **PEAR** | Rating | **76.1%** | Power, Experience, Adjusted Rating composite |
| **Quality** | Statistical | **72.1%** | Pitching + batting quality matchup model |
| Neural | ML | 67.4% | PyTorch NN, 81 features, 2-phase training |
| Elo | Rating | 67.3% | Chess-style, updated per game |
| Prior | Bayesian | 67.2% | Preseason rankings + program history |
| Ensemble | Blend | 66.7% | Dynamic weighted combination of all base models |
| LightGBM | ML | 65.8% | Gradient boosting, 81 features |
| Pythagorean | Formula | 65.8% | Runs scored/allowed expectation |
| Conference | Statistical | 64.6% | Conference strength adjustments |
| Poisson | Statistical | 64.3% | Run distribution with weather adjustments |
| XGBoost | ML | 64.3% | Gradient boosting, 81 features |
| Pitching | Statistical | 63.8% | Staff quality tables (ace, rotation, bullpen depth) |
| Advanced | Statistical | 63.6% | Opponent-adjusted stats, recency-weighted |
| Log5 | Formula | 63.6% | Bill James head-to-head |
| **Meta-Ensemble** | Stacking | **76.7%** | LogReg over all 14 models (walk-forward validated) |

Plus 5 totals models (runs_ensemble 67.9%, runs_poisson 67.2%, runs_advanced 66.9%, nn_slim_totals 64.7%, runs_pitching 51.1%) and a weather adjustment model.

## Features

- **Live scores** — In-progress games update every 15 minutes with current inning
- **12 prediction models** — Statistical, ML, and ensemble approaches
- **Stored predictions** — Sub-100ms page loads (predictions pre-computed daily)

## Data Sources

- **D1Baseball** — Scores, schedules, box scores, player stats (basic + advanced), rankings
- **DraftKings** — Betting lines (moneyline, spreads, totals)
- **ESPN** — Legacy schedule backbone (being phased out)
- **Open-Meteo** — Weather forecasts

## Web Dashboard

Flask app serving 12 pages: Dashboard, Scores, Betting, Teams, Team Detail, Game Detail, Predict, Rankings, Standings, Models, Calendar, Tracker.

API: `/api/best-bets`, `/api/predict`, `/api/runs`, `/api/teams`

## Stack

- Python 3, SQLite (WAL), Flask, Playwright
- PyTorch, XGBoost, LightGBM
- OpenClaw cron jobs for automation
- Cloudflare Tunnel for public access

## Architecture: Schedule Gateway

All schedule writes (creates, scores, finalize, postpone) route through `scripts/schedule_gateway.py` — a single write path with:
- Deterministic game ID generation
- Multi-strategy dedup (exact, legacy suffix, swapped H/A, fuzzy)
- Status hierarchy enforcement (final > in-progress > scheduled)
- Ghost replacement with FK migration across 16 tables
- Structured audit logging

## Structure

```
├── models/          # 23 model files (win prob, runs, features, weather)
├── scripts/         # Data collection, training, predictions, betting
│   └── schedule_gateway.py  # Single write path to games table
├── web/             # Flask app + 16 Jinja2 templates
├── config/          # Team ID mappings (D1BB slugs, ESPN IDs)
├── tasks/           # Lessons learned, task tracking
├── data/            # Database, model weights, configs (local only, not in repo)
└── CONTEXT.md       # Full project documentation
```

## Quick Start

```bash
# Predictions
PYTHONPATH=. python3 scripts/predict_and_track.py predict

# Model accuracy
PYTHONPATH=. python3 scripts/predict_and_track.py accuracy

# Start dashboard
python3 -m flask run --host=0.0.0.0 --port=5000

# Train all models
PYTHONPATH=. python3 scripts/train_all_models.py
```

See `CONTEXT.md` for full documentation — data pipeline, cron schedule, database schema, model details, betting system.

## Betting Risk Engine v1 (Selection Flow)

`scripts/bet_selection_v2.py` now supports a config-driven risk engine with:

- Legacy fixed stake mode (`BET_RISK_ENGINE_MODE = "fixed"`, default)
- Fractional Kelly mode (`"fractional_kelly"`) with bankroll/min/max stake caps
- Drawdown-aware Kelly throttle
- Correlation exposure caps (team / conference / day buckets)

Risk knobs live in `config/model_config.py`. Recommendation outputs include `risk_score`, `kelly_fraction_used`, `suggested_stake`, and `exposure_bucket`.

## Documentation Status

- `README.md` (this file) is for overview + quickstart only.
- `CONTEXT.md` is the canonical operational source of truth (cron flow, runtime behavior, deployment notes).
- `MANIFEST.md` is the canonical path/classification inventory (what files are active, cron-adjacent, one-shot, archived).
- `docs/DASHBOARD.md` is the dashboard route/data dependency reference only.
- Cleanup planning and status tracking:
  - `docs/CLEANUP_AUDIT_2026-02-22.md`
  - `docs/CLEANUP_CHECKLIST.md`
  - `docs/CLEANUP_PROGRESS_2026-02-22.md`
