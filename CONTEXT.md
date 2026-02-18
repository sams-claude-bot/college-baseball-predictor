# College Baseball Predictor — Project Context

> **Read this first.** This is the single source of truth for understanding the project.

## What This Is

NCAA D1 college baseball prediction system with a web dashboard. Collects data, runs 13 models, tracks betting P&L, serves predictions at [baseball.mcdevitt.page](https://baseball.mcdevitt.page).

- **Season:** Feb 14 – June 22, 2026 (CWS in Omaha)
- **Focus:** Mississippi State + Auburn (featured), all SEC, all Power 4, Top 25, full D1 scores
- **Stack:** Python, SQLite, Flask, Playwright, OpenClaw cron jobs
- **Dashboard:** `college-baseball-dashboard.service` on port 5000

## Data Sources

| Source | What | How | Priority |
|--------|------|-----|----------|
| **D1Baseball** | Scores, schedules, box scores, player stats, rankings, advanced stats | Playwright browser scrape | **PRIMARY for everything** |
| **DraftKings** | Betting lines (ML, spreads, totals) | Playwright browser scrape | Primary for odds |
| **ESPN** | Legacy schedule backbone (future games) | API | Being replaced by D1BB as its 7-day window advances |
| **Open-Meteo** | Game weather forecasts | API | Primary for weather |

### ⚠️ Critical Rules
- **D1Baseball is the source of truth** for scores, schedules, and stats
- **DO NOT scrape team athletics sites for schedules** — causes duplicates
- **ESPN future games** are kept as long-range calendar but get replaced when D1BB's sliding window reaches them (dedup logic in `d1bb_schedule.py`)
- Team name mismatches between sources handled via `team_aliases` table

## Database

**Location:** `data/baseball.db` (SQLite, WAL mode, ~20MB)

### Key Tables
| Table | Rows | Purpose |
|-------|------|---------|
| `games` | ~2000 | All games (scheduled + completed) |
| `teams` | 407 | D1 teams with conference, rank |
| `team_aliases` | 700 | Cross-source team name mapping (DK, ESPN, D1BB) |
| `player_stats` | 10k+ | Per-player batting/pitching stats with advanced metrics |
| `model_predictions` | 2000+ | Pre-game predictions from all models, evaluated post-game |
| `elo_ratings` | 315 | Current Elo rating per team |
| `betting_lines` | varies | DraftKings odds per game |
| `tracked_bets` | varies | ML bet tracking with P&L |
| `tracked_bets_spreads` | varies | Spread/total bet tracking |
| `tracked_confident_bets` | varies | High-consensus bet tracking (v2) |
| `game_weather` | 400+ | Weather forecasts for upcoming games |
| `power_rankings` | 382 | Weekly model-generated power rankings |
| `rankings_history` | 50 | D1Baseball Top 25 poll history |

### Game ID Format
`YYYY-MM-DD_away-team_home-team` (e.g. `2026-02-17_cincinnati_auburn`)
Doubleheaders: `_g1`, `_g2` suffixes.

## Models (13 total)

### Win Probability Models (10)
| Model | What It Does |
|-------|-------------|
| `neural` | PyTorch NN, 88% accuracy — **best model** |
| `ensemble` | Weighted blend of all component models |
| `prior` | Preseason rankings + program history (Bayesian) |
| `elo` | Chess-style ratings updated per game |
| `advanced` | Opponent-adjusted stats, recency-weighted |
| `conference` | Conference strength adjustments |
| `log5` | Bill James head-to-head formula |
| `poisson` | Run distribution modeling |
| `pythagorean` | Runs scored/allowed expectation |
| `pitching` | ERA, WHIP, K rates, bullpen depth |
| `momentum` | Post-ensemble modifier (±5% based on last 5-7 games) |

**Ensemble blend:** Neural 60% / Traditional Ensemble 40% (early season weighting)

### Run Projection Models (3+)
| Model | Purpose |
|-------|---------|
| `nn_totals` | Neural net for over/under |
| `nn_spread` | Neural net for run line spreads |
| `nn_dow_totals` | Day-of-week adjusted totals |
| `runs_ensemble` | Weighted blend of runs models |

### Weather Model
Adjusts run projections based on temperature, wind, humidity. Coefficients in `data/weather_coefficients.json`.

## Betting System

### v2 Selection Logic (`bet_selection_v2.py`)
- Prioritizes **consensus bets** (7+/10 models agree)
- **Spreads disabled** (0/5 historical, model not calibrated)
- Higher edge thresholds: 8% favorites, 15% underdogs
- Max 3 bets/day, Kelly-adjusted sizing
- Flat $100 per bet for tracking

### P&L Tables
- `tracked_bets` — Moneyline bets
- `tracked_bets_spreads` — Spread and total bets
- `tracked_confident_bets` — High-consensus bets (v2, started Feb 17)

### P&L starts Feb 16, 2026 (Sam's directive — no Feb 15 data)

## Cron Schedule (OpenClaw)

| Time | Job | What |
|------|-----|------|
| Every 15 min, 12-11 PM | Score Updates | `d1bb_schedule.py --today` |
| 1 AM daily | Nightly D1 Stats + Schedule | D1BB schedule (7 days) + player stats (all D1) |
| 2 AM daily | Nightly Scores + Pipeline | D1BB box scores → evaluate bets → update Elo → predictions |
| 8 AM daily | DraftKings Odds + Weather | Browser scrape odds, fetch weather |
| 9 AM daily | Record Best Bets | `bet_selection_v2.py record` |
| Mon 12 PM | Power Rankings | `power_rankings.py --top 25 --store` |
| Mon 10 PM | D1Baseball Rankings | Browser scrape Top 25 poll |
| Sun 9:30 PM | NN Fine-Tuning | `finetune_weekly.py` (2-week delayed training) |
| Sun 10 PM | Weekly Accuracy Report | Full model comparison + P&L summary |

## Key Scripts

### Daily Operations
```bash
python3 scripts/d1bb_schedule.py --today              # Live score updates
python3 scripts/d1bb_schedule.py --days 7              # Schedule sync (next 7 days)
python3 scripts/d1bb_box_scores.py --date YYYY-MM-DD   # Box scores + game creation
python3 scripts/d1bb_scraper.py --all-d1 --delay 2     # All D1 player stats
python3 scripts/record_daily_bets.py evaluate           # Grade completed bets
python3 scripts/update_elo.py --date YYYY-MM-DD         # Elo ratings
python3 scripts/aggregate_team_stats.py                 # Team aggregates
python3 scripts/weather.py fetch --upcoming             # Weather for next 3 days
PYTHONPATH=. python3 scripts/predict_and_track.py predict   # Generate predictions
PYTHONPATH=. python3 scripts/predict_and_track.py accuracy  # Model accuracy report
python3 scripts/bet_selection_v2.py record              # Record today's best bets
python3 scripts/backup_db.py                            # Database backup
```

### Weekly
```bash
python3 scripts/power_rankings.py --top 25 --store     # Power rankings
python3 scripts/finetune_weekly.py                      # NN fine-tuning
PYTHONPATH=. python3 scripts/rankings.py update         # D1BB rankings
python3 scripts/d1bb_advanced_scraper.py --conference SEC  # Advanced stats (wOBA, FIP, etc.)
```

## Web Dashboard

**Pages:** Dashboard, Teams, Predict, Rankings, Standings, Betting, Scores, Game Detail, Tracker, Models

### Dashboard (`/`)
MSU + Auburn cards, today's best bets, recent results with model accuracy

### Scores (`/scores`)
Defaults to most recent date with scores. Date picker, conference filter, model prediction badges per game.

### Betting (`/betting`)
Best Bets (consensus), Highest EV, Best Totals. v2 badge shows selection logic version.

### API Endpoints
- `/api/best-bets` — JSON best bets for today
- Game detail pages show full model breakdown per matchup

## Dedup Logic

ESPN-sourced games coexist with D1BB games. When D1BB's 7-day schedule window reaches an ESPN game:

1. `d1bb_schedule.py` generates D1BB game ID
2. Checks exact ID match → update if found
3. If no match, fuzzy-searches by `date + home_team_id + away_team_id` (using `team_aliases` for cross-source name resolution + home/away swap detection)
4. If ESPN ghost found → migrates FK data (predictions, bets, weather) → replaces game

**To add a new team name mapping:** Insert into `team_aliases` table. No code change needed.

## File Structure
```
college-baseball-predictor/
├── data/
│   ├── baseball.db              # Main database
│   ├── backups/                 # DB backups
│   ├── p4_team_urls.json        # P4 team stats URLs
│   ├── d1_team_urls.json        # Extended D1 stats URLs
│   ├── weather_coefficients.json
│   └── *.json                   # Various config/progress files
├── config/
│   ├── d1bb_slugs.json          # D1BB team slug mapping
│   ├── espn_team_ids.json       # ESPN team ID mapping
│   └── team_sites.json          # Athletics site configs
├── models/                      # Prediction models (24 .py files)
│   ├── neural_model.py          # Best performer (88%)
│   ├── ensemble_model.py        # Win probability ensemble
│   ├── runs_ensemble.py         # Totals ensemble
│   └── nn_features.py           # Shared NN feature engineering
├── scripts/                     # Data collection & operations (90+ .py files)
│   ├── d1bb_schedule.py         # D1BB score/schedule sync (with dedup)
│   ├── d1bb_box_scores.py       # D1BB box score scraper
│   ├── d1bb_scraper.py          # D1BB player stats scraper
│   ├── d1bb_advanced_scraper.py # D1BB advanced stats (wOBA, FIP)
│   ├── bet_selection_v2.py      # Current bet selection logic
│   ├── predict_and_track.py     # Prediction generation + evaluation
│   ├── team_resolver.py         # Team name normalization
│   └── weather.py               # Open-Meteo weather fetcher
├── web/
│   ├── app.py                   # Flask app (~2000 lines)
│   └── templates/               # 16 Jinja2 templates
├── weights/                     # Trained NN weight files
├── CONTEXT.md                   # This file
└── README.md
```

## Git
- **Repo:** github.com/sams-claude-bot/college-baseball-predictor
- **Commit as:** sams-claude-bot / sams-claude-bot@users.noreply.github.com
- **DO NOT reset/re-backfill Elo** — let it update naturally (Sam's directive)

---
*Last updated: February 17, 2026*
