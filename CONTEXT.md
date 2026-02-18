# College Baseball Predictor — Project Context

> **Read this first.** Single source of truth for understanding the project.
> Last verified: February 18, 2026

## What This Is

NCAA D1 college baseball prediction system with a web dashboard. Collects data from D1Baseball and DraftKings, runs 12 prediction models, tracks betting P&L, and serves everything at [baseball.mcdevitt.page](https://baseball.mcdevitt.page).

- **Season:** Feb 14 – June 22, 2026 (CWS in Omaha)
- **Focus:** Mississippi State + Auburn (featured), all SEC, all Power 4, Top 25, full D1 scores
- **Stack:** Python 3, SQLite (WAL), Flask, Playwright, PyTorch, XGBoost, LightGBM
- **Dashboard:** systemd `college-baseball-dashboard.service` → Flask on port 5000
- **Public URL:** baseball.mcdevitt.page (Cloudflare Tunnel + Access)
- **Repo:** github.com/sams-claude-bot/college-baseball-predictor

## Current Status (as of Feb 18)

| Metric | Value |
|--------|-------|
| Total games tracked | 2,190 |
| Games completed | 392 |
| Games scheduled | 1,793 |
| D1 teams | 407 |
| Teams with player stats | 292 |
| Model predictions made | 4,235 |
| Player stats rows | 10,706 |
| Venues with coordinates | 299 |
| Team aliases | 704 |
| Games with weather | 520 |
| Season date range | Feb 13 – Feb 17 (5 days in) |

---

## Data Sources

| Source | What | Method | Notes |
|--------|------|--------|-------|
| **D1Baseball** | Scores, schedules, box scores, player stats (basic + advanced), rankings | Playwright browser scrape | **PRIMARY for everything**. Requires D1BB subscription (login persists in openclaw browser profile) |
| **DraftKings** | Betting lines (ML, spreads, totals) | Playwright browser scrape | Fragile — NCAA baseball page layout changes frequently |
| **ESPN** | Legacy schedule backbone (future games beyond D1BB's 7-day window) | REST API | Being gradually replaced as D1BB's sliding window advances. Dedup logic handles migration |
| **Open-Meteo** | Game weather forecasts (temp, wind, humidity, precip) | REST API | Free, no API key. Fetches for P4 home games using venue coordinates |

### ⚠️ Critical Data Rules
- **D1Baseball is the source of truth** for scores, schedules, and stats
- **DO NOT scrape team athletics sites** — causes duplicates (learned Feb 14)
- **DO NOT reset/re-backfill Elo** — let it update naturally (Sam's directive)
- ESPN future games get replaced when D1BB's 7-day window reaches them (dedup in `d1bb_schedule.py`)
- Team name mismatches handled via `team_aliases` table (704 entries across DK/ESPN/D1BB/manual)

---

## Database

**Location:** `data/baseball.db` (SQLite, WAL mode)

### Core Tables

| Table | Rows | Purpose |
|-------|------|---------|
| `games` | 2,190 | All games — scheduled, final, postponed, cancelled |
| `teams` | 407 | D1 teams with conference, rank, athletics URL |
| `team_aliases` | 704 | Cross-source name mapping (DK↔ESPN↔D1BB) |
| `player_stats` | 10,706 | Per-player batting/pitching stats + advanced metrics (wOBA, FIP, xFIP, wRC+) |
| `player_stats_snapshots` | varies | Point-in-time stat snapshots for historical tracking |
| `model_predictions` | 4,235 | Pre-game predictions from all models, graded post-game |
| `elo_ratings` | 391 | Current Elo rating per team |
| `elo_history` | varies | Elo rating changes per game |
| `betting_lines` | 78 | DraftKings odds per game (ML, spread, O/U) |
| `tracked_bets` | 1 | Moneyline bet tracking with P&L |
| `tracked_bets_spreads` | 0 | Spread/total bet tracking |
| `tracked_confident_bets` | 2 | High-consensus bet tracking (v2) |
| `game_weather` | 520 | Weather forecasts for games |
| `venues` | 299 | Stadium coordinates, dome status, capacity |
| `power_rankings` | 382 | Model-generated weekly power rankings |
| `rankings_history` | varies | D1Baseball Top 25 poll history |
| `pitcher_game_log` | varies | Per-pitcher box score data per game |
| `pitching_matchups` | varies | Starter assignments per game |
| `ensemble_weights_history` | varies | Historical ensemble weight snapshots |
| `conference_ratings` | varies | Conference strength ratings by season |
| `preseason_priors` | varies | Preseason rankings, projected win%, returning WAR |

### Game ID Format
`YYYY-MM-DD_away-team_home-team` (e.g. `2026-02-17_cincinnati_auburn`)
Doubleheaders: `_g1`, `_g2` suffixes.

### Game Status Values
`scheduled`, `final`, `postponed`, `cancelled`

---

## Models

### Win Probability Models (10)

| Model | Type | Description | Current Accuracy |
|-------|------|-------------|-----------------|
| `advanced` | Statistical | Opponent-adjusted stats, recency-weighted | **81.5%** (234/287) |
| `conference` | Statistical | Conference strength adjustments | 81.2% (233/287) |
| `prior` | Bayesian | Preseason rankings + program history | 80.8% (232/287) |
| `log5` | Formula | Bill James head-to-head formula | 80.5% (231/287) |
| `ensemble` | Blend | Dynamic weighted blend of all component models | 79.8% (229/287) |
| `lightgbm` | ML (GBM) | LightGBM gradient boosting, 81 features | 79.0% (226/286) |
| `elo` | Rating | Chess-style ratings, updated per game result | 78.1% (214/274) |
| `poisson` | Statistical | Run distribution modeling with weather | 77.0% (221/287) |
| `xgboost` | ML (GBM) | XGBoost gradient boosting, 81 features | 76.2% (218/286) |
| `pythagorean` | Formula | Runs scored/allowed expectation | 75.3% (216/287) |
| `neural` | ML (NN) | PyTorch neural net, 81 features, 2-phase training | 74.4% (287/386) |
| `pitching` | Statistical | ERA, WHIP, K rates, bullpen depth | 69.7% (200/287) — **disabled in ensemble (weight=0)** |

**Note:** `momentum` is a post-ensemble modifier (±5% based on last 5-7 games), not a standalone model.

### Ensemble Weights
Dynamic — auto-adjusts based on recency-weighted accuracy. Minimum 5% floor per model. Pitching model at 0 weight. Weights logged to `ensemble_weights_history` table.

### Run Projection Models

| Model | File | Purpose |
|-------|------|---------|
| `nn_totals` | `nn_totals_model.py` | Neural net for over/under totals |
| `nn_spread` | `nn_spread_model.py` | Neural net for run line spreads |
| `nn_dow_totals` | `nn_dow_totals_model.py` | Day-of-week adjusted totals |
| `runs_ensemble` | `runs_ensemble.py` | Weighted blend of runs models |

### Weather Model
`weather_model.py` — Adjusts run projections based on temperature, wind, humidity. Coefficients stored in `data/weather_coefficients.json`.

### Feature Engineering
All trainable models (neural, XGBoost, LightGBM) share the same feature pipeline:
- **`nn_features.py`** → `FeatureComputer` class (live predictions) and `HistoricalFeatureComputer` (training)
- **81 features** with `use_model_predictions=False`
- Features include: Elo ratings, team stats (batting/pitching aggregates), records, conference strength, weather, home/away, ranking, momentum
- Missing data handled with sensible defaults (e.g., 1500 Elo, average weather)

### Training Policy
- **Train set:** Historical (2024-2025) + 2026 games older than 7 days
- **Validation set:** Last 7 days of completed games (fallback to 3 days if <20 games)
- **Neural net:** 2-phase — base at lr=0.001, fine-tune at lr=0.0001. Only save fine-tuned weights if they beat base
- **XGBoost/LightGBM:** Full retrain weekly
- **Schedule:** Sunday 9:30 PM via `train_all_models.py`
- **Model weights stored in:** `data/nn_model.pt`, `data/xgb_moneyline.pkl`, `data/lgb_moneyline.pkl`, etc.

---

## Betting System

### v2 Selection Logic (`bet_selection_v2.py`)
- **Consensus bets:** 7+/10 models must agree on the winner
- **Edge thresholds:** 8% for favorites, 15% for underdogs
- **Spreads:** Disabled (not calibrated — 0/5 historical)
- **Max bets/day:** 3, Kelly-adjusted sizing
- **Flat tracking:** $100 per bet for P&L tracking
- **P&L tracking started:** Feb 18, 2026 (reset after Feb 17 model improvements)

### Bet Recording Flow
1. Morning (8 AM): Scrape DraftKings odds + fetch weather
2. Morning (9 AM): `bet_selection_v2.py record` — analyze and record best bets
3. Pre-game (15 min before first pitch): Optional odds refresh + re-record
4. Nightly (2 AM): `record_daily_bets.py evaluate` — grade completed bets

### P&L Tables
- `tracked_bets` — Moneyline bets (v2 ML picks)
- `tracked_bets_spreads` — Spread and total bets
- `tracked_confident_bets` — High-consensus bets (7+/10 models agree)

---

## Web Dashboard

**2,749 lines** of Flask (web/app.py) serving 16 Jinja2 templates.

### Pages

| Route | Template | Description |
|-------|----------|-------------|
| `/` | `dashboard.html` | MSU + Auburn cards, today's best bets, recent results with model accuracy |
| `/scores` | `scores.html` | Scoreboard by date, conference filter, model prediction badges per game |
| `/betting` | `betting.html` | Best Bets (consensus), Highest EV, Best Totals. v2 badge shows selection logic |
| `/teams` | `teams.html` | All teams list, searchable/filterable by conference |
| `/team/<id>` | `team_detail.html` | Team profile: record, stats, schedule, Elo chart |
| `/game/<id>` | `game.html` | Full model breakdown per matchup, box score if completed |
| `/predict` | `predict.html` | Interactive head-to-head prediction tool |
| `/rankings` | `rankings.html` | D1Baseball Top 25 + model power rankings |
| `/standings` | `standings.html` | Conference standings |
| `/models` | `models.html` | Model accuracy comparison, ensemble weight history |
| `/calendar` | `calendar.html` | Game calendar with date navigation |
| `/tracker` | `tracker.html` | Bet tracking P&L dashboard |
| `/debug` | `debug.html` | Debug flags and bug reports |

### API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/best-bets` | GET | JSON best bets for today |
| `/api/predict` | POST | Run prediction for arbitrary matchup |
| `/api/runs` | POST | Run total projections |
| `/api/teams` | GET | All teams JSON |
| `/api/debug/flag` | POST | Toggle debug flags |
| `/api/bug-report` | POST/PATCH | Submit/update bug reports |

### Service Configuration
```ini
# /home/sam/college-baseball-predictor/web/college-baseball-dashboard.service
[Service]
User=sam
WorkingDirectory=/home/sam/college-baseball-predictor
ExecStart=venv/bin/python -m flask run --host=0.0.0.0 --port=5000
ReadWritePaths=/home/sam/college-baseball-predictor/data
# Security: NoNewPrivileges, ProtectSystem=strict, ProtectHome=read-only
```

---

## Cron Schedule (OpenClaw)

All jobs managed via OpenClaw cron (not system cron). Times are CT (America/Chicago).

### Active Jobs

| Time | Name | Model | What |
|------|------|-------|------|
| **Every 15 min, 12PM-11PM** | Score Updates | default | `d1bb_schedule.py --today` — live score polling |
| **1 AM daily** | Nightly D1 Stats + Schedule | opus | D1BB schedule sync (7 days) + all D1 player stats (~1hr for 311 teams) |
| **2 AM daily** | Nightly Scores + Pipeline | opus | Backup → D1BB box scores → evaluate bets → update Elo → aggregate stats → evaluate predictions → generate new predictions → push git |
| **8 AM daily** | DraftKings Odds + Weather | opus | Browser scrape DK odds + Open-Meteo weather for next 3 days |
| **9 AM daily** | Record Best Bets | opus | `bet_selection_v2.py record` (scrapes DK first if no lines exist) |
| **9:30 AM daily** | Pre-Game Scheduler | default | Creates one-shot job 15min before first pitch for odds refresh |
| **Mon 12 PM** | Power Rankings | default | `power_rankings.py --top 25 --store` + restart dashboard |
| **Mon 10 PM** | D1BB Rankings | opus | Browser scrape Top 25 poll |
| **Sun 9:30 PM** | Weekly Model Training | opus | `train_all_models.py` — NN + XGB + LGB unified training |
| **Sun 10 PM** | Weekly Accuracy Report | opus | Full model comparison + P&L summary |

### Disabled Jobs
- Nashville Rent Tracker (3 jobs) — paused
- Birmingham House Prices (2 jobs) — paused  
- Daily Self-Improvement — paused
- College Baseball Daily Collection — replaced by nightly pipeline
- Nightly Stats Collection (old) — replaced by D1 Stats + Schedule job

### Pipeline Order (nightly)
```
1 AM: Schedule sync (7 days) + Player stats (all D1)
2 AM: DB backup → Box scores (yesterday) → Evaluate bets → Update Elo → Aggregate stats → Evaluate predictions → Generate predictions → Git push
8 AM: DK odds scrape + Weather fetch
9 AM: Record best bets
9:30 AM: Schedule pre-game odds refresh
```

---

## Key Scripts

### Data Collection
| Script | What It Does |
|--------|-------------|
| `d1bb_schedule.py --today` | Live score updates from D1Baseball scoreboard |
| `d1bb_schedule.py --days 7` | Schedule sync — finds new games, time changes, cancellations |
| `d1bb_box_scores.py --date YYYY-MM-DD` | Box score scraper — creates game records + player box scores |
| `d1bb_scraper.py --all-d1 --delay 2` | All D1 player stats (basic + advanced) via Playwright (~1hr) |
| `d1bb_advanced_scraper.py --conference SEC` | Advanced stats by conference (wOBA, FIP, xFIP) — all D1 covered via nightly scraper |
| `weather.py fetch --upcoming` | Open-Meteo weather for next 3 days of P4 home games |

### Predictions & Evaluation
| Script | What It Does |
|--------|-------------|
| `predict_and_track.py predict` | Generate predictions for upcoming games (all 12 models) |
| `predict_and_track.py evaluate` | Grade predictions against final scores |
| `predict_and_track.py accuracy` | Display model accuracy breakdown |

### Betting
| Script | What It Does |
|--------|-------------|
| `bet_selection_v2.py record` | Analyze odds, record today's best bets |
| `record_daily_bets.py evaluate` | Grade completed bets, calculate P&L |

### Ratings & Rankings
| Script | What It Does |
|--------|-------------|
| `update_elo.py --date YYYY-MM-DD` | Update Elo ratings for completed games |
| `aggregate_team_stats.py` | Recompute team aggregate stats from player data |
| `power_rankings.py --top 25 --store` | Generate model power rankings |
| `rankings.py update` | Scrape D1Baseball Top 25 poll |

### Training
| Script | What It Does |
|--------|-------------|
| `train_all_models.py` | Unified weekly training — NN + XGB + LGB with consistent split |
| `train_neural_v2.py` | Neural net only — 2-phase (base + finetune) |
| `train_gradient_boosting.py` | XGBoost + LightGBM only |

### Utilities
| Script | What It Does |
|--------|-------------|
| `team_resolver.py` | Team name normalization (`resolve_team()`, `add_alias()`) |
| `backup_db.py` | Database backup to `data/backups/` |
| `verification_check.py` | Post-collection data integrity checks |
| `database.py` | Shared DB connection and helper queries |
| `compute_historical_features.py` | Generate historical feature vectors for training |
| `infer_starters.py` | Infer probable starters from pitcher game logs |
| `build_pitching_infrastructure.py` | Build pitching matchup data |
| `add_game.py` | Manually add a game to the database |

---

## Dedup Logic (ESPN → D1BB Migration)

ESPN-sourced games coexist with D1BB games. When D1BB's 7-day schedule window reaches an ESPN game:

1. `d1bb_schedule.py` generates D1BB-format game ID
2. Checks exact ID match → update if found
3. If no match, fuzzy-searches by `date + home_team_id + away_team_id` (using `team_aliases` for cross-source name resolution + home/away swap detection)
4. If ESPN ghost found → migrates FK data (predictions, bets, weather) → replaces game

**To add a new team name mapping:** `INSERT INTO team_aliases (alias, team_id, source) VALUES ('DK Name', 'db-team-id', 'draftkings')` — no code change needed.

---

## File Structure
```
college-baseball-predictor/
├── data/
│   ├── baseball.db              # Main database (SQLite WAL)
│   ├── backups/                 # Timestamped DB backups
│   ├── nn_model.pt              # Neural net weights (win prob)
│   ├── nn_totals_model.pt       # Neural net weights (totals)
│   ├── nn_spread_model.pt       # Neural net weights (spreads)
│   ├── nn_dow_totals_model.pt   # Neural net weights (DOW totals)
│   ├── xgb_moneyline.pkl        # XGBoost moneyline model
│   ├── xgb_totals.pkl           # XGBoost totals model
│   ├── xgb_spread.pkl           # XGBoost spread model
│   ├── lgb_moneyline.pkl        # LightGBM moneyline model
│   ├── lgb_totals.pkl           # LightGBM totals model
│   ├── lgb_spread.pkl           # LightGBM spread model
│   ├── weather_coefficients.json
│   ├── config.json              # App configuration
│   ├── draftkings_odds.json     # Cached DK odds
│   ├── preseason_priors.json    # Preseason data
│   └── *.json                   # Various progress/state files
├── config/
│   ├── d1bb_slugs.json          # D1BB team URL slugs
│   ├── espn_team_ids.json       # ESPN team ID mapping
│   └── team_sites.json          # Athletics site configs (unused)
├── models/                      # 23 Python model files
│   ├── base_model.py            # Abstract base class
│   ├── neural_model.py          # PyTorch win probability (81 features)
│   ├── ensemble_model.py        # Dynamic weighted blend
│   ├── xgboost_model.py         # XGBoost (ML, totals, spread)
│   ├── lightgbm_model.py        # LightGBM (ML, totals, spread)
│   ├── nn_features.py           # Feature pipeline (FeatureComputer)
│   ├── nn_totals_model.py       # Over/under neural net
│   ├── nn_spread_model.py       # Spread neural net
│   ├── nn_dow_totals_model.py   # DOW-adjusted totals
│   ├── runs_ensemble.py         # Totals ensemble
│   ├── advanced_model.py        # Opponent-adjusted stats
│   ├── conference_model.py      # Conference strength
│   ├── elo_model.py             # Elo ratings
│   ├── log5_model.py            # Bill James Log5
│   ├── pitching_model.py        # Pitching matchups
│   ├── poisson_model.py         # Run distribution
│   ├── prior_model.py           # Preseason priors
│   ├── pythagorean_model.py     # Pythagorean expectation
│   ├── momentum_model.py        # Post-ensemble momentum modifier
│   ├── weather_model.py         # Weather adjustments
│   ├── predictor_db.py          # DB helpers for models
│   ├── compare_models.py        # Model comparison utilities
│   └── archive/                 # Deprecated models
├── scripts/                     # 30+ active scripts
│   ├── d1bb_schedule.py         # D1BB score/schedule sync (with dedup)
│   ├── d1bb_box_scores.py       # D1BB box score scraper
│   ├── d1bb_scraper.py          # D1BB player stats (basic + advanced)
│   ├── d1bb_advanced_scraper.py # D1BB advanced stats (SEC only)
│   ├── predict_and_track.py     # Prediction generation + evaluation
│   ├── bet_selection_v2.py      # Bet selection (consensus + EV)
│   ├── record_daily_bets.py     # Bet grading + P&L
│   ├── train_all_models.py      # Unified weekly training
│   ├── train_neural_v2.py       # 2-phase NN training
│   ├── train_gradient_boosting.py # XGB + LGB training
│   ├── team_resolver.py         # Team name normalization
│   ├── weather.py               # Open-Meteo weather fetcher
│   ├── update_elo.py            # Elo rating updates
│   ├── aggregate_team_stats.py  # Team stat aggregation
│   ├── power_rankings.py        # Power rankings generator
│   ├── rankings.py              # D1BB Top 25 scraper
│   ├── backup_db.py             # Database backup
│   ├── database.py              # Shared DB connection
│   ├── verification_check.py    # Data integrity checks
│   ├── compute_historical_features.py
│   ├── infer_starters.py
│   ├── build_pitching_infrastructure.py
│   ├── add_game.py
│   ├── finetune_weekly.py       # Legacy weekly fine-tuning
│   ├── archive/                 # 78 deprecated/one-off scripts
│   └── custom_scrapers/         # Team-specific scrapers
├── web/
│   ├── app.py                   # Flask app (2,749 lines)
│   ├── college-baseball-dashboard.service
│   ├── static/                  # CSS, JS, images
│   └── templates/               # 16 Jinja2 templates
├── tasks/
│   ├── lessons.md               # Anti-patterns and past mistakes
│   └── todo.md                  # Current task tracking
├── CONTEXT.md                   # This file
├── MANIFEST.md                  # Original project manifest
└── README.md
```

---

## Git
- **Repo:** github.com/sams-claude-bot/college-baseball-predictor
- **Commit as:** sams-claude-bot / sams-claude-bot@users.noreply.github.com
- **HTTPS auth** (SSH had permission issues)
- Nightly auto-commits after data sync

---

## Known Issues & Technical Debt
1. **DraftKings scraper is fragile** — NCAA baseball page layout changes break parsing regularly
2. **Pitching model underperforms** (69.7%) — disabled in ensemble at 0 weight, needs rework
3. **Neural net accuracy dropped** to 74.4% — was 88% early on, possibly overfitting to small sample
4. **Spreads disabled** in betting — model not calibrated for run lines
5. **5 teams need custom scrapers** for stats: Georgia Tech (PDF rosters), Arkansas, Kentucky, South Carolina, Vanderbilt
6. **`app.py` is 2,749 lines** — could benefit from blueprint refactoring

---

## Lessons Learned (see also `tasks/lessons.md`)
- Always run `evaluate` before `accuracy` — accuracy only displays, evaluate actually grades
- Feature dimensions must match between training (historical) and prediction (live) — both 81
- Don't backfill predictions on completed games — creates data leakage
- Sub-agents will report "ok" on empty tables — always include SQL verification with row counts
- Silent failures (job succeeds but inserts 0 rows) are the worst — always check output counts
