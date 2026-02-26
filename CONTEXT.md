# College Baseball Predictor — Project Context

> **Read this first.** Single source of truth for understanding the project.
> Last verified: February 26, 2026
> 
> ⚠️ Documentation sync is in progress. For active cleanup/status tracking, see:
> - `docs/CLEANUP_AUDIT_2026-02-22.md`
> - `docs/CLEANUP_CHECKLIST.md`
> - `docs/CLEANUP_PROGRESS_2026-02-22.md`

## What This Is

NCAA D1 college baseball prediction system with a web dashboard. Collects data from D1Baseball and DraftKings, runs 14 win-probability models + meta-ensemble, tracks betting P&L, and serves everything at [baseball.mcdevitt.page](https://baseball.mcdevitt.page).

- **Season:** Feb 13 – June 22, 2026 (CWS in Omaha)
- **Focus:** Mississippi State + Auburn (featured), all SEC, all Power 4, Top 25, full D1 scores
- **Stack:** Python 3, SQLite (WAL), Flask, Playwright, PyTorch, XGBoost, LightGBM
- **Dashboard:** systemd `college-baseball-dashboard.service` → Flask on port 5000
- **Public URL:** baseball.mcdevitt.page (Cloudflare Tunnel + Access)
- **Repo:** github.com/sams-claude-bot/college-baseball-predictor

## Documentation Ownership (Canonical)

- `CONTEXT.md` (this file) is the canonical operational reference.
- `README.md` is overview/quickstart only and intentionally omits full operational detail.
- `MANIFEST.md` is the canonical file classification/path inventory.
- `docs/DASHBOARD.md` is limited to dashboard routes/data dependencies (not full cron/runbook docs).

## Current Status (as of Feb 26)

| Metric | Value |
|--------|-------|
| Total games tracked | 8,344 |
| Games completed | 1,160 |
| Games scheduled | ~7,150 |
| D1 teams | 313 |
| Elo ratings | 317 |
| Model predictions made | 95,370 |
| Player stats rows | 12,174 |
| Team aliases | 857 |
| Games with weather | 1,529 |
| Betting lines captured | 374 (329 games) |
| Season date range | Feb 13 – Feb 26 (14 days in) |
| Tests passing | 350/350 |
| Top model (PEAR) | 76.1% accuracy |
| Meta-ensemble (LogReg) | 76.7% walk-forward |
| Totals (runs_ensemble) | 67.9% O/U accuracy |

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
- ESPN future games get replaced when D1BB's 7-day window reaches them (dedup via `schedule_gateway.py`)
- Team name mismatches handled via `team_aliases` table (704 entries across DK/ESPN/D1BB/manual)

---

## Database

**Location:** `data/baseball.db` (SQLite, WAL mode)

### Core Tables

| Table | Rows | Purpose |
|-------|------|---------|
| `games` | 2,190 | All games — scheduled, final, in-progress, postponed, cancelled. `inning_text` for live games |
| `teams` | 407 | D1 teams with conference, rank, athletics URL |
| `team_aliases` | 704 | Cross-source name mapping (DK↔ESPN↔D1BB) |
| `player_stats` | 10,706 | Per-player batting/pitching stats + advanced metrics (wOBA, FIP, xFIP, wRC+) |
| `player_stats_snapshots` | varies | Point-in-time stat snapshots for historical tracking |
| `model_predictions` | 95,370 | Pre-game predictions from all 14+1 models, graded post-game |
| `totals_predictions` | varies | Per-component O/U predictions (runs_poisson, runs_pitching, runs_advanced, runs_ensemble) |
| `team_pitching_quality` | 292 | Staff quality metrics — ace, rotation, bullpen, depth, HHI |
| `team_batting_quality` | 292 | Lineup quality — OPS, wOBA, wRC+, bench depth, concentration |
| `elo_ratings` | 317 | Current Elo rating per team |
| `elo_history` | varies | Elo rating changes per game |
| `betting_lines` | 374 | DraftKings odds per game (ML, spread, O/U) |
| `tracked_bets` | 1 | Moneyline bet tracking with P&L |
| `tracked_bets_spreads` | 0 | Spread/total bet tracking |
| `tracked_confident_bets` | 2 | High-consensus bet tracking (v2) |
| `game_weather` | 1,529 | Weather forecasts for games |
| `venues` | 299 | Stadium coordinates, dome status, capacity |
| `power_rankings` | 382 | Model-generated weekly power rankings |
| `rankings_history` | varies | D1Baseball Top 25 poll history |
| `pitcher_game_log` | varies | Per-pitcher box score data per game |
| `pitching_matchups` | varies | Starter assignments per game |
| `ensemble_weights_history` | varies | Historical ensemble weight snapshots |
| `conference_ratings` | varies | Conference strength ratings by season |
| `preseason_priors` | varies | Preseason rankings, projected win%, returning WAR |

### Active Supporting Tables

| Table | Rows | Purpose |
|-------|------|---------|
| `game_batting_stats` | 2,741 | Per-game team batting stats |
| `game_pitching_stats` | 1,124 | Per-game team pitching stats |
| `game_predictions` | 1,766 | Game-level prediction records |
| `historical_games` | 6,184 | Historical game data (2024-2025 seasons) for training |
| `historical_game_weather` | 5,886 | Historical weather data for training |
| `team_aggregate_stats` | 284 | Computed team aggregate statistics |
| `team_sos` | 222 | Strength of schedule ratings |
| `margin_tracking` | 169 | Margin/spread tracking data |
| `power_rankings_detail` | 2,208 | Per-model power ranking scores |
| `team_batting_quality_snapshots` | 585 | Historical batting quality snapshots |
| `team_pitching_quality_snapshots` | 585 | Historical pitching quality snapshots |
| `ensemble_weight_log` | 25 | Ensemble weight change log |
| `tournaments` | 3 | Tournament/postseason data |

### Legacy/Empty Tables (candidates for cleanup)

`ncaa_individual_stats`, `ncaa_team_stats`, `player_boxscore_batting`, `player_boxscore_pitching`, `predictions`, `spread_predictions`, `team_stats`, `team_stats_snapshots` — all empty, likely from earlier iterations.

### Game ID Format
`YYYY-MM-DD_away-team_home-team` (e.g. `2026-02-17_cincinnati_auburn`)
Doubleheaders: `_g1`, `_g2` suffixes.

### Game Status Values
`scheduled`, `final`, `postponed`, `cancelled`, `in-progress`

### Live Games
- Games with `in-progress` status have scores updated live (every 15 min during game hours)
- `inning_text` column stores current inning ("Top 5", "Bottom 7", etc.)
- `inning_text` is cleared when game goes `final`

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
| `pitching` | Statistical | Uses `team_pitching_quality` + `team_batting_quality` tables; DOW rotation/bullpen blending | **5% weight in ensemble** (re-enabled) |

**Note:** `momentum` is a post-ensemble modifier (±5% based on last 5-7 games), not a standalone model.

### Ensemble Weights
Dynamic — auto-adjusts based on recency-weighted accuracy. Minimum 5% floor per model. Pitching model re-enabled at 5% weight. Weights logged to `ensemble_weights_history` table.

### Run Projection Models

| Model | File | Purpose |
|-------|------|---------|
| `nn_totals` | `nn_totals_model.py` | Neural net for over/under totals |
| `nn_spread` | `nn_spread_model.py` | Neural net for run line spreads |
| `nn_dow_totals` | `nn_dow_totals_model.py` | Day-of-week adjusted totals |
| `runs_ensemble` | `runs_ensemble.py` | Stats-only blend: Poisson 35%, Pitching 35%, Advanced 30% (no Elo/Pythagorean). Auto-weight adjustment after 20+ games |

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

Flask app refactored into blueprints: `web/app.py` (110 lines) + `web/blueprints/` (2,189 lines across 9 modules). Serves 18 Jinja2 templates.

### Pages

| Route | Template | Description |
|-------|----------|-------------|
| `/` | `dashboard.html` | MSU + Auburn cards, today's best bets, recent results with model accuracy |
| `/scores` | `scores.html` | Scoreboard by date: Live (in-progress with inning), Final, Scheduled. Conference filter |
| `/betting` | `betting.html` | Best Bets (consensus), Highest EV, Best Totals. v2 badge shows selection logic |
| `/teams` | `teams.html` | All teams list, searchable/filterable by conference |
| `/team/<id>` | `team_detail.html` | Team profile: record, stats, schedule, Elo chart |
| `/game/<id>` | `game.html` | Full model breakdown per matchup, box score if completed (only page with live model calls) |
| `/predict` | `predict.html` | Interactive head-to-head prediction tool |
| `/rankings` | `rankings.html` | D1Baseball Top 25 + model power rankings |
| `/standings` | `standings.html` | Conference standings |
| `/models` | `models.html` | Model accuracy comparison, ensemble weight history |
| `/calendar` | `calendar.html` | Game calendar with date navigation |
| `/tracker` | `tracker.html` | Bet tracking P&L dashboard |
| `/debug` | `debug.html` | Debug flags and bug reports |
| `/model-testing` | `model_testing.html` | Model A/B testing tool |
| `/model-trends` | `model_trends.html` | Model accuracy trends over time |

**Other templates:** `base.html` (layout), `404.html`, `500.html` (error pages)

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

> Canonical repo service unit source: `web/college-baseball-dashboard.service`.
> `config/baseball-dashboard.service` is retained as a legacy/copy reference during cleanup (no behavior change in this pass).
> Consolidation/move decisions remain tracked in `docs/CLEANUP_CHECKLIST.md`.

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

## Cron Schedule

Jobs are split between **system cron** (bash scripts, no AI) and **OpenClaw cron** (need AI/browser).

Note: `cron/` contains both active entrypoints and retained legacy/overlap scripts (for example `01_schedule_sync.sh`, `01b_late_scores.sh`). The merged nightly step is `01_schedule_and_finalize.sh`. Treat active crontab entries as authoritative when in doubt.

### System Cron (bash scripts in `cron/`)

| Time (CT) | Script | What |
|-----------|--------|------|
| **5:00 AM** | `01_schedule_and_finalize.sh` | Merged schedule sync + finalize yesterday + late catchup verification |
| **1:00 AM** | `02_stats_scrape.sh` | All D1 player stats via D1BB (~25 min) |
| **1:45 AM** | `03_derived_stats.sh` | Quality tables + aggregates + snapshots |
| **2:30 AM** | `04_nightly_eval.sh` | Backup → evaluate bets → Elo → evaluate predictions |
| **3:30 AM** | `full_train.sh` | Full model retraining (`train_all_models.py --full-train`) |
| **8:15 AM** | `05_morning_pipeline.sh` | Weather + predictions + bet selection |
| **Sun 9:30 PM** | `weekly_training.sh` | NN + XGB + LGB unified training |
| **Sun 10 PM** | `weekly_accuracy.sh` | Model accuracy report |
| **Mon 12 PM** | `weekly_power_rankings.sh` | Power rankings generation |

Also in system cron (not bash scripts):
- **Every 15 min, 12PM-11PM**: `d1bb_schedule.py --today` — live score polling
- **Mon 6 AM**: Lineup scrape
- **3 AM daily**: Verification job

### OpenClaw Cron (need AI/browser — 3 active jobs)

| Time (CT) | Name | Model | What |
|-----------|------|-------|------|
| **8 AM daily** | Odds Scrape | opus | Browser reads DraftKings NCAA page → JSON → `dk_odds_scraper.py load` |
| **9:30 AM daily** | Pre-Game Scheduler | default | Creates one-shot odds refresh 15 min before first pitch |
| **Mon 10 PM** | D1BB Rankings | default | Browser reads d1baseball.com Top 25 poll |

### Disabled OpenClaw Jobs
- Nashville Rent Tracker (3 jobs) — paused
- Birmingham House Prices (2 jobs) — paused
- Daily Self-Improvement — paused
- Old nightly pipeline jobs (6 jobs) — replaced by system cron

### Pipeline Order (nightly)
```
5:00 AM: Schedule + finalize + late catchup (merged step)
1:00 AM: Player stats (all D1, ~25 min)
1:45 AM: Quality tables + aggregates + snapshots
2:30 AM: Backup → evaluate bets → Elo → evaluate predictions
3:30 AM: Full retrain (optional/active system cron entrypoint)
8:00 AM: DK odds scrape (OpenClaw, browser)
8:15 AM: Weather + predictions + bet selection
9:30 AM: Schedule pre-game odds refresh (OpenClaw)
```

---

## Key Scripts

### Data Collection
| Script | What It Does |
|--------|-------------|
| `schedule_gateway.py` | **Single write path to games table** — all schedule scripts route through this |
| `d1bb_team_sync.py` | Nightly schedule sync for all 311 teams via HTTP (no browser) → ScheduleGateway |
| `d1bb_schedule.py --today` | Live score updates from D1Baseball scoreboard → ScheduleGateway |
| `espn_live_scores.py` | ESPN API live scores (every 2 min during game hours) → ScheduleGateway |
| `finalize_games.py --date YYYY-MM-DD` | Finalize/postpone yesterday's games → ScheduleGateway |
| `d1bb_box_scores.py --date YYYY-MM-DD` | Box score scraper — creates game records + player box scores |
| `d1bb_scraper.py --all-d1 --delay 2` | All D1 player stats (basic + advanced) via Playwright (~1hr) |
| `weather.py fetch --upcoming` | Open-Meteo weather for next 3 days of P4 home games |

### Predictions & Evaluation
| Script | What It Does |
|--------|-------------|
| `predict_and_track.py predict` | Generate predictions for upcoming games (all 14 models + meta-ensemble) |
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

## Schedule Gateway & Dedup Logic

**All schedule writes route through `scripts/schedule_gateway.py`** (since Feb 26). This eliminates the prior problem of 7 scripts writing to `games` with 3 different upsert strategies.

### Write Path
```
d1bb_team_sync.py ──┐
d1bb_schedule.py ───┤
espn_live_scores.py ┼──→ ScheduleGateway.upsert_game() ──→ games table
finalize_games.py ──┤
```

### Dedup Strategy (in `find_existing_game()`)
1. **Exact canonical ID** — `{date}_{away}_{home}` (no suffix for game 1, `_gm2` for game 2)
2. **Legacy suffix variants** — `_g1`, `_gm1` for game 1 (historical artifacts)
3. **Swapped home/away** — same date, teams reversed (neutral-site labeling differences)
4. **Fuzzy match** — same date + same two teams in any order (catches ESPN ghost IDs)

### Ghost Replacement
When a match is found with a different ID (e.g., ESPN ghost), the gateway:
1. Migrates FK rows across 16 tables (predictions, betting lines, weather, etc.)
2. Deletes the old game row
3. Creates the new game with the canonical ID

### Status Hierarchy
`final` > `in-progress` > `scheduled` > `postponed` > `cancelled`  
A lower-status update cannot overwrite a higher-status game.

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
│   ├── neural_model_v3.py       # On hold — new features/params, needs more data (revisit mid-season)
│   ├── lightgbm_model_v2.py     # On hold — same as above
│   ├── xgboost_model_v2.py      # On hold — same as above
│   ├── nn_features_enhanced.py  # Enhanced features for v2/v3 models
│   └── archive/                 # Deprecated models
├── scripts/                     # 30+ active scripts
│   ├── d1bb_schedule.py         # D1BB score/schedule sync (with dedup)
│   ├── d1bb_box_scores.py       # D1BB box score scraper
│   ├── d1bb_scraper.py          # D1BB player stats (basic + advanced)
│   ├── # d1bb_advanced_scraper.py — archived Feb 19 (redundant with d1bb_scraper.py)
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
│   ├── dk_odds_scraper.py       # DK odds loading from JSON
│   ├── run_utils.py             # ScriptRunner, logging utils
│   ├── snapshot_stats.py        # Daily stat snapshots
│   ├── compute_batting_quality.py  # Batting quality tables
│   ├── compute_pitching_quality.py # Pitching quality tables
│   ├── d1bb_full_schedule_overwrite.py # Full season schedule tool
│   ├── d1bb_lineups.py          # D1Baseball lineup scraper
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
2. **Pitching model v2 re-enabled** at 5% weight — now uses quality tables instead of raw ERA/WHIP
3. **Neural net accuracy dropped** to 74.4% — was 88% early on, possibly overfitting to small sample
4. **Spreads disabled** in betting — model not calibrated for run lines
5. **5 teams need custom scrapers** for stats: Georgia Tech (PDF rosters), Arkansas, Kentucky, South Carolina, Vanderbilt

## Performance Notes
- **Web pages use stored predictions only** — reads from `model_predictions` and `totals_predictions` tables (no live model calls except `/game/<id>` detail page)
- **`predict_and_track.py`** runs daily (morning cron) to populate predictions
- Page load times: <0.1s (was 10-30s with live model calls)

---

## Lessons Learned (see also `tasks/lessons.md`)
- Always run `evaluate` before `accuracy` — accuracy only displays, evaluate actually grades
- Feature dimensions must match between training (historical) and prediction (live) — both 81
- Don't backfill predictions on completed games — creates data leakage
- Sub-agents will report "ok" on empty tables — always include SQL verification with row counts
- Silent failures (job succeeds but inserts 0 rows) are the worst — always check output counts
