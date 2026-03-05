# Dashboard Pages & Data Sources

> Scope: dashboard routes, page data dependencies, and freshness expectations.
> Last refreshed: 2026-03-05.

**URL:** `baseball.mcdevitt.page`  
**Service:** `college-baseball-dashboard.service` (Flask on port 5000)

---

## Pages

### `/` — Dashboard Home
- **Shows:** Overview cards, today’s games, quick snapshots
- **Data:** `games`, `model_predictions`, `elo_ratings`
- **Freshness:** score jobs + prediction pipeline

### `/scores`
- **Shows:** date-filtered games + predictions vs outcomes
- **Data:** `games`, `model_predictions`, `totals_predictions`
- **Freshness:** score update cron + prediction pipeline

### `/calendar`
- **Shows:** monthly schedule and completed games
- **Data:** `games`

### `/game/<game_id>`
- **Shows:** game-level model breakdown, WP, weather, matchup context
- **Data:** `games`, model classes, `game_weather`, `pitching_matchups`
- **Notes:** one of the few routes that can run live model logic

### `/predict`
- **Shows:** ad-hoc matchup prediction (live inference)
- **Data:** model classes + DB features

### `/teams` and `/team/<team_id>`
- **Shows:** team records, quality tables, schedule, roster/stats
- **Data:** `teams`, `games`, `elo_ratings`, `team_batting_quality`, `team_pitching_quality`, `player_stats`

### `/models`
- **Shows:** model performance table with active vs legacy context
- **Data:** `model_predictions` (`was_correct IS NOT NULL`), meta feature importances
- **Current active stack:**
  - Base (12): `elo, pythagorean, lightgbm, poisson, xgboost, pitching, pear, quality, neural, venue, rest_travel, upset`
  - Stacker: `meta_ensemble`

### `/models/trends`
- **Shows:** accuracy trends over time by model
- **Data:** `model_predictions` + timestamps

### `/rankings`
- **Shows:** composite rankings + model disagreement
- **Data:** `power_rankings`

### `/standings`
- **Shows:** conference standings
- **Data:** `games`, `teams`

### `/betting`
- **Shows:** today’s value picks vs books
- **Data:** `betting_lines`, `model_predictions`
- **Notes:** depends on DK/FD odds scrape freshness

### `/tracker`
- **Shows:** bet history, P&L, and **CLV summary**
- **Data:** `tracked_bets`, `tracked_confident_bets`, `tracked_parlays`, `betting_line_history`
- **CLV fields:** `closing_ml`, `clv_implied`, `clv_cents`

### `/debug` and `/debug/model-testing`
- **Shows:** diagnostics, model test sandbox
- **Data:** mixed validation and runtime model calls

---

## API Endpoints (selected)

- `POST /api/predict` — live matchup prediction
- `POST /api/runs` — live runs projection
- `GET /api/teams`
- `GET /api/best-bets`

---

## Pipeline/Data Flow (high level)

Typical day:
1. Schedule sync/finalize
2. Stats + derived quality updates
3. Model retraining windows (weekly/full)
4. Morning prediction pipeline (`predict_and_track`)
5. Odds scrapes + pregame refresh
6. Score updates and grading
7. CLV closing line capture (system cron)

CLV capture script:
- `scripts/capture_closing_lines.py`
- intended cadence: every 15 minutes during game hours

---

## Performance Design Notes

- Most list pages use **stored predictions** for speed.
- `/game/<id>` and `/predict` are the main routes with live model calls.
- Strict benchmark/replay work should rely on scripts in `scripts/` + `artifacts/`, not dashboard snapshots alone.
