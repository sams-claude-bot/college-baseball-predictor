# Dashboard Pages & Data Sources

**URL:** baseball.mcdevitt.page  
**Service:** `college-baseball-dashboard.service` → Flask on port 5000  
**Code:** `web/blueprints/`

---

## Pages

### `/` — Dashboard Home
- **Shows:** Overview, today's games, quick stats
- **Data:** `games`, `model_predictions`, `elo_ratings`
- **Updated by:** Score cron (15-min), morning pipeline

### `/scores` — Today's Scores
- **Shows:** All games for a given date with live scores, predictions vs actuals
- **Data:** `games`, `model_predictions`, `totals_predictions`
- **Updated by:** 15-min score cron (`d1bb_schedule.py --today`), morning predictions
- **Note:** Uses stored pre-game predictions. Falls back to live model for untracked games.

### `/calendar` — Schedule Calendar
- **Shows:** Monthly calendar view of all scheduled/completed games
- **Data:** `games`
- **Updated by:** `01_schedule_sync.sh` (12:30 AM, 7-day lookahead)

### `/game/<game_id>` — Game Detail
- **Shows:** Full breakdown per model, win probabilities, run projections, weather, starters
- **Data:** `games`, `game_weather`, `pitching_matchups`, `player_stats`, all model classes
- **Updated by:** Runs models LIVE (single game, ~1-2s). Weather from `05_morning_pipeline.sh`.
- **Note:** Only page that calls models at request time.

### `/predict` — Interactive Prediction Tool
- **Shows:** Pick two teams, get live prediction from all models
- **Data:** All model classes, `player_stats`, `elo_ratings`, `game_weather`
- **Updated by:** Models run live on demand. Stats from nightly scrape.

### `/teams` — Team List
- **Shows:** All teams with W/L record, Elo rating
- **Data:** `teams`, `games`, `elo_ratings`
- **Updated by:** `04_nightly_eval.sh` (Elo), score cron (records)

### `/team/<team_id>` — Team Detail
- **Shows:** Schedule, results, roster stats, Elo history, pitching/batting quality
- **Data:** `games`, `player_stats`, `elo_ratings`, `team_batting_quality`, `team_pitching_quality`, `lineup_history`
- **Updated by:** `02_stats_scrape.sh` (1 AM), `03_derived_stats.sh` (1:45 AM), `04_nightly_eval.sh` (Elo)

### `/models` — Model Performance
- **Shows:** Accuracy per model, prediction counts, calibration
- **Data:** `model_predictions` (where `was_correct IS NOT NULL`)
- **Updated by:** `05_morning_pipeline.sh` (stores predictions), `04_nightly_eval.sh` (evaluates)

### `/models/trends` — Model Accuracy Trends
- **Shows:** Accuracy over time per model, rolling averages
- **Data:** `model_predictions` with `predicted_at` timestamps
- **Updated by:** Same as `/models`

### `/rankings` — Power Rankings
- **Shows:** Multi-model composite power rankings, per-model ranks, disagreements
- **Data:** `power_rankings`
- **Updated by:** `weekly_power_rankings.sh` (Monday 12 PM)
- **Gap:** Only updates weekly. Stale by weekend.

### `/standings` — Conference Standings
- **Shows:** W/L records grouped by conference
- **Data:** `games`, `teams` (conference field)
- **Updated by:** Score cron (automatic from game results)

### `/betting` — Best Bets
- **Shows:** Today's value picks comparing model predictions vs DraftKings odds
- **Data:** `betting_lines`, `model_predictions`
- **Updated by:** DK odds scrape (8 AM OpenClaw cron, opus), `05_morning_pipeline.sh`
- **Depends on:** DK odds being available. No odds = no edge calculations.

### `/tracker` — Bet Tracker / P&L
- **Shows:** Historical bet performance, profit/loss over time
- **Data:** `tracked_bets`, `tracked_confident_bets`, `tracked_bets_spreads`
- **Updated by:** `05_morning_pipeline.sh` (records bets), `04_nightly_eval.sh` (evaluates results)

### `/debug` — Debug Dashboard
- **Shows:** Data quality flags, model errors, missing data
- **Data:** Various tables, validation queries
- **Updated by:** On demand
- **Note:** `bug_reports` table missing — page may partially error

### `/debug/model-testing` — Model Testing Sandbox
- **Shows:** Side-by-side model comparison for specific games
- **Data:** Live model calls
- **Updated by:** On demand

---

## API Endpoints

| Endpoint | Method | Returns |
|----------|--------|---------|
| `/api/predict` | POST | Live prediction for a matchup |
| `/api/runs` | POST | Live runs projection for a matchup |
| `/api/teams` | GET | All teams JSON |
| `/api/best-bets` | GET | Today's best bets JSON |
| `/api/debug/flag` | POST | Flag data quality issue |
| `/api/bug-report` | POST | Submit bug report |

---

## Data Flow Summary

```
12:30 AM  01_schedule_sync     → games (schedule + phantom pruning)
 1:00 AM  02_stats_scrape      → player_stats
 1:45 AM  03_derived_stats     → team_batting_quality, team_pitching_quality, snapshots
 2:30 AM  04_nightly_eval      → elo_ratings, model_predictions (eval), tracked_bets (eval)
 3:30 AM  full_train           → nn_slim_model.pt, xgb_*.pkl, lgb_*.pkl (model weights)
 6:00 AM  scrape_all_lineups   → lineup_history (Mondays)
 8:00 AM  DK odds scrape (AI)  → betting_lines
 8:15 AM  05_morning_pipeline  → game_weather, model_predictions (new), tracked_bets (new)
 9:30 AM  pre-game scheduler   → one-shot odds refresh before first pitch
12-11 PM  score updates (15m)  → games (live scores)
12 PM Mon power_rankings       → power_rankings
10 PM Mon D1BB rankings (AI)   → rankings table
```

## Key Design Decisions
- **No live model calls on list pages** — all pages read stored predictions for speed (<0.1s)
- **Only `/game/<id>` and `/predict` run models live** — acceptable for single-game views
- **Predictions stored once daily** — games added after 8:15 AM won't have predictions until next day
- **DK odds are AI-scraped** — DraftKings layout changes frequently, browser-based approach is intentional
