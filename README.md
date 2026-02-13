# College Baseball Prediction Model

**Season:** 2026 NCAA Division I Baseball
**Primary Focus:** SEC Conference, Mississippi State Bulldogs
**Started:** February 13, 2026 (Opening Day)

## Goals
- Predict game and series winners
- Project runs scored per game
- Track player performance and trends
- Build historical data for model training

## Data Sources
- **NCAA.com** - Official stats, standings, schedules
- **Team Athletics Sites** - Rosters, schedules (hailstate.com, etc.)
- **D1Baseball** - Rankings, analysis (requires browser scraping)
- **Baseball Reference** - Historical stats

## Directory Structure
```
data/
  teams/        # Team info, rosters, season stats
  games/        # Game results and box scores
  players/      # Individual player stats
  rankings/     # Weekly rankings snapshots
  snapshots/    # Daily data snapshots
scripts/        # Data collection and processing
models/         # Prediction model code
reports/        # Generated analysis reports
artifacts/      # Charts, exports, etc.
```

## SEC Teams (2026)
**East:** Florida, Georgia, Kentucky, Missouri, South Carolina, Tennessee, Texas A&M, Vanderbilt
**West:** Alabama, Arkansas, Auburn, LSU, Mississippi State, Ole Miss, Oklahoma, Texas

## Priority Tracking
1. **Mississippi State** - Full roster, game-by-game tracking
2. **SEC Teams** - Conference games, key matchups
3. **Top 25 Teams** - Weekly monitoring
4. **All D1** - Results for model training

## Automation
- Daily: Collect game results, update stats
- Weekly: Rankings updates, model retraining
- Pre-game: Generate predictions for upcoming matchups
