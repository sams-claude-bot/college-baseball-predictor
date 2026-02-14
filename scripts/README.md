# Scripts Organization

## Directory Structure

```
scripts/
├── core/           # Essential daily operations
│   ├── database.py         # DB connection and queries
│   ├── daily_collection.py # Main orchestrator (cron job)
│   ├── add_game.py         # Add/update game results
│   └── rankings.py         # Top 25 management
│
├── reports/        # Report generation
│   ├── generate_report.py  # PDF weekend preview
│   ├── matchup_report.py   # Detailed matchup analysis
│   └── model_accuracy.py   # Weekly accuracy tracking
│
├── data/           # Data collection
│   ├── collect_box_scores.py  # ESPN/SEC box scores
│   ├── betting_lines.py       # DraftKings lines
│   ├── player_stats.py        # Roster/stats management
│   └── track_starters.py      # Pitching rotation tracking
│
├── loaders/        # One-time setup (run once per season)
│   ├── load_*_schedules.py    # Conference schedule loaders
│   └── scrape_*_rosters.py    # Roster scrapers
│
└── archive/        # Deprecated/one-off scripts
    └── (old migration scripts, etc.)
```

## Daily Usage

```bash
# Run daily collection (11 PM cron)
python scripts/core/daily_collection.py

# Add a game result
python scripts/core/add_game.py "Mississippi State" "Hofstra" --home-score 5 --away-score 2

# Generate weekend preview
python scripts/reports/generate_report.py
```

## Season Setup

```bash
# Load schedules (once per season)
python scripts/loaders/load_sec_schedules.py
python scripts/loaders/load_big12_schedules.py
# etc.
```
