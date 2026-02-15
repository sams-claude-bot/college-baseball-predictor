# P4 Stats Collection System

Automated biweekly collection of batting and pitching stats for all 67 Power 4 conference teams.

## Overview

This system scrapes cumulative season stats from official team athletics websites (SIDEARM Sports platform) and stores them in the `player_stats` table for use in prediction models.

**Schedule:**
- **Sunday 11 PM CST** - Post-game collection after weekend series complete
- **Thursday 8 AM CST** - Pre-weekend refresh for weekend predictions

## Architecture

```
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│  Cron Job (OpenClaw)│────▶│  Browser Automation  │────▶│  Team Stats URL │
│  (isolated session) │     │  (Playwright/openclaw)│     │  (SIDEARM Sports)│
└─────────────────────┘     └──────────────────────┘     └─────────────────┘
                                      │
                                      ▼
                            ┌──────────────────────┐
                            │  Parse HTML Tables   │
                            │  - Batting stats     │
                            │  - Pitching stats    │
                            └──────────────────────┘
                                      │
                                      ▼
                            ┌──────────────────────┐
                            │  SQLite Database     │
                            │  player_stats table  │
                            └──────────────────────┘
```

## Components

### 1. Team URL Mapping (`data/p4_team_urls.json`)

JSON file containing stats page URLs for all 67 P4 teams:

```json
{
  "teams": {
    "alabama": "https://rolltide.com/sports/baseball/stats/2026",
    "mississippi-state": "https://hailstate.com/sports/baseball/stats/2026",
    ...
  }
}
```

**URL Pattern:** Most schools use SIDEARM Sports with the format:
```
https://{school-domain}/sports/baseball/stats/{year}
```

### 2. Scraper Script (`scripts/p4_stats_scraper.py`)

Browser-based scraper that:
1. Opens each team's stats page
2. Waits for JavaScript to render tables
3. Extracts batting stats from the default view
4. Switches to Pitching view via dropdown
5. Extracts pitching stats
6. Updates `player_stats` table

**Usage:**
```bash
cd /home/sam/college-baseball-predictor

# List all teams
python3 scripts/p4_stats_scraper.py --list

# Scrape single conference
python3 scripts/p4_stats_scraper.py --conference SEC

# Scrape single team
python3 scripts/p4_stats_scraper.py --team alabama

# Dry run (no database updates)
python3 scripts/p4_stats_scraper.py --dry-run

# Resume interrupted run
python3 scripts/p4_stats_scraper.py --resume
```

### 3. Cron Jobs (OpenClaw)

Two isolated session jobs handle the actual browser automation:

| Job | Schedule | Purpose |
|-----|----------|---------|
| P4 Stats Collection (Sunday) | `0 23 * * 0` | Post-game stats after weekend |
| P4 Stats Collection (Thursday) | `0 8 * * 4` | Fresh data before weekend series |

## Data Collected

### Batting Stats
| Column | Description |
|--------|-------------|
| `batting_avg` | Batting average |
| `ops` | On-base plus slugging |
| `at_bats` | At bats |
| `runs` | Runs scored |
| `hits` | Hits |
| `doubles` | Doubles |
| `triples` | Triples |
| `home_runs` | Home runs |
| `rbi` | Runs batted in |
| `walks` | Walks (BB) |
| `strikeouts` | Strikeouts (SO) |
| `stolen_bases` | Stolen bases |
| `obp` | On-base percentage |
| `slg` | Slugging percentage |

### Pitching Stats
| Column | Description |
|--------|-------------|
| `era` | Earned run average |
| `whip` | Walks + hits per inning pitched |
| `wins` | Wins |
| `losses` | Losses |
| `saves` | Saves |
| `innings_pitched` | Innings pitched |
| `hits_allowed` | Hits allowed |
| `runs_allowed` | Runs allowed |
| `earned_runs` | Earned runs |
| `walks_allowed` | Walks allowed |
| `strikeouts_pitched` | Strikeouts |
| `games_pitched` | Games pitched |
| `games_started` | Games started |

## Database Schema

The `player_stats` table stores both batting and pitching stats:

```sql
CREATE TABLE player_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id TEXT NOT NULL,
    name TEXT NOT NULL,
    number INTEGER,
    position TEXT,
    year TEXT,
    -- Batting stats
    games INTEGER DEFAULT 0,
    at_bats INTEGER DEFAULT 0,
    runs INTEGER DEFAULT 0,
    hits INTEGER DEFAULT 0,
    doubles INTEGER DEFAULT 0,
    triples INTEGER DEFAULT 0,
    home_runs INTEGER DEFAULT 0,
    rbi INTEGER DEFAULT 0,
    walks INTEGER DEFAULT 0,
    strikeouts INTEGER DEFAULT 0,
    stolen_bases INTEGER DEFAULT 0,
    batting_avg REAL DEFAULT 0,
    obp REAL DEFAULT 0,
    slg REAL DEFAULT 0,
    ops REAL DEFAULT 0,
    -- Pitching stats
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    era REAL DEFAULT 0,
    games_pitched INTEGER DEFAULT 0,
    games_started INTEGER DEFAULT 0,
    saves INTEGER DEFAULT 0,
    innings_pitched REAL DEFAULT 0,
    hits_allowed INTEGER DEFAULT 0,
    runs_allowed INTEGER DEFAULT 0,
    earned_runs INTEGER DEFAULT 0,
    walks_allowed INTEGER DEFAULT 0,
    strikeouts_pitched INTEGER DEFAULT 0,
    whip REAL DEFAULT 0,
    -- Metadata
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(team_id, name)
);
```

## Browser Automation Details

### Why Browser Automation?

SIDEARM Sports pages use JavaScript to render stats tables. Simple HTTP requests (`requests`, `curl`) only get the page skeleton without data. Browser automation (Playwright via OpenClaw) renders the full page.

### Page Structure (SIDEARM Sports)

```
┌─────────────────────────────────────────┐
│  Stats Page Header                      │
├─────────────────────────────────────────┤
│  [Player Stats] [Team Stats] [Game-By-Game] ... │  ← Tabs
├─────────────────────────────────────────┤
│  [Overall ▼]  [Batting ▼]               │  ← Dropdowns
├─────────────────────────────────────────┤
│  # | Player | AVG | OPS | GP-GS | AB ...│  ← Table header
│  3 | Reese, Ace | .417 | 1.379 | 3-3 ...│  ← Player rows
│  7 | Stallman, Reed | .375 | ...        │
│  ...                                     │
│  Total | .236 | .733 | 3-3 | 89 | ...   │  ← Summary row
└─────────────────────────────────────────┘
```

### Scraping Flow

1. **Open URL** - Navigate to team stats page
2. **Wait for load** - 3-5 seconds for JS rendering
3. **Snapshot batting** - Take accessibility tree snapshot
4. **Parse table** - Extract rows from snapshot
5. **Click dropdown** - Switch to Pitching view
6. **Snapshot pitching** - Take second snapshot
7. **Parse & save** - Update database
8. **Delay** - Wait 15 seconds before next team

## Rate Limiting

To avoid overwhelming servers and getting blocked:

- **15 second delay** between teams
- **Progress tracking** in `data/stats_scraper_progress.json`
- **Resume capability** if interrupted
- **~2-3 hours** for full 67-team run

## Conferences Covered

| Conference | Teams | Example Schools |
|------------|-------|-----------------|
| SEC | 16 | Alabama, LSU, Mississippi State, Texas |
| Big Ten | 18 | Michigan, Ohio State, UCLA, USC |
| ACC | 17 | Clemson, Florida State, Miami, Duke |
| Big 12 | 16 | Texas Tech, Oklahoma State, TCU, Arizona |

## Adding New Teams

1. Add team to `teams` table in database:
```sql
INSERT INTO teams (id, name, nickname, conference) 
VALUES ('new-team', 'New Team', 'Mascots', 'Conference');
```

2. Add URL to `data/p4_team_urls.json`:
```json
"new-team": "https://newteam.com/sports/baseball/stats/2026"
```

3. Verify URL works by testing single team:
```bash
python3 scripts/p4_stats_scraper.py --team new-team --dry-run
```

## Troubleshooting

### Common Issues

**Page doesn't load tables:**
- Some sites may have different SIDEARM versions
- Check if URL pattern is correct (some use `/stats/2026`, others `/stats`)
- May need to wait longer for JS rendering

**Stats not parsing correctly:**
- Table column order may differ between schools
- Check snapshot output for actual table structure
- May need to adjust column mappings in scraper

**Rate limited / blocked:**
- Increase delay between teams
- Try running at off-peak hours
- Check if site requires login

### Manual Verification

To verify a team's stats URL works:
```bash
# Open in browser and check
python3 -c "
import subprocess
subprocess.run(['xdg-open', 'https://hailstate.com/sports/baseball/stats/2026'])
"
```

## Integration with Prediction Models

The collected stats feed into multiple prediction models:

- **Pitching model** - Uses ERA, WHIP, K/9 for run projections
- **Advanced model** - Incorporates batting OPS, team slugging
- **Ensemble** - Weights multiple models based on stat availability

Fresh stats collected Thursday morning inform weekend predictions generated Thursday night.

## Files

```
college-baseball-predictor/
├── data/
│   ├── p4_team_urls.json          # Team URL mappings
│   ├── stats_scraper_progress.json # Progress tracking
│   └── baseball.db                 # SQLite database
├── scripts/
│   └── p4_stats_scraper.py        # Scraper script
└── docs/
    └── P4_STATS_COLLECTION.md     # This file
```

## Changelog

- **2026-02-15**: Initial setup - 67 P4 teams, biweekly cron jobs
- Added Colorado and Iowa State to Big 12 (conference expansion)
- Tested SIDEARM Sports format with Mississippi State
