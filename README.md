# College Baseball Prediction System

**Season:** 2026 NCAA Division I Baseball  
**Started:** February 13, 2026 (Opening Day)  
**GitHub:** [sams-claude-bot/college-baseball-predictor](https://github.com/sams-claude-bot/college-baseball-predictor)

## Overview

Full-stack college baseball prediction system with 7 statistical models, an ensemble engine, automated data pipelines, PDF reports with betting analysis, and a live web dashboard.

### Coverage
- **SEC** — 16 teams (primary focus, Mississippi State #4)
- **ACC** — 17 teams
- **Big 12** — 16 teams
- **Big Ten** — 18 teams
- **Top 25** — Auto-tracked weekly with new team detection
- **270+ teams** total (opponents auto-added as encountered)

## Prediction Models (7 + Ensemble)

| Model | Description | Weight |
|-------|-------------|--------|
| **Advanced** | Opponent-adjusted, recency-weighted, SOS-aware | ~25% |
| **Elo** | FiveThirtyEight-style ratings, margin-of-victory adjusted | ~20% |
| **Prior** | Preseason rankings + Bayesian blending (cold start solver) | ~15% |
| **Pitching** | Starting pitcher matchup + bullpen state + fatigue | ~15% |
| **Conference** | Conference strength adjustments (SEC boost, etc.) | ~10% |
| **Log5** | Bill James Log5 head-to-head formula | ~8% |
| **Pythagorean** | Bill James Pythagorean runs scored/allowed | ~7% |
| **Ensemble** | Dynamic weighted combination — auto-adjusts based on accuracy | ★ |

### Cold Start Solution
Bayesian blending shifts weight from preseason priors to actual performance:
- Games 0-5: 80% prior / 20% actual
- Games 5-15: Linear blend to 30/70
- Games 15+: 20% prior / 80% actual (priors never fully disappear)

## Features

### PDF Reports (automated via cron)
- **Weekend Preview** — All 7 model predictions with charts
- **EV vs DraftKings** — Edge %, EV per $100, totals analysis
- **Best Bets** — Games ranked by model-vs-sportsbook disagreement
- **Top 5 Picks** — AI commentary, confidence bars, contextual disclaimers
- **Rankings Movement** — Weekly risers, fallers, new entries
- **SEC/Top 25 Game Summaries**

### Web Dashboard (Flask)
- **Dashboard** — Today's games, value picks, Top 25, quick stats
- **Teams** — Sortable list, click into team detail with roster/schedule/Elo
- **Predictions** — Interactive matchup tool, pick any two teams
- **Rankings** — Top 25 with movement, historical view
- **Betting** — DK lines with model edges, best bets sorted by value
- **Models** — Ensemble weights, accuracy tracking

### Data Pipelines
- SEC roster scraper (all 16 teams)
- Multi-source schedule loader (ESPN, team sites)
- Box score collection (post-game stats)
- Starting pitcher tracking (days rest, pitch counts)
- Weekly Top 25 scraper with auto-tracking
- NCAA.com stats integration
- DraftKings line comparison

## Tech Stack

- **Language:** Python 3
- **Database:** SQLite (`data/baseball.db`)
- **Web:** Flask + Bootstrap 5 + Chart.js
- **Reports:** fpdf2 + matplotlib
- **Automation:** OpenClaw cron jobs (Thu/Mon/Sun)

## Directory Structure

```
├── models/              # 7 prediction models + ensemble
│   ├── advanced_model.py
│   ├── elo_model.py
│   ├── pitching_model.py
│   ├── conference_model.py
│   ├── prior_model.py
│   ├── log5_model.py
│   ├── pythagorean_model.py
│   ├── ensemble_model.py
│   └── predictor_db.py
├── scripts/             # Data collection & management
│   ├── daily_collection.py    # Nightly cron orchestrator
│   ├── generate_report.py     # PDF report generator
│   ├── betting_lines.py       # DK line comparison
│   ├── scrape_sec_rosters.py  # SEC roster scraper
│   ├── scrape_rankings.py     # Weekly Top 25
│   ├── collect_box_scores.py  # Post-game stats
│   ├── collect_all_stats.py   # Multi-source pipeline
│   ├── track_starters.py      # Pitcher tracking
│   ├── add_game.py            # Quick game entry
│   └── database.py            # DB operations
├── web/                 # Flask dashboard
│   ├── app.py
│   └── templates/
├── data/                # SQLite database + JSON configs
├── reports/             # Generated PDF reports
└── README.md
```

## Quick Start

```bash
# Run predictions
python3 models/predictor_db.py "Mississippi State" "Hofstra"
python3 models/predictor_db.py "Mississippi State" "UCLA" --neutral --compare

# Generate PDF report
python3 scripts/generate_report.py

# Start web dashboard
python3 web/app.py  # → http://0.0.0.0:5000

# Add game result
python3 scripts/add_game.py "Mississippi State" "Hofstra" 8 3

# Check betting value
python3 scripts/betting_lines.py compare "Mississippi State" "Hofstra" -450 +350

# View rankings
python3 scripts/scrape_rankings.py show
```

## Cron Schedule

| Time | Day | Task |
|------|-----|------|
| 11 PM | Thursday | Weekend preview — predictions, DK lines, value picks |
| 11 PM | Monday | Weekend recap + midweek preview — results, Elo updates |
| 10 PM | Sunday | Weekly accuracy report — model performance, profitability |

## Colors

- **Maroon:** #5D1725 (Mississippi State)
- **Grey:** #777777
