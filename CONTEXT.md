# College Baseball Predictor - Context & Methodology

## Project Philosophy

**Accuracy over speed. Always.**

This project prioritizes getting predictions right over being first. We collect data methodically, verify against multiple sources, and accept that some information will be delayed or incomplete rather than risk corrupting our models with bad data.

## Prediction Architecture

### Two Prediction Systems

**1. Win Probability Ensemble (Moneylines)**
Predicts which team will win using 8 weighted models + momentum adjustment.

**2. Runs Ensemble (Totals)**
Predicts total runs scored for over/under betting analysis using 5 models.

---

## Win Probability Models

### Component Models (8 total)

| Model | Weight | Approach |
|-------|--------|----------|
| **Advanced** | 20% | Opponent-adjusted stats, recency-weighted, strength of schedule |
| **Poisson** | 18% | Run distribution modeling, Poisson PMF for scoring probabilities |
| **Pitching** | 15% | ERA, WHIP, strikeout rates, bullpen depth analysis |
| **Elo** | 12% | Chess-style ratings that update after each game based on margin + opponent |
| **Prior** | 12% | Preseason rankings + historical program strength (Bayesian prior) |
| **Log5** | 8% | Bill James head-to-head formula using win percentages |
| **Conference** | 8% | Conference strength adjustments (SEC/ACC/Big 12 vs mid-majors) |
| **Pythagorean** | 7% | Bill James runs scored/allowed expectation |

### Momentum Adjustment
Applied as a post-ensemble modifier (not a weighted component):
- Tracks last 5-7 games with exponential recency weighting
- Calculates win streak, weighted run differential, recent win%
- Outputs score from -1.0 to +1.0
- Adjusts final probability by up to ±5%

### Why These Weights?
- **Early season** (Feb-Mar): Prior and Elo weighted higher due to limited game data
- **Mid-season** (Apr-May): Advanced and Pitching become more reliable
- **Weights adjust automatically** based on rolling accuracy

---

## Runs Ensemble (Totals Model)

### Component Models (5 total)

| Model | Weight | Why |
|-------|--------|-----|
| **Poisson** | 30% | Best for run distributions; models exact scoring probabilities |
| **Advanced** | 25% | Good overall projections with opponent adjustments |
| **Elo** | 20% | Solid baseline run expectations |
| **Pythagorean** | 15% | Classic approach using historical RS/RA |
| **Pitching** | 10% | Useful but tends to inflate totals; down-weighted |

### How It Works
1. Each model projects home/away runs independently
2. Weighted average produces ensemble projection
3. Poisson distribution calculates O/U probabilities for any line
4. 90% confidence interval shows projection uncertainty
5. Model agreement score (0-1) indicates consensus

### Edge Calculation
```
edge = |model_total - line| × 8%
```
~8% edge per run of difference. Capped at 50%.

---

## Data Collection Strategy

### Sources (Priority Order)
1. **ESPN** — Game scores, box scores, player stats
2. **DraftKings** — Betting lines (moneylines, run lines, totals)
3. **D1Baseball** — Top 25 rankings
4. **Team Athletics Sites** — Cumulative stats (SIDEARM Sports format)
5. **Conference Portals** — Backup verification

### Collection Schedule
| Time | Job | Data |
|------|-----|------|
| 8 AM | DraftKings | Today's betting lines + totals predictions |
| 2 AM | ESPN | Final scores, box scores, prediction evaluation |
| Mon 10 PM | D1Baseball | Updated Top 25 rankings |
| Sun/Thu | Team Sites | Power 4 player stats (67 teams) |

### Browser Automation
- **Playwright** with `openclaw` profile for automated scraping
- **SIDEARM Sports** format consistent across most P4 schools
- 15-second delays between teams to avoid rate limiting
- Progress tracking for resume capability

---

## Betting Analysis Approach

### Value Detection
We compare model probabilities to DraftKings implied odds:

```
DK implied prob = remove vig from moneyline
Model prob = ensemble prediction
Edge = (model_prob - dk_implied) × 100
```

### Best Bets Criteria
- **Moneylines**: 5%+ edge
- **Totals**: 15%+ edge

### EV Calculation
```
EV = (win_prob × payout) - (lose_prob × stake)
```
Displayed as "EV per $100 wagered"

---

## Accuracy Tracking

### Moneyline Predictions
- Stored in `model_predictions` table
- Each model's prediction recorded pre-game
- Evaluated against actual winner post-game
- Broken down by: model, confidence level, home/away

### Totals Predictions
- Stored in `totals_predictions` table
- Tracked by: prediction type (OVER/UNDER), edge bucket
- Edge buckets: 30%+, 20-30%, 10-20%, <10%

### Current Performance (Feb 15, 2026)
- **Elo/Prior**: 88.2% (leading)
- **Ensemble**: 82.4%
- **Poisson**: 81.3%
- Sample size still small (17 games)

---

## Key Design Decisions

### 1. Ensemble Over Single Model
No single model captures everything. Pythagorean is elegant but ignores pitching matchups. Elo is robust but slow to adjust. The ensemble leverages each model's strengths.

### 2. Momentum as Adjustment, Not Component
Momentum is applied *after* the ensemble combines predictions. This prevents it from being washed out by other models while still having meaningful impact on hot/cold teams.

### 3. Separate Runs Ensemble
Win probability and run totals require different approaches. A team can be favored to win while the total goes under. The runs ensemble is purpose-built for totals analysis.

### 4. Conservative Pitching Weight for Totals
The pitching model tends to inflate run projections (predicting higher scores than actually occur). Down-weighted to 10% in the runs ensemble.

### 5. Bayesian Prior for Early Season
With limited game data in Feb/Mar, the Prior model (using preseason rankings and historical program strength) provides stability. Its influence fades as the season progresses.

---

## Web Dashboard

**URL:** http://192.168.1.101:5000

### Pages
- **Dashboard** — Today's games, Top 25, quick stats
- **Teams** — All teams with records, Elo, conference
- **Predict** — Manual matchup predictor
- **Rankings** — Top 25 with historical tracking
- **Betting** — DK lines vs model, Best Bets, Best Totals
- **Calendar** — Historical games by date with model accuracy
- **Models** — Ensemble weights, accuracy tracking, totals performance

### Features
- Conference filtering on all pages
- Model prediction badges (✓/✗) on game results
- Real-time EV calculations
- Over/under probability percentages

---

## Future Improvements

### Planned
- Weather integration for totals (wind, temperature)
- Bullpen fatigue tracking (game-by-game usage)
- Regression detection (hot starts that normalize)
- Public hosting via Cloudflare Tunnel

### Under Consideration
- Live game probability updates
- Pitcher-specific models (vs L/R splits)
- Conference tournament simulations
- CWS bracket projections

---

## File Structure

```
college-baseball-predictor/
├── data/
│   ├── baseball.db          # SQLite database
│   ├── p4_team_urls.json    # 67 P4 team stats URLs
│   └── snapshots/           # Historical data snapshots
├── models/
│   ├── ensemble_model.py    # Main win probability ensemble
│   ├── runs_ensemble.py     # Totals prediction ensemble
│   ├── poisson_model.py     # Run distribution model
│   ├── momentum_model.py    # Hot/cold team tracking
│   ├── advanced_model.py    # Opponent-adjusted stats
│   ├── elo_model.py         # Rating system
│   ├── pitching_model.py    # Pitching matchup analysis
│   └── [others...]
├── scripts/
│   ├── predict_and_track.py # ML prediction recording
│   ├── track_totals.py      # Totals prediction tracking
│   ├── betting_lines.py     # DraftKings line management
│   ├── p4_stats_scraper.py  # Player stats collection
│   └── [others...]
├── web/
│   ├── app.py               # Flask application
│   └── templates/           # Jinja2 templates
├── docs/
│   └── P4_STATS_COLLECTION.md
├── CONTEXT.md               # This file
└── README.md
```

---

*Last updated: February 15, 2026*
