"""
Shared configuration for prediction models.

Centralizes magic numbers and tunable parameters to avoid
scattering constants across multiple files.
"""

# =============================================================================
# Probability Bounds
# =============================================================================
PROB_FLOOR = 0.02    # Minimum win probability (2%)
PROB_CEILING = 0.98  # Maximum win probability (98%)

# =============================================================================
# Home Field Advantage
# =============================================================================
HOME_ADVANTAGE_PROB = 0.035      # Win probability boost for home team (~3.5%)
HOME_ADVANTAGE_ELO = 50          # Elo points added for home team
HOME_ADVANTAGE_RUNS = 0.02       # Run multiplier for home team (1.02x)
AWAY_DISADVANTAGE_RUNS = 0.98    # Run multiplier for away team (0.98x)

# =============================================================================
# Elo Rating System
# =============================================================================
ELO_BASE_RATING = 1500           # Starting Elo for unknown teams
ELO_K_FACTOR = 32                # How much ratings change per game
ELO_MOV_MULTIPLIER_CAP = 2.0     # Max margin-of-victory K multiplier

# Conference-tiered starting Elo (used when team first appears)
ELO_CONFERENCE_TIERS = {
    'SEC': 1600, 'ACC': 1580, 'Big 12': 1560, 'Big Ten': 1550,
    'AAC': 1520, 'Sun Belt': 1510, 'C-USA': 1500, 'MWC': 1490,
    'Big East': 1490, 'WCC': 1480, 'A-10': 1470, 'CAA': 1470,
    'MVC': 1460, 'SoCon': 1460, 'ASUN': 1460, 'Big West': 1460,
    'MAC': 1450, 'OVC': 1430, 'Southland': 1430, 'Summit': 1420,
    'WAC': 1420, 'Big South': 1420, 'NEC': 1410, 'Patriot': 1410,
    'Horizon': 1410, 'America East': 1400, 'Ivy': 1400,
    'MEAC': 1380, 'SWAC': 1370,
}

# =============================================================================
# Neural Network
# =============================================================================
NN_FEATURE_COUNT = 81            # Number of features for NN/XGB/LGB
NN_CALIBRATION_STRENGTH = 1.5    # Platt scaling strength
NN_BASE_LR = 0.001               # Base learning rate
NN_FINETUNE_LR = 0.0001          # Fine-tuning learning rate

# =============================================================================
# Ensemble Weights
# =============================================================================
ENSEMBLE_MIN_WEIGHT = 0.05       # Minimum weight for any model (5%)
ENSEMBLE_ADJUSTMENT_RATE = 0.3   # How fast weights shift toward accuracy-based targets
ENSEMBLE_MIN_GAMES = 20          # Minimum games before auto-adjusting weights

# =============================================================================
# Betting Thresholds
# =============================================================================
BET_ML_EDGE_FAVORITE = 8.0       # Min edge % for favorite ML bets
BET_ML_EDGE_UNDERDOG = 15.0      # Min edge % for underdog ML bets
BET_UNDERDOG_DISCOUNT = 0.5      # Discount underdog edges by 50%
BET_CONSENSUS_BONUS = 1.0        # +1% edge per model above 5 agreeing
BET_ML_MAX_FAVORITE = -200       # Don't bet favorites juicier than -200
BET_ML_MIN_UNDERDOG = 250        # Don't bet underdogs longer than +250
BET_TOTALS_EDGE_RUNS = 3.0       # Min runs edge for totals bets
BET_MODEL_PROB_FLOOR = 0.55      # Don't bet below 55% model probability
BET_MODEL_PROB_CEILING = 0.88    # Cap model probability at 88%

# =============================================================================
# Run Projections
# =============================================================================
RUNS_LEAGUE_AVG_DEFAULT = 6.5    # Default league average runs per team
RUNS_TOTAL_DEFAULT = 11.0        # Default total runs per game
RUNS_FLOOR = 0.5                 # Minimum expected runs per team
RUNS_CEILING = 15.0              # Maximum expected runs per team (sanity check)

# Day-of-week rotation expectations (0=Mon, 6=Sun)
# Weight for rotation vs bullpen ERA when projecting runs
DOW_ROTATION_WEIGHT = {
    0: 0.45,  # Monday (midweek) - bullpen/spot starter
    1: 0.45,  # Tuesday
    2: 0.50,  # Wednesday
    3: 0.50,  # Thursday
    4: 0.85,  # Friday - ace day
    5: 0.75,  # Saturday - #2 starter
    6: 0.55,  # Sunday - mix of #3 and bullpen
}

# =============================================================================
# Data Collection
# =============================================================================
SCRAPE_DELAY_SECONDS = 2.0       # Delay between page loads
SCRAPE_TIMEOUT_MS = 30000        # Page load timeout

# =============================================================================
# Training
# =============================================================================
TRAIN_VAL_DAYS = 7               # Days of recent games for validation set
TRAIN_VAL_FALLBACK_DAYS = 3      # Fallback if <20 games in last 7 days
TRAIN_MIN_VAL_GAMES = 20         # Minimum validation games before using fallback
