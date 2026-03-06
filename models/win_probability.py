#!/usr/bin/env python3
"""
Live Win Probability Calculator for College Baseball

Computes P(home_win) at any game state using:
  1. Pre-game team strength (from model predictions or Elo)
  2. Current score differential
  3. Innings remaining
  4. Base-out state (run expectancy for current half-inning)
  5. Poisson distribution for remaining run scoring

The model:
  - Each team's remaining runs ~ Poisson(λ) where λ is based on
    innings remaining × team run rate, adjusted for base-out state
  - P(home_win) = P(home_final > away_final) using convolution of
    Poisson distributions for remaining runs
  - Pre-game strength encoded as asymmetric run rates

MLB run expectancy matrix (24 base-out states) provides expected
additional runs in the current inning given the situation.

Usage:
    from models.win_probability import WinProbabilityModel

    wp = WinProbabilityModel()
    prob = wp.calculate(
        home_score=3, away_score=2,
        inning=5, inning_half='top', outs=1,
        on_first=True, on_second=False, on_third=False,
        home_team_id='louisville', away_team_id='morehead-state',
        pregame_home_prob=0.85
    )
"""

import math
import sqlite3
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.stats import poisson

DATA_DIR = Path(__file__).parent.parent / 'data'

# ============================================================
# MLB Run Expectancy Matrix (2019-2023 average)
# Expected additional runs scored in remainder of inning
# from each of the 24 base-out states.
# Rows: base state (0=empty, 1=1B, 2=2B, 3=12, 4=3B, 5=13, 6=23, 7=123)
# Cols: outs (0, 1, 2)
# ============================================================
RUN_EXPECTANCY = np.array([
    # 0 outs  1 out   2 outs
    [0.481,  0.254,  0.098],   # bases empty
    [0.859,  0.509,  0.224],   # runner on 1st
    [1.100,  0.664,  0.319],   # runner on 2nd
    [1.437,  0.884,  0.429],   # runners on 1st & 2nd
    [1.350,  0.950,  0.353],   # runner on 3rd
    [1.784,  1.130,  0.478],   # runners on 1st & 3rd
    [1.920,  1.376,  0.580],   # runners on 2nd & 3rd
    [2.282,  1.541,  0.752],   # bases loaded
])

# College baseball averages ~43% more runs than MLB (12.9 vs 9.0 per game)
# Scale the RE matrix accordingly
COLLEGE_RUN_SCALE = 1.43
COLLEGE_RE = RUN_EXPECTANCY * COLLEGE_RUN_SCALE

# Average runs per half-inning in college baseball
# 12.87 total runs / 18 half-innings = 0.715 per half-inning
AVG_RUNS_PER_HALF_INNING = 0.715


def _base_state_index(on_first, on_second, on_third):
    """Convert base occupancy to index (0-7) for RE matrix."""
    return (int(on_first) << 0) | (int(on_second) << 1) | (int(on_third) << 2)


def _expected_runs_this_inning(outs, on_first, on_second, on_third):
    """Expected additional runs in current half-inning from this state."""
    if outs >= 3:
        return 0.0
    idx = _base_state_index(on_first, on_second, on_third)
    return float(COLLEGE_RE[idx, outs])


class WinProbabilityModel:
    """Calculate live win probability for college baseball games."""

    def __init__(self, db_path=None):
        self.db_path = db_path or str(DATA_DIR / 'baseball.db')

    def _get_pregame_prob(self, game_id):
        """Fetch pre-game home win probability from model_predictions."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.row_factory = sqlite3.Row
            # Prefer meta_ensemble, fall back to ensemble
            for model in ('meta_ensemble', 'ensemble'):
                row = conn.execute(
                    'SELECT predicted_home_prob FROM model_predictions '
                    'WHERE game_id=? AND model_name=? '
                    'ORDER BY predicted_at DESC LIMIT 1',
                    (game_id, model)
                ).fetchone()
                if row:
                    conn.close()
                    return float(row['predicted_home_prob'])
            conn.close()
        except Exception:
            pass
        return 0.5  # neutral fallback

    def _team_run_rates(self, pregame_home_prob, total_innings=9):
        """Convert pre-game win probability to per-half-inning run rates.

        Uses the relationship between win probability and run scoring:
        if home team is stronger, they score more runs per inning on average.

        Returns (home_rate, away_rate) per half-inning.
        """
        # Total expected runs per game ~ 12.87 for college baseball
        total_runs_per_game = 12.87

        # Convert win probability to expected run ratio
        # Using log5-style: if P(home) = 0.7, home scores ~58% of runs
        # Logistic mapping: run_share = sigmoid(logit(p) * dampening)
        p = max(0.05, min(0.95, pregame_home_prob))
        logit_p = math.log(p / (1 - p))
        # Dampen the effect — run differential is less extreme than win prob
        dampened = logit_p * 0.6
        home_share = 1.0 / (1.0 + math.exp(-dampened))

        home_runs_per_game = total_runs_per_game * home_share
        away_runs_per_game = total_runs_per_game * (1 - home_share)

        home_rate = home_runs_per_game / total_innings
        away_rate = away_runs_per_game / total_innings

        return home_rate, away_rate

    def calculate(self, home_score, away_score, inning, inning_half,
                  outs, on_first=False, on_second=False, on_third=False,
                  home_team_id=None, away_team_id=None,
                  pregame_home_prob=None, game_id=None, total_innings=9):
        """Calculate P(home_win) at current game state.

        Args:
            home_score: Current home team score
            away_score: Current away team score
            inning: Current inning (1-9+)
            inning_half: 'top' or 'bottom'
            outs: Outs in current half-inning (0-2)
            on_first/second/third: Baserunner occupancy
            pregame_home_prob: Pre-game P(home_win), fetched from DB if None
            game_id: Game ID for DB lookup of pre-game prob
            total_innings: Regulation innings (default 9)

        Returns:
            float: P(home_win) in [0.001, 0.999]
        """
        if home_score is None or away_score is None:
            return 0.5
        if inning is None:
            return pregame_home_prob or 0.5

        home_score = int(home_score)
        away_score = int(away_score)
        inning = int(inning)
        outs = int(outs) if outs is not None else 0

        # Get pre-game probability
        if pregame_home_prob is None:
            if game_id:
                pregame_home_prob = self._get_pregame_prob(game_id)
            else:
                pregame_home_prob = 0.5

        # Team run rates per half-inning
        home_rate, away_rate = self._team_run_rates(pregame_home_prob, total_innings)

        # Calculate remaining half-innings for each team
        is_top = (inning_half == 'top')

        if is_top:
            # Away team is batting
            # Remaining full half-innings after current:
            #   Away: rest of current + (total_innings - inning) more full innings
            #   Home: (total_innings - inning + 1) full innings (hasn't batted this inning yet)
            away_full_remaining = total_innings - inning  # full future innings
            home_full_remaining = total_innings - inning  # includes this inning (bottom)
            # Plus partial current half-inning for away
            away_current_re = _expected_runs_this_inning(outs, on_first, on_second, on_third)
        else:
            # Home team is batting
            # Away: (total_innings - inning) more full innings
            # Home: rest of current + (total_innings - inning) more full innings
            away_full_remaining = total_innings - inning
            home_full_remaining = total_innings - inning  # full future innings
            # Plus partial current half-inning for home
            away_current_re = 0
            home_current_re = _expected_runs_this_inning(outs, on_first, on_second, on_third)

        # Expected remaining runs for each team
        if is_top:
            away_expected = away_current_re * (away_rate / AVG_RUNS_PER_HALF_INNING) + \
                           away_full_remaining * away_rate
            home_expected = (home_full_remaining + 1) * home_rate  # +1 for this inning's bottom
        else:
            away_expected = away_full_remaining * away_rate
            home_expected = _expected_runs_this_inning(outs, on_first, on_second, on_third) * \
                           (home_rate / AVG_RUNS_PER_HALF_INNING) + \
                           home_full_remaining * home_rate

        # Handle late-game / extra innings
        if inning > total_innings:
            # Extra innings: each team expected to score ~0.5 runs per extra inning
            if is_top:
                away_expected = away_current_re * (away_rate / AVG_RUNS_PER_HALF_INNING)
                home_expected = home_rate
            else:
                away_expected = 0
                home_expected = _expected_runs_this_inning(outs, on_first, on_second, on_third) * \
                               (home_rate / AVG_RUNS_PER_HALF_INNING)

        # Walk-off handling: if bottom of last inning and home is ahead, no more at-bats
        if not is_top and inning >= total_innings and home_score > away_score:
            return 0.999  # home has already won (or is winning in walk-off position)

        # Compute P(home_win) using Poisson convolution
        # P(home_win) = P(home_final > away_final)
        # home_final = home_score + home_remaining
        # away_final = away_score + away_remaining
        # Need: P(home_score + H > away_score + A) where H ~ Poisson(home_expected), A ~ Poisson(away_expected)
        # Equivalently: P(H - A > away_score - home_score) = P(H - A > deficit)
        deficit = away_score - home_score  # positive means home is behind

        prob = self._poisson_win_prob(home_expected, away_expected, deficit)

        # Handle bottom of 9th (or later) with home leading — walk-off not needed
        if not is_top and inning >= total_innings:
            if home_score > away_score:
                return 0.999
            # Home batting with chance to walk off
            # Include tie scenario: if currently tied, home just needs 1 run
            # Already handled by Poisson calculation

        return max(0.001, min(0.999, prob))

    def _poisson_win_prob(self, home_lambda, away_lambda, deficit, max_runs=25):
        """Compute P(home_remaining - away_remaining > deficit) using Poisson PMFs.

        Args:
            home_lambda: Expected remaining home runs
            away_lambda: Expected remaining away runs
            deficit: (away_score - home_score), positive = home trailing
        """
        # Clamp lambdas
        home_lambda = max(0.01, home_lambda)
        away_lambda = max(0.01, away_lambda)

        # Compute PMFs
        home_pmf = np.array([poisson.pmf(k, home_lambda) for k in range(max_runs + 1)])
        away_pmf = np.array([poisson.pmf(k, away_lambda) for k in range(max_runs + 1)])

        # P(H - A > deficit) = P(home_final > away_final)
        # = sum over all h,a where h - a > deficit
        prob_win = 0.0
        prob_tie = 0.0
        for h in range(max_runs + 1):
            for a in range(max_runs + 1):
                joint = home_pmf[h] * away_pmf[a]
                if h - a > deficit:
                    prob_win += joint
                elif h - a == deficit:
                    prob_tie += joint

        # Tie probability: roughly 50/50 but slight home advantage in extras
        # In college baseball, home team wins ~52% of extra-inning games
        prob_win += prob_tie * 0.52

        return prob_win

    def game_wp_timeline(self, game_id):
        """Generate win probability timeline for a completed or in-progress game.

        Returns list of dicts: [{timestamp, inning, half, outs, home_score,
        away_score, home_wp, on_first, on_second, on_third}, ...]
        """
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row

        # Get pre-game probability
        pregame_prob = self._get_pregame_prob(game_id)

        # Get deduplicated state transitions
        rows = conn.execute("""
            SELECT 
                MIN(created_at) as timestamp,
                json_extract(data_json, '$.inning') as inning,
                json_extract(data_json, '$.inning_half') as half,
                json_extract(data_json, '$.outs') as outs,
                json_extract(data_json, '$.home_score') as home_score,
                json_extract(data_json, '$.visitor_score') as away_score,
                json_extract(data_json, '$.on_first') as on_first,
                json_extract(data_json, '$.on_second') as on_second,
                json_extract(data_json, '$.on_third') as on_third
            FROM live_events
            WHERE game_id = ? AND event_type IN ('sb_situation', 'sa_situation')
              AND json_extract(data_json, '$.inning') IS NOT NULL
            GROUP BY inning, half, outs, home_score, away_score, on_first, on_second, on_third
            ORDER BY MIN(id)
        """, (game_id,)).fetchall()

        # Get team info
        game_info = conn.execute("""
            SELECT home_team_id, away_team_id, h.name as home_name, a.name as away_name,
                   home_score as final_home, away_score as final_away, status
            FROM games g
            JOIN teams h ON g.home_team_id = h.id
            JOIN teams a ON g.away_team_id = a.id
            WHERE g.id = ?
        """, (game_id,)).fetchone()
        conn.close()

        if not game_info:
            return {'error': 'Game not found', 'timeline': []}

        timeline = []

        # Add pre-game point
        timeline.append({
            'timestamp': None,
            'inning': 0,
            'half': 'pre',
            'outs': 0,
            'home_score': 0,
            'away_score': 0,
            'home_wp': round(pregame_prob, 4),
            'label': 'Pre-game',
        })

        for row in rows:
            inning = row['inning']
            half = row['half']
            outs = row['outs']
            home_score = row['home_score']
            away_score = row['away_score']

            if inning is None or home_score is None or away_score is None:
                continue

            wp = self.calculate(
                home_score=home_score,
                away_score=away_score,
                inning=inning,
                inning_half=half,
                outs=int(outs) if outs is not None else 0,
                on_first=bool(row['on_first']),
                on_second=bool(row['on_second']),
                on_third=bool(row['on_third']),
                pregame_home_prob=pregame_prob,
                game_id=game_id,
            )

            timeline.append({
                'timestamp': row['timestamp'],
                'inning': int(inning),
                'half': half,
                'outs': int(outs) if outs is not None else 0,
                'home_score': int(home_score),
                'away_score': int(away_score),
                'home_wp': round(wp, 4),
            })

        # Add final state if game is complete
        if game_info['status'] == 'final':
            final_wp = 1.0 if game_info['final_home'] > game_info['final_away'] else 0.0
            if timeline and timeline[-1]['home_wp'] != final_wp:
                timeline.append({
                    'timestamp': timeline[-1]['timestamp'] if timeline else None,
                    'inning': timeline[-1]['inning'] if timeline else 9,
                    'half': 'final',
                    'outs': 3,
                    'home_score': int(game_info['final_home']),
                    'away_score': int(game_info['final_away']),
                    'home_wp': final_wp,
                    'label': 'Final',
                })

        return {
            'game_id': game_id,
            'home_team': game_info['home_name'],
            'away_team': game_info['away_name'],
            'home_team_id': game_info['home_team_id'],
            'away_team_id': game_info['away_team_id'],
            'pregame_home_wp': round(pregame_prob, 4),
            'status': game_info['status'],
            'timeline': timeline,
        }


if __name__ == '__main__':
    import sys
    import json

    wp = WinProbabilityModel()

    if len(sys.argv) > 1:
        game_id = sys.argv[1]
    else:
        # Demo with a recent game
        game_id = '2026-03-03_morehead-state_louisville'

    result = wp.game_wp_timeline(game_id)
    print(f"\n{'='*60}")
    print(f"  Win Probability: {result.get('away_team', '?')} @ {result.get('home_team', '?')}")
    print(f"  Pre-game: {result['pregame_home_wp']*100:.1f}% {result.get('home_team', 'Home')}")
    print(f"{'='*60}\n")

    for pt in result['timeline']:
        inn = pt['inning']
        half = pt['half']
        if inn == 0:
            label = 'Pre-game'
        else:
            arrow = '▲' if half == 'top' else '▼' if half == 'bottom' else '■'
            label = f"{arrow}{inn} {pt['outs']}out"
        score = f"{pt['away_score']}-{pt['home_score']}"
        wp_pct = pt['home_wp'] * 100
        bar = '█' * int(wp_pct / 2) + '░' * (50 - int(wp_pct / 2))
        print(f"  {label:<14} {score:>5}  {bar} {wp_pct:5.1f}%")
