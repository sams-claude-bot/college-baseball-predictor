#!/usr/bin/env python3
"""
Neural Network Feature Pipeline

Extracts feature vectors for game predictions. Takes two teams and a date,
returns a numpy array of features suitable for the neural network.

Handles missing data gracefully with sensible defaults.
"""

import sys
import math
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.database import get_connection

# Feature names for documentation and debugging
FEATURE_NAMES = []

# --- Constants ---
DEFAULT_ELO = 1500
ELO_K = 32
ELO_HOME_ADV = 50


class FeatureComputer:
    """Computes features for a single game given team IDs and date."""

    def __init__(self, use_model_predictions=True):
        """
        Args:
            use_model_predictions: Whether to include predictions from other
                models as meta-features (model stacking). Set False for
                historical feature computation where models aren't available.
        """
        self.use_model_predictions = use_model_predictions
        self._models = None

    def _get_models(self):
        """Lazy-load prediction models for meta-features."""
        if self._models is None and self.use_model_predictions:
            try:
                from models.elo_model import EloModel
                from models.advanced_model import AdvancedModel
                from models.pitching_model import PitchingModel
                from models.pythagorean_model import PythagoreanModel
                from models.log5_model import Log5Model
                self._models = {
                    'elo': EloModel(),
                    'advanced': AdvancedModel(),
                    'pitching': PitchingModel(),
                    'pythagorean': PythagoreanModel(),
                    'log5': Log5Model(),
                }
            except Exception:
                self._models = {}
        return self._models or {}

    def get_feature_names(self):
        """Return ordered list of feature names matching the feature vector."""
        names = []
        for prefix in ['home_', 'away_']:
            # Team strength
            names.extend([
                f'{prefix}elo',
                f'{prefix}win_pct_all',
                f'{prefix}win_pct_last10',
                f'{prefix}win_pct_last20',
                f'{prefix}run_diff_per_game',
                f'{prefix}pythag_win_pct',
            ])
            # Batting
            names.extend([
                f'{prefix}batting_avg',
                f'{prefix}obp',
                f'{prefix}slg',
                f'{prefix}ops',
                f'{prefix}hr_per_game',
                f'{prefix}bb_per_game',
                f'{prefix}so_per_game',
                f'{prefix}runs_per_game',
            ])
            # Pitching
            names.extend([
                f'{prefix}era',
                f'{prefix}whip',
                f'{prefix}k_per_9',
                f'{prefix}bb_per_9',
                f'{prefix}hr_per_9',
                f'{prefix}opp_batting_avg',
            ])
            # Situational (per-team)
            names.extend([
                f'{prefix}days_rest',
                f'{prefix}sos',
            ])
        # Situational (game-level)
        names.extend([
            'is_neutral_site',
            'is_conference_game',
        ])
        # Meta features
        if self.use_model_predictions:
            names.extend([
                'meta_elo_prob',
                'meta_advanced_prob',
                'meta_pitching_prob',
                'meta_pythagorean_prob',
                'meta_log5_prob',
            ])
        return names

    def get_num_features(self):
        """Return the number of features in the vector."""
        return len(self.get_feature_names())

    def compute_features(self, home_team_id, away_team_id, game_date=None,
                         neutral_site=False, is_conference=False):
        """
        Compute feature vector for a game.

        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            game_date: Date string (YYYY-MM-DD) or datetime. If None, uses today.
            neutral_site: Whether game is at neutral site
            is_conference: Whether it's a conference game

        Returns:
            numpy array of features
        """
        if game_date is None:
            game_date = datetime.now().strftime('%Y-%m-%d')
        elif isinstance(game_date, datetime):
            game_date = game_date.strftime('%Y-%m-%d')

        conn = get_connection()
        features = []

        for team_id, opp_id in [(home_team_id, away_team_id),
                                 (away_team_id, home_team_id)]:
            features.extend(self._team_strength_features(conn, team_id, game_date))
            features.extend(self._batting_features(conn, team_id, game_date))
            features.extend(self._pitching_features(conn, team_id, game_date))
            features.extend(self._situational_team_features(conn, team_id, game_date))

        # Game-level situational
        features.append(1.0 if neutral_site else 0.0)
        features.append(1.0 if is_conference else 0.0)

        # Meta features from other models
        if self.use_model_predictions:
            features.extend(self._meta_features(home_team_id, away_team_id, neutral_site))

        conn.close()
        return np.array(features, dtype=np.float32)

    # ---- Team Strength ----

    def _team_strength_features(self, conn, team_id, game_date):
        """Elo, win%, run diff, pythagorean."""
        c = conn.cursor()

        # Elo rating
        c.execute("SELECT rating FROM elo_ratings WHERE team_id = ?", (team_id,))
        row = c.fetchone()
        elo = row['rating'] if row else DEFAULT_ELO

        # Get completed games before this date
        games = self._get_team_games(conn, team_id, game_date)

        total = len(games)
        wins_all = sum(1 for g in games if g['won'])
        win_pct_all = wins_all / total if total > 0 else 0.5

        last10 = games[-10:]
        wins10 = sum(1 for g in last10 if g['won'])
        win_pct_10 = wins10 / len(last10) if last10 else 0.5

        last20 = games[-20:]
        wins20 = sum(1 for g in last20 if g['won'])
        win_pct_20 = wins20 / len(last20) if last20 else 0.5

        rs_total = sum(g['runs_scored'] for g in games)
        ra_total = sum(g['runs_allowed'] for g in games)
        run_diff = (rs_total - ra_total) / total if total > 0 else 0.0

        # Pythagorean win%
        if rs_total > 0 or ra_total > 0:
            rs2 = rs_total ** 2
            ra2 = ra_total ** 2
            pythag = rs2 / (rs2 + ra2) if (rs2 + ra2) > 0 else 0.5
        else:
            pythag = 0.5

        return [elo, win_pct_all, win_pct_10, win_pct_20, run_diff, pythag]

    # ---- Batting ----

    def _batting_features(self, conn, team_id, game_date):
        """Team batting stats."""
        c = conn.cursor()

        # Try team_stats table for batting_avg and runs
        c.execute("""
            SELECT * FROM team_stats WHERE team_id = ?
            ORDER BY season DESC LIMIT 1
        """, (team_id,))
        row = c.fetchone()

        # Try to get detailed stats from player aggregation
        c.execute("""
            SELECT SUM(at_bats) as ab, SUM(hits) as h, SUM(home_runs) as hr,
                   SUM(walks) as bb, SUM(strikeouts) as so,
                   SUM(doubles) as d2, SUM(triples) as d3, SUM(rbis) as rbi
            FROM players WHERE team_id = ? AND at_bats > 0
        """, (team_id,))
        prow = c.fetchone()

        games = self._get_team_games(conn, team_id, game_date)
        total = len(games)
        rpg = sum(g['runs_scored'] for g in games) / total if total > 0 else 4.5

        if prow and prow['ab'] and prow['ab'] > 0:
            ab = prow['ab']
            hits = prow['h'] or 0
            hr = prow['hr'] or 0
            bb = prow['bb'] or 0
            so = prow['so'] or 0
            d2 = prow['d2'] or 0
            d3 = prow['d3'] or 0
            gp = max(total, 1)

            avg = hits / ab
            obp = (hits + bb) / (ab + bb) if (ab + bb) > 0 else 0.320
            tb = hits + d2 + 2 * d3 + 3 * hr
            slg = tb / ab
            ops = obp + slg

            return [avg, obp, slg, ops, hr / gp, bb / gp, so / gp, rpg]

        # Use team_stats batting_avg if available
        avg = row['batting_avg'] if row and row['batting_avg'] else 0.260

        return [avg, 0.330, 0.400, 0.730, 0.8, 3.5, 7.0, rpg]

    # ---- Pitching ----

    def _pitching_features(self, conn, team_id, game_date):
        """Team pitching stats."""
        c = conn.cursor()

        # Aggregate from player pitching stats
        c.execute("""
            SELECT SUM(innings_pitched) as ip, SUM(earned_runs) as er,
                   SUM(strikeouts_pitched) as k, SUM(walks_pitched) as bb
            FROM players WHERE team_id = ? AND innings_pitched > 0
        """, (team_id,))
        prow = c.fetchone()

        if prow and prow['ip'] and prow['ip'] > 0:
            ip = prow['ip']
            er = prow['er'] or 0
            k = prow['k'] or 0
            bb = prow['bb'] or 0

            era = (er * 9.0 / ip)
            whip_est = 1.35  # can't compute hits allowed from this schema
            k9 = (k * 9.0 / ip)
            bb9 = (bb * 9.0 / ip)

            return [era, whip_est, k9, bb9, 0.8, 0.260]

        # Try team_stats ERA
        c.execute("""
            SELECT era FROM team_stats WHERE team_id = ?
            ORDER BY season DESC LIMIT 1
        """, (team_id,))
        row = c.fetchone()
        era = row['era'] if row and row['era'] else 4.50

        # Estimate from runs allowed
        games = self._get_team_games(conn, team_id, game_date)
        if games:
            rapg = sum(g['runs_allowed'] for g in games) / len(games)
            era = min(era, rapg * 0.9)  # ERA slightly less than RA

        return [era, 1.35, 7.5, 3.5, 0.8, 0.260]

    # ---- Situational (per-team) ----

    def _situational_team_features(self, conn, team_id, game_date):
        """Rest days, strength of schedule."""
        c = conn.cursor()

        # Days since last game
        c.execute("""
            SELECT date FROM games
            WHERE (home_team_id = ? OR away_team_id = ?)
            AND status = 'final' AND date < ?
            ORDER BY date DESC LIMIT 1
        """, (team_id, team_id, game_date))
        row = c.fetchone()

        if row:
            try:
                last_date = datetime.strptime(row['date'], '%Y-%m-%d')
                current = datetime.strptime(game_date, '%Y-%m-%d')
                days_rest = (current - last_date).days
            except (ValueError, TypeError):
                days_rest = 2.0
        else:
            days_rest = 3.0

        # Strength of schedule: average Elo of opponents faced
        games = self._get_team_games(conn, team_id, game_date)
        if games:
            opp_elos = []
            for g in games:
                c.execute("SELECT rating FROM elo_ratings WHERE team_id = ?",
                          (g['opponent_id'],))
                r = c.fetchone()
                opp_elos.append(r['rating'] if r else DEFAULT_ELO)
            sos = sum(opp_elos) / len(opp_elos)
        else:
            sos = DEFAULT_ELO

        return [float(days_rest), sos]

    # ---- Meta (model stacking) ----

    def _meta_features(self, home_team_id, away_team_id, neutral_site):
        """Predictions from existing models as features."""
        models = self._get_models()
        probs = []
        for name in ['elo', 'advanced', 'pitching', 'pythagorean', 'log5']:
            model = models.get(name)
            if model:
                try:
                    pred = model.predict_game(home_team_id, away_team_id,
                                              neutral_site=neutral_site)
                    probs.append(pred.get('home_win_probability', 0.5))
                except Exception:
                    probs.append(0.5)
            else:
                probs.append(0.5)
        return probs

    # ---- Helpers ----

    def _get_team_games(self, conn, team_id, before_date):
        """Get all completed games for a team before given date."""
        c = conn.cursor()
        c.execute("""
            SELECT home_team_id, away_team_id, home_score, away_score, date
            FROM games
            WHERE (home_team_id = ? OR away_team_id = ?)
            AND status = 'final' AND date < ?
            ORDER BY date ASC
        """, (team_id, team_id, before_date))

        games = []
        for row in c.fetchall():
            is_home = row['home_team_id'] == team_id
            games.append({
                'date': row['date'],
                'runs_scored': row['home_score'] if is_home else row['away_score'],
                'runs_allowed': row['away_score'] if is_home else row['home_score'],
                'won': (row['home_score'] > row['away_score']) == is_home,
                'opponent_id': row['away_team_id'] if is_home else row['home_team_id'],
            })
        return games


class HistoricalFeatureComputer:
    """
    Computes features from historical_games table for training.
    Processes games chronologically, maintaining rolling state per team.
    No data leakage: features for game N use only games 0..N-1.
    """

    def __init__(self):
        self.elo = defaultdict(lambda: DEFAULT_ELO)
        self.team_stats = defaultdict(lambda: {
            'wins': 0, 'losses': 0,
            'runs_scored': 0, 'runs_allowed': 0,
            'games': 0,
            'recent_results': [],  # list of bools (win/loss), last 20
            'last_game_date': None,
            'opponents': [],  # list of opponent team names
        })

    def reset(self):
        """Reset all state (call between seasons if desired)."""
        self.elo = defaultdict(lambda: DEFAULT_ELO)
        self.team_stats = defaultdict(lambda: {
            'wins': 0, 'losses': 0,
            'runs_scored': 0, 'runs_allowed': 0,
            'games': 0,
            'recent_results': [],
            'last_game_date': None,
            'opponents': [],
        })

    def compute_game_features(self, game_row):
        """
        Compute features for a historical game BEFORE updating state.

        Args:
            game_row: dict with keys: home_team, away_team, home_score,
                      away_score, date, neutral_site, season

        Returns:
            (features_array, label) where label is 1.0 if home won, 0.0 otherwise
        """
        home = game_row['home_team']
        away = game_row['away_team']
        date_str = game_row['date']
        neutral = game_row.get('neutral_site', 0)

        features = []

        for team, opp in [(home, away), (away, home)]:
            s = self.team_stats[team]
            gp = s['games']

            # Elo
            features.append(self.elo[team])

            # Win %
            total_wins = s['wins']
            win_pct_all = total_wins / gp if gp > 0 else 0.5
            features.append(win_pct_all)

            recent = s['recent_results']
            last10 = recent[-10:]
            last20 = recent[-20:]
            features.append(sum(last10) / len(last10) if last10 else 0.5)
            features.append(sum(last20) / len(last20) if last20 else 0.5)

            # Run differential per game
            rd = (s['runs_scored'] - s['runs_allowed']) / gp if gp > 0 else 0.0
            features.append(rd)

            # Pythagorean
            rs2 = s['runs_scored'] ** 2
            ra2 = s['runs_allowed'] ** 2
            features.append(rs2 / (rs2 + ra2) if (rs2 + ra2) > 0 else 0.5)

            # Batting (computed from historical, limited info)
            rpg = s['runs_scored'] / gp if gp > 0 else 4.5
            # We don't have detailed batting from historical_games, use defaults
            features.extend([0.260, 0.330, 0.400, 0.730,  # avg, obp, slg, ops
                             0.8, 3.5, 7.0,  # hr/g, bb/g, so/g
                             rpg])

            # Pitching (from run allowed)
            rapg = s['runs_allowed'] / gp if gp > 0 else 4.5
            era_est = rapg  # rough approximation
            features.extend([era_est, 1.35, 7.5, 3.5, 0.8, 0.260])

            # Situational per-team
            if s['last_game_date']:
                try:
                    last_d = datetime.strptime(s['last_game_date'], '%Y-%m-%d')
                    curr_d = datetime.strptime(date_str, '%Y-%m-%d')
                    days_rest = (curr_d - last_d).days
                except (ValueError, TypeError):
                    days_rest = 2.0
            else:
                days_rest = 3.0
            features.append(float(days_rest))

            # SOS: avg opponent elo
            if s['opponents']:
                sos = sum(self.elo[o] for o in s['opponents']) / len(s['opponents'])
            else:
                sos = DEFAULT_ELO
            features.append(sos)

        # Game-level
        features.append(1.0 if neutral else 0.0)
        features.append(0.0)  # conference flag unknown in historical data

        # No meta features for historical
        # (those would be added by live FeatureComputer only)

        label = 1.0 if game_row['home_score'] > game_row['away_score'] else 0.0

        return np.array(features, dtype=np.float32), label

    def update_state(self, game_row):
        """Update rolling state AFTER computing features for this game."""
        home = game_row['home_team']
        away = game_row['away_team']
        hs = game_row['home_score']
        aws = game_row['away_score']
        date_str = game_row['date']
        neutral = game_row.get('neutral_site', 0)

        home_won = hs > aws

        # Update team stats
        for team, opp, rs, ra, won in [
            (home, away, hs, aws, home_won),
            (away, home, aws, hs, not home_won),
        ]:
            s = self.team_stats[team]
            s['games'] += 1
            s['runs_scored'] += rs
            s['runs_allowed'] += ra
            if won:
                s['wins'] += 1
            else:
                s['losses'] += 1
            s['recent_results'].append(1.0 if won else 0.0)
            if len(s['recent_results']) > 50:
                s['recent_results'] = s['recent_results'][-50:]
            s['last_game_date'] = date_str
            s['opponents'].append(opp)

        # Update Elo
        home_elo = self.elo[home]
        away_elo = self.elo[away]
        elo_diff = home_elo - away_elo + (0 if neutral else ELO_HOME_ADV)
        expected_home = 1.0 / (1.0 + 10 ** (-elo_diff / 400))
        actual_home = 1.0 if home_won else 0.0

        # Margin of victory multiplier
        mov = abs(hs - aws)
        mov_mult = math.log(max(mov, 1) + 1) * (2.2 / (elo_diff * 0.001 + 2.2))

        self.elo[home] += ELO_K * mov_mult * (actual_home - expected_home)
        self.elo[away] += ELO_K * mov_mult * (expected_home - actual_home)
