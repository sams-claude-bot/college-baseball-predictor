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

# Weather defaults (when data is missing)
DEFAULT_WEATHER = {
    'temp_f': 65.0,
    'humidity_pct': 55.0,
    'wind_speed_mph': 6.0,
    'wind_direction_deg': 180,
    'precip_prob_pct': 5.0,
    'is_dome': 0,
}

# Normalization constants for weather (mean, std)
WEATHER_NORM = {
    'temp_f': (65.0, 15.0),
    'humidity_pct': (55.0, 20.0),
    'wind_speed_mph': (7.0, 5.0),
    'precip_prob_pct': (10.0, 15.0),
}


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
            # Advanced batting
            names.extend([
                f'{prefix}adv_wrc_plus',
                f'{prefix}adv_woba',
                f'{prefix}adv_iso',
                f'{prefix}adv_babip',
                f'{prefix}adv_k_pct',
                f'{prefix}adv_bb_pct',
                f'{prefix}adv_gb_pct',
                f'{prefix}adv_fb_pct',
                f'{prefix}adv_ld_pct',
            ])
            # Advanced pitching
            names.extend([
                f'{prefix}adv_fip',
                f'{prefix}adv_xfip',
                f'{prefix}adv_siera',
                f'{prefix}adv_gb_pct_pitch',
                f'{prefix}adv_fb_pct_pitch',
            ])
            # Staff quality (from team_pitching_quality table)
            names.extend([
                f'{prefix}ace_era',
                f'{prefix}ace_whip',
                f'{prefix}ace_k9',
                f'{prefix}ace_fip',
                f'{prefix}rotation_era',
                f'{prefix}rotation_whip',
                f'{prefix}rotation_k9',
                f'{prefix}rotation_fip',
                f'{prefix}bullpen_era',
                f'{prefix}bullpen_whip',
                f'{prefix}bullpen_k9',
                f'{prefix}staff_depth',        # normalized staff size
                f'{prefix}ace_ip_pct',         # innings concentration
                f'{prefix}innings_hhi',        # Herfindahl index
                f'{prefix}quality_arms_pct',   # quality arms / staff size
            ])
        # Situational (game-level)
        names.extend([
            'is_neutral_site',
            'is_conference_game',
        ])
        # Weather features
        names.extend([
            'weather_temp_norm',
            'weather_humidity_norm',
            'weather_wind_speed_norm',
            'weather_wind_dir_sin',
            'weather_wind_dir_cos',
            'weather_precip_prob_norm',
            'weather_is_dome',
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
                         neutral_site=False, is_conference=False, game_id=None,
                         weather_data=None):
        """
        Compute feature vector for a game.

        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            game_date: Date string (YYYY-MM-DD) or datetime. If None, uses today.
            neutral_site: Whether game is at neutral site
            is_conference: Whether it's a conference game
            game_id: Optional game ID to lookup weather data
            weather_data: Optional dict with weather (overrides lookup)

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
            features.extend(self._advanced_batting_features(conn, team_id))
            features.extend(self._advanced_pitching_features(conn, team_id))
            features.extend(self._staff_quality_features(conn, team_id))

        # Game-level situational
        features.append(1.0 if neutral_site else 0.0)
        features.append(1.0 if is_conference else 0.0)

        # Weather features
        features.extend(self._weather_features(conn, game_id, weather_data))

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

    # ---- Advanced Stats ----

    # League-average defaults
    _ADV_BAT_DEFAULTS = [100.0, 0.320, 0.140, 0.300, 20.0, 8.5, 43.0, 36.0, 21.0]
    _ADV_PITCH_DEFAULTS = [4.00, 4.00, 4.00, 43.0, 36.0]

    def _advanced_batting_features(self, conn, team_id):
        """Team advanced batting stats aggregated from player_stats (AB-weighted)."""
        c = conn.cursor()
        c.execute("""
            SELECT
                SUM(at_bats * wrc_plus) / NULLIF(SUM(CASE WHEN wrc_plus IS NOT NULL THEN at_bats END), 0) as wrc_plus,
                SUM(at_bats * woba) / NULLIF(SUM(CASE WHEN woba IS NOT NULL THEN at_bats END), 0) as woba,
                SUM(at_bats * iso) / NULLIF(SUM(CASE WHEN iso IS NOT NULL THEN at_bats END), 0) as iso,
                SUM(at_bats * babip) / NULLIF(SUM(CASE WHEN babip IS NOT NULL THEN at_bats END), 0) as babip,
                SUM(at_bats * k_pct) / NULLIF(SUM(CASE WHEN k_pct IS NOT NULL THEN at_bats END), 0) as k_pct,
                SUM(at_bats * bb_pct) / NULLIF(SUM(CASE WHEN bb_pct IS NOT NULL THEN at_bats END), 0) as bb_pct,
                SUM(at_bats * gb_pct) / NULLIF(SUM(CASE WHEN gb_pct IS NOT NULL THEN at_bats END), 0) as gb_pct,
                SUM(at_bats * fb_pct) / NULLIF(SUM(CASE WHEN fb_pct IS NOT NULL THEN at_bats END), 0) as fb_pct,
                SUM(at_bats * ld_pct) / NULLIF(SUM(CASE WHEN ld_pct IS NOT NULL THEN at_bats END), 0) as ld_pct
            FROM player_stats
            WHERE team_id = ? AND at_bats > 0
        """, (team_id,))
        row = c.fetchone()
        if row and row['wrc_plus'] is not None:
            return [
                row['wrc_plus'] or 100.0,
                row['woba'] or 0.320,
                row['iso'] or 0.140,
                row['babip'] or 0.300,
                row['k_pct'] or 20.0,
                row['bb_pct'] or 8.5,
                row['gb_pct'] or 43.0,
                row['fb_pct'] or 36.0,
                row['ld_pct'] or 21.0,
            ]
        return list(self._ADV_BAT_DEFAULTS)

    def _advanced_pitching_features(self, conn, team_id):
        """Team advanced pitching stats aggregated from player_stats (IP-weighted)."""
        c = conn.cursor()
        c.execute("""
            SELECT
                SUM(innings_pitched * fip) / NULLIF(SUM(CASE WHEN fip IS NOT NULL THEN innings_pitched END), 0) as fip,
                SUM(innings_pitched * xfip) / NULLIF(SUM(CASE WHEN xfip IS NOT NULL THEN innings_pitched END), 0) as xfip,
                SUM(innings_pitched * siera) / NULLIF(SUM(CASE WHEN siera IS NOT NULL THEN innings_pitched END), 0) as siera,
                SUM(innings_pitched * gb_pct_pitch) / NULLIF(SUM(CASE WHEN gb_pct_pitch IS NOT NULL THEN innings_pitched END), 0) as gb_pct_pitch,
                SUM(innings_pitched * fb_pct_pitch) / NULLIF(SUM(CASE WHEN fb_pct_pitch IS NOT NULL THEN innings_pitched END), 0) as fb_pct_pitch
            FROM player_stats
            WHERE team_id = ? AND innings_pitched > 0
        """, (team_id,))
        row = c.fetchone()
        if row and row['fip'] is not None:
            return [
                row['fip'] or 4.00,
                row['xfip'] or 4.00,
                row['siera'] or 4.00,
                row['gb_pct_pitch'] or 43.0,
                row['fb_pct_pitch'] or 36.0,
            ]
        return list(self._ADV_PITCH_DEFAULTS)

    # ---- Staff Quality (from compute_pitching_quality.py) ----

    # Defaults when team_pitching_quality data is missing
    _STAFF_QUALITY_DEFAULTS = [
        4.50,  # ace_era
        1.35,  # ace_whip
        7.5,   # ace_k9
        4.50,  # ace_fip
        4.50,  # rotation_era
        1.35,  # rotation_whip
        7.5,   # rotation_k9
        4.50,  # rotation_fip
        4.50,  # bullpen_era
        1.35,  # bullpen_whip
        7.5,   # bullpen_k9
        0.5,   # staff_depth (normalized)
        0.25,  # ace_ip_pct
        0.15,  # innings_hhi
        0.2,   # quality_arms_pct
    ]

    def _staff_quality_features(self, conn, team_id):
        """Staff-level pitching quality from team_pitching_quality table."""
        c = conn.cursor()
        try:
            c.execute("""
                SELECT ace_era, ace_whip, ace_k_per_9, ace_fip,
                       rotation_era, rotation_whip, rotation_k_per_9, rotation_fip,
                       bullpen_era, bullpen_whip, bullpen_k_per_9,
                       staff_size, ace_ip_pct, innings_hhi, quality_arms
                FROM team_pitching_quality WHERE team_id = ?
            """, (team_id,))
            row = c.fetchone()
        except Exception:
            return list(self._STAFF_QUALITY_DEFAULTS)

        if not row:
            return list(self._STAFF_QUALITY_DEFAULTS)

        staff_size = row['staff_size'] or 10
        quality_arms = row['quality_arms'] or 0

        return [
            row['ace_era'] or 4.50,
            row['ace_whip'] or 1.35,
            row['ace_k_per_9'] or 7.5,
            row['ace_fip'] or 4.50,
            row['rotation_era'] or 4.50,
            row['rotation_whip'] or 1.35,
            row['rotation_k_per_9'] or 7.5,
            row['rotation_fip'] or 4.50,
            row['bullpen_era'] or 4.50,
            row['bullpen_whip'] or 1.35,
            row['bullpen_k_per_9'] or 7.5,
            min(staff_size / 20.0, 1.0),  # normalize: 20 pitchers = 1.0
            row['ace_ip_pct'] or 0.25,
            row['innings_hhi'] or 0.15,
            quality_arms / max(staff_size, 1),  # quality arms as pct of staff
        ]

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

    def _weather_features(self, conn, game_id=None, weather_data=None):
        """
        Compute normalized weather features for a game.

        Returns 7 features: temp, humidity, wind_speed, wind_dir_sin, 
                           wind_dir_cos, precip_prob, is_dome
        """
        w = dict(DEFAULT_WEATHER)

        if weather_data:
            w.update({k: v for k, v in weather_data.items() if v is not None})
        elif game_id:
            c = conn.cursor()
            c.execute("""
                SELECT temp_f, humidity_pct, wind_speed_mph, wind_direction_deg,
                       precip_prob_pct, is_dome
                FROM game_weather WHERE game_id = ?
            """, (game_id,))
            row = c.fetchone()
            if row:
                if row['temp_f'] is not None:
                    w['temp_f'] = row['temp_f']
                if row['humidity_pct'] is not None:
                    w['humidity_pct'] = row['humidity_pct']
                if row['wind_speed_mph'] is not None:
                    w['wind_speed_mph'] = row['wind_speed_mph']
                if row['wind_direction_deg'] is not None:
                    w['wind_direction_deg'] = row['wind_direction_deg']
                if row['precip_prob_pct'] is not None:
                    w['precip_prob_pct'] = row['precip_prob_pct']
                if row['is_dome'] is not None:
                    w['is_dome'] = row['is_dome']

        # Normalize
        temp_norm = (w['temp_f'] - WEATHER_NORM['temp_f'][0]) / WEATHER_NORM['temp_f'][1]
        humidity_norm = (w['humidity_pct'] - WEATHER_NORM['humidity_pct'][0]) / WEATHER_NORM['humidity_pct'][1]
        wind_speed_norm = (w['wind_speed_mph'] - WEATHER_NORM['wind_speed_mph'][0]) / WEATHER_NORM['wind_speed_mph'][1]
        precip_norm = (w['precip_prob_pct'] - WEATHER_NORM['precip_prob_pct'][0]) / WEATHER_NORM['precip_prob_pct'][1]

        # Circular encoding for wind direction
        wind_dir_rad = math.radians(w['wind_direction_deg'])
        wind_dir_sin = math.sin(wind_dir_rad)
        wind_dir_cos = math.cos(wind_dir_rad)

        is_dome = 1.0 if w['is_dome'] else 0.0

        return [temp_norm, humidity_norm, wind_speed_norm,
                wind_dir_sin, wind_dir_cos, precip_norm, is_dome]

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

    def compute_game_features(self, game_row, weather_row=None):
        """
        Compute features for a historical game BEFORE updating state.

        Args:
            game_row: dict with keys: home_team, away_team, home_score,
                      away_score, date, neutral_site, season
            weather_row: optional dict with weather data from historical_game_weather

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

            # Elo — use default value for historical (no reliable cross-season Elo)
            features.append(1500.0)

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
            features.append(float(min(days_rest, 14)))  # Cap at 14

            # SOS — use default Elo for historical
            features.append(1500.0)

            # Advanced batting defaults (no player_stats in historical)
            features.extend([100.0, 0.320, 0.140, 0.300, 20.0, 8.5, 43.0, 36.0, 21.0])
            # Advanced pitching defaults
            features.extend([4.00, 4.00, 4.00, 43.0, 36.0])

        # Game-level
        features.append(1.0 if neutral else 0.0)
        features.append(0.0)  # conference flag unknown in historical data

        # Weather features (7 features)
        features.extend(self._compute_weather_features(weather_row))

        label = 1.0 if game_row['home_score'] > game_row['away_score'] else 0.0

        return np.array(features, dtype=np.float32), label

    def _compute_weather_features(self, weather_row):
        """Compute normalized weather features (7 features)."""
        w = dict(DEFAULT_WEATHER)

        if weather_row:
            for key in ['temp_f', 'humidity_pct', 'wind_speed_mph', 
                        'wind_direction_deg', 'precip_prob_pct', 'is_dome']:
                if weather_row.get(key) is not None:
                    w[key] = weather_row[key]

        # Normalize
        temp_norm = (w['temp_f'] - WEATHER_NORM['temp_f'][0]) / WEATHER_NORM['temp_f'][1]
        humidity_norm = (w['humidity_pct'] - WEATHER_NORM['humidity_pct'][0]) / WEATHER_NORM['humidity_pct'][1]
        wind_speed_norm = (w['wind_speed_mph'] - WEATHER_NORM['wind_speed_mph'][0]) / WEATHER_NORM['wind_speed_mph'][1]
        precip_norm = (w['precip_prob_pct'] - WEATHER_NORM['precip_prob_pct'][0]) / WEATHER_NORM['precip_prob_pct'][1]

        wind_dir_rad = math.radians(w['wind_direction_deg'])
        wind_dir_sin = math.sin(wind_dir_rad)
        wind_dir_cos = math.cos(wind_dir_rad)

        is_dome = 1.0 if w['is_dome'] else 0.0

        return [temp_norm, humidity_norm, wind_speed_norm,
                wind_dir_sin, wind_dir_cos, precip_norm, is_dome]

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
