#!/usr/bin/env python3
"""
Pitching-Specialized Feature Pipeline

Extracts ONLY pitching/defense features for XGBoost moneyline model.
Designed to be orthogonal to BattingFeatureComputer (used by LightGBM).

Sources:
  - team_pitching_quality table (rotation, bullpen, staff depth)
  - ncaa_team_stats table (NCAA-reported pitching/fielding stats)
  - pitcher_game_log / pitching_matchups (starting pitcher stats)
  - games table (runs allowed per game)
  - elo_ratings table (baseline strength signal)

Target: ~40 features, ALL pitching/defense focused.
"""

import sys
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.database import get_connection

DEFAULT_ELO = 1500


class PitchingFeatureComputer:
    """Computes pitching/defense-only features for a single game."""

    def __init__(self):
        pass

    def get_feature_names(self):
        """Return ordered list of feature names matching the feature vector."""
        names = []
        for prefix in ['home_', 'away_']:
            names.extend([
                f'{prefix}rotation_era',
                f'{prefix}rotation_whip',
                f'{prefix}rotation_fip',
                f'{prefix}bullpen_era',
                f'{prefix}bullpen_whip',
                f'{prefix}bullpen_fip',
                f'{prefix}staff_k_per_9',
                f'{prefix}staff_bb_per_9',
                f'{prefix}quality_arms_pct',
                f'{prefix}innings_hhi',
                f'{prefix}rapg',
                f'{prefix}ncaa_era',
                f'{prefix}ncaa_k_per_9',
                f'{prefix}ncaa_fielding_pct',
                f'{prefix}starter_era',
                f'{prefix}starter_whip',
                f'{prefix}starter_known',
            ])
        names.extend([
            'era_diff',
            'fip_diff',
            'elo_diff',
            'is_neutral',
            'is_conference',
            'is_early_season',
        ])
        return names

    def get_num_features(self):
        return len(self.get_feature_names())

    def compute(self, *args, **kwargs):
        return self.compute_features(*args, **kwargs)

    def compute_features(self, home_team_id, away_team_id, game_date=None,
                         neutral_site=False, is_conference=False, game_id=None,
                         weather_data=None):
        """Compute pitching feature vector for a game."""
        if game_date is None:
            game_date = datetime.now().strftime('%Y-%m-%d')
        elif isinstance(game_date, datetime):
            game_date = game_date.strftime('%Y-%m-%d')

        conn = get_connection()

        home_feats = self._team_pitching_features(
            conn, home_team_id, game_date, game_id, is_home=True)
        away_feats = self._team_pitching_features(
            conn, away_team_id, game_date, game_id, is_home=False)

        features = list(home_feats) + list(away_feats)

        # Differentials (home - away, lower is better for pitching)
        features.append(home_feats[11] - away_feats[11])  # era_diff (ncaa_era)
        features.append(home_feats[2] - away_feats[2])    # fip_diff (rotation_fip)

        # Game context
        home_elo = self._get_elo(conn, home_team_id)
        away_elo = self._get_elo(conn, away_team_id)
        features.append(home_elo - away_elo)  # elo_diff

        features.append(1.0 if neutral_site else 0.0)
        features.append(1.0 if is_conference else 0.0)

        # Early season: before March 1
        try:
            month = int(game_date.split('-')[1])
            day = int(game_date.split('-')[2])
            features.append(1.0 if (month < 3 or (month == 3 and day == 1)) else 0.0)
        except (ValueError, IndexError):
            features.append(0.0)

        conn.close()
        return np.array(features, dtype=np.float32)

    def _team_pitching_features(self, conn, team_id, game_date, game_id, is_home):
        """Extract 17 pitching features for a single team."""
        c = conn.cursor()

        # Defaults (league average)
        rotation_era = 4.50
        rotation_whip = 1.35
        rotation_fip = 4.50
        bullpen_era = 4.50
        bullpen_whip = 1.35
        bullpen_fip = 4.50
        staff_k_per_9 = 7.5
        staff_bb_per_9 = 3.5
        quality_arms_pct = 0.20
        innings_hhi = 0.15

        # From team_pitching_quality
        try:
            c.execute("""
                SELECT rotation_era, rotation_whip, rotation_fip,
                       bullpen_era, bullpen_whip, bullpen_fip,
                       staff_k_per_9, staff_bb_per_9,
                       quality_arms, staff_size, innings_hhi
                FROM team_pitching_quality WHERE team_id = ?
            """, (team_id,))
            row = c.fetchone()
            if row:
                rotation_era = row['rotation_era'] or 4.50
                rotation_whip = row['rotation_whip'] or 1.35
                rotation_fip = row['rotation_fip'] or 4.50
                bullpen_era = row['bullpen_era'] or 4.50
                bullpen_whip = row['bullpen_whip'] or 1.35
                bullpen_fip = row['bullpen_fip'] or 4.50
                staff_k_per_9 = row['staff_k_per_9'] or 7.5
                staff_bb_per_9 = row['staff_bb_per_9'] or 3.5
                staff_size = row['staff_size'] or 10
                qa = row['quality_arms'] or 0
                quality_arms_pct = qa / max(staff_size, 1)
                innings_hhi = row['innings_hhi'] or 0.15
        except Exception:
            pass

        # RAPG from completed games
        games = self._get_team_games(conn, team_id, game_date)
        total = len(games)
        rapg = sum(g['runs_allowed'] for g in games) / total if total > 0 else 4.5

        # NCAA stats
        ncaa_era = 4.50
        ncaa_k_per_9 = 7.5
        ncaa_fielding_pct = 0.965
        try:
            c.execute("""
                SELECT stat_name, stat_value FROM ncaa_team_stats
                WHERE team_id = ? AND stat_name IN ('era', 'k_per_9', 'fielding_pct')
                ORDER BY season DESC
            """, (team_id,))
            for row in c.fetchall():
                if row['stat_name'] == 'era' and row['stat_value'] is not None:
                    ncaa_era = row['stat_value']
                elif row['stat_name'] == 'k_per_9' and row['stat_value'] is not None:
                    ncaa_k_per_9 = row['stat_value']
                elif row['stat_name'] == 'fielding_pct' and row['stat_value'] is not None:
                    ncaa_fielding_pct = row['stat_value']
        except Exception:
            pass

        # Starting pitcher
        starter_era, starter_whip, starter_known = self._starter_features(
            conn, team_id, game_id, is_home)

        return (rotation_era, rotation_whip, rotation_fip,
                bullpen_era, bullpen_whip, bullpen_fip,
                staff_k_per_9, staff_bb_per_9, quality_arms_pct, innings_hhi,
                rapg, ncaa_era, ncaa_k_per_9, ncaa_fielding_pct,
                starter_era, starter_whip, starter_known)

    def _starter_features(self, conn, team_id, game_id, is_home):
        """Get starting pitcher stats (era, whip, known)."""
        c = conn.cursor()

        # Team average defaults
        def_era = 4.50
        def_whip = 1.35
        try:
            c.execute("""
                SELECT AVG(era) as avg_era, AVG(whip) as avg_whip
                FROM player_stats
                WHERE team_id = ? AND innings_pitched > 0 AND games_started > 0
            """, (team_id,))
            row = c.fetchone()
            if row and row['avg_era'] is not None:
                def_era = row['avg_era']
                def_whip = row['avg_whip'] or 1.35
        except Exception:
            pass

        if not game_id:
            return (def_era, def_whip, 0.0)

        # Look up starter from pitching_matchups
        col = 'home_starter_id' if is_home else 'away_starter_id'
        try:
            c.execute(
                f"SELECT {col} FROM pitching_matchups WHERE game_id = ?",
                (game_id,))
            row = c.fetchone()
        except Exception:
            return (def_era, def_whip, 0.0)

        if not row or not row[col]:
            return (def_era, def_whip, 0.0)

        starter_id = row[col]

        # Get starter's season stats
        try:
            c.execute("""
                SELECT era, whip FROM player_stats
                WHERE id = ? AND team_id = ?
            """, (starter_id, team_id))
            ps = c.fetchone()
        except Exception:
            return (def_era, def_whip, 1.0)

        if not ps:
            return (def_era, def_whip, 1.0)

        era = ps['era'] if ps['era'] is not None else def_era
        whip = ps['whip'] if ps['whip'] is not None else def_whip

        return (era, whip, 1.0)

    def _get_elo(self, conn, team_id):
        c = conn.cursor()
        c.execute("SELECT rating FROM elo_ratings WHERE team_id = ?", (team_id,))
        row = c.fetchone()
        return row['rating'] if row else DEFAULT_ELO

    def _get_team_games(self, conn, team_id, before_date):
        """Get all completed games for a team before given date."""
        c = conn.cursor()
        c.execute("""
            SELECT home_team_id, away_team_id, home_score, away_score, date
            FROM games
            WHERE (home_team_id = ? OR away_team_id = ?)
              AND status = 'final'
              AND home_score IS NOT NULL
              AND away_score IS NOT NULL
              AND date < ?
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
            })
        return games


class HistoricalPitchingFeatureComputer:
    """
    Computes pitching features from historical_games for training.
    Processes games chronologically with rolling state. No data leakage.
    """

    def __init__(self):
        self.elo = defaultdict(lambda: DEFAULT_ELO)
        self.team_stats = defaultdict(lambda: {
            'wins': 0, 'losses': 0,
            'runs_scored': 0, 'runs_allowed': 0,
            'games': 0,
        })

    def reset(self):
        self.elo = defaultdict(lambda: DEFAULT_ELO)
        self.team_stats = defaultdict(lambda: {
            'wins': 0, 'losses': 0,
            'runs_scored': 0, 'runs_allowed': 0,
            'games': 0,
        })

    def compute_game_features(self, game_row, weather_row=None):
        """Compute pitching features for a historical game BEFORE updating state."""
        home = game_row['home_team']
        away = game_row['away_team']
        date_str = game_row['date']
        neutral = game_row.get('neutral_site', 0)

        features = []

        # Per-team features (17 each, defaults for most)
        team_feats = {}
        for team in [home, away]:
            s = self.team_stats[team]
            gp = s['games']
            rapg = s['runs_allowed'] / gp if gp > 0 else 4.5

            team_feats[team] = (
                4.50,    # rotation_era (default)
                1.35,    # rotation_whip
                4.50,    # rotation_fip
                4.50,    # bullpen_era
                1.35,    # bullpen_whip
                4.50,    # bullpen_fip
                7.5,     # staff_k_per_9
                3.5,     # staff_bb_per_9
                0.20,    # quality_arms_pct
                0.15,    # innings_hhi
                rapg,    # rapg (from rolling state)
                4.50,    # ncaa_era
                7.5,     # ncaa_k_per_9
                0.965,   # ncaa_fielding_pct
                4.50,    # starter_era
                1.35,    # starter_whip
                0.0,     # starter_known
            )

        features.extend(team_feats[home])
        features.extend(team_feats[away])

        # Differentials
        features.append(team_feats[home][11] - team_feats[away][11])  # era_diff
        features.append(team_feats[home][2] - team_feats[away][2])    # fip_diff

        # Game context
        features.append(self.elo[home] - self.elo[away])  # elo_diff
        features.append(1.0 if neutral else 0.0)
        features.append(0.0)  # is_conference (unknown)

        # Early season
        try:
            month = int(date_str.split('-')[1])
            day = int(date_str.split('-')[2])
            features.append(1.0 if (month < 3 or (month == 3 and day == 1)) else 0.0)
        except (ValueError, IndexError):
            features.append(0.0)

        label = 1.0 if game_row['home_score'] > game_row['away_score'] else 0.0
        return np.array(features, dtype=np.float32), label

    def update_state(self, game_row):
        """Update rolling state AFTER computing features."""
        home = game_row['home_team']
        away = game_row['away_team']
        hs = game_row['home_score']
        aws = game_row['away_score']
        neutral = game_row.get('neutral_site', 0)
        home_won = hs > aws

        for team, rs, ra, won in [
            (home, hs, aws, home_won),
            (away, aws, hs, not home_won),
        ]:
            s = self.team_stats[team]
            s['games'] += 1
            s['runs_scored'] += rs
            s['runs_allowed'] += ra
            if won:
                s['wins'] += 1
            else:
                s['losses'] += 1

        # Update Elo
        import math
        home_elo = self.elo[home]
        away_elo = self.elo[away]
        elo_diff = home_elo - away_elo + (0 if neutral else 50)
        expected_home = 1.0 / (1.0 + 10 ** (-elo_diff / 400))
        actual_home = 1.0 if home_won else 0.0
        mov = abs(hs - aws)
        mov_mult = math.log(max(mov, 1) + 1) * (2.2 / (elo_diff * 0.001 + 2.2))
        self.elo[home] += 32 * mov_mult * (actual_home - expected_home)
        self.elo[away] += 32 * mov_mult * (expected_home - actual_home)
