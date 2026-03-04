#!/usr/bin/env python3
"""
Batting-Specialized Feature Pipeline

Extracts ONLY batting/offense features for LightGBM moneyline model.
Designed to be orthogonal to PitchingFeatureComputer (used by XGBoost).

Sources:
  - team_batting_quality table (lineup stats, power, bench depth)
  - ncaa_team_stats table (NCAA-reported batting stats)
  - games table (actual run production, recent form)
  - elo_ratings table (baseline strength signal)

Target: ~35 features, ALL batting-focused.
"""

import sys
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.database import get_connection

DEFAULT_ELO = 1500


class BattingFeatureComputer:
    """Computes batting-only features for a single game."""

    def __init__(self):
        pass

    def get_feature_names(self):
        """Return ordered list of feature names matching the feature vector."""
        names = []
        for prefix in ['home_', 'away_']:
            names.extend([
                f'{prefix}lineup_ops',
                f'{prefix}lineup_woba',
                f'{prefix}lineup_iso',
                f'{prefix}lineup_babip',
                f'{prefix}lineup_k_pct',
                f'{prefix}lineup_bb_pct',
                f'{prefix}hr_per_game',
                f'{prefix}extra_base_hit_pct',
                f'{prefix}bench_ops',
                f'{prefix}elite_bats_pct',
                f'{prefix}rpg',
                f'{prefix}runs_last10',
                f'{prefix}ncaa_scoring',
                f'{prefix}ncaa_obp',
            ])
        names.extend([
            'ops_diff',
            'woba_diff',
            'rpg_diff',
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
        """Compute batting feature vector for a game."""
        if game_date is None:
            game_date = datetime.now().strftime('%Y-%m-%d')
        elif isinstance(game_date, datetime):
            game_date = game_date.strftime('%Y-%m-%d')

        conn = get_connection()

        home_feats = self._team_batting_features(conn, home_team_id, game_date)
        away_feats = self._team_batting_features(conn, away_team_id, game_date)

        features = list(home_feats) + list(away_feats)

        # Differentials (home - away)
        features.append(home_feats[0] - away_feats[0])   # ops_diff
        features.append(home_feats[1] - away_feats[1])   # woba_diff
        features.append(home_feats[10] - away_feats[10])  # rpg_diff

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

    def _team_batting_features(self, conn, team_id, game_date):
        """Extract 14 batting features for a single team."""
        c = conn.cursor()

        # Defaults (league average)
        lineup_ops = 0.730
        lineup_woba = 0.320
        lineup_iso = 0.140
        lineup_babip = 0.300
        lineup_k_pct = 20.0
        lineup_bb_pct = 8.5
        hr_per_game = 0.8
        extra_base_hit_pct = 0.30
        bench_ops = 0.650
        elite_bats_pct = 0.15

        # From team_batting_quality
        try:
            c.execute("""
                SELECT lineup_ops, lineup_woba, lineup_iso, lineup_babip,
                       lineup_k_pct, lineup_bb_pct, hr_per_game,
                       extra_base_hit_pct, bench_ops,
                       elite_bats, solid_bats, weak_bats
                FROM team_batting_quality WHERE team_id = ?
            """, (team_id,))
            row = c.fetchone()
            if row:
                lineup_ops = row['lineup_ops'] or 0.730
                lineup_woba = row['lineup_woba'] or 0.320
                lineup_iso = row['lineup_iso'] or 0.140
                lineup_babip = row['lineup_babip'] or 0.300
                lineup_k_pct = row['lineup_k_pct'] or 20.0
                lineup_bb_pct = row['lineup_bb_pct'] or 8.5
                hr_per_game = row['hr_per_game'] or 0.8
                extra_base_hit_pct = row['extra_base_hit_pct'] or 0.30
                bench_ops = row['bench_ops'] or 0.650
                elite = row['elite_bats'] or 0
                solid = row['solid_bats'] or 0
                weak = row['weak_bats'] or 0
                total_bats = elite + solid + weak
                elite_bats_pct = elite / max(total_bats, 1)
        except Exception:
            pass

        # RPG and recent runs from completed games
        games = self._get_team_games(conn, team_id, game_date)
        total = len(games)
        rpg = sum(g['runs_scored'] for g in games) / total if total > 0 else 4.5
        last10 = games[-10:]
        runs_last10 = sum(g['runs_scored'] for g in last10) / len(last10) if last10 else 4.5

        # NCAA stats
        ncaa_scoring = 5.0
        ncaa_obp = 0.340
        try:
            c.execute("""
                SELECT stat_name, stat_value FROM ncaa_team_stats
                WHERE team_id = ? AND stat_name IN ('scoring', 'obp')
                ORDER BY season DESC
            """, (team_id,))
            for row in c.fetchall():
                if row['stat_name'] == 'scoring' and row['stat_value'] is not None:
                    ncaa_scoring = row['stat_value']
                elif row['stat_name'] == 'obp' and row['stat_value'] is not None:
                    ncaa_obp = row['stat_value']
        except Exception:
            pass

        return (lineup_ops, lineup_woba, lineup_iso, lineup_babip,
                lineup_k_pct, lineup_bb_pct, hr_per_game, extra_base_hit_pct,
                bench_ops, elite_bats_pct, rpg, runs_last10,
                ncaa_scoring, ncaa_obp)

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


class HistoricalBattingFeatureComputer:
    """
    Computes batting features from historical_games for training.
    Processes games chronologically with rolling state. No data leakage.
    """

    def __init__(self):
        self.elo = defaultdict(lambda: DEFAULT_ELO)
        self.team_stats = defaultdict(lambda: {
            'wins': 0, 'losses': 0,
            'runs_scored': 0, 'runs_allowed': 0,
            'games': 0,
            'recent_runs': [],  # last 10 games runs scored
        })

    def reset(self):
        self.elo = defaultdict(lambda: DEFAULT_ELO)
        self.team_stats = defaultdict(lambda: {
            'wins': 0, 'losses': 0,
            'runs_scored': 0, 'runs_allowed': 0,
            'games': 0,
            'recent_runs': [],
        })

    def compute_game_features(self, game_row, weather_row=None):
        """Compute batting features for a historical game BEFORE updating state."""
        home = game_row['home_team']
        away = game_row['away_team']
        date_str = game_row['date']
        neutral = game_row.get('neutral_site', 0)

        features = []

        # Per-team features (14 each, defaults for most since no batting quality data)
        team_feats = {}
        for team in [home, away]:
            s = self.team_stats[team]
            gp = s['games']
            rpg = s['runs_scored'] / gp if gp > 0 else 4.5
            recent = s['recent_runs']
            runs_last10 = sum(recent[-10:]) / len(recent[-10:]) if recent else 4.5

            team_feats[team] = (
                0.730,   # lineup_ops (default)
                0.320,   # lineup_woba
                0.140,   # lineup_iso
                0.300,   # lineup_babip
                20.0,    # lineup_k_pct
                8.5,     # lineup_bb_pct
                0.8,     # hr_per_game
                0.30,    # extra_base_hit_pct
                0.650,   # bench_ops
                0.15,    # elite_bats_pct
                rpg,     # rpg (from rolling state)
                runs_last10,  # runs_last10
                5.0,     # ncaa_scoring
                0.340,   # ncaa_obp
            )

        features.extend(team_feats[home])
        features.extend(team_feats[away])

        # Differentials
        features.append(team_feats[home][0] - team_feats[away][0])   # ops_diff
        features.append(team_feats[home][1] - team_feats[away][1])   # woba_diff
        features.append(team_feats[home][10] - team_feats[away][10])  # rpg_diff

        # Game context
        features.append(self.elo[home] - self.elo[away])  # elo_diff
        features.append(1.0 if neutral else 0.0)
        features.append(0.0)  # is_conference (unknown in historical)

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
            s['recent_runs'].append(rs)
            if len(s['recent_runs']) > 20:
                s['recent_runs'] = s['recent_runs'][-20:]

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
