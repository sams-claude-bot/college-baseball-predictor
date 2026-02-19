#!/usr/bin/env python3
"""
Enhanced Feature Computer - adds interaction and momentum features
on top of the base FeatureComputer (opt-in, backward compatible).
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.nn_features import FeatureComputer
from scripts.database import get_connection


class EnhancedFeatureComputer(FeatureComputer):
    """
    Extends FeatureComputer with:
    - Batting quality × pitching quality interaction features
    - Momentum (recent win rates: last 5 and last 10 games)
    """

    def __init__(self, use_model_predictions=False):
        super().__init__(use_model_predictions=use_model_predictions)

    def get_feature_names(self):
        names = super().get_feature_names()
        names.extend([
            'interact_home_bat_x_away_pitch',
            'interact_away_bat_x_home_pitch',
            'home_momentum_5',
            'home_momentum_10',
            'away_momentum_5',
            'away_momentum_10',
        ])
        return names

    def get_num_features(self):
        return len(self.get_feature_names())

    def _get_batting_quality(self, conn, team_id):
        """Get composite batting quality score for a team."""
        row = conn.execute(
            "SELECT lineup_ops, lineup_wrc_plus, runs_per_game FROM team_batting_quality WHERE team_id = ?",
            (team_id,)
        ).fetchone()
        if row is None:
            return 0.0
        ops = row['lineup_ops'] or 0.0
        wrc = (row['lineup_wrc_plus'] or 100.0) / 100.0
        rpg = (row['runs_per_game'] or 5.0) / 5.0
        return (ops + wrc + rpg) / 3.0

    def _get_pitching_quality(self, conn, team_id):
        """Get composite pitching quality score (lower ERA = higher quality)."""
        row = conn.execute(
            "SELECT rotation_era, rotation_fip, bullpen_era FROM team_pitching_quality WHERE team_id = ?",
            (team_id,)
        ).fetchone()
        if row is None:
            return 0.0
        era = row['rotation_era'] or 4.5
        fip = row['rotation_fip'] or 4.5
        bp_era = row['bullpen_era'] or 5.0
        # Invert so higher = better pitching; normalize around 1.0
        composite = (4.5 / max(era, 1.0) + 4.5 / max(fip, 1.0) + 5.0 / max(bp_era, 1.0)) / 3.0
        return composite

    def _get_momentum(self, conn, team_id, game_date, n_games):
        """Get win rate over last n completed games."""
        rows = conn.execute("""
            SELECT home_team_id, away_team_id, home_score, away_score
            FROM games
            WHERE status = 'final'
              AND date < ?
              AND (home_team_id = ? OR away_team_id = ?)
            ORDER BY date DESC
            LIMIT ?
        """, (game_date, team_id, team_id, n_games)).fetchall()
        if not rows:
            return 0.5
        wins = 0
        for r in rows:
            if r['home_team_id'] == team_id and (r['home_score'] or 0) > (r['away_score'] or 0):
                wins += 1
            elif r['away_team_id'] == team_id and (r['away_score'] or 0) > (r['home_score'] or 0):
                wins += 1
        return wins / len(rows)

    def compute_features(self, home_team_id, away_team_id, game_date=None,
                         neutral_site=False, is_conference=False, game_id=None,
                         weather_data=None):
        base = super().compute_features(
            home_team_id, away_team_id, game_date=game_date,
            neutral_site=neutral_site, is_conference=is_conference,
            game_id=game_id, weather_data=weather_data
        )

        if game_date is None:
            game_date = datetime.now().strftime('%Y-%m-%d')
        elif isinstance(game_date, datetime):
            game_date = game_date.strftime('%Y-%m-%d')

        conn = get_connection()

        home_bat = self._get_batting_quality(conn, home_team_id)
        away_bat = self._get_batting_quality(conn, away_team_id)
        home_pitch = self._get_pitching_quality(conn, home_team_id)
        away_pitch = self._get_pitching_quality(conn, away_team_id)

        interact_home = home_bat * away_pitch  # home batting vs away pitching
        interact_away = away_bat * home_pitch

        home_mom5 = self._get_momentum(conn, home_team_id, game_date, 5)
        home_mom10 = self._get_momentum(conn, home_team_id, game_date, 10)
        away_mom5 = self._get_momentum(conn, away_team_id, game_date, 5)
        away_mom10 = self._get_momentum(conn, away_team_id, game_date, 10)

        conn.close()

        enhanced = np.append(base, [
            interact_home, interact_away,
            home_mom5, home_mom10,
            away_mom5, away_mom10,
        ])
        return enhanced
