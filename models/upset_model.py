#!/usr/bin/env python3
"""
Upset Model — Contrarian Signal

Trained to detect when favorites are vulnerable:
- Elo gap magnitude
- Favorite's declining recent form vs season average
- Favorite's shrinking margins
- Underdog's improving form
- Conference game flag (rivals know each other)
- Day of week (midweek = weaker rotations)
- Underdog's away win %
- Favorite's home loss count
- Historical upset rate for elo gap bucket

Trained ONLY on games with a clear favorite (>60% elo-implied).
Uses Random Forest for non-linear interaction detection.
"""

import math
import pickle
from pathlib import Path

from models.base_model import BaseModel
from scripts.database import get_connection

MODEL_PATH = Path(__file__).parent.parent / "data" / "upset_model.pkl"

# Elo-based win probability
ELO_HOME_ADV = 50  # elo points for home advantage


def elo_expected(rating_a, rating_b, home_adv=0):
    """Expected score for A vs B with optional home advantage."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a - home_adv) / 400))


class UpsetModel(BaseModel):
    name = "upset"
    version = "1.0"
    description = "Contrarian upset-detection model (identifies vulnerable favorites)"

    FEATURE_NAMES = [
        'elo_gap', 'fav_recent_form_vs_avg', 'fav_margin_trend',
        'dog_recent_form_vs_avg', 'is_conference', 'is_midweek',
        'dog_away_win_pct', 'fav_home_losses', 'hist_upset_rate',
    ]

    def __init__(self):
        self.model = None
        self._elo_cache = {}
        self._load_model()

    def _load_model(self):
        if MODEL_PATH.exists():
            with open(MODEL_PATH, 'rb') as f:
                data = pickle.load(f)
            self.model = data['model']
            self._elo_gap_upset_rates = data.get('elo_gap_upset_rates', {})

    def _get_elo(self, team_id):
        if team_id in self._elo_cache:
            return self._elo_cache[team_id]
        conn = get_connection()
        c = conn.cursor()
        c.execute("SELECT rating FROM elo_ratings WHERE team_id = ?",
                  (team_id,))
        row = c.fetchone()
        conn.close()
        rating = row['rating'] if row else 1500.0
        self._elo_cache[team_id] = rating
        return rating

    def _get_team_form(self, team_id):
        """Get recent form (last 5) vs season average, margins, etc."""
        conn = get_connection()
        c = conn.cursor()

        # Season record
        c.execute("""
            SELECT
                COUNT(*) as games,
                SUM(CASE WHEN winner_id = ? THEN 1 ELSE 0 END) as wins,
                AVG(CASE WHEN home_team_id = ?
                    THEN home_score - away_score
                    ELSE away_score - home_score END) as avg_margin
            FROM games
            WHERE (home_team_id = ? OR away_team_id = ?)
              AND status = 'final' AND home_score IS NOT NULL
        """, (team_id, team_id, team_id, team_id))
        season = c.fetchone()

        # Last 5 games
        c.execute("""
            SELECT winner_id,
                   CASE WHEN home_team_id = ?
                       THEN home_score - away_score
                       ELSE away_score - home_score END as margin
            FROM games
            WHERE (home_team_id = ? OR away_team_id = ?)
              AND status = 'final' AND home_score IS NOT NULL
            ORDER BY date DESC LIMIT 5
        """, (team_id, team_id, team_id))
        recent = [dict(r) for r in c.fetchall()]

        # Home losses
        c.execute("""
            SELECT COUNT(*) as home_losses
            FROM games
            WHERE home_team_id = ? AND status = 'final'
              AND winner_id != ? AND winner_id IS NOT NULL
        """, (team_id, team_id))
        hl_row = c.fetchone()

        # Away win %
        c.execute("""
            SELECT COUNT(*) as away_games,
                   SUM(CASE WHEN winner_id = ? THEN 1 ELSE 0 END) as away_wins
            FROM games
            WHERE away_team_id = ? AND status = 'final'
              AND winner_id IS NOT NULL
        """, (team_id, team_id))
        away_row = c.fetchone()

        conn.close()

        games = season['games'] or 0
        season_win_pct = (season['wins'] or 0) / games if games > 0 else 0.5
        season_avg_margin = season['avg_margin'] or 0.0

        if recent:
            recent_win_pct = sum(1 for r in recent if r['winner_id'] == team_id) / len(recent)
            recent_avg_margin = sum(r['margin'] for r in recent) / len(recent)
        else:
            recent_win_pct = season_win_pct
            recent_avg_margin = season_avg_margin

        away_games = away_row['away_games'] or 0
        away_win_pct = ((away_row['away_wins'] or 0) / away_games
                        if away_games > 0 else 0.5)

        return {
            'season_win_pct': season_win_pct,
            'recent_win_pct': recent_win_pct,
            'form_vs_avg': recent_win_pct - season_win_pct,
            'season_avg_margin': season_avg_margin,
            'recent_avg_margin': recent_avg_margin,
            'margin_trend': recent_avg_margin - season_avg_margin,
            'home_losses': hl_row['home_losses'] or 0,
            'away_win_pct': away_win_pct,
        }

    def _get_historical_upset_rate(self, elo_gap):
        """Lookup upset rate for this elo gap bucket from training data."""
        if hasattr(self, '_elo_gap_upset_rates') and self._elo_gap_upset_rates:
            # Bucket by 50-elo increments
            bucket = int(abs(elo_gap) // 50) * 50
            return self._elo_gap_upset_rates.get(bucket, 0.30)
        # Default rates by rough elo gap
        gap = abs(elo_gap)
        if gap < 50:
            return 0.42
        elif gap < 100:
            return 0.38
        elif gap < 150:
            return 0.33
        elif gap < 200:
            return 0.28
        elif gap < 300:
            return 0.22
        else:
            return 0.15

    def _build_features(self, home_team_id, away_team_id, game_date=None):
        """Build upset detection features. Returns (features, fav_is_home)."""
        home_elo = self._get_elo(home_team_id)
        away_elo = self._get_elo(away_team_id)

        # Determine favorite (with home advantage)
        home_exp = elo_expected(home_elo, away_elo, ELO_HOME_ADV)
        fav_is_home = home_exp >= 0.5

        if fav_is_home:
            fav_id, dog_id = home_team_id, away_team_id
            fav_elo, dog_elo = home_elo, away_elo
            elo_gap = (fav_elo + ELO_HOME_ADV) - dog_elo
        else:
            fav_id, dog_id = away_team_id, home_team_id
            fav_elo, dog_elo = away_elo, home_elo
            elo_gap = fav_elo - (dog_elo + ELO_HOME_ADV)

        fav_form = self._get_team_form(fav_id)
        dog_form = self._get_team_form(dog_id)

        # Conference game check
        conn = get_connection()
        c = conn.cursor()
        c.execute("SELECT conference FROM teams WHERE id = ?", (home_team_id,))
        h_conf = c.fetchone()
        c.execute("SELECT conference FROM teams WHERE id = ?", (away_team_id,))
        a_conf = c.fetchone()
        conn.close()
        is_conference = 1 if (h_conf and a_conf and
                              h_conf['conference'] and a_conf['conference'] and
                              h_conf['conference'] == a_conf['conference']) else 0

        # Day of week
        if game_date:
            from datetime import datetime
            if isinstance(game_date, str):
                game_date = datetime.strptime(game_date, '%Y-%m-%d').date()
            dow = game_date.weekday()
        else:
            from datetime import datetime
            dow = datetime.now().weekday()
        is_midweek = 1 if dow in (1, 2, 3) else 0

        features = [
            elo_gap,
            fav_form['form_vs_avg'],
            fav_form['margin_trend'],
            dog_form['form_vs_avg'],
            is_conference,
            is_midweek,
            dog_form['away_win_pct'],
            fav_form['home_losses'],
            self._get_historical_upset_rate(elo_gap),
        ]

        return features, fav_is_home, home_exp

    def predict_game(self, home_team_id, away_team_id, neutral_site=False,
                     game_date=None):
        features, fav_is_home, base_home_exp = self._build_features(
            home_team_id, away_team_id, game_date)

        if self.model is not None:
            import numpy as np
            X = np.array([features])
            p_upset = float(self.model.predict_proba(X)[0, 1])
        else:
            # Heuristic fallback
            p_upset = features[8]  # historical upset rate
            # Adjust for form
            p_upset += features[1] * -0.1  # fav declining → more upsets
            p_upset += features[3] * 0.1   # dog improving → more upsets
            p_upset = max(0.05, min(0.60, p_upset))

        # Convert P(upset) to standard home/away probabilities
        if fav_is_home:
            home_prob = 1 - p_upset  # Favorite is home; upset = away wins
        else:
            home_prob = p_upset  # Favorite is away; upset = home wins

        if neutral_site:
            home_prob = 0.5 + (home_prob - 0.5) * 0.5

        home_prob = max(0.02, min(0.98, home_prob))

        avg_total = 11.0
        home_runs = avg_total * home_prob
        away_runs = avg_total * (1 - home_prob)

        run_line = self.calculate_run_line(home_runs, away_runs)

        return {
            "model": self.name,
            "home_win_probability": round(home_prob, 3),
            "away_win_probability": round(1 - home_prob, 3),
            "projected_home_runs": round(home_runs, 1),
            "projected_away_runs": round(away_runs, 1),
            "projected_total": round(avg_total, 1),
            "run_line": run_line,
            "inputs": {
                "elo_gap": round(features[0], 1),
                "fav_is_home": fav_is_home,
                "p_upset": round(p_upset, 3),
                "fav_form_vs_avg": round(features[1], 3),
                "dog_form_vs_avg": round(features[3], 3),
                "fav_margin_trend": round(features[2], 2),
                "is_conference": bool(features[4]),
                "hist_upset_rate": round(features[8], 3),
            },
        }
