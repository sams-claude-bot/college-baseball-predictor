#!/usr/bin/env python3
"""
PEAR Ratings Model

Uses PEARatings.com's composite metrics (TSR, Elo, fWAR, Pythag)
to predict games via Log5 with blended adjustments.
"""

import math
from datetime import datetime

from models.base_model import BaseModel
from scripts.database import get_connection

HOME_ADVANTAGE = 0.07


class PearModel(BaseModel):
    name = "pear"
    version = "1.0"
    description = "PEAR Ratings composite model using external TSR/ELO/fWAR metrics"

    def __init__(self):
        self._cache = {}
        self._percentiles = None
        self._season = datetime.now().year

    def _load_team(self, team_id):
        """Load PEAR data for a team, cached per instance."""
        if team_id in self._cache:
            return self._cache[team_id]

        conn = get_connection()
        row = conn.execute(
            "SELECT * FROM pear_ratings WHERE team_id = ? AND season = ?",
            (team_id, self._season),
        ).fetchone()
        conn.close()

        if row:
            self._cache[team_id] = dict(row)
        else:
            self._cache[team_id] = None
        return self._cache[team_id]

    def _load_percentiles(self):
        """Compute TSR percentile lookup from all teams this season.

        Teams with the same rating get the same percentile (average rank).
        """
        if self._percentiles is not None:
            return

        conn = get_connection()
        rows = conn.execute(
            "SELECT team_id, rating FROM pear_ratings WHERE season = ? AND rating IS NOT NULL ORDER BY rating",
            (self._season,),
        ).fetchall()
        conn.close()

        if not rows:
            self._percentiles = {}
            return

        n = len(rows)
        # Group teams by rating, assign average percentile rank to ties
        from collections import defaultdict
        rating_groups = defaultdict(list)
        for i, row in enumerate(rows):
            rating_groups[row["rating"]].append((i, row["team_id"]))

        self._percentiles = {}
        for rating, members in rating_groups.items():
            avg_rank = sum(i + 1 for i, _ in members) / len(members)
            pct = avg_rank / (n + 1)
            for _, team_id in members:
                self._percentiles[team_id] = pct

    def _tsr_to_strength(self, team_id):
        """Convert TSR rating to implied strength (0-1) via percentile."""
        self._load_percentiles()
        pct = self._percentiles.get(team_id, 0.5)
        # Map percentile to a reasonable win probability range
        # A team at 50th percentile is .500, at 100th is ~.800
        return 0.2 + 0.6 * pct

    @staticmethod
    def _log5(pa, pb):
        """Log5 formula: P(A beats B) given true strength pA, pB."""
        denom = pa + pb - 2 * pa * pb
        if denom == 0:
            return 0.5
        return (pa - pa * pb) / denom

    def predict_game(self, home_team_id, away_team_id, neutral_site=False):
        home = self._load_team(home_team_id)
        away = self._load_team(away_team_id)

        if home is None or away is None:
            return None

        # --- Primary: Log5 on TSR percentile ---
        home_str = self._tsr_to_strength(home_team_id)
        away_str = self._tsr_to_strength(away_team_id)
        base_prob = self._log5(home_str, away_str)

        # --- Adjustment: Elo differential ---
        home_elo = home.get("elo") or 1500
        away_elo = away.get("elo") or 1500
        elo_diff = home_elo - away_elo
        # ~400 Elo diff ≈ 0.9 expected, scale contribution gently
        elo_adj = elo_diff / 400 * 0.05  # max ~±0.05

        # --- Adjustment: fWAR differential ---
        home_fwar = home.get("fwar") or 0
        away_fwar = away.get("fwar") or 0
        fwar_diff = home_fwar - away_fwar
        fwar_adj = fwar_diff * 0.015  # scale gently

        # --- Adjustment: Pythag blend ---
        home_pythag = home.get("pythag") or 0.5
        away_pythag = away.get("pythag") or 0.5
        pythag_prob = self._log5(home_pythag, away_pythag)
        # Blend pythag at 15% weight
        blended = 0.85 * base_prob + 0.15 * pythag_prob

        # --- Adjustment: Killshot ratio ---
        home_kshot = home.get("kshot_ratio") or 1.0
        away_kshot = away.get("kshot_ratio") or 1.0
        kshot_diff = home_kshot - away_kshot
        kshot_adj = kshot_diff * 0.005  # very subtle

        # Combine
        home_prob = blended + elo_adj + fwar_adj + kshot_adj

        # Home advantage
        if not neutral_site:
            home_prob += HOME_ADVANTAGE

        # Clamp
        home_prob = max(0.02, min(0.98, home_prob))

        # --- Project runs ---
        home_rpg = home.get("rpg") or 5.5
        away_rpg = away.get("rpg") or 5.5
        home_era = home.get("era") or 5.0
        away_era = away.get("era") or 5.0

        # Home team scores: blend their RPG with opponent's ERA
        projected_home_runs = 0.6 * home_rpg + 0.4 * away_era
        projected_away_runs = 0.6 * away_rpg + 0.4 * home_era

        if not neutral_site:
            projected_home_runs *= 1.03
            projected_away_runs *= 0.97

        run_line = self.calculate_run_line(projected_home_runs, projected_away_runs)

        return {
            "model": self.name,
            "home_win_probability": round(home_prob, 3),
            "away_win_probability": round(1 - home_prob, 3),
            "projected_home_runs": round(projected_home_runs, 1),
            "projected_away_runs": round(projected_away_runs, 1),
            "projected_total": round(projected_home_runs + projected_away_runs, 1),
            "run_line": run_line,
            "inputs": {
                "home_tsr": home.get("rating"),
                "away_tsr": away.get("rating"),
                "home_elo": home_elo,
                "away_elo": away_elo,
                "home_fwar": home_fwar,
                "away_fwar": away_fwar,
                "home_pythag": home_pythag,
                "away_pythag": away_pythag,
            },
        }
