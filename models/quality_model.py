#!/usr/bin/env python3
"""
Quality Metrics Model

Heuristic model that combines batting quality, pitching quality, and
dominance metrics via composite z-score differentials. Uses current-season
data only — no training required.
"""

import math

from models.base_model import BaseModel
from scripts.database import get_connection

HOME_ADVANTAGE = 0.07
LOGISTIC_K = 0.8  # scaling so 1-stdev advantage ≈ 62% win prob


class QualityModel(BaseModel):
    name = "quality"
    version = "1.0"
    description = "Advanced quality metrics model using batting/pitching quality, dominance, and resume"

    def __init__(self):
        self._team_scores = None  # {team_id: {offense_z, pitching_z, dominance_z, team_quality, rpg, staff_era}}

    def _load_all(self):
        """Load all team stats and compute league-wide z-scores once."""
        if self._team_scores is not None:
            return

        conn = get_connection()

        # Batting quality
        batting_rows = conn.execute(
            "SELECT team_id, lineup_ops, lineup_woba, lineup_wrc_plus, runs_per_game FROM team_batting_quality"
        ).fetchall()
        batting = {r["team_id"]: dict(r) for r in batting_rows}

        # Pitching quality
        pitching_rows = conn.execute(
            "SELECT team_id, staff_era, staff_whip, staff_fip, staff_k_per_9, quality_arms, shutdown_arms "
            "FROM team_pitching_quality"
        ).fetchall()
        pitching = {r["team_id"]: dict(r) for r in pitching_rows}

        # Aggregate stats (dominance)
        agg_rows = conn.execute(
            "SELECT team_id, games, blowout_wins, blowout_losses, win_pct, "
            "close_games_wins, close_games_losses, pythagorean_pct "
            "FROM team_aggregate_stats"
        ).fetchall()
        agg = {r["team_id"]: dict(r) for r in agg_rows}

        conn.close()

        # Only include teams present in ALL three tables
        common_teams = set(batting) & set(pitching) & set(agg)
        if not common_teams:
            self._team_scores = {}
            return

        # Collect raw values for z-scoring
        off_ops = []
        off_woba = []
        off_wrc = []
        pitch_era = []
        pitch_whip = []
        pitch_fip = []
        pitch_k9 = []
        dom_vals = []
        team_ids = []

        for tid in common_teams:
            b = batting[tid]
            p = pitching[tid]
            a = agg[tid]
            if b["lineup_ops"] is None or p["staff_era"] is None:
                continue
            games = a.get("games") or 1
            dom = ((a.get("blowout_wins") or 0) - (a.get("blowout_losses") or 0)) / games

            team_ids.append(tid)
            off_ops.append(b["lineup_ops"] or 0)
            off_woba.append(b["lineup_woba"] or 0)
            off_wrc.append(b["lineup_wrc_plus"] or 0)
            pitch_era.append(p["staff_era"] or 0)
            pitch_whip.append(p["staff_whip"] or 0)
            pitch_fip.append(p["staff_fip"] or 0)
            pitch_k9.append(p["staff_k_per_9"] or 0)
            dom_vals.append(dom)

        if not team_ids:
            self._team_scores = {}
            return

        # Z-score helper
        def z_scores(vals):
            n = len(vals)
            mean = sum(vals) / n
            var = sum((v - mean) ** 2 for v in vals) / n
            std = math.sqrt(var) if var > 0 else 1.0
            return [(v - mean) / std for v in vals]

        z_ops = z_scores(off_ops)
        z_woba = z_scores(off_woba)
        z_wrc = z_scores(off_wrc)
        # Invert ERA/WHIP/FIP (lower is better)
        z_era = [-z for z in z_scores(pitch_era)]
        z_whip = [-z for z in z_scores(pitch_whip)]
        z_fip = [-z for z in z_scores(pitch_fip)]
        z_k9 = z_scores(pitch_k9)
        z_dom = z_scores(dom_vals)

        self._team_scores = {}
        for i, tid in enumerate(team_ids):
            offense_z = (z_ops[i] + z_woba[i] + z_wrc[i]) / 3.0
            pitching_z = (z_era[i] + z_whip[i] + z_fip[i] + z_k9[i]) / 4.0
            dominance_z = z_dom[i]
            team_quality = 0.45 * offense_z + 0.45 * pitching_z + 0.10 * dominance_z

            b = batting[tid]
            p = pitching[tid]
            self._team_scores[tid] = {
                "offense_z": offense_z,
                "pitching_z": pitching_z,
                "dominance_z": dominance_z,
                "team_quality": team_quality,
                "rpg": b.get("runs_per_game") or 5.5,
                "staff_era": p.get("staff_era") or 5.0,
            }

    def predict_game(self, home_team_id, away_team_id, neutral_site=False):
        self._load_all()

        home = self._team_scores.get(home_team_id)
        away = self._team_scores.get(away_team_id)

        if home is None or away is None:
            return None

        # Quality differential → logistic probability
        quality_diff = home["team_quality"] - away["team_quality"]
        home_prob = 1.0 / (1.0 + math.exp(-quality_diff * LOGISTIC_K))

        # Home advantage
        if not neutral_site:
            home_prob += HOME_ADVANTAGE

        # Clamp
        home_prob = max(0.05, min(0.95, home_prob))

        # Project runs
        projected_home_runs = 0.6 * home["rpg"] + 0.4 * (10 - away["staff_era"])
        projected_away_runs = 0.6 * away["rpg"] + 0.4 * (10 - home["staff_era"])

        # Floor at 1.0
        projected_home_runs = max(1.0, projected_home_runs)
        projected_away_runs = max(1.0, projected_away_runs)

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
                "home_quality": round(home["team_quality"], 3),
                "away_quality": round(away["team_quality"], 3),
                "home_offense_z": round(home["offense_z"], 3),
                "home_pitching_z": round(home["pitching_z"], 3),
                "away_offense_z": round(away["offense_z"], 3),
                "away_pitching_z": round(away["pitching_z"], 3),
            },
        }
