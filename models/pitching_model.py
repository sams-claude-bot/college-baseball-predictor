#!/usr/bin/env python3
"""
Pitching Staff Quality Model

Predicts game outcomes based on team pitching staff quality metrics.
Uses data from team_pitching_quality table (computed by compute_pitching_quality.py).

Key signals:
- Rotation quality (top 3 starters by IP, IP-weighted)
- Bullpen quality (non-starters)
- Staff depth (number of quality arms, innings concentration)
- Ace vs bullpen ERA gap (fragility indicator)
- Day-of-week adjustments (ace likely Friday, bullpen Sunday)

Does NOT try to identify specific starters — uses staff profiles instead.
"""

import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from models.base_model import BaseModel
from scripts.database import get_connection


class PitchingModel(BaseModel):
    """Pitching staff quality prediction model"""

    name = "pitching"
    version = "2.0"
    description = "Staff quality + depth + day-of-week model"

    HOME_ADVANTAGE = 0.035

    # Day-of-week rotation expectations
    # Friday = ace, Saturday = #2, Sunday = #3/#bullpen, midweek = #4+
    # We blend rotation vs bullpen ERA based on who's likely pitching
    DOW_ROTATION_WEIGHT = {
        4: 0.85,  # Friday - mostly ace/rotation
        5: 0.75,  # Saturday - rotation
        6: 0.55,  # Sunday - mix of #3 starter and bullpen
        0: 0.45,  # Monday (midweek) - bullpen/spot starter
        1: 0.45,  # Tuesday
        2: 0.50,  # Wednesday
        3: 0.50,  # Thursday
    }

    def _get_staff_quality(self, team_id):
        """Fetch team pitching quality metrics. Returns dict or None."""
        conn = get_connection()
        c = conn.cursor()
        c.execute("""
            SELECT ace_era, ace_whip, ace_k_per_9, ace_bb_per_9, ace_fip, ace_innings,
                   rotation_era, rotation_whip, rotation_k_per_9, rotation_bb_per_9, rotation_fip, rotation_innings,
                   bullpen_era, bullpen_whip, bullpen_k_per_9, bullpen_bb_per_9, bullpen_fip, bullpen_innings,
                   staff_size, starter_count, ace_ip_pct, top3_ip_pct, innings_hhi,
                   staff_era, staff_whip, staff_k_per_9, staff_bb_per_9, staff_fip, staff_total_ip,
                   quality_arms, shutdown_arms, liability_arms
            FROM team_pitching_quality WHERE team_id = ?
        """, (team_id,))
        row = c.fetchone()
        conn.close()
        if row:
            return dict(row)
        return None

    def _get_team_hitting(self, team_id):
        """Get team batting stats for matchup context."""
        conn = get_connection()
        c = conn.cursor()
        c.execute("""
            SELECT AVG(batting_avg) as ba, AVG(obp) as obp, AVG(slg) as slg,
                   AVG(ops) as ops
            FROM player_stats WHERE team_id = ? AND at_bats > 10
        """, (team_id,))
        row = c.fetchone()
        conn.close()
        if row and row['ba']:
            return {'ba': row['ba'], 'obp': row['obp'], 'slg': row['slg'], 'ops': row['ops']}
        return {'ba': 0.265, 'obp': 0.340, 'slg': 0.400, 'ops': 0.740}

    def _effective_era(self, staff, dow):
        """
        Compute expected effective ERA for this day of week.
        Blends rotation ERA and bullpen ERA based on who's likely pitching.
        """
        if not staff:
            return 4.50

        rot_weight = self.DOW_ROTATION_WEIGHT.get(dow, 0.50)
        bp_weight = 1.0 - rot_weight

        rot_era = staff.get('rotation_era') or 4.50
        bp_era = staff.get('bullpen_era') or 4.50

        return rot_era * rot_weight + bp_era * bp_weight

    def _effective_whip(self, staff, dow):
        """Compute expected effective WHIP for this day of week."""
        if not staff:
            return 1.35

        rot_weight = self.DOW_ROTATION_WEIGHT.get(dow, 0.50)
        bp_weight = 1.0 - rot_weight

        rot_whip = staff.get('rotation_whip') or 1.35
        bp_whip = staff.get('bullpen_whip') or 1.35

        return rot_whip * rot_weight + bp_whip * bp_weight

    def _effective_k9(self, staff, dow):
        """Compute expected K/9 for this day of week."""
        if not staff:
            return 7.5

        rot_weight = self.DOW_ROTATION_WEIGHT.get(dow, 0.50)
        bp_weight = 1.0 - rot_weight

        rot_k9 = staff.get('rotation_k_per_9') or 7.5
        bp_k9 = staff.get('bullpen_k_per_9') or 7.5

        return rot_k9 * rot_weight + bp_k9 * bp_weight

    def _staff_score(self, staff, dow, opp_hitting):
        """
        Compute a 0-1 quality score for a pitching staff on a given day.

        Components:
        - ERA quality (35%): lower = better
        - WHIP quality (20%): lower = better
        - K rate (15%): higher = better
        - Depth (15%): more quality arms = better
        - Fragility penalty (15%): big ace-bullpen gap = risky
        """
        if not staff:
            return 0.50

        # ERA score (0-1, league avg ERA ~4.50 for D1)
        eff_era = self._effective_era(staff, dow)
        era_score = max(0.1, min(0.9, 1.0 - (eff_era / 9.0)))

        # WHIP score
        eff_whip = self._effective_whip(staff, dow)
        whip_score = max(0.1, min(0.9, 1.0 - (eff_whip / 2.5)))

        # K/9 score
        eff_k9 = self._effective_k9(staff, dow)
        k_score = max(0.1, min(0.9, eff_k9 / 15.0))

        # Depth score: quality arms as fraction of staff
        staff_size = staff.get('staff_size') or 10
        quality = staff.get('quality_arms') or 0
        shutdown = staff.get('shutdown_arms') or 0
        depth_score = min(0.9, (quality * 0.15 + shutdown * 0.1))
        depth_score = max(0.1, depth_score)

        # Fragility: big gap between rotation and bullpen ERA = risky
        rot_era = staff.get('rotation_era') or 4.50
        bp_era = staff.get('bullpen_era') or 4.50
        era_gap = abs(bp_era - rot_era)
        # HHI captures innings concentration (1 guy throws everything = fragile)
        hhi = staff.get('innings_hhi') or 0.15
        fragility = 1.0 - min(1.0, (era_gap / 6.0) * 0.5 + hhi * 2.0)
        fragility = max(0.1, min(0.9, fragility))

        # Opponent hitting adjustment: better offense slightly lowers pitcher score
        opp_ops = opp_hitting.get('ops', 0.740)
        opp_adj = (0.740 - opp_ops) * 0.15  # +/- small amount

        score = (era_score * 0.35 +
                 whip_score * 0.20 +
                 k_score * 0.15 +
                 depth_score * 0.15 +
                 fragility * 0.15 +
                 opp_adj)

        return max(0.10, min(0.90, score))

    def predict_game(self, home_team_id, away_team_id, neutral_site=False,
                     game_date=None, game_id=None, weather_data=None, **kwargs):
        """Predict game based on pitching staff quality."""
        if game_date is None:
            game_date = datetime.now().strftime('%Y-%m-%d')
        elif isinstance(game_date, datetime):
            game_date = game_date.strftime('%Y-%m-%d')

        try:
            dow = datetime.strptime(game_date, '%Y-%m-%d').weekday()
        except (ValueError, TypeError):
            dow = 4  # default to Friday

        # Get staff quality
        home_staff = self._get_staff_quality(home_team_id)
        away_staff = self._get_staff_quality(away_team_id)

        # Get opponent hitting for matchup adjustment
        home_hitting = self._get_team_hitting(home_team_id)
        away_hitting = self._get_team_hitting(away_team_id)

        # Score each staff (home pitching vs away hitting, etc.)
        home_pitch_score = self._staff_score(home_staff, dow, away_hitting)
        away_pitch_score = self._staff_score(away_staff, dow, home_hitting)

        # Convert scores to probability
        total = home_pitch_score + away_pitch_score
        if total > 0:
            home_prob = home_pitch_score / total
        else:
            home_prob = 0.50

        # Home advantage
        if not neutral_site:
            home_prob += self.HOME_ADVANTAGE

        home_prob = max(0.10, min(0.90, home_prob))

        # Project runs from effective ERA
        home_eff_era = self._effective_era(home_staff, dow)
        away_eff_era = self._effective_era(away_staff, dow)

        # Runs projected = opponent's effective ERA scaled, adjusted for opponent hitting
        home_runs = away_eff_era * 0.65 + home_hitting.get('ops', 0.740) * 4.0
        away_runs = home_eff_era * 0.65 + away_hitting.get('ops', 0.740) * 4.0

        if not neutral_site:
            home_runs *= 1.02
            away_runs *= 0.98

        run_line = self.calculate_run_line(home_runs, away_runs)

        return {
            "model": self.name,
            "home_win_probability": round(home_prob, 3),
            "away_win_probability": round(1 - home_prob, 3),
            "projected_home_runs": round(home_runs, 1),
            "projected_away_runs": round(away_runs, 1),
            "projected_total": round(home_runs + away_runs, 1),
            "run_line": run_line,
            "inputs": {
                "home_pitch_score": round(home_pitch_score, 3),
                "away_pitch_score": round(away_pitch_score, 3),
                "home_eff_era": round(home_eff_era, 2),
                "away_eff_era": round(away_eff_era, 2),
                "home_rotation_era": round((home_staff or {}).get('rotation_era', 4.50), 2),
                "away_rotation_era": round((away_staff or {}).get('rotation_era', 4.50), 2),
                "home_bullpen_era": round((home_staff or {}).get('bullpen_era', 4.50), 2),
                "away_bullpen_era": round((away_staff or {}).get('bullpen_era', 4.50), 2),
                "home_quality_arms": (home_staff or {}).get('quality_arms', 0),
                "away_quality_arms": (away_staff or {}).get('quality_arms', 0),
                "dow": dow,
                "day_name": ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][dow],
            },
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Pitching Staff Quality Model')
    parser.add_argument('home_team', nargs='?', help='Home team ID')
    parser.add_argument('away_team', nargs='?', help='Away team ID')
    parser.add_argument('--neutral', action='store_true')
    parser.add_argument('--date', type=str, default=None)
    args = parser.parse_args()

    model = PitchingModel()

    if args.home_team and args.away_team:
        home = args.home_team.lower().replace(" ", "-")
        away = args.away_team.lower().replace(" ", "-")
        pred = model.predict_game(home, away, args.neutral, game_date=args.date)

        print(f"\n{'='*55}")
        print(f"  PITCHING MODEL v2: {away} @ {home}")
        print(f"{'='*55}")
        print(f"\nHome Win Prob: {pred['home_win_probability']*100:.1f}%")
        print(f"Projected: {pred['projected_away_runs']:.1f} - {pred['projected_home_runs']:.1f}")
        print(f"\nInputs:")
        for k, v in pred['inputs'].items():
            print(f"  {k}: {v}")
    else:
        print("Usage: python pitching_model.py <home_team> <away_team> [--neutral] [--date YYYY-MM-DD]")
