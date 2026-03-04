#!/usr/bin/env python3
"""
Pitching Staff Quality Model v3

Predicts game outcomes using a trained logistic regression on matchup
differentials (pitching quality + opponent batting quality).

Key improvements over v2:
- Trained LogisticRegression instead of hand-tuned scoring function
- kwFIP computed from K/BB/IP instead of NULL-filled FIP
- Starter-aware: blends known starter stats with bullpen when available
- No unvalidated day-of-week weights
- Proper matchup differentials as features
"""

import math
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from models.base_model import BaseModel
from scripts.database import get_connection

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_PATH = DATA_DIR / "pitching_logreg.pkl"
KWFIP_CONSTANT = 6.434


class PitchingModel(BaseModel):
    """Pitching staff quality prediction model using trained logistic regression."""

    name = "pitching"
    version = "3.0"
    description = "Trained LogReg on pitching/batting matchup differentials"

    HOME_ADVANTAGE = 0.02  # Small additive; LogReg already captures most HFA via training data

    def __init__(self):
        self._model = None
        self._feature_names = None
        self._load_model()

    def _load_model(self):
        """Load trained logistic regression pipeline."""
        try:
            with open(MODEL_PATH, "rb") as f:
                data = pickle.load(f)
            self._model = data["pipeline"]
            self._feature_names = data["feature_names"]
        except (FileNotFoundError, KeyError, Exception):
            self._model = None
            self._feature_names = None

    def _get_staff_quality(self, team_id):
        """Fetch team pitching quality metrics."""
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
        """Get team batting quality metrics."""
        conn = get_connection()
        c = conn.cursor()
        try:
            c.execute("""
                SELECT lineup_avg as ba, lineup_obp as obp, lineup_slg as slg,
                       lineup_ops, lineup_woba, lineup_wrc_plus,
                       lineup_iso as iso, lineup_k_pct, lineup_bb_pct,
                       runs_per_game, elite_bats, solid_bats, weak_bats
                FROM team_batting_quality WHERE team_id = ?
            """, (team_id,))
            row = c.fetchone()
            if row and row['ba']:
                conn.close()
                return dict(row)
        except Exception:
            pass
        # Fallback to raw player averages
        c.execute("""
            SELECT AVG(batting_avg) as ba, AVG(obp) as obp, AVG(slg) as slg,
                   AVG(ops) as ops
            FROM player_stats WHERE team_id = ? AND at_bats > 10
        """, (team_id,))
        row = c.fetchone()
        conn.close()
        if row and row['ba']:
            d = dict(row)
            d.setdefault('lineup_ops', d.get('ops', 0.740))
            d.setdefault('lineup_woba', 0.320)
            d.setdefault('lineup_wrc_plus', 100)
            d.setdefault('lineup_k_pct', 0.20)
            d.setdefault('lineup_bb_pct', 0.10)
            d.setdefault('runs_per_game', 5.5)
            return d
        return {
            'ba': 0.265, 'obp': 0.340, 'slg': 0.400,
            'lineup_ops': 0.740, 'lineup_woba': 0.320, 'lineup_wrc_plus': 100,
            'lineup_k_pct': 0.20, 'lineup_bb_pct': 0.10, 'runs_per_game': 5.5,
        }

    def _get_starter_stats(self, game_id, team_id, is_home):
        """Get individual starter stats if we know who's pitching."""
        if not game_id:
            return None
        try:
            conn = get_connection()
            c = conn.cursor()

            col = 'home_starter_id' if is_home else 'away_starter_id'
            name_col = 'home_starter_name' if is_home else 'away_starter_name'

            c.execute(f'SELECT {col}, {name_col} FROM pitching_matchups WHERE game_id = ?',
                      (game_id,))
            row = c.fetchone()
            if not row:
                conn.close()
                return None

            starter_id = row[0]
            starter_name = row[1]

            if not starter_id and starter_name:
                # Try name match
                c.execute("""
                    SELECT id FROM player_stats
                    WHERE team_id = ? AND LOWER(name) LIKE ?
                    LIMIT 1
                """, (team_id, f'%{starter_name.split()[-1].lower()}%'))
                r2 = c.fetchone()
                if r2:
                    starter_id = r2['id']

            if not starter_id:
                conn.close()
                return None

            c.execute("""
                SELECT era, whip, k_per_9, bb_per_9, innings_pitched,
                       strikeouts_pitched, walks_allowed
                FROM player_stats
                WHERE id = ? AND innings_pitched >= 3
            """, (starter_id,))
            stats = c.fetchone()
            conn.close()

            if stats and stats['innings_pitched']:
                ip = stats['innings_pitched']
                k = stats['strikeouts_pitched']
                bb = stats['walks_allowed']
                if ip > 0 and k is not None and bb is not None:
                    kwfip = (3 * bb - 2 * k) / ip + KWFIP_CONSTANT
                else:
                    kwfip = stats['era'] or 4.50

                return {
                    'name': starter_name,
                    'era': stats['era'] or 4.50,
                    'whip': stats['whip'] or 1.35,
                    'kwfip': kwfip,
                    'innings': ip,
                }
            conn.close()
            return None
        except Exception:
            return None

    def _build_feature_vector(self, home_pitch, away_pitch, home_hit, away_hit,
                              home_starter=None, away_starter=None):
        """Build feature vector for prediction (same as training)."""
        if home_pitch is None or away_pitch is None:
            return None

        hh = home_hit or {}
        ah = away_hit or {}

        def gp(d, k, default):
            v = d.get(k)
            return v if v is not None else default

        # When a starter is known, blend their stats into rotation metrics
        # 60% starter, 40% staff rotation (starter goes ~5-6 of 9 innings)
        STARTER_WEIGHT = 0.6

        h_rot_era = gp(home_pitch, "rotation_era", 4.50)
        a_rot_era = gp(away_pitch, "rotation_era", 4.50)
        h_rot_whip = gp(home_pitch, "rotation_whip", 1.35)
        a_rot_whip = gp(away_pitch, "rotation_whip", 1.35)
        h_rot_fip = gp(home_pitch, "rotation_fip", 4.50)
        a_rot_fip = gp(away_pitch, "rotation_fip", 4.50)

        # Blend starter stats into rotation when known
        if home_starter:
            h_rot_era = STARTER_WEIGHT * home_starter['era'] + (1 - STARTER_WEIGHT) * h_rot_era
            h_rot_whip = STARTER_WEIGHT * home_starter['whip'] + (1 - STARTER_WEIGHT) * h_rot_whip
            h_rot_fip = STARTER_WEIGHT * home_starter['kwfip'] + (1 - STARTER_WEIGHT) * h_rot_fip
        if away_starter:
            a_rot_era = STARTER_WEIGHT * away_starter['era'] + (1 - STARTER_WEIGHT) * a_rot_era
            a_rot_whip = STARTER_WEIGHT * away_starter['whip'] + (1 - STARTER_WEIGHT) * a_rot_whip
            a_rot_fip = STARTER_WEIGHT * away_starter['kwfip'] + (1 - STARTER_WEIGHT) * a_rot_fip

        d_rot_era = h_rot_era - a_rot_era
        d_bp_era = gp(home_pitch, "bullpen_era", 4.50) - gp(away_pitch, "bullpen_era", 4.50)
        d_rot_whip = h_rot_whip - a_rot_whip
        d_bp_whip = gp(home_pitch, "bullpen_whip", 1.35) - gp(away_pitch, "bullpen_whip", 1.35)
        d_rot_k9 = gp(home_pitch, "rotation_k_per_9", 7.5) - gp(away_pitch, "rotation_k_per_9", 7.5)
        d_bp_k9 = gp(home_pitch, "bullpen_k_per_9", 7.5) - gp(away_pitch, "bullpen_k_per_9", 7.5)
        d_rot_bb9 = gp(home_pitch, "rotation_bb_per_9", 3.5) - gp(away_pitch, "rotation_bb_per_9", 3.5)
        d_bp_bb9 = gp(home_pitch, "bullpen_bb_per_9", 3.5) - gp(away_pitch, "bullpen_bb_per_9", 3.5)
        d_rot_fip = h_rot_fip - a_rot_fip
        d_bp_fip = gp(home_pitch, "bullpen_fip", 4.50) - gp(away_pitch, "bullpen_fip", 4.50)
        d_quality = gp(home_pitch, "quality_arms", 0) - gp(away_pitch, "quality_arms", 0)
        d_shutdown = gp(home_pitch, "shutdown_arms", 0) - gp(away_pitch, "shutdown_arms", 0)
        d_liability = gp(home_pitch, "liability_arms", 0) - gp(away_pitch, "liability_arms", 0)
        d_hhi = gp(home_pitch, "innings_hhi", 0.15) - gp(away_pitch, "innings_hhi", 0.15)

        d_ops = gp(hh, "lineup_ops", 0.740) - gp(ah, "lineup_ops", 0.740)
        d_woba = gp(hh, "lineup_woba", 0.320) - gp(ah, "lineup_woba", 0.320)
        d_wrc = gp(hh, "lineup_wrc_plus", 100) - gp(ah, "lineup_wrc_plus", 100)
        d_kpct = gp(hh, "lineup_k_pct", 0.20) - gp(ah, "lineup_k_pct", 0.20)
        d_bbpct = gp(hh, "lineup_bb_pct", 0.10) - gp(ah, "lineup_bb_pct", 0.10)
        d_rpg = gp(hh, "runs_per_game", 5.5) - gp(ah, "runs_per_game", 5.5)

        home_rot_vs_away_ops = h_rot_era * gp(ah, "lineup_ops", 0.740)
        away_rot_vs_home_ops = a_rot_era * gp(hh, "lineup_ops", 0.740)

        # Starter features
        starter_known = 0
        d_starter_era = 0.0
        d_starter_whip = 0.0
        d_starter_kwfip = 0.0
        if home_starter and away_starter:
            starter_known = 1
            d_starter_era = home_starter['era'] - away_starter['era']
            d_starter_whip = home_starter['whip'] - away_starter['whip']
            d_starter_kwfip = home_starter['kwfip'] - away_starter['kwfip']

        return np.array([
            d_rot_era, d_bp_era, d_rot_whip, d_bp_whip,
            d_rot_k9, d_bp_k9, d_rot_bb9, d_bp_bb9,
            d_rot_fip, d_bp_fip,
            d_quality, d_shutdown, d_liability, d_hhi,
            d_ops, d_woba, d_wrc, d_kpct, d_bbpct, d_rpg,
            home_rot_vs_away_ops, away_rot_vs_home_ops,
            d_starter_era, d_starter_whip, d_starter_kwfip, starter_known,
        ], dtype=np.float64)

    def _fallback_predict(self, home_staff, away_staff, home_hitting, away_hitting,
                          neutral_site):
        """Heuristic fallback when trained model is not available."""
        def score(staff, opp_hit):
            if not staff:
                return 0.50
            era = staff.get('rotation_era') or 4.50
            whip = staff.get('rotation_whip') or 1.35
            fip = staff.get('rotation_fip') or 4.50
            era_s = max(0.1, min(0.9, 1.0 - era / 9.0))
            whip_s = max(0.1, min(0.9, 1.0 - whip / 2.5))
            fip_s = max(0.1, min(0.9, 1.0 - fip / 9.0))
            return era_s * 0.4 + whip_s * 0.3 + fip_s * 0.3

        hs = score(home_staff, away_hitting)
        aws = score(away_staff, home_hitting)
        total = hs + aws
        prob = hs / total if total > 0 else 0.50
        if not neutral_site:
            prob += self.HOME_ADVANTAGE
        return max(0.10, min(0.90, prob))

    def predict_game(self, home_team_id, away_team_id, neutral_site=False,
                     game_date=None, game_id=None, weather_data=None, **kwargs):
        """Predict game based on pitching/batting matchup differentials."""
        if game_date is None:
            game_date = datetime.now().strftime('%Y-%m-%d')
        elif isinstance(game_date, datetime):
            game_date = game_date.strftime('%Y-%m-%d')

        try:
            dow = datetime.strptime(game_date, '%Y-%m-%d').weekday()
        except (ValueError, TypeError):
            dow = 4

        # Get team-level data
        home_staff = self._get_staff_quality(home_team_id)
        away_staff = self._get_staff_quality(away_team_id)
        home_hitting = self._get_team_hitting(home_team_id)
        away_hitting = self._get_team_hitting(away_team_id)

        # Get starter data
        home_starter = self._get_starter_stats(game_id, home_team_id, is_home=True)
        away_starter = self._get_starter_stats(game_id, away_team_id, is_home=False)

        # Predict with trained model
        if self._model is not None:
            fv = self._build_feature_vector(
                home_staff, away_staff, home_hitting, away_hitting,
                home_starter, away_starter,
            )
            if fv is not None:
                prob = self._model.predict_proba(fv.reshape(1, -1))[0, 1]
                # Apply home advantage on top (model was trained on raw outcomes
                # which already include some HFA, so use a small additive)
                if not neutral_site:
                    prob += self.HOME_ADVANTAGE
                home_prob = max(0.10, min(0.90, prob))
            else:
                home_prob = self._fallback_predict(
                    home_staff, away_staff, home_hitting, away_hitting, neutral_site)
        else:
            home_prob = self._fallback_predict(
                home_staff, away_staff, home_hitting, away_hitting, neutral_site)

        # Project runs
        home_runs, away_runs = self._project_runs(
            home_staff, away_staff, home_hitting, away_hitting,
            home_starter, away_starter, neutral_site)

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
                "home_rotation_era": round((home_staff or {}).get('rotation_era', 4.50), 2),
                "away_rotation_era": round((away_staff or {}).get('rotation_era', 4.50), 2),
                "home_bullpen_era": round((home_staff or {}).get('bullpen_era', 4.50), 2),
                "away_bullpen_era": round((away_staff or {}).get('bullpen_era', 4.50), 2),
                "home_rotation_fip": round((home_staff or {}).get('rotation_fip', 4.50), 2),
                "away_rotation_fip": round((away_staff or {}).get('rotation_fip', 4.50), 2),
                "home_quality_arms": (home_staff or {}).get('quality_arms', 0),
                "away_quality_arms": (away_staff or {}).get('quality_arms', 0),
                "home_lineup_ops": round((home_hitting or {}).get('lineup_ops', 0.740), 3),
                "away_lineup_ops": round((away_hitting or {}).get('lineup_ops', 0.740), 3),
                "home_wrc_plus": round((home_hitting or {}).get('lineup_wrc_plus', 100.0) or 100.0, 1),
                "away_wrc_plus": round((away_hitting or {}).get('lineup_wrc_plus', 100.0) or 100.0, 1),
                "dow": dow,
                "day_name": ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][dow],
                "home_starter": home_starter.get('name') if home_starter else None,
                "away_starter": away_starter.get('name') if away_starter else None,
                "home_starter_era": home_starter.get('era') if home_starter else None,
                "away_starter_era": away_starter.get('era') if away_starter else None,
                "model_version": self.version,
                "trained_model": self._model is not None,
            },
        }

    def _project_runs(self, home_staff, away_staff, home_hitting, away_hitting,
                      home_starter, away_starter, neutral_site):
        """Project runs for each team using pitching vs hitting matchup."""
        # Get league average runs per game
        try:
            conn = get_connection()
            r = conn.cursor().execute(
                'SELECT AVG(home_score + away_score) / 2.0 FROM games '
                'WHERE home_score IS NOT NULL AND status = "final"'
            ).fetchone()
            conn.close()
            base_rpg = r[0] if r and r[0] else 5.5
        except Exception:
            base_rpg = 5.5

        league_avg_era = 4.50

        # Get effective ERA for each side (pitcher facing opponent)
        def get_eff_era(staff, starter):
            if starter:
                # Blend starter (60%) with bullpen (40%)
                s_era = starter.get('era', 4.50)
                bp_era = (staff or {}).get('bullpen_era', 4.50) or 4.50
                return 0.6 * s_era + 0.4 * bp_era
            if staff:
                rot = staff.get('rotation_era') or 4.50
                bp = staff.get('bullpen_era') or 4.50
                return 0.55 * rot + 0.45 * bp
            return league_avg_era

        away_eff_era = get_eff_era(away_staff, away_starter)  # home team faces away pitching
        home_eff_era = get_eff_era(home_staff, home_starter)  # away team faces home pitching

        # Hitting quality multiplier
        home_wrc = min(max((home_hitting or {}).get('lineup_wrc_plus', 100) or 100, 70), 160)
        away_wrc = min(max((away_hitting or {}).get('lineup_wrc_plus', 100) or 100, 70), 160)

        # Home team's runs = base adjusted by away pitching quality + home hitting quality
        home_runs = base_rpg * (away_eff_era / league_avg_era) * (home_wrc / 100.0)
        away_runs = base_rpg * (home_eff_era / league_avg_era) * (away_wrc / 100.0)

        # Clamp
        home_runs = max(1.0, min(15.0, home_runs))
        away_runs = max(1.0, min(15.0, away_runs))

        if not neutral_site:
            home_runs *= 1.03
            away_runs *= 0.97

        return home_runs, away_runs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Pitching Staff Quality Model')
    parser.add_argument('home_team', nargs='?', help='Home team ID')
    parser.add_argument('away_team', nargs='?', help='Away team ID')
    parser.add_argument('--neutral', action='store_true')
    parser.add_argument('--date', type=str, default=None)
    parser.add_argument('--game-id', type=str, default=None)
    args = parser.parse_args()

    model = PitchingModel()

    if args.home_team and args.away_team:
        home = args.home_team.lower().replace(" ", "-")
        away = args.away_team.lower().replace(" ", "-")
        pred = model.predict_game(home, away, args.neutral,
                                  game_date=args.date, game_id=args.game_id)

        print(f"\n{'='*55}")
        print(f"  PITCHING MODEL v3: {away} @ {home}")
        print(f"{'='*55}")
        print(f"\nHome Win Prob: {pred['home_win_probability']*100:.1f}%")
        print(f"Projected: {pred['projected_away_runs']:.1f} - {pred['projected_home_runs']:.1f}")
        print(f"\nInputs:")
        for k, v in pred['inputs'].items():
            print(f"  {k}: {v}")
    else:
        print("Usage: python pitching_model.py <home_team> <away_team> [--neutral] [--date YYYY-MM-DD] [--game-id ID]")
