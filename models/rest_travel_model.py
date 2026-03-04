#!/usr/bin/env python3
"""
Rest & Travel Model — Schedule Fatigue

Predicts based on WHEN teams last played and HOW FAR they traveled:
- Days since last game (back-to-back is hardest)
- Games in last 7 days / 3 days (fatigue proxy)
- Midweek game flag (weaker pitching rotations)
- Travel distance from last game venue
- Home stand / road trip length
- Cross-timezone travel flag

Uses logistic regression on rest/travel feature differentials.
"""

import math
import pickle
from datetime import datetime, timedelta
from pathlib import Path

from models.base_model import BaseModel
from scripts.database import get_connection

MODEL_PATH = Path(__file__).parent.parent / "data" / "rest_travel_model.pkl"

# Defaults
DEFAULT_LATITUDE = 34.0
DEFAULT_LONGITUDE = -86.0


def haversine(lat1, lon1, lat2, lon2):
    """Distance in miles between two lat/lng points."""
    R = 3959
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class RestTravelModel(BaseModel):
    name = "rest_travel"
    version = "1.0"
    description = "Schedule fatigue model (rest days, travel distance, road trips)"

    FEATURE_NAMES = [
        'days_since_last_game', 'games_in_last_7', 'games_in_last_3',
        'is_midweek', 'travel_distance', 'home_stand_length',
        'road_trip_length', 'crossed_timezone',
    ]

    def __init__(self):
        self.model = None
        self._venue_cache = {}
        self._load_model()

    def _load_model(self):
        if MODEL_PATH.exists():
            with open(MODEL_PATH, 'rb') as f:
                data = pickle.load(f)
            self.model = data['model']
            self._feature_means = data.get('feature_means')
            self._feature_stds = data.get('feature_stds')

    def _get_venue_location(self, team_id):
        if team_id in self._venue_cache:
            return self._venue_cache[team_id]
        conn = get_connection()
        c = conn.cursor()
        c.execute("SELECT latitude, longitude FROM venues WHERE team_id = ?",
                  (team_id,))
        row = c.fetchone()
        conn.close()
        if row and row['latitude']:
            loc = (row['latitude'], row['longitude'])
        else:
            loc = (DEFAULT_LATITUDE, DEFAULT_LONGITUDE)
        self._venue_cache[team_id] = loc
        return loc

    def _get_team_schedule_features(self, team_id, game_date, is_home):
        """Compute rest/travel features for a team relative to a game date."""
        if isinstance(game_date, str):
            game_date = datetime.strptime(game_date, '%Y-%m-%d').date()

        conn = get_connection()
        c = conn.cursor()
        # Get recent games for this team (up to 14 days back)
        lookback = (game_date - timedelta(days=14)).strftime('%Y-%m-%d')
        game_date_str = game_date.strftime('%Y-%m-%d')
        c.execute("""
            SELECT date, home_team_id, away_team_id
            FROM games
            WHERE (home_team_id = ? OR away_team_id = ?)
              AND status = 'final' AND date >= ? AND date < ?
            ORDER BY date DESC
        """, (team_id, team_id, lookback, game_date_str))
        recent = [dict(r) for r in c.fetchall()]
        conn.close()

        # days_since_last_game
        if recent:
            last_date = datetime.strptime(recent[0]['date'], '%Y-%m-%d').date()
            days_since = (game_date - last_date).days
        else:
            days_since = 7  # No recent games = well-rested

        days_since = min(days_since, 7)  # Cap at 7

        # games_in_last_7 and games_in_last_3
        cutoff_7 = game_date - timedelta(days=7)
        cutoff_3 = game_date - timedelta(days=3)
        games_7 = sum(1 for g in recent
                      if datetime.strptime(g['date'], '%Y-%m-%d').date() > cutoff_7)
        games_3 = sum(1 for g in recent
                      if datetime.strptime(g['date'], '%Y-%m-%d').date() > cutoff_3)

        # is_midweek (Tue=1, Wed=2, Thu=3)
        dow = game_date.weekday()
        is_midweek = 1 if dow in (1, 2, 3) else 0

        # Travel distance from last game venue to today's venue
        today_venue_team = team_id if is_home else None
        if recent:
            last_game = recent[0]
            was_home = last_game['home_team_id'] == team_id
            last_venue_team = last_game['home_team_id']
        else:
            was_home = True
            last_venue_team = team_id

        # Get today's game venue (home team's venue)
        # For this team: if they're home, venue is their home. If away, we need
        # the opponent's home venue—but we only know team_id and is_home here.
        # The caller passes is_home, so we use team_id's venue if home,
        # otherwise the travel is computed in _build_features with opponent info.
        last_loc = self._get_venue_location(last_venue_team)
        if is_home:
            today_loc = self._get_venue_location(team_id)
        else:
            # Will be overridden by caller with actual opponent venue
            today_loc = self._get_venue_location(team_id)

        travel = haversine(last_loc[0], last_loc[1], today_loc[0], today_loc[1])

        # home_stand_length / road_trip_length (consecutive home or away)
        home_stand = 0
        road_trip = 0
        for g in recent:
            g_home = g['home_team_id'] == team_id
            if is_home and g_home:
                home_stand += 1
            elif not is_home and not g_home:
                road_trip += 1
            else:
                break

        # crossed_timezone (>500 miles from last venue)
        crossed_tz = 1 if travel > 500 else 0

        return [days_since, games_7, games_3, is_midweek,
                travel, home_stand, road_trip, crossed_tz]

    def _build_features_for_game(self, home_team_id, away_team_id, game_date):
        """Build differential features: home - away."""
        home_feats = self._get_team_schedule_features(
            home_team_id, game_date, is_home=True)
        away_feats = self._get_team_schedule_features(
            away_team_id, game_date, is_home=False)

        # Fix away travel: from away team's last venue to home team's venue
        home_loc = self._get_venue_location(home_team_id)
        away_loc = self._get_venue_location(away_team_id)

        # Recalculate away travel distance to today's game (home team venue)
        conn = get_connection()
        c = conn.cursor()
        lookback = (game_date - timedelta(days=14)).strftime('%Y-%m-%d') if isinstance(
            game_date, str) is False else game_date
        if isinstance(game_date, str):
            game_date_obj = datetime.strptime(game_date, '%Y-%m-%d').date()
        else:
            game_date_obj = game_date
        lookback = (game_date_obj - timedelta(days=14)).strftime('%Y-%m-%d')
        c.execute("""
            SELECT home_team_id FROM games
            WHERE (home_team_id = ? OR away_team_id = ?)
              AND status = 'final' AND date >= ? AND date < ?
            ORDER BY date DESC LIMIT 1
        """, (away_team_id, away_team_id, lookback,
              game_date_obj.strftime('%Y-%m-%d')))
        row = c.fetchone()
        conn.close()

        if row:
            last_venue_team = row['home_team_id']
            last_loc = self._get_venue_location(last_venue_team)
        else:
            last_loc = away_loc
        away_feats[4] = haversine(last_loc[0], last_loc[1],
                                  home_loc[0], home_loc[1])
        away_feats[7] = 1 if away_feats[4] > 500 else 0

        # Return both raw feature sets (for the model: differential)
        diff = [h - a for h, a in zip(home_feats, away_feats)]
        return diff, home_feats, away_feats

    def predict_game(self, home_team_id, away_team_id, neutral_site=False,
                     game_date=None):
        if game_date is None:
            game_date = datetime.now().strftime('%Y-%m-%d')

        diff, home_feats, away_feats = self._build_features_for_game(
            home_team_id, away_team_id, game_date)

        if self.model is not None:
            import numpy as np
            X = np.array([diff])
            if self._feature_means is not None:
                X = (X - self._feature_means) / (self._feature_stds + 1e-8)
            home_prob = float(self.model.predict_proba(X)[0, 1])
        else:
            # Heuristic: more rest and less travel → advantage
            rest_adv = (home_feats[0] - away_feats[0]) * 0.02  # days since
            fatigue_adv = (away_feats[1] - home_feats[1]) * 0.01  # games in 7
            travel_adv = (away_feats[4] - home_feats[4]) / 5000 * 0.05
            home_prob = 0.55 + rest_adv + fatigue_adv + travel_adv

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
                "home_days_rest": home_feats[0],
                "away_days_rest": away_feats[0],
                "home_games_7d": home_feats[1],
                "away_games_7d": away_feats[1],
                "home_travel_mi": round(home_feats[4], 1),
                "away_travel_mi": round(away_feats[4], 1),
                "is_midweek": bool(home_feats[3]),
            },
        }
