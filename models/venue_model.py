#!/usr/bin/env python3
"""
Venue Model — Park Factors + Environment

Predicts based on WHERE the game is played:
- Elevation (higher = more offense)
- Dome status (indoor = consistent conditions)
- Venue capacity (proxy for program size)
- Latitude (climate proxy)
- Historical home advantage at venue
- Average total runs at venue (park factor)
- Opponent travel distance (haversine)

Uses logistic regression on feature differentials.
"""

import math
import pickle
from pathlib import Path

from models.base_model import BaseModel
from scripts.database import get_connection

MODEL_PATH = Path(__file__).parent.parent / "data" / "venue_model.pkl"

# Defaults for missing venue data (league averages)
DEFAULT_ELEVATION = 500
DEFAULT_LATITUDE = 34.0
DEFAULT_LONGITUDE = -86.0
DEFAULT_CAPACITY = 3000
DEFAULT_HOME_WIN_PCT = 0.58
DEFAULT_AVG_TOTAL_RUNS = 11.0


def haversine(lat1, lon1, lat2, lon2):
    """Distance in miles between two lat/lng points."""
    R = 3959  # Earth radius in miles
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class VenueModel(BaseModel):
    name = "venue"
    version = "1.0"
    description = "Park factors + environment model (elevation, dome, travel distance)"

    FEATURE_NAMES = [
        'elevation_ft', 'is_dome', 'capacity', 'latitude',
        'home_win_pct_at_venue', 'avg_total_runs_at_venue',
        'travel_distance_miles',
    ]

    def __init__(self):
        self.model = None
        self._venue_cache = {}
        self._venue_stats_cache = {}
        self._load_model()

    def _load_model(self):
        if MODEL_PATH.exists():
            with open(MODEL_PATH, 'rb') as f:
                data = pickle.load(f)
            self.model = data['model']
            self._feature_means = data.get('feature_means')
            self._feature_stds = data.get('feature_stds')

    def _get_venue(self, team_id):
        if team_id in self._venue_cache:
            return self._venue_cache[team_id]
        conn = get_connection()
        c = conn.cursor()
        c.execute(
            "SELECT latitude, longitude, elevation_ft, is_dome, capacity "
            "FROM venues WHERE team_id = ?", (team_id,))
        row = c.fetchone()
        conn.close()
        if row:
            self._venue_cache[team_id] = dict(row)
        else:
            self._venue_cache[team_id] = None
        return self._venue_cache[team_id]

    def _get_venue_stats(self, team_id):
        """Home win % and avg total runs at this team's home venue."""
        if team_id in self._venue_stats_cache:
            return self._venue_stats_cache[team_id]
        conn = get_connection()
        c = conn.cursor()
        c.execute("""
            SELECT COUNT(*) as games,
                   SUM(CASE WHEN winner_id = home_team_id THEN 1 ELSE 0 END) as home_wins,
                   AVG(home_score + away_score) as avg_total
            FROM games
            WHERE home_team_id = ? AND status = 'final'
              AND home_score IS NOT NULL AND away_score IS NOT NULL
        """, (team_id,))
        row = c.fetchone()
        conn.close()
        games = row['games'] or 0
        if games > 0:
            stats = {
                'home_win_pct': (row['home_wins'] or 0) / games,
                'avg_total_runs': row['avg_total'] or DEFAULT_AVG_TOTAL_RUNS,
            }
        else:
            stats = {
                'home_win_pct': DEFAULT_HOME_WIN_PCT,
                'avg_total_runs': DEFAULT_AVG_TOTAL_RUNS,
            }
        self._venue_stats_cache[team_id] = stats
        return stats

    def _build_features(self, home_team_id, away_team_id):
        home_venue = self._get_venue(home_team_id)
        away_venue = self._get_venue(away_team_id)
        venue_stats = self._get_venue_stats(home_team_id)

        hv = home_venue or {}
        av = away_venue or {}

        elevation = hv.get('elevation_ft') or DEFAULT_ELEVATION
        is_dome = hv.get('is_dome') or 0
        capacity = hv.get('capacity') or DEFAULT_CAPACITY
        latitude = hv.get('latitude') or DEFAULT_LATITUDE

        home_lat = hv.get('latitude') or DEFAULT_LATITUDE
        home_lon = hv.get('longitude') or DEFAULT_LONGITUDE
        away_lat = av.get('latitude') or DEFAULT_LATITUDE
        away_lon = av.get('longitude') or DEFAULT_LONGITUDE
        travel = haversine(away_lat, away_lon, home_lat, home_lon)

        return [
            elevation,
            is_dome,
            capacity,
            latitude,
            venue_stats['home_win_pct'],
            venue_stats['avg_total_runs'],
            travel,
        ]

    def predict_game(self, home_team_id, away_team_id, neutral_site=False):
        features = self._build_features(home_team_id, away_team_id)

        if self.model is not None:
            import numpy as np
            X = np.array([features])
            if self._feature_means is not None:
                X = (X - self._feature_means) / (self._feature_stds + 1e-8)
            home_prob = float(self.model.predict_proba(X)[0, 1])
        else:
            # Fallback heuristic when no trained model
            home_prob = features[4]  # venue home win pct

        if neutral_site:
            home_prob = 0.5 + (home_prob - 0.5) * 0.5

        home_prob = max(0.02, min(0.98, home_prob))

        avg_total = features[5]
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
                "elevation_ft": features[0],
                "is_dome": features[1],
                "capacity": features[2],
                "travel_distance_miles": round(features[6], 1),
                "venue_home_win_pct": round(features[4], 3),
                "avg_total_runs": round(features[5], 1),
            },
        }
