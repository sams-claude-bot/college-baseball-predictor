#!/usr/bin/env python3
"""
Slim Neural Network Feature Pipeline — v3

v3 changes:
  - 9 NCAA team stats per team (×2 = 18 new features)
  - Residual block architecture with GELU activation
  - Total: 58 features

Per team (×2 = 48):
  - elo, win_pct_all, win_pct_last10, win_pct_last20
  - run_diff_per_game, pythag_win_pct
  - runs_per_game, era_estimate (from runs allowed)
  - days_rest
  - win_streak, home_win_pct, away_win_pct
  - scoring_trend, opp_adj_win_pct, mov_trend
  - ncaa_batting_avg, ncaa_era, ncaa_fielding_pct, ncaa_scoring
  - ncaa_slugging, ncaa_k_per_9, ncaa_whip, ncaa_k_bb_ratio, ncaa_obp
Game-level (3):
  - elo_diff, is_neutral_site, is_conference_game
Weather (7):
  - temp, humidity, wind_speed, wind_dir_sin, wind_dir_cos, precip, is_dome
"""

import math
import sys
import re
import numpy as np
from datetime import datetime
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.database import get_connection

DEFAULT_ELO = 1500
ELO_K = 32
ELO_HOME_ADV = 90

DEFAULT_WEATHER = {
    'temp_f': 65.0, 'humidity_pct': 55.0, 'wind_speed_mph': 6.0,
    'wind_direction_deg': 180, 'precip_prob_pct': 5.0, 'is_dome': 0,
}
WEATHER_NORM = {
    'temp_f': (65.0, 15.0), 'humidity_pct': (55.0, 20.0),
    'wind_speed_mph': (7.0, 5.0), 'precip_prob_pct': (10.0, 15.0),
}

# The 9 NCAA stat types in canonical order
NCAA_STAT_NAMES = [
    'batting_avg', 'era', 'fielding_pct', 'scoring',
    'slugging', 'k_per_9', 'whip', 'k_bb_ratio', 'obp',
]

FEATURE_NAMES = []
for prefix in ['home_', 'away_']:
    FEATURE_NAMES.extend([
        # Core strength (6)
        f'{prefix}elo', f'{prefix}win_pct_all',
        f'{prefix}win_pct_last10', f'{prefix}win_pct_last20',
        f'{prefix}run_diff_per_game', f'{prefix}pythag_win_pct',
        # Scoring (2)
        f'{prefix}runs_per_game', f'{prefix}era_estimate',
        # Situational (1)
        f'{prefix}days_rest',
        # Derived features (6)
        f'{prefix}win_streak',
        f'{prefix}home_win_pct',
        f'{prefix}away_win_pct',
        f'{prefix}scoring_trend',
        f'{prefix}opp_adj_win_pct',
        f'{prefix}mov_trend',
        # NCAA stats (9)  — z-score normalized within season
        *[f'{prefix}ncaa_{s}' for s in NCAA_STAT_NAMES],
    ])
# Game-level (3)
FEATURE_NAMES.extend(['elo_diff', 'is_neutral_site', 'is_conference_game'])
# Weather (7)
FEATURE_NAMES.extend([
    'weather_temp_norm', 'weather_humidity_norm', 'weather_wind_speed_norm',
    'weather_wind_dir_sin', 'weather_wind_dir_cos',
    'weather_precip_prob_norm', 'weather_is_dome',
])

NUM_FEATURES = len(FEATURE_NAMES)  # 58


# ===========================================================
# Residual Block for v3 architecture
# ===========================================================

class ResidualBlock(nn.Module):
    """Two-layer residual block: Linear→BN→GELU→Drop→Linear→BN + skip → GELU→Drop."""
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.drop(F.gelu(self.bn1(self.fc1(x))))
        out = self.bn2(self.fc2(out))
        out = out + residual
        out = self.drop(F.gelu(out))
        return out


class SlimBaseballNet(nn.Module):
    """Network for slim feature set — v3 with residual connections and GELU.

    The .net attribute may be overwritten by build_model() during training
    with a custom architecture. The saved checkpoint's state_dict is
    authoritative — just ensure input_size matches.

    When input_size < NUM_FEATURES, uses a backward-compatible v2 architecture
    (64→32→16) so old checkpoints can still be loaded.
    """
    def __init__(self, input_size=NUM_FEATURES):
        super().__init__()
        self.input_size = input_size

        if input_size < NUM_FEATURES:
            # Backward-compatible v2 architecture for loading old checkpoints
            self.net = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 16),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(16, 1),
                nn.Sigmoid(),
            )
        else:
            # v3 default architecture (overwritten by build_model during training)
            self.net = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.BatchNorm1d(128),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _weather_features(weather_data=None):
    """Compute 7 normalized weather features."""
    w = dict(DEFAULT_WEATHER)
    if weather_data:
        w.update({k: v for k, v in weather_data.items() if v is not None})

    temp_norm = (w['temp_f'] - WEATHER_NORM['temp_f'][0]) / WEATHER_NORM['temp_f'][1]
    humidity_norm = (w['humidity_pct'] - WEATHER_NORM['humidity_pct'][0]) / WEATHER_NORM['humidity_pct'][1]
    wind_norm = (w['wind_speed_mph'] - WEATHER_NORM['wind_speed_mph'][0]) / WEATHER_NORM['wind_speed_mph'][1]
    precip_norm = (w['precip_prob_pct'] - WEATHER_NORM['precip_prob_pct'][0]) / WEATHER_NORM['precip_prob_pct'][1]
    wind_rad = math.radians(w['wind_direction_deg'])

    return [temp_norm, humidity_norm, wind_norm,
            math.sin(wind_rad), math.cos(wind_rad),
            precip_norm, 1.0 if w['is_dome'] else 0.0]


# ===========================================================
# NCAA Stats Loader — shared between live and historical
# ===========================================================

def _load_ncaa_stats_from_db(conn=None):
    """Load ALL ncaa_team_stats into a dict keyed by (team_id, season, stat_name) → stat_value.

    Also computes per-(season, stat_name) mean and std for z-score normalization.
    Returns: (stats_dict, norm_dict)
      stats_dict: {(team_id, season): {stat_name: stat_value}}
      norm_dict:  {(season, stat_name): (mean, std)}
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    c = conn.cursor()
    c.execute("SELECT team_id, season, stat_name, stat_value FROM ncaa_team_stats")

    raw = defaultdict(dict)  # (team_id, season) → {stat_name: value}
    by_season_stat = defaultdict(list)  # (season, stat_name) → [values]

    for row in c.fetchall():
        tid = row['team_id']
        season = row['season']
        sname = row['stat_name']
        sval = row['stat_value']
        if sval is not None:
            raw[(tid, season)][sname] = sval
            by_season_stat[(season, sname)].append(sval)

    if close_conn:
        conn.close()

    # Compute per-(season, stat_name) mean/std
    norm_dict = {}
    for key, vals in by_season_stat.items():
        arr = np.array(vals, dtype=np.float64)
        m = arr.mean()
        s = arr.std()
        if s < 1e-8:
            s = 1.0
        norm_dict[key] = (m, s)

    return dict(raw), norm_dict


def _get_ncaa_features(team_id, season, stats_dict, norm_dict):
    """Return 9 z-score normalized NCAA stat features for a team+season.

    Falls back to 0.0 (league average) if stat is missing.
    """
    team_stats = stats_dict.get((team_id, season), {})
    features = []
    for sname in NCAA_STAT_NAMES:
        raw_val = team_stats.get(sname)
        norm_key = (season, sname)
        if raw_val is not None and norm_key in norm_dict:
            mean, std = norm_dict[norm_key]
            features.append((raw_val - mean) / std)
        else:
            features.append(0.0)  # league average fallback
    return features


def _build_name_to_id_mapping(conn=None):
    """Build a mapping from historical_games team names to ncaa_team_stats team_ids.

    Strategy:
    1. Use team_aliases table (via TeamResolver) for direct lookup
    2. Fall back to slug-based matching (e.g., 'Arizona Wildcats' → 'arizona')
    """
    close_conn = False
    if conn is None:
        conn = get_connection()
        close_conn = True

    mapping = {}

    # Load all team_aliases for reverse lookup
    c = conn.cursor()
    try:
        c.execute("SELECT alias, team_id FROM team_aliases")
        alias_map = {}
        for row in c.fetchall():
            alias_map[row['alias'].lower()] = row['team_id']
    except Exception:
        alias_map = {}

    # Load all known team IDs from ncaa_team_stats
    c.execute("SELECT DISTINCT team_id FROM ncaa_team_stats")
    valid_ids = {row['team_id'] for row in c.fetchall()}

    # Load distinct team names from historical_games
    c.execute("SELECT DISTINCT home_team FROM historical_games UNION SELECT DISTINCT away_team FROM historical_games")
    hist_names = [row[0] for row in c.fetchall()]

    for name in hist_names:
        if not name:
            continue
        key = name.lower().strip()

        # 1) Direct alias lookup
        if key in alias_map:
            tid = alias_map[key]
            if tid in valid_ids:
                mapping[name] = tid
                continue

        # 2) Try slug: take the first word (school name), lowercase, hyphenate
        # 'Arizona Wildcats' → 'arizona', 'Mississippi State Bulldogs' → 'mississippi-state'
        slug = _name_to_slug(name)
        if slug in valid_ids:
            mapping[name] = slug
            continue

        # 3) Try alias of slug
        if slug in alias_map:
            tid = alias_map[slug]
            if tid in valid_ids:
                mapping[name] = tid
                continue

    if close_conn:
        conn.close()

    return mapping


def _name_to_slug(name):
    """Convert a full team name like 'Mississippi State Bulldogs' to slug 'mississippi-state'.

    Drops the last word (mascot) and slugifies the rest.
    """
    parts = name.strip().split()
    if len(parts) <= 1:
        return name.lower().strip()

    # Drop last word (mascot) — e.g., 'Arizona Wildcats' → 'Arizona'
    school_parts = parts[:-1]
    slug = '-'.join(school_parts).lower()
    # Clean non-alphanumeric (keep hyphens)
    slug = re.sub(r'[^a-z0-9-]', '', slug)
    slug = re.sub(r'-+', '-', slug).strip('-')
    return slug


# ===========================================================
# Live Feature Computer (for 2026 predictions)
# ===========================================================

class SlimFeatureComputer:
    """Compute slim features from the current season games table."""

    def __init__(self):
        # Pre-load NCAA stats for live predictions
        conn = get_connection()
        self._ncaa_stats, self._ncaa_norm = _load_ncaa_stats_from_db(conn)
        conn.close()

    def get_feature_names(self):
        return list(FEATURE_NAMES)

    def get_num_features(self):
        return NUM_FEATURES

    def compute_features(self, home_team_id, away_team_id, game_date=None,
                         neutral_site=False, is_conference=False, game_id=None,
                         weather_data=None):
        if game_date is None:
            game_date = datetime.now().strftime('%Y-%m-%d')
        elif isinstance(game_date, datetime):
            game_date = game_date.strftime('%Y-%m-%d')

        # Determine season from game_date
        try:
            season = int(game_date[:4])
        except (ValueError, TypeError):
            season = datetime.now().year

        conn = get_connection()
        features = []

        home_feats = self._team_features(conn, home_team_id, game_date)
        away_feats = self._team_features(conn, away_team_id, game_date)
        features.extend(home_feats)
        # NCAA stats for home team
        features.extend(_get_ncaa_features(home_team_id, season, self._ncaa_stats, self._ncaa_norm))
        features.extend(away_feats)
        # NCAA stats for away team
        features.extend(_get_ncaa_features(away_team_id, season, self._ncaa_stats, self._ncaa_norm))

        # Game-level: elo_diff (home - away)
        home_elo = home_feats[0]
        away_elo = away_feats[0]
        features.append(home_elo - away_elo)

        features.append(1.0 if neutral_site else 0.0)
        features.append(1.0 if is_conference else 0.0)

        # Weather
        w_data = weather_data
        if w_data is None and game_id:
            c = conn.cursor()
            c.execute("""SELECT temp_f, humidity_pct, wind_speed_mph, wind_direction_deg,
                         precip_prob_pct, is_dome FROM game_weather WHERE game_id = ?""", (game_id,))
            row = c.fetchone()
            if row:
                w_data = {k: row[k] for k in ['temp_f', 'humidity_pct', 'wind_speed_mph',
                                                'wind_direction_deg', 'precip_prob_pct', 'is_dome']
                          if row[k] is not None}
        features.extend(_weather_features(w_data))

        conn.close()
        return np.array(features, dtype=np.float32)

    def _team_features(self, conn, team_id, game_date):
        """15 base features per team from current season games."""
        c = conn.cursor()

        # Elo
        c.execute("SELECT rating FROM elo_ratings WHERE team_id = ?", (team_id,))
        row = c.fetchone()
        elo = row['rating'] if row else DEFAULT_ELO

        # Get completed games before date
        c.execute("""
            SELECT home_team_id, away_team_id, home_score, away_score, date
            FROM games
            WHERE (home_team_id = ? OR away_team_id = ?)
              AND status = 'final'
              AND home_score IS NOT NULL
              AND away_score IS NOT NULL
              AND date < ?
            ORDER BY date ASC
        """, (team_id, team_id, game_date))

        games = []
        for row in c.fetchall():
            is_home = row['home_team_id'] == team_id
            opp_id = row['away_team_id'] if is_home else row['home_team_id']
            games.append({
                'rs': row['home_score'] if is_home else row['away_score'],
                'ra': row['away_score'] if is_home else row['home_score'],
                'won': (row['home_score'] > row['away_score']) == is_home,
                'is_home': is_home,
                'opp_id': opp_id,
                'date': row['date'],
            })

        gp = len(games)
        wins = sum(1 for g in games if g['won'])
        win_pct_all = wins / gp if gp > 0 else 0.5

        last10 = games[-10:]
        win_pct_10 = sum(1 for g in last10 if g['won']) / len(last10) if last10 else 0.5
        last20 = games[-20:]
        win_pct_20 = sum(1 for g in last20 if g['won']) / len(last20) if last20 else 0.5

        rs_total = sum(g['rs'] for g in games)
        ra_total = sum(g['ra'] for g in games)
        run_diff = (rs_total - ra_total) / gp if gp > 0 else 0.0

        rs2 = rs_total ** 2
        ra2 = ra_total ** 2
        pythag = rs2 / (rs2 + ra2) if (rs2 + ra2) > 0 else 0.5

        rpg = rs_total / gp if gp > 0 else 4.5
        era_est = (ra_total / gp) if gp > 0 else 4.5

        # Days rest
        if games:
            try:
                last_d = datetime.strptime(games[-1]['date'], '%Y-%m-%d')
                curr_d = datetime.strptime(game_date, '%Y-%m-%d')
                days_rest = min((curr_d - last_d).days, 14)
            except (ValueError, TypeError):
                days_rest = 2.0
        else:
            days_rest = 3.0

        # Win streak
        streak = 0
        if games:
            streak_dir = 1 if games[-1]['won'] else -1
            for g in reversed(games):
                if (g['won'] and streak_dir > 0) or (not g['won'] and streak_dir < 0):
                    streak += streak_dir
                else:
                    break

        # Home/away win% splits
        home_games = [g for g in games if g['is_home']]
        away_games = [g for g in games if not g['is_home']]
        home_wp = sum(1 for g in home_games if g['won']) / len(home_games) if home_games else 0.5
        away_wp = sum(1 for g in away_games if g['won']) / len(away_games) if away_games else 0.5

        # Scoring trend
        last5 = games[-5:]
        last5_rpg = sum(g['rs'] for g in last5) / len(last5) if last5 else rpg
        scoring_trend = last5_rpg - rpg

        # Opponent-adjusted win%
        opp_adj_wp = 0.5
        if games:
            weighted_wins = 0.0
            total_opp_elo = 0.0
            for g in games:
                c.execute("SELECT rating FROM elo_ratings WHERE team_id = ?", (g['opp_id'],))
                opp_row = c.fetchone()
                opp_elo = opp_row['rating'] if opp_row else DEFAULT_ELO
                total_opp_elo += opp_elo
                if g['won']:
                    weighted_wins += opp_elo
            opp_adj_wp = weighted_wins / total_opp_elo if total_opp_elo > 0 else 0.5

        # MOV trend
        margins = [g['rs'] - g['ra'] for g in games]
        season_mov = sum(margins) / len(margins) if margins else 0.0
        last5_margins = margins[-5:]
        last5_mov = sum(last5_margins) / len(last5_margins) if last5_margins else 0.0
        mov_trend = last5_mov - season_mov

        return [elo, win_pct_all, win_pct_10, win_pct_20,
                run_diff, pythag, rpg, era_est, float(days_rest),
                float(streak), home_wp, away_wp, scoring_trend,
                opp_adj_wp, mov_trend]


# ===========================================================
# Historical Feature Computer (for training on historical_games)
# ===========================================================

class SlimHistoricalFeatureComputer:
    """Compute slim features from historical_games, maintaining rolling state.

    v3: Also loads NCAA team stats at init and appends 9 z-score-normalized
    features per team from ncaa_team_stats.
    """

    def __init__(self):
        # Load NCAA stats once at init
        conn = get_connection()
        self._ncaa_stats, self._ncaa_norm = _load_ncaa_stats_from_db(conn)
        self._name_to_id = _build_name_to_id_mapping(conn)
        conn.close()
        self.reset()

    def reset(self):
        self.elo = defaultdict(lambda: DEFAULT_ELO)
        self.stats = defaultdict(lambda: {
            'wins': 0, 'losses': 0, 'rs': 0, 'ra': 0, 'games': 0,
            'recent': [],       # last 50 W/L results (bool)
            'recent_rs': [],    # last 50 runs scored
            'recent_margins': [],  # last 50 margins
            'home_w': 0, 'home_g': 0,
            'away_w': 0, 'away_g': 0,
            'opp_elo_sum': 0.0, 'weighted_wins': 0.0,
            'last_date': None,
        })

    def compute_game_features(self, game_row, weather_row=None):
        """Compute features BEFORE updating state. Returns (features, label)."""
        home = game_row['home_team']
        away = game_row['away_team']
        date_str = game_row['date']
        neutral = game_row.get('neutral_site', 0)
        season = game_row.get('season')
        if season is None:
            try:
                season = int(date_str[:4])
            except (ValueError, TypeError):
                season = 2025

        features = []
        home_feats = self._team_features(home, date_str)
        away_feats = self._team_features(away, date_str)
        features.extend(home_feats)
        # NCAA stats for home team
        home_tid = self._name_to_id.get(home)
        features.extend(_get_ncaa_features(home_tid, season, self._ncaa_stats, self._ncaa_norm))
        features.extend(away_feats)
        # NCAA stats for away team
        away_tid = self._name_to_id.get(away)
        features.extend(_get_ncaa_features(away_tid, season, self._ncaa_stats, self._ncaa_norm))

        # Game-level: elo_diff
        features.append(home_feats[0] - away_feats[0])

        features.append(1.0 if neutral else 0.0)
        features.append(0.0)  # conference unknown in historical

        features.extend(_weather_features(weather_row))

        label = 1.0 if game_row['home_score'] > game_row['away_score'] else 0.0
        return np.array(features, dtype=np.float32), label

    def _team_features(self, team, date_str):
        """15 base features per team from rolling state."""
        s = self.stats[team]
        gp = s['games']

        elo = self.elo[team]
        win_pct_all = s['wins'] / gp if gp > 0 else 0.5

        recent = s['recent']
        last10 = recent[-10:]
        last20 = recent[-20:]
        win_pct_10 = sum(last10) / len(last10) if last10 else 0.5
        win_pct_20 = sum(last20) / len(last20) if last20 else 0.5

        run_diff = (s['rs'] - s['ra']) / gp if gp > 0 else 0.0

        rs2 = s['rs'] ** 2
        ra2 = s['ra'] ** 2
        pythag = rs2 / (rs2 + ra2) if (rs2 + ra2) > 0 else 0.5

        rpg = s['rs'] / gp if gp > 0 else 4.5
        era_est = s['ra'] / gp if gp > 0 else 4.5

        if s['last_date']:
            try:
                last_d = datetime.strptime(s['last_date'], '%Y-%m-%d')
                curr_d = datetime.strptime(date_str, '%Y-%m-%d')
                days_rest = min((curr_d - last_d).days, 14)
            except (ValueError, TypeError):
                days_rest = 2.0
        else:
            days_rest = 3.0

        # Win streak
        streak = 0
        if recent:
            streak_dir = 1 if recent[-1] > 0.5 else -1
            for r in reversed(recent):
                if (r > 0.5 and streak_dir > 0) or (r <= 0.5 and streak_dir < 0):
                    streak += streak_dir
                else:
                    break

        # Home/away win% splits
        home_wp = s['home_w'] / s['home_g'] if s['home_g'] > 0 else 0.5
        away_wp = s['away_w'] / s['away_g'] if s['away_g'] > 0 else 0.5

        # Scoring trend
        recent_rs = s['recent_rs']
        last5_rs = recent_rs[-5:]
        last5_rpg = sum(last5_rs) / len(last5_rs) if last5_rs else rpg
        scoring_trend = last5_rpg - rpg

        # Opponent-adjusted win%
        opp_adj_wp = s['weighted_wins'] / s['opp_elo_sum'] if s['opp_elo_sum'] > 0 else 0.5

        # MOV trend
        margins = s['recent_margins']
        season_mov = sum(margins) / len(margins) if margins else 0.0
        last5_m = margins[-5:]
        last5_mov = sum(last5_m) / len(last5_m) if last5_m else 0.0
        mov_trend = last5_mov - season_mov

        return [elo, win_pct_all, win_pct_10, win_pct_20,
                run_diff, pythag, rpg, era_est, float(days_rest),
                float(streak), home_wp, away_wp, scoring_trend,
                opp_adj_wp, mov_trend]

    def update_state(self, game_row):
        """Update rolling state AFTER computing features."""
        home = game_row['home_team']
        away = game_row['away_team']
        hs = game_row['home_score']
        aws = game_row['away_score']
        date_str = game_row['date']
        neutral = game_row.get('neutral_site', 0)
        home_won = hs > aws

        # Get opponent Elos before updating
        home_opp_elo = self.elo[away]
        away_opp_elo = self.elo[home]

        for team, opp, rs, ra, won, is_home, opp_elo in [
            (home, away, hs, aws, home_won, True, home_opp_elo),
            (away, home, aws, hs, not home_won, False, away_opp_elo),
        ]:
            s = self.stats[team]
            s['games'] += 1
            s['rs'] += rs
            s['ra'] += ra
            if won:
                s['wins'] += 1
            else:
                s['losses'] += 1
            s['recent'].append(1.0 if won else 0.0)
            if len(s['recent']) > 50:
                s['recent'] = s['recent'][-50:]
            s['recent_rs'].append(rs)
            if len(s['recent_rs']) > 50:
                s['recent_rs'] = s['recent_rs'][-50:]
            s['recent_margins'].append(rs - ra)
            if len(s['recent_margins']) > 50:
                s['recent_margins'] = s['recent_margins'][-50:]
            s['last_date'] = date_str

            # Home/away splits
            if is_home:
                s['home_g'] += 1
                if won:
                    s['home_w'] += 1
            else:
                s['away_g'] += 1
                if won:
                    s['away_w'] += 1

            # Opponent-adjusted win%
            s['opp_elo_sum'] += opp_elo
            if won:
                s['weighted_wins'] += opp_elo

        # Update Elo
        home_elo = self.elo[home]
        away_elo = self.elo[away]
        elo_diff = home_elo - away_elo + (0 if neutral else ELO_HOME_ADV)
        expected = 1.0 / (1.0 + 10 ** (-elo_diff / 400))
        actual = 1.0 if home_won else 0.0

        mov = abs(hs - aws)
        mov_mult = math.log(max(mov, 1) + 1) * (2.2 / (elo_diff * 0.001 + 2.2))

        self.elo[home] += ELO_K * mov_mult * (actual - expected)
        self.elo[away] += ELO_K * mov_mult * (expected - actual)
