"""Cross-matchup builder: Team A Offense vs Team B Pitching with percentile ranks."""

from typing import Dict, List, Optional, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.database import get_connection

# Stats to display and whether lower is better
BATTING_STATS = [
    ('AVG', 'lineup_avg', False),
    ('OBP', 'lineup_obp', False),
    ('SLG', 'lineup_slg', False),
    ('OPS', 'lineup_ops', False),
    ('wOBA', 'lineup_woba', False),
    ('ISO', 'lineup_iso', False),
    ('K%', 'lineup_k_pct', True),
    ('BB%', 'lineup_bb_pct', False),
    ('RPG', 'runs_per_game', False),
    ('HR/G', 'hr_per_game', False),
]

PITCHING_STATS = [
    ('ERA', 'staff_era', True),
    ('WHIP', 'staff_whip', True),
    ('K/9', 'staff_k_per_9', False),
    ('BB/9', 'staff_bb_per_9', True),
    ('FIP', 'staff_fip', True),
]


def percentile_color(pct: int) -> str:
    """Return hex color for a percentile value."""
    if pct >= 90:
        return '#1a7431'
    if pct >= 75:
        return '#2196F3'
    if pct >= 50:
        return '#607D8B'
    if pct >= 25:
        return '#FF9800'
    return '#d32f2f'


def _compute_percentiles(all_values: List[float], value: float, lower_is_better: bool) -> int:
    """Compute percentile rank (1-99) for value within all_values."""
    if not all_values:
        return 50
    n = len(all_values)
    if lower_is_better:
        count_worse = sum(1 for v in all_values if v > value)
    else:
        count_worse = sum(1 for v in all_values if v < value)
    pct = int(round(count_worse / n * 100))
    return max(1, min(99, pct))


def _load_all_batting(conn) -> List[dict]:
    """Load all team batting quality rows."""
    c = conn.cursor()
    cols = [col for _, col, _ in BATTING_STATS]
    c.execute(f'SELECT team_id, {", ".join(cols)} FROM team_batting_quality')
    return [dict(row) for row in c.fetchall()]


def _load_all_pitching(conn) -> List[dict]:
    """Load all team pitching quality rows."""
    c = conn.cursor()
    cols = [col for _, col, _ in PITCHING_STATS]
    c.execute(f'SELECT team_id, {", ".join(cols)} FROM team_pitching_quality')
    return [dict(row) for row in c.fetchall()]


def _build_stat_block(team_row: dict, all_rows: List[dict], stat_defs: List[tuple]) -> List[dict]:
    """Build list of stat dicts with name, value, percentile, color."""
    result = []
    for display_name, col, lower_is_better in stat_defs:
        value = team_row.get(col)
        if value is None:
            continue
        all_values = [r[col] for r in all_rows if r.get(col) is not None]
        pct = _compute_percentiles(all_values, value, lower_is_better)
        result.append({
            'name': display_name,
            'value': value,
            'percentile': pct,
            'color': percentile_color(pct),
        })
    return result


def build_cross_matchup(home_team_id: str, away_team_id: str) -> Optional[dict]:
    """
    Build cross-matchup data: each team's offense vs opponent's pitching.

    Returns dict with keys:
      home_offense, away_offense, home_pitching, away_pitching
    Each is a list of stat dicts with name, value, percentile, color.
    Returns None if data is missing for either team.
    """
    conn = get_connection()
    try:
        all_batting = _load_all_batting(conn)
        all_pitching = _load_all_pitching(conn)
    except Exception:
        conn.close()
        return None

    batting_by_team = {r['team_id']: r for r in all_batting}
    pitching_by_team = {r['team_id']: r for r in all_pitching}

    if (home_team_id not in batting_by_team or away_team_id not in batting_by_team
            or home_team_id not in pitching_by_team or away_team_id not in pitching_by_team):
        conn.close()
        return None

    result = {
        'home_offense': _build_stat_block(batting_by_team[home_team_id], all_batting, BATTING_STATS),
        'away_offense': _build_stat_block(batting_by_team[away_team_id], all_batting, BATTING_STATS),
        'home_pitching': _build_stat_block(pitching_by_team[home_team_id], all_pitching, PITCHING_STATS),
        'away_pitching': _build_stat_block(pitching_by_team[away_team_id], all_pitching, PITCHING_STATS),
    }
    conn.close()
    return result
