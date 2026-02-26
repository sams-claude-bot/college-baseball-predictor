"""Team percentile rankings for batting and pitching quality stats."""

from typing import Dict, List, Optional

from scripts.database import get_connection
from web.services.cross_matchup import _compute_percentiles, percentile_color

# (display_name, column, format_str, lower_is_better)
BATTING_STATS = [
    ('AVG', 'lineup_avg', '.3f', False),
    ('OBP', 'lineup_obp', '.3f', False),
    ('SLG', 'lineup_slg', '.3f', False),
    ('OPS', 'lineup_ops', '.3f', False),
    ('wOBA', 'lineup_woba', '.3f', False),
    ('wRC+', 'lineup_wrc_plus', '.0f', False),
    ('ISO', 'lineup_iso', '.3f', False),
    ('K%', 'lineup_k_pct', '.1f%', True),
    ('BB%', 'lineup_bb_pct', '.1f%', False),
    ('RPG', 'runs_per_game', '.1f', False),
    ('HR/G', 'hr_per_game', '.2f', False),
    ('BABIP', 'lineup_babip', '.3f', False),
]

PITCHING_STATS = [
    ('Staff ERA', 'staff_era', '.2f', True),
    ('Staff WHIP', 'staff_whip', '.2f', True),
    ('Staff K/9', 'staff_k_per_9', '.1f', False),
    ('Staff BB/9', 'staff_bb_per_9', '.1f', True),
    ('Staff FIP', 'staff_fip', '.2f', True),
    ('Rotation ERA', 'rotation_era', '.2f', True),
    ('Bullpen ERA', 'bullpen_era', '.2f', True),
    ('Quality Arms', 'quality_arms', '.0f', False),
    ('Shutdown Arms', 'shutdown_arms', '.0f', False),
]


def _format_value(value: float, fmt: str) -> str:
    """Format a stat value using its format string."""
    if fmt.endswith('%'):
        return f'{value:{fmt[:-1]}}%'
    return f'{value:{fmt}}'


def _build_percentile_list(team_row: dict, all_rows: List[dict],
                           stat_defs: list) -> List[dict]:
    """Build list of stat dicts with name, value, formatted, percentile, color."""
    result = []
    for display_name, col, fmt, lower_is_better in stat_defs:
        value = team_row.get(col)
        if value is None:
            continue
        all_values = [r[col] for r in all_rows if r.get(col) is not None]
        pct = _compute_percentiles(all_values, value, lower_is_better)
        result.append({
            'name': display_name,
            'value': value,
            'formatted': _format_value(value, fmt),
            'percentile': pct,
            'color': percentile_color(pct),
        })
    return result


def get_team_percentiles(team_id: str) -> Optional[dict]:
    """
    Get team's batting and pitching quality stats with percentile ranks.

    Returns dict with:
      batting: list of {name, value, formatted, percentile, color}
      pitching: list of {name, value, formatted, percentile, color}
    Returns None if team has no quality data.
    """
    conn = get_connection()
    try:
        c = conn.cursor()

        # Load all batting quality
        bat_cols = [col for _, col, _, _ in BATTING_STATS]
        c.execute(f'SELECT team_id, {", ".join(bat_cols)} FROM team_batting_quality')
        all_batting = [dict(row) for row in c.fetchall()]

        # Load all pitching quality
        pitch_cols = [col for _, col, _, _ in PITCHING_STATS]
        c.execute(f'SELECT team_id, {", ".join(pitch_cols)} FROM team_pitching_quality')
        all_pitching = [dict(row) for row in c.fetchall()]
    except Exception:
        conn.close()
        return None

    conn.close()

    batting_by_team = {r['team_id']: r for r in all_batting}
    pitching_by_team = {r['team_id']: r for r in all_pitching}

    bat_row = batting_by_team.get(team_id)
    pitch_row = pitching_by_team.get(team_id)

    if not bat_row and not pitch_row:
        return None

    result = {}
    if bat_row:
        result['batting'] = _build_percentile_list(bat_row, all_batting, BATTING_STATS)
    else:
        result['batting'] = []

    if pitch_row:
        result['pitching'] = _build_percentile_list(pitch_row, all_pitching, PITCHING_STATS)
    else:
        result['pitching'] = []

    return result
