"""Tests for team percentile rankings service."""

import pytest
from unittest.mock import patch, MagicMock

from web.services.team_percentiles import (
    get_team_percentiles,
    _build_percentile_list,
    _format_value,
    BATTING_STATS,
    PITCHING_STATS,
)
from web.services.cross_matchup import _compute_percentiles, percentile_color


# ── _format_value ──

def test_format_value_3f():
    assert _format_value(0.312, '.3f') == '0.312'


def test_format_value_0f():
    assert _format_value(115.0, '.0f') == '115'


def test_format_value_pct():
    assert _format_value(18.3, '.1f%') == '18.3%'


def test_format_value_2f():
    assert _format_value(3.45, '.2f') == '3.45'


# ── _build_percentile_list ──

def _make_batting_rows():
    """Three fake batting rows for testing."""
    bat_cols = ['team_id'] + [col for _, col, _, _ in BATTING_STATS]
    return [
        dict(zip(bat_cols, ['team_a', .310, .410, .520, .930, .395, 130, .210, 16.0, 12.0, 8.0, 1.8, .320])),
        dict(zip(bat_cols, ['team_b', .250, .330, .380, .710, .310, 85, .130, 26.0, 7.0, 4.5, 0.6, .290])),
        dict(zip(bat_cols, ['team_c', .275, .360, .440, .800, .345, 105, .165, 21.0, 9.5, 6.0, 1.1, .305])),
    ]


def _make_pitching_rows():
    """Three fake pitching rows for testing."""
    pitch_cols = ['team_id'] + [col for _, col, _, _ in PITCHING_STATS]
    return [
        dict(zip(pitch_cols, ['team_a', 2.80, 1.05, 10.5, 2.3, 3.00, 2.50, 3.20, 6, 3])),
        dict(zip(pitch_cols, ['team_b', 5.20, 1.55, 6.8, 4.5, 5.40, 5.00, 5.50, 2, 0])),
        dict(zip(pitch_cols, ['team_c', 3.90, 1.28, 8.5, 3.3, 4.10, 3.70, 4.20, 4, 1])),
    ]


def test_build_percentile_list_structure():
    """Output has expected keys per stat."""
    rows = _make_batting_rows()
    result = _build_percentile_list(rows[0], rows, BATTING_STATS)
    assert len(result) == len(BATTING_STATS)
    for stat in result:
        assert 'name' in stat
        assert 'value' in stat
        assert 'formatted' in stat
        assert 'percentile' in stat
        assert 'color' in stat
        assert 1 <= stat['percentile'] <= 99


def test_build_percentile_list_lower_is_better():
    """K% uses lower-is-better: lowest K% gets highest percentile."""
    rows = _make_batting_rows()
    # team_a has K% 16.0 (lowest/best), team_b has 26.0 (highest/worst)
    block_a = _build_percentile_list(rows[0], rows, BATTING_STATS)
    block_b = _build_percentile_list(rows[1], rows, BATTING_STATS)
    k_a = next(s for s in block_a if s['name'] == 'K%')
    k_b = next(s for s in block_b if s['name'] == 'K%')
    assert k_a['percentile'] > k_b['percentile']


def test_build_percentile_list_pitching_lower_is_better():
    """Staff ERA uses lower-is-better: lowest ERA gets highest percentile."""
    rows = _make_pitching_rows()
    block_a = _build_percentile_list(rows[0], rows, PITCHING_STATS)
    block_b = _build_percentile_list(rows[1], rows, PITCHING_STATS)
    era_a = next(s for s in block_a if s['name'] == 'Staff ERA')
    era_b = next(s for s in block_b if s['name'] == 'Staff ERA')
    assert era_a['percentile'] > era_b['percentile']


def test_build_percentile_list_skips_none():
    """Stats with None values are skipped."""
    rows = _make_batting_rows()
    rows[0]['lineup_babip'] = None
    result = _build_percentile_list(rows[0], rows, BATTING_STATS)
    names = [s['name'] for s in result]
    assert 'BABIP' not in names


# ── get_team_percentiles (with mock DB) ──

def _make_mock_conn(batting_rows, pitching_rows):
    """Create a mock connection returning given rows."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor

    call_count = [0]

    def execute_side_effect(query, *args):
        pass

    def fetchall_side_effect():
        call_count[0] += 1
        if call_count[0] == 1:
            return batting_rows
        return pitching_rows

    cursor.execute = execute_side_effect
    cursor.fetchall = fetchall_side_effect
    return conn


def test_get_team_percentiles_returns_structure():
    """Returns dict with batting and pitching lists."""
    bat_rows = _make_batting_rows()
    pitch_rows = _make_pitching_rows()
    mock_conn = _make_mock_conn(bat_rows, pitch_rows)

    with patch('web.services.team_percentiles.get_connection', return_value=mock_conn):
        result = get_team_percentiles('team_a')

    assert result is not None
    assert 'batting' in result
    assert 'pitching' in result
    assert len(result['batting']) == len(BATTING_STATS)
    assert len(result['pitching']) == len(PITCHING_STATS)


def test_get_team_percentiles_missing_team():
    """Returns None for unknown team."""
    bat_rows = _make_batting_rows()
    pitch_rows = _make_pitching_rows()
    mock_conn = _make_mock_conn(bat_rows, pitch_rows)

    with patch('web.services.team_percentiles.get_connection', return_value=mock_conn):
        result = get_team_percentiles('nonexistent')

    assert result is None


def test_get_team_percentiles_values_in_range():
    """All percentile values are between 1 and 99."""
    bat_rows = _make_batting_rows()
    pitch_rows = _make_pitching_rows()
    mock_conn = _make_mock_conn(bat_rows, pitch_rows)

    with patch('web.services.team_percentiles.get_connection', return_value=mock_conn):
        result = get_team_percentiles('team_a')

    for stat in result['batting'] + result['pitching']:
        assert 1 <= stat['percentile'] <= 99


def test_get_team_percentiles_colors_valid():
    """All colors match known percentile_color outputs."""
    valid_colors = {'#1a7431', '#2196F3', '#607D8B', '#FF9800', '#d32f2f'}
    bat_rows = _make_batting_rows()
    pitch_rows = _make_pitching_rows()
    mock_conn = _make_mock_conn(bat_rows, pitch_rows)

    with patch('web.services.team_percentiles.get_connection', return_value=mock_conn):
        result = get_team_percentiles('team_a')

    for stat in result['batting'] + result['pitching']:
        assert stat['color'] in valid_colors


# ── Integration test with real DB ──

def test_get_team_percentiles_real_db_known_team():
    """With the real database, a known team returns data."""
    result = get_team_percentiles('texas')
    if result is None:
        pytest.skip('No quality data for texas in database')
    assert len(result['batting']) > 0
    assert len(result['pitching']) > 0
    for stat in result['batting'] + result['pitching']:
        assert 1 <= stat['percentile'] <= 99
        assert stat['formatted']  # non-empty string


def test_get_team_percentiles_real_db_missing_team():
    """With the real database, a nonexistent team returns None."""
    result = get_team_percentiles('zzz-nonexistent-team-zzz')
    assert result is None
