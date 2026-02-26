"""Tests for the cross-matchup service (offense vs pitching percentiles)."""

import sqlite3
import pytest
from unittest.mock import patch, MagicMock

from web.services.cross_matchup import (
    _compute_percentiles,
    percentile_color,
    _build_stat_block,
    build_cross_matchup,
    BATTING_STATS,
    PITCHING_STATS,
)


# ── percentile_color ──

def test_color_elite():
    assert percentile_color(95) == '#1a7431'
    assert percentile_color(90) == '#1a7431'


def test_color_good():
    assert percentile_color(80) == '#2196F3'
    assert percentile_color(75) == '#2196F3'


def test_color_average():
    assert percentile_color(55) == '#607D8B'
    assert percentile_color(50) == '#607D8B'


def test_color_below_avg():
    assert percentile_color(30) == '#FF9800'
    assert percentile_color(25) == '#FF9800'


def test_color_poor():
    assert percentile_color(10) == '#d32f2f'
    assert percentile_color(1) == '#d32f2f'


# ── _compute_percentiles ──

def test_percentile_higher_is_better():
    """Best value gets highest percentile."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    # 5.0 is best (higher is better) — 4 out of 5 are worse
    pct = _compute_percentiles(values, 5.0, lower_is_better=False)
    assert pct == 80
    # 1.0 is worst — 0 out of 5 are worse
    pct = _compute_percentiles(values, 1.0, lower_is_better=False)
    assert pct == 1  # clamped from 0


def test_percentile_lower_is_better():
    """Lowest value gets highest percentile when lower_is_better=True."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    # 1.0 is best (lower is better) — 4 out of 5 have higher (worse) values
    pct = _compute_percentiles(values, 1.0, lower_is_better=True)
    assert pct == 80
    # 5.0 is worst — 0 out of 5 are worse
    pct = _compute_percentiles(values, 5.0, lower_is_better=True)
    assert pct == 1


def test_percentile_clamped_1_to_99():
    """Percentile is always between 1 and 99."""
    values = [1.0, 2.0, 3.0]
    pct_best = _compute_percentiles(values, 3.0, lower_is_better=False)
    assert 1 <= pct_best <= 99
    pct_worst = _compute_percentiles(values, 1.0, lower_is_better=False)
    assert 1 <= pct_worst <= 99


def test_percentile_empty_list():
    assert _compute_percentiles([], 5.0, lower_is_better=False) == 50


def test_percentile_with_many_teams():
    """Simulate ~310 teams and check percentile makes sense."""
    values = list(range(1, 311))  # 1 through 310
    # The best (310) should have ~99th percentile
    pct = _compute_percentiles(values, 310, lower_is_better=False)
    assert pct >= 95
    # The worst (1) should have low percentile
    pct = _compute_percentiles(values, 1, lower_is_better=False)
    assert pct <= 5
    # A median value
    pct = _compute_percentiles(values, 155, lower_is_better=False)
    assert 45 <= pct <= 55


# ── _build_stat_block ──

def test_build_stat_block_structure():
    """Output has expected keys per stat."""
    all_rows = [
        {'team_id': 'a', 'lineup_avg': .300, 'lineup_obp': .400, 'lineup_slg': .500,
         'lineup_ops': .900, 'lineup_woba': .380, 'lineup_iso': .200,
         'lineup_k_pct': 20.0, 'lineup_bb_pct': 10.0, 'runs_per_game': 7.0, 'hr_per_game': 1.5},
        {'team_id': 'b', 'lineup_avg': .250, 'lineup_obp': .330, 'lineup_slg': .400,
         'lineup_ops': .730, 'lineup_woba': .320, 'lineup_iso': .150,
         'lineup_k_pct': 25.0, 'lineup_bb_pct': 8.0, 'runs_per_game': 5.0, 'hr_per_game': 0.8},
        {'team_id': 'c', 'lineup_avg': .270, 'lineup_obp': .360, 'lineup_slg': .450,
         'lineup_ops': .810, 'lineup_woba': .350, 'lineup_iso': .180,
         'lineup_k_pct': 22.0, 'lineup_bb_pct': 9.0, 'runs_per_game': 6.0, 'hr_per_game': 1.0},
    ]
    block = _build_stat_block(all_rows[0], all_rows, BATTING_STATS)
    assert len(block) == 10
    for stat in block:
        assert 'name' in stat
        assert 'value' in stat
        assert 'percentile' in stat
        assert 'color' in stat
        assert 1 <= stat['percentile'] <= 99


def test_build_stat_block_lower_is_better():
    """K% uses lower-is-better: team with lowest K% gets highest percentile."""
    all_rows = [
        {'team_id': 'a', 'lineup_avg': .300, 'lineup_obp': .400, 'lineup_slg': .500,
         'lineup_ops': .900, 'lineup_woba': .380, 'lineup_iso': .200,
         'lineup_k_pct': 15.0, 'lineup_bb_pct': 10.0, 'runs_per_game': 7.0, 'hr_per_game': 1.5},
        {'team_id': 'b', 'lineup_avg': .250, 'lineup_obp': .330, 'lineup_slg': .400,
         'lineup_ops': .730, 'lineup_woba': .320, 'lineup_iso': .150,
         'lineup_k_pct': 25.0, 'lineup_bb_pct': 8.0, 'runs_per_game': 5.0, 'hr_per_game': 0.8},
        {'team_id': 'c', 'lineup_avg': .270, 'lineup_obp': .360, 'lineup_slg': .450,
         'lineup_ops': .810, 'lineup_woba': .350, 'lineup_iso': .180,
         'lineup_k_pct': 30.0, 'lineup_bb_pct': 9.0, 'runs_per_game': 6.0, 'hr_per_game': 1.0},
    ]
    # Team 'a' has lowest K% (15.0) → should have highest K% percentile
    block_a = _build_stat_block(all_rows[0], all_rows, BATTING_STATS)
    # Team 'c' has highest K% (30.0) → should have lowest K% percentile
    block_c = _build_stat_block(all_rows[2], all_rows, BATTING_STATS)

    k_pct_a = next(s for s in block_a if s['name'] == 'K%')
    k_pct_c = next(s for s in block_c if s['name'] == 'K%')
    assert k_pct_a['percentile'] > k_pct_c['percentile']


# ── build_cross_matchup ──

def _make_mock_conn(batting_rows, pitching_rows):
    """Create a mock connection that returns given batting/pitching rows."""
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


def test_build_cross_matchup_missing_team():
    """Returns None if a team is missing from the data."""
    batting_cols = ['team_id'] + [c for _, c, _ in BATTING_STATS]
    pitching_cols = ['team_id'] + [c for _, c, _ in PITCHING_STATS]

    bat_row = dict(zip(batting_cols, ['team_a', .280, .360, .450, .810, .350, .170, 22.0, 9.0, 6.0, 1.0]))
    pitch_row = dict(zip(pitching_cols, ['team_a', 3.50, 1.20, 9.0, 3.0, 3.80]))

    mock_conn = _make_mock_conn([bat_row], [pitch_row])

    with patch('web.services.cross_matchup.get_connection', return_value=mock_conn):
        result = build_cross_matchup('team_a', 'team_b')
        assert result is None


def test_build_cross_matchup_full():
    """Returns all four stat blocks when both teams present."""
    batting_cols = ['team_id'] + [c for _, c, _ in BATTING_STATS]
    pitching_cols = ['team_id'] + [c for _, c, _ in PITCHING_STATS]

    bat_rows = [
        dict(zip(batting_cols, ['team_a', .300, .400, .500, .900, .380, .200, 20.0, 10.0, 7.0, 1.5])),
        dict(zip(batting_cols, ['team_b', .250, .330, .400, .730, .320, .150, 25.0, 8.0, 5.0, 0.8])),
        dict(zip(batting_cols, ['team_c', .270, .360, .450, .810, .350, .180, 22.0, 9.0, 6.0, 1.0])),
    ]
    pitch_rows = [
        dict(zip(pitching_cols, ['team_a', 3.00, 1.10, 10.0, 2.5, 3.20])),
        dict(zip(pitching_cols, ['team_b', 4.50, 1.40, 7.0, 4.0, 4.80])),
        dict(zip(pitching_cols, ['team_c', 3.80, 1.25, 8.5, 3.2, 4.00])),
    ]

    mock_conn = _make_mock_conn(bat_rows, pitch_rows)

    with patch('web.services.cross_matchup.get_connection', return_value=mock_conn):
        result = build_cross_matchup('team_a', 'team_b')

    assert result is not None
    assert 'home_offense' in result
    assert 'away_offense' in result
    assert 'home_pitching' in result
    assert 'away_pitching' in result

    # home = team_a, away = team_b
    assert len(result['home_offense']) == 10  # all batting stats
    assert len(result['away_offense']) == 10
    assert len(result['home_pitching']) == 5  # all pitching stats
    assert len(result['away_pitching']) == 5

    # team_a has best batting → should have high percentiles
    avg_stat = result['home_offense'][0]
    assert avg_stat['name'] == 'AVG'
    assert avg_stat['percentile'] >= 50

    # team_a has best pitching (lowest ERA) → high ERA percentile
    era_stat = result['home_pitching'][0]
    assert era_stat['name'] == 'ERA'
    assert era_stat['percentile'] >= 50


def test_build_cross_matchup_pitching_lower_is_better():
    """ERA, WHIP, BB/9, FIP: lower values → higher percentiles."""
    batting_cols = ['team_id'] + [c for _, c, _ in BATTING_STATS]
    pitching_cols = ['team_id'] + [c for _, c, _ in PITCHING_STATS]

    bat_rows = [
        dict(zip(batting_cols, ['h', .270, .350, .440, .790, .340, .170, 22.0, 9.0, 5.5, 0.9])),
        dict(zip(batting_cols, ['a', .270, .350, .440, .790, .340, .170, 22.0, 9.0, 5.5, 0.9])),
        dict(zip(batting_cols, ['x', .270, .350, .440, .790, .340, .170, 22.0, 9.0, 5.5, 0.9])),
    ]
    pitch_rows = [
        dict(zip(pitching_cols, ['h', 2.00, 0.90, 11.0, 1.5, 2.50])),  # elite
        dict(zip(pitching_cols, ['a', 6.00, 1.80, 5.0, 5.0, 6.00])),  # bad
        dict(zip(pitching_cols, ['x', 4.00, 1.30, 8.0, 3.0, 4.00])),  # mid
    ]

    mock_conn = _make_mock_conn(bat_rows, pitch_rows)

    with patch('web.services.cross_matchup.get_connection', return_value=mock_conn):
        result = build_cross_matchup('h', 'a')

    # home pitching (team h, ERA 2.00) should have higher ERA percentile than away (team a, ERA 6.00)
    home_era = next(s for s in result['home_pitching'] if s['name'] == 'ERA')
    away_era = next(s for s in result['away_pitching'] if s['name'] == 'ERA')
    assert home_era['percentile'] > away_era['percentile']

    home_whip = next(s for s in result['home_pitching'] if s['name'] == 'WHIP')
    away_whip = next(s for s in result['away_pitching'] if s['name'] == 'WHIP')
    assert home_whip['percentile'] > away_whip['percentile']

    home_fip = next(s for s in result['home_pitching'] if s['name'] == 'FIP')
    away_fip = next(s for s in result['away_pitching'] if s['name'] == 'FIP')
    assert home_fip['percentile'] > away_fip['percentile']

    home_bb9 = next(s for s in result['home_pitching'] if s['name'] == 'BB/9')
    away_bb9 = next(s for s in result['away_pitching'] if s['name'] == 'BB/9')
    assert home_bb9['percentile'] > away_bb9['percentile']
