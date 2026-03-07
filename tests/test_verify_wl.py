#!/usr/bin/env python3
"""
Tests for scripts/verify_wl_records.py — HTML parsing and W-L comparison.
"""

import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from scripts.verify_wl_records import fetch_conference_standings, compute_db_wl


# Sample HTML mimicking D1BB conference standings table
SAMPLE_STANDINGS_HTML = """
<html><body>
<table class="standings-table">
<thead><tr><th>Team</th><th>Conf</th><th>Win%</th><th>GB</th><th>Overall</th><th>Overall%</th><th>Streak</th></tr></thead>
<tbody>
<tr>
  <td><a href="/team/tennessee/">Tennessee</a></td>
  <td>5-1</td>
  <td>.833</td>
  <td>-</td>
  <td>14-2</td>
  <td>.875</td>
  <td>W5</td>
</tr>
<tr>
  <td><a href="/team/ole-miss/">Ole Miss</a></td>
  <td>4-2</td>
  <td>.667</td>
  <td>1.0</td>
  <td>13-3</td>
  <td>.813</td>
  <td>L1</td>
</tr>
<tr>
  <td><a href="/team/vanderbilt/">Vanderbilt</a></td>
  <td>3-3</td>
  <td>.500</td>
  <td>2.0</td>
  <td>10-6</td>
  <td>.625</td>
  <td>W2</td>
</tr>
</tbody>
</table>
</body></html>
"""


class TestParseConferenceStandings:
    """test_parse_conference_standings — mock HTML, verify parsing."""

    @patch('scripts.verify_wl_records.requests.get')
    def test_parse(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.text = SAMPLE_STANDINGS_HTML
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        results = fetch_conference_standings('sec', year=2026)

        assert len(results) == 3

        assert results[0]['d1bb_slug'] == 'tennessee'
        assert results[0]['wins'] == 14
        assert results[0]['losses'] == 2
        assert results[0]['conf_record'] == '5-1'
        assert results[0]['streak'] == 'W5'

        assert results[1]['d1bb_slug'] == 'ole-miss'
        assert results[1]['wins'] == 13
        assert results[1]['losses'] == 3

        assert results[2]['d1bb_slug'] == 'vanderbilt'
        assert results[2]['wins'] == 10
        assert results[2]['losses'] == 6


def _make_db_with_games():
    """Create an in-memory DB with some final games."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE games (
            id TEXT PRIMARY KEY,
            date TEXT,
            home_team_id TEXT,
            away_team_id TEXT,
            home_score INTEGER,
            away_score INTEGER,
            winner_id TEXT,
            status TEXT DEFAULT 'scheduled'
        )
    """)
    return conn


class TestWlComparisonMatch:
    """test_wl_comparison_match — DB matches D1BB → no mismatch."""

    def test_match(self):
        db = _make_db_with_games()
        # 3 wins for tennessee (2 home wins, 1 away win)
        db.execute("""INSERT INTO games (id, date, home_team_id, away_team_id,
                      home_score, away_score, winner_id, status)
                      VALUES ('g1', '2025-02-21', 'tennessee', 'auburn',
                              5, 3, 'tennessee', 'final')""")
        db.execute("""INSERT INTO games (id, date, home_team_id, away_team_id,
                      home_score, away_score, winner_id, status)
                      VALUES ('g2', '2025-02-22', 'tennessee', 'auburn',
                              7, 1, 'tennessee', 'final')""")
        db.execute("""INSERT INTO games (id, date, home_team_id, away_team_id,
                      home_score, away_score, winner_id, status)
                      VALUES ('g3', '2025-02-23', 'auburn', 'tennessee',
                              2, 4, 'tennessee', 'final')""")
        # 1 loss
        db.execute("""INSERT INTO games (id, date, home_team_id, away_team_id,
                      home_score, away_score, winner_id, status)
                      VALUES ('g4', '2025-02-28', 'tennessee', 'florida',
                              1, 3, 'florida', 'final')""")
        db.commit()

        wins, losses = compute_db_wl(db, 'tennessee')
        assert wins == 3
        assert losses == 1


class TestWlComparisonMismatch:
    """test_wl_comparison_mismatch — DB differs → mismatch detected."""

    def test_mismatch(self):
        db = _make_db_with_games()
        # DB has 2-1 for tennessee
        db.execute("""INSERT INTO games (id, date, home_team_id, away_team_id,
                      home_score, away_score, winner_id, status)
                      VALUES ('g1', '2025-02-21', 'tennessee', 'auburn',
                              5, 3, 'tennessee', 'final')""")
        db.execute("""INSERT INTO games (id, date, home_team_id, away_team_id,
                      home_score, away_score, winner_id, status)
                      VALUES ('g2', '2025-02-22', 'tennessee', 'auburn',
                              7, 1, 'tennessee', 'final')""")
        db.execute("""INSERT INTO games (id, date, home_team_id, away_team_id,
                      home_score, away_score, winner_id, status)
                      VALUES ('g3', '2025-02-23', 'auburn', 'tennessee',
                              6, 4, 'auburn', 'final')""")
        db.commit()

        wins, losses = compute_db_wl(db, 'tennessee')
        # DB says 2-1, but if D1BB says 3-1, that's a mismatch
        d1bb_wins, d1bb_losses = 3, 1
        assert wins != d1bb_wins  # Mismatch detected
        assert losses == d1bb_losses
