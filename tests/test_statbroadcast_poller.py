#!/usr/bin/env python3
"""
Tests for StatBroadcast live poller — polling loop, filetime tracking,
situation_json merge, live_events insertion, and score updates.
"""

import json
import sqlite3
import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from statbroadcast_poller import (
    merge_situation,
    StatBroadcastPoller,
    ensure_live_events_table,
    ENSURE_LIVE_EVENTS_SQL,
)
from statbroadcast_discovery import ensure_table as ensure_sb_table


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db():
    """Create an in-memory DB with all required tables."""
    conn = sqlite3.connect(':memory:')
    conn.execute("""
        CREATE TABLE games (
            id TEXT PRIMARY KEY,
            date TEXT,
            home_team_id TEXT,
            away_team_id TEXT,
            status TEXT DEFAULT 'scheduled',
            home_score INTEGER,
            away_score INTEGER,
            situation_json TEXT,
            linescore_json TEXT,
            inning_text TEXT,
            time TEXT,
            winner_id TEXT,
            updated_at TEXT,
            innings INTEGER
        )
    """)
    conn.execute(ENSURE_LIVE_EVENTS_SQL)
    ensure_sb_table(conn)
    conn.commit()
    return conn


def _seed_active_game(conn, game_id='2026-02-26_byu_washington-state',
                      sb_event_id=652739,
                      xml_file='wsu/652739.xml'):
    """Insert a game and its SB event mapping."""
    conn.execute(
        "INSERT INTO games (id, date, home_team_id, away_team_id, status) VALUES (?, ?, ?, ?, ?)",
        (game_id, '2026-02-26', 'washington-state', 'byu', 'in-progress')
    )
    conn.execute("""
        INSERT INTO statbroadcast_events
            (sb_event_id, game_id, home_team, visitor_team, xml_file, completed)
        VALUES (?, ?, ?, ?, ?, 0)
    """, (sb_event_id, game_id, 'Washington State', 'BYU', xml_file))
    conn.commit()


SAMPLE_HTML = '''
<span class="sb-teamnameV"><b>BYU</b></span>
<span class="sb-teamnameH"><b>Washington State</b></span>
<span class="sb-teamscore">1</span>
<span class="sb-teamscore">4</span>
<div class="font-size-125 mb-1">Top 8th</div>
<span class="mr-2">OUTS</span><span class="no-access"><i class="sbicon d-none d-sm-inline noaccess">ZZ</i></span><span class="d-inline d-sm-none">1</span>
<div class="font-size-125">0-1</div>
At Bat for BYU: #29 Erickson,Ridge [C]</div>
<div class="card-header card-title">
Pitching For WSU: #54 Haider, Rylan</div>
<div class="card-header">Runners On Base</div><table><tr><th>Base</th><th>Runner</th></tr><tr><td>1B</td><td>Roy,Gavin</td></tr></table>
<script>cscore = "BYU 1, WSU 4 - T8th"</script>
'''


# ---------------------------------------------------------------------------
# Merge situation tests
# ---------------------------------------------------------------------------

class TestMergeSituation:
    def test_merge_empty_existing(self):
        """Merges SB data into empty existing JSON."""
        sb = {'outs': 1, 'count': '2-1', 'batter_name': 'Smith', 'pitcher_name': 'Jones'}
        result = json.loads(merge_situation(None, sb))
        assert result['sb_outs'] == 1
        assert result['sb_count'] == '2-1'
        assert result['sb_batter'] == 'Smith'
        assert result['sb_pitcher'] == 'Jones'
        assert 'sb_updated_at' in result

    def test_merge_preserves_espn_data(self):
        """SB merge does not overwrite existing non-sb_ keys."""
        existing = json.dumps({
            'espn_inning': 'Top 8th',
            'espn_score': '4-1',
            'runners': [1, 0, 0],
        })
        sb = {'outs': 2, 'batter_name': 'Brown'}
        result = json.loads(merge_situation(existing, sb))
        # ESPN data preserved
        assert result['espn_inning'] == 'Top 8th'
        assert result['espn_score'] == '4-1'
        assert result['runners'] == [1, 0, 0]
        # SB data added
        assert result['sb_outs'] == 2
        assert result['sb_batter'] == 'Brown'

    def test_merge_updates_sb_fields(self):
        """Subsequent SB merges update existing sb_ fields."""
        existing = json.dumps({
            'sb_outs': 0,
            'sb_batter': 'OldBatter',
            'espn_data': 'keep',
        })
        sb = {'outs': 2, 'batter_name': 'NewBatter'}
        result = json.loads(merge_situation(existing, sb))
        assert result['sb_outs'] == 2
        assert result['sb_batter'] == 'NewBatter'
        assert result['espn_data'] == 'keep'

    def test_merge_with_invalid_json(self):
        """Handles invalid existing JSON gracefully."""
        sb = {'outs': 1}
        result = json.loads(merge_situation('not-json{{', sb))
        assert result['sb_outs'] == 1

    def test_merge_includes_scores(self):
        """Merges score data from SB situation."""
        sb = {'home_score': 5, 'visitor_score': 3, 'inning': 7, 'inning_half': 'bottom'}
        result = json.loads(merge_situation(None, sb))
        assert result['sb_home_score'] == 5
        assert result['sb_visitor_score'] == 3
        assert result['sb_inning'] == 7
        assert result['sb_inning_half'] == 'bottom'


# ---------------------------------------------------------------------------
# Live events table tests
# ---------------------------------------------------------------------------

class TestLiveEventsTable:
    def test_ensure_creates_table(self):
        """ensure_live_events_table creates the table."""
        conn = sqlite3.connect(':memory:')
        ensure_live_events_table(conn)
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='live_events'")
        assert c.fetchone() is not None

    def test_ensure_idempotent(self):
        """Can call multiple times without error."""
        conn = sqlite3.connect(':memory:')
        ensure_live_events_table(conn)
        ensure_live_events_table(conn)


# ---------------------------------------------------------------------------
# Poller tests
# ---------------------------------------------------------------------------

class TestPoller:
    def test_poll_once_no_events(self):
        """poll_once returns 0 when no active events."""
        conn = _make_db()
        poller = StatBroadcastPoller(conn)
        assert poller.poll_once() == 0

    def test_poll_once_with_update(self):
        """poll_once processes a game and returns 1."""
        conn = _make_db()
        _seed_active_game(conn)

        mock_client = MagicMock()
        mock_client.get_live_stats.return_value = (SAMPLE_HTML, 12345)

        poller = StatBroadcastPoller(conn, client=mock_client)
        result = poller.poll_once()
        assert result == 1

        # Verify situation_json was updated
        c = conn.cursor()
        row = c.execute(
            "SELECT situation_json FROM games WHERE id = '2026-02-26_byu_washington-state'"
        ).fetchone()
        assert row is not None
        sit = json.loads(row[0])
        assert sit['sb_outs'] == 1
        assert sit['sb_count'] == '0-1'
        assert sit['sb_batter'] == 'Erickson,Ridge'
        assert sit['sb_pitcher'] == 'Haider, Rylan'

    def test_poll_once_304_no_change(self):
        """poll_once returns 0 on 304 (no change)."""
        conn = _make_db()
        _seed_active_game(conn)

        mock_client = MagicMock()
        mock_client.get_live_stats.return_value = (None, 12345)  # 304

        poller = StatBroadcastPoller(conn, client=mock_client)
        result = poller.poll_once()
        assert result == 0

    def test_filetime_tracking(self):
        """Poller sends updated filetime on subsequent polls."""
        conn = _make_db()
        _seed_active_game(conn)

        mock_client = MagicMock()
        # First call returns data, second returns 304
        mock_client.get_live_stats.side_effect = [
            (SAMPLE_HTML, 12345),
            (None, 12345),
        ]

        poller = StatBroadcastPoller(conn, client=mock_client)

        poller.poll_once()
        assert poller._filetimes[652739] == 12345

        poller.poll_once()
        # Second call should use filetime from first
        calls = mock_client.get_live_stats.call_args_list
        assert calls[1][1].get('filetime', calls[1][0][2] if len(calls[1][0]) > 2 else 0) == 12345

    def test_live_event_inserted(self):
        """poll_once inserts a live_event record."""
        conn = _make_db()
        _seed_active_game(conn)

        mock_client = MagicMock()
        mock_client.get_live_stats.return_value = (SAMPLE_HTML, 12345)

        poller = StatBroadcastPoller(conn, client=mock_client)
        poller.poll_once()

        c = conn.cursor()
        row = c.execute(
            "SELECT event_type, data_json FROM live_events WHERE game_id = '2026-02-26_byu_washington-state'"
        ).fetchone()
        assert row is not None
        assert row[0] == 'sb_situation'
        data = json.loads(row[1])
        assert data['source'] == 'statbroadcast'
        assert data['outs'] == 1
        assert data['batter'] == 'Erickson,Ridge'

    def test_scores_updated(self):
        """poll_once updates game scores from SB data."""
        conn = _make_db()
        _seed_active_game(conn)

        mock_client = MagicMock()
        mock_client.get_live_stats.return_value = (SAMPLE_HTML, 12345)

        poller = StatBroadcastPoller(conn, client=mock_client)
        poller.poll_once()

        c = conn.cursor()
        row = c.execute(
            "SELECT home_score, away_score, status FROM games WHERE id = '2026-02-26_byu_washington-state'"
        ).fetchone()
        assert row[0] == 4  # home (WSU)
        assert row[1] == 1  # away (BYU)
        assert row[2] == 'in-progress'

    def test_completed_game_marked(self):
        """Games with 'Final' in title get marked completed."""
        conn = _make_db()
        _seed_active_game(conn)

        final_html = SAMPLE_HTML.replace(
            'cscore = "BYU 1, WSU 4 - T8th"',
            'cscore = "BYU 1, WSU 4 - Final"'
        )
        mock_client = MagicMock()
        mock_client.get_live_stats.return_value = (final_html, 99999)

        poller = StatBroadcastPoller(conn, client=mock_client)
        poller.poll_once()

        c = conn.cursor()
        row = c.execute(
            "SELECT completed FROM statbroadcast_events WHERE sb_event_id = 652739"
        ).fetchone()
        assert row[0] == 1

    def test_error_handling_continues(self):
        """Errors on one game don't stop polling other games."""
        conn = _make_db()
        _seed_active_game(conn, game_id='game-a', sb_event_id=100, xml_file='a/100.xml')
        _seed_active_game(conn, game_id='game-b', sb_event_id=101, xml_file='b/101.xml')

        mock_client = MagicMock()

        def fake_stats(event_id, xml_file, filetime=0):
            if event_id == 100:
                raise ConnectionError("Network error")
            return (SAMPLE_HTML, 12345)

        mock_client.get_live_stats.side_effect = fake_stats

        poller = StatBroadcastPoller(conn, client=mock_client)
        result = poller.poll_once()
        # game-b should still update even though game-a errored
        assert result == 1

    def test_stop_flag(self):
        """Poller stops when stop() is called."""
        conn = _make_db()
        poller = StatBroadcastPoller(conn, interval=1)
        poller.stop()
        assert poller._running is False

    def test_missing_xml_file(self):
        """Skips events with no xml_file."""
        conn = _make_db()
        _seed_active_game(conn, xml_file='')

        mock_client = MagicMock()
        poller = StatBroadcastPoller(conn, client=mock_client)
        result = poller.poll_once()
        assert result == 0
        mock_client.get_live_stats.assert_not_called()
