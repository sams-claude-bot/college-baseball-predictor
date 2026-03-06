#!/usr/bin/env python3
"""Tests for shared game notification dispatcher."""

import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from scripts.game_notifications import GameNotificationDispatcher

# The dispatcher does `from notifications import send_team_notification, ...`
# inside its methods, so we patch at the `notifications` module level.
PATCH_TEAM = 'notifications.send_team_notification'
PATCH_GAME = 'notifications.send_game_notification'
PATCH_CONF = 'notifications.send_conference_notification'
PATCH_ENSURE = 'notifications.ensure_tables'


def _make_db(tmp_path):
    """Create a minimal test DB with games + teams tables."""
    db_path = tmp_path / 'test_notif.db'
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    conn.executescript("""
        CREATE TABLE teams (
            id TEXT PRIMARY KEY,
            name TEXT,
            conference TEXT
        );
        CREATE TABLE games (
            id TEXT PRIMARY KEY,
            home_team_id TEXT,
            away_team_id TEXT,
            home_score INTEGER,
            away_score INTEGER,
            innings INTEGER,
            status TEXT,
            FOREIGN KEY (home_team_id) REFERENCES teams(id),
            FOREIGN KEY (away_team_id) REFERENCES teams(id)
        );

        INSERT INTO teams VALUES ('auburn', 'Auburn', 'SEC');
        INSERT INTO teams VALUES ('tennessee', 'Tennessee', 'SEC');
        INSERT INTO teams VALUES ('oregon-state', 'Oregon State', 'Pac-12');

        INSERT INTO games VALUES (
            '2026-03-06_auburn_tennessee', 'auburn', 'tennessee',
            5, 3, NULL, 'in-progress'
        );
        INSERT INTO games VALUES (
            '2026-03-06_auburn_oregon-state', 'auburn', 'oregon-state',
            7, 4, 9, 'final'
        );
    """)
    conn.commit()
    return conn


@pytest.fixture
def notif_db(tmp_path):
    return _make_db(tmp_path)


def test_first_sighting_seeds_state_no_notifications(notif_db):
    """First call should seed state without sending notifications."""
    dispatcher = GameNotificationDispatcher(notif_db)

    with patch(PATCH_TEAM) as mock_team, \
         patch(PATCH_GAME) as mock_game:
        dispatcher.check('2026-03-06_auburn_tennessee', {
            'inning': 3, 'inning_half': 'top',
            'home_score': 2, 'visitor_score': 1,
        })
        mock_team.assert_not_called()
        mock_game.assert_not_called()


def test_score_change_fires_score_change_alerts(notif_db):
    """Score change should fire score_change to both teams + game."""
    dispatcher = GameNotificationDispatcher(notif_db)
    game_id = '2026-03-06_auburn_tennessee'

    # Seed
    dispatcher.check(game_id, {
        'inning': 3, 'inning_half': 'top',
        'home_score': 2, 'visitor_score': 1,
    })

    with patch(PATCH_TEAM) as mock_team, \
         patch(PATCH_GAME) as mock_game, \
         patch(PATCH_ENSURE):
        dispatcher.check(game_id, {
            'inning': 3, 'inning_half': 'top',
            'home_score': 3, 'visitor_score': 1,
        })

        # Score change → team notifications for both teams
        score_change_calls = [
            c for c in mock_team.call_args_list
            if c[0][1] == 'score_change'
        ]
        assert len(score_change_calls) == 2

        # Score change → game notification too
        game_score_calls = [
            c for c in mock_game.call_args_list
            if c[0][1] == 'score_change'
        ]
        assert len(game_score_calls) == 1


def test_half_inning_scoring_recap(notif_db):
    """Half-inning transition with scoring should fire recap + game_update."""
    dispatcher = GameNotificationDispatcher(notif_db)
    game_id = '2026-03-06_auburn_tennessee'

    # Seed at top 3, score 1-2
    dispatcher.check(game_id, {
        'inning': 3, 'inning_half': 'top',
        'home_score': 2, 'visitor_score': 1,
    })

    with patch(PATCH_TEAM) as mock_team, \
         patch(PATCH_GAME) as mock_game, \
         patch(PATCH_ENSURE):
        # Transition to bot 3, away scored 2 runs
        dispatcher.check(game_id, {
            'inning': 3, 'inning_half': 'bottom',
            'home_score': 2, 'visitor_score': 3,
        })

        # Should have: game_update (2 teams) + game_update_scoring (2 teams) + score_change (2 teams)
        update_calls = [c for c in mock_team.call_args_list if c[0][1] == 'game_update']
        scoring_recap_calls = [c for c in mock_team.call_args_list if c[0][1] == 'game_update_scoring']
        score_change_calls = [c for c in mock_team.call_args_list if c[0][1] == 'score_change']

        assert len(update_calls) == 2  # Both teams
        assert len(scoring_recap_calls) == 2  # Both teams
        assert len(score_change_calls) == 2  # Both teams

        # Recap should also go to game-follow subscribers
        game_recap_calls = [c for c in mock_game.call_args_list if c[0][1] == 'game_update_scoring']
        assert len(game_recap_calls) == 1


def test_half_transition_no_scoring_no_recap(notif_db):
    """Half-inning transition without scoring should NOT fire recap."""
    dispatcher = GameNotificationDispatcher(notif_db)
    game_id = '2026-03-06_auburn_tennessee'

    dispatcher.check(game_id, {
        'inning': 3, 'inning_half': 'top',
        'home_score': 2, 'visitor_score': 1,
    })

    with patch(PATCH_TEAM) as mock_team, \
         patch(PATCH_GAME) as mock_game, \
         patch(PATCH_ENSURE):
        # Transition to bot 3, NO scoring
        dispatcher.check(game_id, {
            'inning': 3, 'inning_half': 'bottom',
            'home_score': 2, 'visitor_score': 1,
        })

        # game_update fires (half-inning transition), but NO recap
        update_calls = [c for c in mock_team.call_args_list if c[0][1] == 'game_update']
        scoring_calls = [c for c in mock_team.call_args_list if c[0][1] == 'game_update_scoring']
        score_change_calls = [c for c in mock_team.call_args_list if c[0][1] == 'score_change']

        assert len(update_calls) == 2
        assert len(scoring_calls) == 0
        assert len(score_change_calls) == 0


def test_check_final_cleans_up_game_follows(tmp_path):
    """check_final should remove game-follow prefs and account favorites after sending."""
    conn = _make_db(tmp_path)
    game_id = '2026-03-06_auburn_oregon-state'

    # Set up alert_preferences and account_favorite_games for this game
    conn.executescript(f"""
        CREATE TABLE IF NOT EXISTS push_subscriptions (
            id INTEGER PRIMARY KEY, endpoint TEXT, keys_json TEXT, active INTEGER DEFAULT 1,
            account_id INTEGER, created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS alert_preferences (
            id INTEGER PRIMARY KEY, subscription_id INTEGER, alert_type TEXT,
            team_id TEXT, conference TEXT, game_id TEXT, enabled INTEGER DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS account_favorite_games (
            account_id INTEGER, game_id TEXT, meta_json TEXT,
            created_at TEXT, updated_at TEXT, PRIMARY KEY (account_id, game_id)
        );
        CREATE TABLE IF NOT EXISTS notification_log (
            id INTEGER PRIMARY KEY, game_id TEXT, alert_type TEXT,
            message TEXT, recipients INTEGER, dedup_key TEXT UNIQUE, created_at TEXT DEFAULT (datetime('now'))
        );

        INSERT INTO push_subscriptions VALUES (1, 'https://push.test/x', '{{"p256dh":"a","auth":"b"}}', 1, NULL, datetime('now'));
        INSERT INTO alert_preferences VALUES (1, 1, 'game_update_scoring', NULL, NULL, '{game_id}', 1);
        INSERT INTO account_favorite_games VALUES (1, '{game_id}', NULL, datetime('now'), datetime('now'));
    """)
    conn.commit()

    # Verify data exists before
    assert conn.execute("SELECT COUNT(*) FROM alert_preferences WHERE game_id=?", (game_id,)).fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM account_favorite_games WHERE game_id=?", (game_id,)).fetchone()[0] == 1

    dispatcher = GameNotificationDispatcher(conn)

    with patch(PATCH_TEAM), patch(PATCH_GAME), patch(PATCH_ENSURE):
        dispatcher.check_final(game_id)

    # Both should be cleaned up
    assert conn.execute("SELECT COUNT(*) FROM alert_preferences WHERE game_id=?", (game_id,)).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM account_favorite_games WHERE game_id=?", (game_id,)).fetchone()[0] == 0


def test_check_final_sends_to_team_and_game_subscribers(notif_db):
    """check_final should send to both team-follow AND game-follow subscribers."""
    dispatcher = GameNotificationDispatcher(notif_db)
    game_id = '2026-03-06_auburn_oregon-state'

    with patch(PATCH_TEAM) as mock_team, \
         patch(PATCH_GAME) as mock_game, \
         patch(PATCH_ENSURE):
        dispatcher.check_final(game_id)

        # Team notifications for both teams
        assert mock_team.call_count == 2
        team_ids = {c[0][0] for c in mock_team.call_args_list}
        assert team_ids == {'auburn', 'oregon-state'}

        # Game notification for game-follow subscribers
        assert mock_game.call_count == 1
        assert mock_game.call_args[0][0] == game_id
        assert mock_game.call_args[0][1] == 'final_score'


def test_cleanup_game_removes_state(notif_db):
    """cleanup_game should clear cached state."""
    dispatcher = GameNotificationDispatcher(notif_db)
    game_id = '2026-03-06_auburn_tennessee'

    dispatcher.check(game_id, {
        'inning': 1, 'inning_half': 'top',
        'home_score': 0, 'visitor_score': 0,
    })

    assert game_id in dispatcher._last_half_state
    dispatcher.cleanup_game(game_id)
    assert game_id not in dispatcher._last_half_state
    assert game_id not in dispatcher._last_score_state
    assert game_id not in dispatcher._half_start_score


def test_no_notification_on_same_state_repeated(notif_db):
    """Repeated same state should not fire notifications."""
    dispatcher = GameNotificationDispatcher(notif_db)
    game_id = '2026-03-06_auburn_tennessee'

    # Seed
    dispatcher.check(game_id, {
        'inning': 3, 'inning_half': 'top',
        'home_score': 2, 'visitor_score': 1,
    })

    with patch(PATCH_TEAM) as mock_team:
        # Same state again
        dispatcher.check(game_id, {
            'inning': 3, 'inning_half': 'top',
            'home_score': 2, 'visitor_score': 1,
        })
        mock_team.assert_not_called()
