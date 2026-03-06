#!/usr/bin/env python3
"""Alerts blueprint API tests (subscriptions, game follows, preference persistence)."""

import sqlite3
import sys
from pathlib import Path

import pytest
from flask import Flask

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import web.blueprints.alerts as alerts_module


def _connection_factory(db_path):
    """Return a get_connection-compatible factory bound to a sqlite file."""

    def _get_connection():
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn

    return _get_connection


@pytest.fixture
def alerts_client(tmp_path, monkeypatch):
    """Minimal Flask app with alerts blueprint backed by a temp sqlite DB."""
    db_path = tmp_path / 'alerts-test.db'
    monkeypatch.setattr(alerts_module, 'get_connection', _connection_factory(db_path))

    app = Flask(__name__)
    app.config['TESTING'] = True
    app.register_blueprint(alerts_module.alerts_bp)

    return app.test_client(), db_path


def _subscription(endpoint='https://push.example/sub-1'):
    return {
        'endpoint': endpoint,
        'keys': {
            'p256dh': 'p256dh-key',
            'auth': 'auth-key',
        },
    }


def test_subscribe_saves_team_and_game_preferences(alerts_client):
    client, db_path = alerts_client

    payload = {
        'subscription': _subscription(),
        'preferences': [
            {'alert_type': 'game_update_scoring', 'team_id': 'auburn'},
            {'alert_type': 'score_change', 'team_id': 'auburn'},
            {'alert_type': 'final_score', 'team_id': 'auburn'},
            {'alert_type': 'game_update_scoring', 'game_id': '2026-03-05_auburn_tennessee'},
            {'alert_type': 'upset_watch', 'conference': 'SEC'},
        ],
    }

    resp = client.post('/api/push/subscribe', json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['ok'] is True
    assert isinstance(data['subscription_id'], int)

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT alert_type, team_id, conference, game_id, enabled
        FROM alert_preferences
        ORDER BY alert_type, COALESCE(team_id, ''), COALESCE(game_id, '')
        """
    ).fetchall()
    conn.close()

    assert len(rows) == 5
    assert (
        'game_update_scoring',
        'auburn',
        None,
        None,
        1,
    ) in rows
    assert (
        'game_update_scoring',
        None,
        None,
        '2026-03-05_auburn_tennessee',
        1,
    ) in rows
    assert ('upset_watch', None, 'SEC', None, 1) in rows


def test_subscribe_replaces_existing_preferences_for_same_subscription(alerts_client):
    client, db_path = alerts_client

    first = {
        'subscription': _subscription('https://push.example/sub-replace'),
        'preferences': [
            {'alert_type': 'game_update_scoring', 'team_id': 'auburn'},
            {'alert_type': 'final_score', 'team_id': 'auburn'},
        ],
    }
    second = {
        'subscription': _subscription('https://push.example/sub-replace'),
        'preferences': [
            {'alert_type': 'score_change', 'team_id': 'tennessee'},
        ],
    }

    resp1 = client.post('/api/push/subscribe', json=first)
    resp2 = client.post('/api/push/subscribe', json=second)

    assert resp1.status_code == 200
    assert resp2.status_code == 200

    conn = sqlite3.connect(db_path)
    prefs = conn.execute(
        "SELECT alert_type, team_id, game_id FROM alert_preferences ORDER BY id"
    ).fetchall()
    subs = conn.execute(
        "SELECT endpoint, active FROM push_subscriptions"
    ).fetchall()
    conn.close()

    # Same endpoint should remain a single active subscription.
    assert subs == [('https://push.example/sub-replace', 1)]
    # Preferences from first save are replaced by second save.
    assert prefs == [('score_change', 'tennessee', None)]


def test_game_follow_preference_upsert_and_remove(alerts_client):
    client, db_path = alerts_client

    sub = _subscription('https://push.example/sub-follow')
    game_id = '2026-03-05_auburn_tennessee'

    resp_add = client.post('/api/push/game-follow', json={
        'subscription': sub,
        'game_id': game_id,
        'enabled': True,
        'alert_type': 'game_update_scoring',
    })
    assert resp_add.status_code == 200
    assert resp_add.get_json()['ok'] is True

    conn = sqlite3.connect(db_path)
    rows_after_add = conn.execute(
        """
        SELECT alert_type, team_id, game_id, enabled
        FROM alert_preferences
        WHERE game_id = ?
        """,
        (game_id,),
    ).fetchall()
    assert rows_after_add == [('game_update_scoring', None, game_id, 1)]

    resp_remove = client.post('/api/push/game-follow', json={
        'subscription': sub,
        'game_id': game_id,
        'enabled': False,
        'alert_type': 'game_update_scoring',
    })
    assert resp_remove.status_code == 200
    assert resp_remove.get_json()['ok'] is True

    rows_after_remove = conn.execute(
        "SELECT COUNT(*) FROM alert_preferences WHERE game_id = ?",
        (game_id,),
    ).fetchone()[0]
    active_subs = conn.execute(
        "SELECT COUNT(*) FROM push_subscriptions WHERE active = 1"
    ).fetchone()[0]
    conn.close()

    assert rows_after_remove == 0
    # Subscription remains active even after unfollowing a specific game.
    assert active_subs == 1


def test_save_preferences_preserves_unrelated_game_follows(alerts_client):
    """Game-follow prefs set via /api/push/game-follow should survive
    a full save_preferences call that doesn't include that game_id."""
    client, db_path = alerts_client

    sub = _subscription('https://push.example/sub-preserve')

    # Step 1: Follow a game via game-follow endpoint
    resp1 = client.post('/api/push/game-follow', json={
        'subscription': sub,
        'game_id': '2026-03-05_auburn_tennessee',
        'enabled': True,
        'alert_type': 'game_update_scoring',
    })
    assert resp1.status_code == 200

    # Step 2: Save preferences (team-level) — should NOT wipe game follow
    resp2 = client.post('/api/push/subscribe', json={
        'subscription': sub,
        'preferences': [
            {'alert_type': 'game_update_scoring', 'team_id': 'lsu'},
            {'alert_type': 'final_score', 'team_id': 'lsu'},
        ],
    })
    assert resp2.status_code == 200

    # Step 3: Verify game follow survived
    conn = sqlite3.connect(db_path)
    game_prefs = conn.execute(
        "SELECT alert_type, game_id FROM alert_preferences WHERE game_id IS NOT NULL"
    ).fetchall()
    team_prefs = conn.execute(
        "SELECT alert_type, team_id FROM alert_preferences WHERE team_id IS NOT NULL"
    ).fetchall()
    conn.close()

    assert len(game_prefs) == 1
    assert game_prefs[0] == ('game_update_scoring', '2026-03-05_auburn_tennessee')
    assert len(team_prefs) == 2


def test_game_follow_rejects_unsupported_alert_type(alerts_client):
    client, _ = alerts_client

    resp = client.post('/api/push/game-follow', json={
        'subscription': _subscription('https://push.example/sub-invalid'),
        'game_id': '2026-03-05_auburn_tennessee',
        'enabled': True,
        'alert_type': 'final_score',
    })

    assert resp.status_code == 400
    data = resp.get_json()
    assert data['error'] == 'Unsupported alert_type'
