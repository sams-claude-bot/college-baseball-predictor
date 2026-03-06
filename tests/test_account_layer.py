#!/usr/bin/env python3
"""Tests for lightweight account layer (favorites sync + push linkage)."""

import sqlite3
import sys
from pathlib import Path

import pytest
from flask import Flask

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import web.blueprints.account as account_module
import web.blueprints.alerts as alerts_module
from scripts.account_store import SESSION_COOKIE_NAME


def _connection_factory(db_path):
    def _get_connection():
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn

    return _get_connection


@pytest.fixture
def account_test_app(tmp_path, monkeypatch):
    db_path = tmp_path / 'account-layer.db'
    conn_factory = _connection_factory(db_path)

    monkeypatch.setattr(account_module, 'get_connection', conn_factory)
    monkeypatch.setattr(alerts_module, 'get_connection', conn_factory)

    app = Flask(__name__)
    app.config['TESTING'] = True
    app.register_blueprint(account_module.account_bp)
    app.register_blueprint(alerts_module.alerts_bp)
    return app, db_path


def test_bootstrap_creates_single_account_session(account_test_app):
    app, db_path = account_test_app
    client = app.test_client()

    resp1 = client.post('/api/account/bootstrap')
    assert resp1.status_code == 200
    data1 = resp1.get_json()
    assert data1['ok'] is True
    assert data1['account']['id']
    assert SESSION_COOKIE_NAME in (resp1.headers.get('Set-Cookie') or '')

    # Repeated bootstrap in same client should reuse session/account.
    resp2 = client.post('/api/account/bootstrap')
    assert resp2.status_code == 200
    data2 = resp2.get_json()
    assert data2['ok'] is True
    assert data2['account']['id'] == data1['account']['id']

    conn = sqlite3.connect(db_path)
    account_count = conn.execute("SELECT COUNT(*) FROM user_accounts").fetchone()[0]
    session_count = conn.execute("SELECT COUNT(*) FROM account_sessions").fetchone()[0]
    conn.close()

    assert account_count == 1
    assert session_count == 1


def test_favorites_roundtrip(account_test_app):
    app, _db_path = account_test_app
    client = app.test_client()

    assert client.post('/api/account/bootstrap').status_code == 200

    payload = {
        'teams': ['auburn', 'lsu', 'auburn'],
        'games': {
            '2026-03-05_auburn_tennessee': {
                'gameId': '2026-03-05_auburn_tennessee',
                'homeTeamId': 'auburn',
                'awayTeamId': 'tennessee',
            }
        },
    }

    save_resp = client.post('/api/account/favorites', json=payload)
    assert save_resp.status_code == 200
    save_data = save_resp.get_json()
    assert save_data['ok'] is True
    assert save_data['teams_count'] == 2
    assert save_data['games_count'] == 1

    state_resp = client.get('/api/account/state')
    assert state_resp.status_code == 200
    state = state_resp.get_json()

    assert set(state['teams']) == {'auburn', 'lsu'}
    assert '2026-03-05_auburn_tennessee' in state['games']


def test_link_code_merges_favorites_and_switches_device(account_test_app):
    app, _db_path = account_test_app
    client_a = app.test_client()
    client_b = app.test_client()

    # Device A
    assert client_a.post('/api/account/bootstrap').status_code == 200
    assert client_a.post('/api/account/favorites', json={
        'teams': ['auburn'],
        'games': {'g-a': {'gameId': 'g-a'}},
    }).status_code == 200
    code_resp = client_a.post('/api/account/link-code')
    assert code_resp.status_code == 200
    code = code_resp.get_json()['code']
    assert code

    # Device B (separate account first)
    assert client_b.post('/api/account/bootstrap').status_code == 200
    assert client_b.post('/api/account/favorites', json={
        'teams': ['lsu'],
        'games': {'g-b': {'gameId': 'g-b'}},
    }).status_code == 200

    redeem_resp = client_b.post('/api/account/link-code/redeem', json={'code': code})
    assert redeem_resp.status_code == 200
    redeem_data = redeem_resp.get_json()
    assert redeem_data['ok'] is True

    # Device B should now see merged state under device A account.
    state_b = client_b.get('/api/account/state').get_json()
    assert set(state_b['teams']) == {'auburn', 'lsu'}
    assert set(state_b['games'].keys()) == {'g-a', 'g-b'}


def test_push_subscription_links_to_account_when_session_exists(account_test_app):
    app, db_path = account_test_app
    client = app.test_client()

    assert client.post('/api/account/bootstrap').status_code == 200

    subscribe_payload = {
        'subscription': {
            'endpoint': 'https://push.example/sub-account-link',
            'keys': {
                'p256dh': 'p256dh-key',
                'auth': 'auth-key',
            },
        },
        'preferences': [
            {'alert_type': 'game_update_scoring', 'team_id': 'auburn'},
        ],
    }

    resp = client.post('/api/push/subscribe', json=subscribe_payload)
    assert resp.status_code == 200
    assert resp.get_json()['ok'] is True

    conn = sqlite3.connect(db_path)
    session_account_id = conn.execute(
        "SELECT account_id FROM account_sessions LIMIT 1"
    ).fetchone()[0]
    sub_account_id = conn.execute(
        "SELECT account_id FROM push_subscriptions WHERE endpoint = ?",
        ('https://push.example/sub-account-link',),
    ).fetchone()[0]
    conn.close()

    assert sub_account_id == session_account_id
