"""Light account/session API for syncing favorites and alerts across devices."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from scripts.account_store import (
    SESSION_COOKIE_NAME,
    SESSION_MAX_AGE_SECONDS,
    create_link_code,
    ensure_account_session,
    ensure_tables,
    get_account_state,
    get_public_id,
    redeem_link_code,
    save_account_state,
    set_session_account,
)
from scripts.database import get_connection

account_bp = Blueprint('account', __name__)


def _json_with_session_cookie(payload: dict, session_token: str, old_token: str | None):
    response = jsonify(payload)
    if session_token != old_token:
        response.set_cookie(
            SESSION_COOKIE_NAME,
            session_token,
            max_age=SESSION_MAX_AGE_SECONDS,
            httponly=True,
            samesite='Lax',
            path='/',
        )
    return response


@account_bp.route('/api/account/bootstrap', methods=['POST'])
def bootstrap_account():
    """Ensure this browser has an account session and return summary."""
    conn = get_connection()
    ensure_tables(conn)

    incoming = request.cookies.get(SESSION_COOKIE_NAME)
    account_id, public_id, session_token, created = ensure_account_session(conn, incoming)
    state = get_account_state(conn, account_id)

    payload = {
        'ok': True,
        'account': {
            'id': public_id,
            'created': created,
            'teams_count': len(state['teams']),
            'games_count': len(state['games']),
        },
    }
    conn.close()
    return _json_with_session_cookie(payload, session_token, incoming)


@account_bp.route('/api/account/state', methods=['GET'])
def account_state():
    """Fetch server-side favorites state for this account."""
    conn = get_connection()
    ensure_tables(conn)

    incoming = request.cookies.get(SESSION_COOKIE_NAME)
    account_id, public_id, session_token, _created = ensure_account_session(conn, incoming)
    state = get_account_state(conn, account_id)

    payload = {
        'ok': True,
        'account': {'id': public_id},
        'teams': state['teams'],
        'games': state['games'],
    }
    conn.close()
    return _json_with_session_cookie(payload, session_token, incoming)


@account_bp.route('/api/account/favorites', methods=['POST'])
def save_favorites():
    """Replace server-side favorites (teams + games) for this account."""
    data = request.get_json() or {}
    teams = data.get('teams', [])
    games = data.get('games', {})

    conn = get_connection()
    ensure_tables(conn)

    incoming = request.cookies.get(SESSION_COOKIE_NAME)
    account_id, public_id, session_token, _created = ensure_account_session(conn, incoming)
    counts = save_account_state(conn, account_id, teams, games)

    payload = {
        'ok': True,
        'account': {'id': public_id},
        **counts,
    }
    conn.close()
    return _json_with_session_cookie(payload, session_token, incoming)


@account_bp.route('/api/account/link-code', methods=['POST'])
def create_account_link_code():
    """Create one-time code to link another device to this account."""
    conn = get_connection()
    ensure_tables(conn)

    incoming = request.cookies.get(SESSION_COOKIE_NAME)
    account_id, public_id, session_token, _created = ensure_account_session(conn, incoming)
    code, expires_at = create_link_code(conn, account_id)

    payload = {
        'ok': True,
        'account': {'id': public_id},
        'code': code,
        'expires_at': expires_at,
    }
    conn.close()
    return _json_with_session_cookie(payload, session_token, incoming)


@account_bp.route('/api/account/link-code/redeem', methods=['POST'])
def redeem_account_link_code():
    """Redeem one-time code and switch this device to the source account."""
    data = request.get_json() or {}
    code = str(data.get('code') or '').strip()
    if not code:
        return jsonify({'error': 'Missing code'}), 400

    conn = get_connection()
    ensure_tables(conn)

    incoming = request.cookies.get(SESSION_COOKIE_NAME)
    current_account_id, _public_id, session_token, _created = ensure_account_session(conn, incoming)

    target_account_id = redeem_link_code(conn, code, current_account_id)
    if not target_account_id:
        conn.close()
        return jsonify({'error': 'Invalid or expired code'}), 400

    set_session_account(conn, session_token, target_account_id)
    public_id = get_public_id(conn, target_account_id)
    state = get_account_state(conn, target_account_id)

    payload = {
        'ok': True,
        'account': {'id': public_id},
        'teams': state['teams'],
        'games': state['games'],
    }
    conn.close()
    return _json_with_session_cookie(payload, session_token, incoming)
