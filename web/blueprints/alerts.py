"""Alerts blueprint — push notification subscription and preferences."""

import json
from flask import Blueprint, render_template, request, jsonify
from scripts.database import get_connection
from scripts.notifications import (
    ensure_tables, save_subscription, save_preferences,
    set_game_alert_preference,
    remove_subscription, send_push, _load_vapid
)
from scripts.account_store import (
    SESSION_COOKIE_NAME,
    ensure_tables as ensure_account_tables,
    get_account_id_for_session,
)

alerts_bp = Blueprint('alerts', __name__)


@alerts_bp.route('/alerts')
def alerts_page():
    """Alert subscription page."""
    conn = get_connection()
    ensure_tables(conn)
    ensure_account_tables(conn)

    # Load VAPID public key for the JS client
    try:
        vapid = _load_vapid()
        vapid_public_key = vapid['public_key']
    except Exception:
        vapid_public_key = ''

    # Get subscriber count for social proof
    sub_count = conn.execute(
        "SELECT COUNT(*) FROM push_subscriptions WHERE active = 1"
    ).fetchone()[0]
    conn.close()

    return render_template('alerts.html',
                           vapid_public_key=vapid_public_key,
                           subscriber_count=sub_count)


@alerts_bp.route('/api/push/subscribe', methods=['POST'])
def subscribe():
    """Register a push subscription + alert preferences."""
    data = request.get_json()
    if not data or 'subscription' not in data:
        return jsonify({'error': 'Missing subscription data'}), 400

    sub = data['subscription']
    endpoint = sub.get('endpoint')
    keys = sub.get('keys', {})

    if not endpoint or not keys.get('p256dh') or not keys.get('auth'):
        return jsonify({'error': 'Invalid subscription'}), 400

    conn = get_connection()
    ensure_tables(conn)
    ensure_account_tables(conn)

    account_id = get_account_id_for_session(
        conn,
        request.cookies.get(SESSION_COOKIE_NAME),
    )

    # Save subscription
    sub_id = save_subscription(endpoint, keys, conn, account_id=account_id)

    # Save preferences (empty list is valid and clears existing prefs)
    preferences = data.get('preferences', [])
    if preferences is None:
        preferences = []
    if not isinstance(preferences, list):
        conn.close()
        return jsonify({'error': 'preferences must be a list'}), 400

    save_preferences(sub_id, preferences, conn)

    conn.close()
    return jsonify({'ok': True, 'subscription_id': sub_id})


@alerts_bp.route('/api/push/game-follow', methods=['POST'])
def game_follow_preference():
    """Upsert/remove game-follow alert preference for an active subscription."""
    data = request.get_json() or {}
    sub = data.get('subscription') or {}

    endpoint = sub.get('endpoint')
    keys = sub.get('keys', {})
    if not endpoint or not keys.get('p256dh') or not keys.get('auth'):
        return jsonify({'error': 'Invalid subscription'}), 400

    game_id = (data.get('game_id') or '').strip()
    if not game_id:
        return jsonify({'error': 'Missing game_id'}), 400

    alert_type = (data.get('alert_type') or 'game_update_scoring').strip()
    if alert_type not in {'game_update_scoring', 'score_change'}:
        return jsonify({'error': 'Unsupported alert_type'}), 400

    enabled = bool(data.get('enabled', True))

    conn = get_connection()
    ensure_tables(conn)
    ensure_account_tables(conn)

    account_id = get_account_id_for_session(
        conn,
        request.cookies.get(SESSION_COOKIE_NAME),
    )

    sub_id = save_subscription(endpoint, keys, conn, account_id=account_id)
    set_game_alert_preference(
        sub_id,
        game_id,
        enabled=enabled,
        alert_type=alert_type,
        conn=conn,
    )
    conn.close()

    return jsonify({
        'ok': True,
        'subscription_id': sub_id,
        'game_id': game_id,
        'alert_type': alert_type,
        'enabled': enabled,
    })


@alerts_bp.route('/api/push/unsubscribe', methods=['POST'])
def unsubscribe():
    """Remove a push subscription."""
    data = request.get_json()
    endpoint = data.get('endpoint') if data else None
    if not endpoint:
        return jsonify({'error': 'Missing endpoint'}), 400

    conn = get_connection()
    remove_subscription(endpoint, conn)
    conn.close()
    return jsonify({'ok': True})


@alerts_bp.route('/api/push/test', methods=['POST'])
def test_push():
    """Send a test notification to verify push is working."""
    data = request.get_json()
    sub = data.get('subscription') if data else None
    if not sub:
        return jsonify({'error': 'Missing subscription'}), 400

    payload = {
        'title': '⚾ Test Alert',
        'body': 'Push notifications are working! You\'ll get alerts for your teams.',
        'url': '/alerts',
        'tag': 'test'
    }

    conn = get_connection()
    success = send_push(sub['endpoint'], sub['keys'], payload, conn)
    conn.close()

    if success:
        return jsonify({'ok': True})
    else:
        return jsonify({'error': 'Push delivery failed'}), 500


@alerts_bp.route('/api/push/vapid-key')
def vapid_key():
    """Return the VAPID public key for client-side subscription."""
    try:
        vapid = _load_vapid()
        return jsonify({'publicKey': vapid['public_key']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
