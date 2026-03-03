"""Alerts blueprint — push notification subscription and preferences."""

import json
from flask import Blueprint, render_template, request, jsonify
from scripts.database import get_connection
from scripts.notifications import (
    ensure_tables, save_subscription, save_preferences,
    remove_subscription, send_push, _load_vapid
)

alerts_bp = Blueprint('alerts', __name__)


@alerts_bp.before_app_first_request
def _init_tables():
    conn = get_connection()
    ensure_tables(conn)
    conn.close()


@alerts_bp.route('/alerts')
def alerts_page():
    """Alert subscription page."""
    conn = get_connection()

    # Get all teams grouped by conference for the picker
    teams = conn.execute("""
        SELECT id, name, conference, primary_color
        FROM teams
        ORDER BY conference, name
    """).fetchall()
    conn.close()

    # Group by conference
    conferences = {}
    for t in teams:
        conf = t['conference'] or 'Independent'
        if conf not in conferences:
            conferences[conf] = []
        conferences[conf].append({
            'id': t['id'],
            'name': t['name'],
            'color': t['primary_color'] or '#64748b'
        })

    # Sort conferences: power conferences first
    power_order = ['SEC', 'ACC', 'Big 12', 'Big Ten', 'Pac-12']
    sorted_confs = []
    for pc in power_order:
        if pc in conferences:
            sorted_confs.append((pc, conferences.pop(pc)))
    for conf in sorted(conferences.keys()):
        sorted_confs.append((conf, conferences[conf]))

    # Load VAPID public key for the JS client
    try:
        vapid = _load_vapid()
        vapid_public_key = vapid['public_key']
    except Exception:
        vapid_public_key = ''

    # Get subscriber count for social proof
    conn = get_connection()
    sub_count = conn.execute(
        "SELECT COUNT(*) FROM push_subscriptions WHERE active = 1"
    ).fetchone()[0]
    conn.close()

    return render_template('alerts.html',
                           conferences=sorted_confs,
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

    # Save subscription
    sub_id = save_subscription(endpoint, keys, conn)

    # Save preferences
    preferences = data.get('preferences', [])
    if preferences:
        save_preferences(sub_id, preferences, conn)

    conn.close()
    return jsonify({'ok': True, 'subscription_id': sub_id})


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
