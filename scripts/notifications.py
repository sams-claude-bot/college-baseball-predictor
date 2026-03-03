#!/usr/bin/env python3
"""
Push notification system for college baseball alerts.

Handles:
  - Push subscription storage (Web Push API)
  - Alert preference management
  - Sending notifications via pywebpush

Alert types:
  - game_update: Half-inning summaries for subscribed teams
  - upset_watch: SEC (or tracked conf) team >75% likely to lose
  - final_score: Game final results for subscribed teams
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

from pywebpush import webpush, WebPushException

logger = logging.getLogger('notifications')

DATA_DIR = Path(__file__).parent.parent / 'data'
DB_PATH = DATA_DIR / 'baseball.db'
VAPID_PATH = DATA_DIR / 'vapid_keys.json'


def _load_vapid():
    """Load VAPID keys from file."""
    with open(VAPID_PATH) as f:
        return json.load(f)


def ensure_tables(conn=None):
    """Create notification tables if they don't exist."""
    close = False
    if conn is None:
        conn = sqlite3.connect(str(DB_PATH))
        close = True

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS push_subscriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            endpoint TEXT UNIQUE NOT NULL,
            keys_json TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            last_used_at TEXT,
            active INTEGER DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS alert_preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subscription_id INTEGER NOT NULL,
            alert_type TEXT NOT NULL,  -- 'game_update', 'upset_watch', 'final_score'
            team_id TEXT,             -- NULL means all teams for that type
            conference TEXT,          -- for upset_watch: which conference to watch
            enabled INTEGER DEFAULT 1,
            FOREIGN KEY (subscription_id) REFERENCES push_subscriptions(id),
            UNIQUE(subscription_id, alert_type, team_id)
        );

        CREATE TABLE IF NOT EXISTS notification_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT,
            alert_type TEXT,
            message TEXT,
            sent_at TEXT DEFAULT (datetime('now')),
            recipients INTEGER DEFAULT 0,
            dedup_key TEXT UNIQUE  -- prevents duplicate notifications
        );
    """)

    if close:
        conn.commit()
        conn.close()


def save_subscription(endpoint, keys, conn=None):
    """Save or update a push subscription."""
    close = False
    if conn is None:
        conn = sqlite3.connect(str(DB_PATH))
        close = True

    conn.execute("""
        INSERT INTO push_subscriptions (endpoint, keys_json)
        VALUES (?, ?)
        ON CONFLICT(endpoint) DO UPDATE SET
            keys_json = excluded.keys_json,
            active = 1,
            last_used_at = datetime('now')
    """, (endpoint, json.dumps(keys)))
    conn.commit()

    # Get the subscription ID
    row = conn.execute(
        "SELECT id FROM push_subscriptions WHERE endpoint = ?", (endpoint,)
    ).fetchone()
    sub_id = row[0] if row else None

    if close:
        conn.close()
    return sub_id


def save_preferences(subscription_id, preferences, conn=None):
    """Save alert preferences for a subscription.

    preferences: list of dicts with keys: alert_type, team_id (optional), conference (optional)
    """
    close = False
    if conn is None:
        conn = sqlite3.connect(str(DB_PATH))
        close = True

    # Clear existing preferences for this subscription
    conn.execute("DELETE FROM alert_preferences WHERE subscription_id = ?",
                 (subscription_id,))

    for pref in preferences:
        conn.execute("""
            INSERT INTO alert_preferences (subscription_id, alert_type, team_id, conference)
            VALUES (?, ?, ?, ?)
        """, (subscription_id, pref['alert_type'],
              pref.get('team_id'), pref.get('conference')))

    conn.commit()
    if close:
        conn.close()


def remove_subscription(endpoint, conn=None):
    """Deactivate a push subscription."""
    close = False
    if conn is None:
        conn = sqlite3.connect(str(DB_PATH))
        close = True

    conn.execute("UPDATE push_subscriptions SET active = 0 WHERE endpoint = ?",
                 (endpoint,))
    conn.commit()
    if close:
        conn.close()


def get_subscribers_for_team(team_id, alert_type, conn=None):
    """Get active push subscriptions that want alerts for a specific team."""
    close = False
    if conn is None:
        conn = sqlite3.connect(str(DB_PATH))
        close = True

    rows = conn.execute("""
        SELECT ps.endpoint, ps.keys_json
        FROM push_subscriptions ps
        JOIN alert_preferences ap ON ps.id = ap.subscription_id
        WHERE ps.active = 1
          AND ap.enabled = 1
          AND ap.alert_type = ?
          AND (ap.team_id = ? OR ap.team_id IS NULL)
    """, (alert_type, team_id)).fetchall()

    if close:
        conn.close()
    return [(r[0], json.loads(r[1])) for r in rows]


def get_subscribers_for_conference(conference, alert_type, conn=None):
    """Get active push subscriptions that want alerts for a conference."""
    close = False
    if conn is None:
        conn = sqlite3.connect(str(DB_PATH))
        close = True

    rows = conn.execute("""
        SELECT DISTINCT ps.endpoint, ps.keys_json
        FROM push_subscriptions ps
        JOIN alert_preferences ap ON ps.id = ap.subscription_id
        WHERE ps.active = 1
          AND ap.enabled = 1
          AND ap.alert_type = ?
          AND (ap.conference = ? OR ap.conference IS NULL)
    """, (alert_type, conference)).fetchall()

    if close:
        conn.close()
    return [(r[0], json.loads(r[1])) for r in rows]


def _already_sent(dedup_key, conn):
    """Check if a notification with this dedup key was already sent."""
    row = conn.execute(
        "SELECT id FROM notification_log WHERE dedup_key = ?", (dedup_key,)
    ).fetchone()
    return row is not None


def send_push(endpoint, keys, payload, conn=None):
    """Send a single push notification.

    payload: dict with keys: title, body, url (optional), icon (optional), tag (optional)
    """
    try:
        vapid = _load_vapid()
        subscription_info = {
            "endpoint": endpoint,
            "keys": keys
        }
        webpush(
            subscription_info=subscription_info,
            data=json.dumps(payload),
            vapid_private_key=vapid['private_pem'],
            vapid_claims={"sub": vapid['contact']}
        )
        return True
    except WebPushException as e:
        logger.warning("Push failed for %s: %s", endpoint[:50], e)
        # If subscription expired (410 Gone), deactivate it
        if hasattr(e, 'response') and e.response and e.response.status_code == 410:
            if conn:
                conn.execute(
                    "UPDATE push_subscriptions SET active = 0 WHERE endpoint = ?",
                    (endpoint,))
                conn.commit()
        return False
    except Exception as e:
        logger.error("Push error: %s", e)
        return False


def send_team_notification(team_id, alert_type, payload, dedup_key=None, conn=None):
    """Send a notification to all subscribers of a team.

    Returns number of successful sends.
    """
    close = False
    if conn is None:
        conn = sqlite3.connect(str(DB_PATH), timeout=10)
        close = True

    if dedup_key and _already_sent(dedup_key, conn):
        if close:
            conn.close()
        return 0

    subscribers = get_subscribers_for_team(team_id, alert_type, conn)
    sent = 0
    for endpoint, keys in subscribers:
        if send_push(endpoint, keys, payload, conn):
            sent += 1

    # Log the notification
    if dedup_key:
        conn.execute("""
            INSERT OR IGNORE INTO notification_log (game_id, alert_type, message, recipients, dedup_key)
            VALUES (?, ?, ?, ?, ?)
        """, (payload.get('game_id'), alert_type,
              payload.get('body', ''), sent, dedup_key))
        conn.commit()

    if close:
        conn.close()
    return sent


def send_conference_notification(conference, alert_type, payload, dedup_key=None, conn=None):
    """Send a notification to all subscribers watching a conference."""
    close = False
    if conn is None:
        conn = sqlite3.connect(str(DB_PATH), timeout=10)
        close = True

    if dedup_key and _already_sent(dedup_key, conn):
        if close:
            conn.close()
        return 0

    subscribers = get_subscribers_for_conference(conference, alert_type, conn)
    sent = 0
    for endpoint, keys in subscribers:
        if send_push(endpoint, keys, payload, conn):
            sent += 1

    if dedup_key:
        conn.execute("""
            INSERT OR IGNORE INTO notification_log (game_id, alert_type, message, recipients, dedup_key)
            VALUES (?, ?, ?, ?, ?)
        """, (payload.get('game_id'), alert_type,
              payload.get('body', ''), sent, dedup_key))
        conn.commit()

    if close:
        conn.close()
    return sent
