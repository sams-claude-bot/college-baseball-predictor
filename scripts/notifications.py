#!/usr/bin/env python3
"""
Push notification system for college baseball alerts.

Handles:
  - Push subscription storage (Web Push API)
  - Alert preference management
  - Sending notifications via pywebpush

Alert types:
  - game_update: Legacy half-inning summaries (all transitions)
  - game_update_scoring: Half-inning recaps only when runs score
  - score_change: Instant alerts whenever score changes
  - upset_watch: SEC (or tracked conf) team >75% likely to lose
  - final_score: Game final results for subscribed teams
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

from py_vapid import Vapid
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
            account_id INTEGER,
            created_at TEXT DEFAULT (datetime('now')),
            last_used_at TEXT,
            active INTEGER DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS alert_preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subscription_id INTEGER NOT NULL,
            alert_type TEXT NOT NULL,  -- 'game_update', 'game_update_scoring', 'score_change', 'upset_watch', 'final_score'
            team_id TEXT,             -- team-specific alerts
            conference TEXT,          -- conference-specific alerts (e.g. upset_watch)
            game_id TEXT,             -- game-specific alerts (e.g. followed game inning recaps)
            enabled INTEGER DEFAULT 1,
            FOREIGN KEY (subscription_id) REFERENCES push_subscriptions(id),
            UNIQUE(subscription_id, alert_type, team_id, game_id)
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

    # Lightweight migration path for older DBs.
    push_cols = [r[1] for r in conn.execute("PRAGMA table_info(push_subscriptions)").fetchall()]
    if 'account_id' not in push_cols:
        conn.execute("ALTER TABLE push_subscriptions ADD COLUMN account_id INTEGER")

    alert_pref_cols = [r[1] for r in conn.execute("PRAGMA table_info(alert_preferences)").fetchall()]
    if 'game_id' not in alert_pref_cols:
        conn.execute("ALTER TABLE alert_preferences ADD COLUMN game_id TEXT")

    conn.executescript("""
        CREATE INDEX IF NOT EXISTS idx_push_sub_account
            ON push_subscriptions(account_id);
        CREATE INDEX IF NOT EXISTS idx_alert_pref_team_type
            ON alert_preferences(alert_type, team_id);
        CREATE INDEX IF NOT EXISTS idx_alert_pref_game_type
            ON alert_preferences(alert_type, game_id);
    """)

    if close:
        conn.commit()
        conn.close()


def save_subscription(endpoint, keys, conn=None, account_id=None):
    """Save or update a push subscription.

    account_id is optional and links endpoint to a lightweight server account.
    """
    close = False
    if conn is None:
        conn = sqlite3.connect(str(DB_PATH))
        close = True

    conn.execute("""
        INSERT INTO push_subscriptions (endpoint, keys_json, account_id)
        VALUES (?, ?, ?)
        ON CONFLICT(endpoint) DO UPDATE SET
            keys_json = excluded.keys_json,
            active = 1,
            last_used_at = datetime('now'),
            account_id = COALESCE(excluded.account_id, push_subscriptions.account_id)
    """, (endpoint, json.dumps(keys), account_id))
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

    preferences: list of dicts with keys:
      - alert_type (required)
      - team_id (optional)
      - conference (optional)
      - game_id (optional)

    Game-follow prefs from /api/push/game-follow that aren't explicitly
    included in the new preference list are preserved (not deleted).
    """
    close = False
    if conn is None:
        conn = sqlite3.connect(str(DB_PATH))
        close = True

    # Collect game_ids being saved in this batch so we know which to preserve.
    incoming_game_ids = set()
    for pref in preferences:
        gid = pref.get('game_id')
        if gid:
            incoming_game_ids.add(gid)

    # Delete non-game prefs (team/conference/wildcard) and game prefs that
    # ARE in the incoming batch (they'll be re-created below). Preserve
    # game-follow prefs NOT in the batch — those were set via the separate
    # /api/push/game-follow endpoint.
    if incoming_game_ids:
        conn.execute(
            "DELETE FROM alert_preferences WHERE subscription_id = ? AND game_id IS NULL",
            (subscription_id,),
        )
        placeholders = ','.join('?' for _ in incoming_game_ids)
        conn.execute(
            f"DELETE FROM alert_preferences WHERE subscription_id = ? AND game_id IN ({placeholders})",
            (subscription_id, *incoming_game_ids),
        )
    else:
        # No game prefs in batch — only clear non-game prefs.
        conn.execute(
            "DELETE FROM alert_preferences WHERE subscription_id = ? AND game_id IS NULL",
            (subscription_id,),
        )

    for pref in preferences:
        alert_type = (pref.get('alert_type') or '').strip()
        team_id = pref.get('team_id')
        conference = pref.get('conference')
        game_id = pref.get('game_id')

        # Hard guard: skip malformed team/game alerts with no scope.
        if alert_type in {'game_update', 'game_update_scoring', 'score_change', 'final_score'}:
            if not team_id and not game_id:
                continue

        # Sensible default for upset_watch when omitted.
        if alert_type == 'upset_watch' and not conference:
            conference = 'SEC'

        conn.execute("""
            INSERT INTO alert_preferences (subscription_id, alert_type, team_id, conference, game_id)
            VALUES (?, ?, ?, ?, ?)
        """, (
            subscription_id,
            alert_type,
            team_id,
            conference,
            game_id,
        ))

    conn.commit()
    if close:
        conn.close()


def set_game_alert_preference(subscription_id, game_id, enabled=True,
                              alert_type='game_update_scoring', conn=None):
    """Upsert/remove a game-specific alert preference for a subscription."""
    if not subscription_id or not game_id:
        return

    close = False
    if conn is None:
        conn = sqlite3.connect(str(DB_PATH))
        close = True

    # Ensure idempotency even on legacy schemas where UNIQUE with NULLs is weak.
    conn.execute(
        "DELETE FROM alert_preferences WHERE subscription_id = ? AND alert_type = ? AND game_id = ?",
        (subscription_id, alert_type, game_id),
    )

    if enabled:
        conn.execute("""
            INSERT INTO alert_preferences (subscription_id, alert_type, team_id, conference, game_id, enabled)
            VALUES (?, ?, NULL, NULL, ?, 1)
        """, (subscription_id, alert_type, game_id))

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
        SELECT DISTINCT ps.endpoint, ps.keys_json
        FROM push_subscriptions ps
        JOIN alert_preferences ap ON ps.id = ap.subscription_id
        WHERE ps.active = 1
          AND ap.enabled = 1
          AND ap.alert_type = ?
          AND ap.game_id IS NULL
          AND ap.team_id = ?
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
          AND ap.game_id IS NULL
          AND ap.conference = ?
    """, (alert_type, conference)).fetchall()

    if close:
        conn.close()
    return [(r[0], json.loads(r[1])) for r in rows]


def get_subscribers_for_game(game_id, alert_type, conn=None):
    """Get active push subscriptions that want alerts for a specific game.

    Following a game means you want ALL notification types for it,
    so we match any active preference for this game_id (ignoring alert_type).
    """
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
          AND ap.game_id = ?
    """, (game_id,)).fetchall()

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
        # Build Vapid instance from PEM (from_string expects raw b64, not PEM)
        vv = Vapid.from_pem(vapid['private_pem'].encode())
        webpush(
            subscription_info=subscription_info,
            data=json.dumps(payload),
            vapid_private_key=vv,
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


def send_game_notification(game_id, alert_type, payload, dedup_key=None, conn=None):
    """Send a notification to all subscribers watching a specific game."""
    close = False
    if conn is None:
        conn = sqlite3.connect(str(DB_PATH), timeout=10)
        close = True

    if dedup_key and _already_sent(dedup_key, conn):
        if close:
            conn.close()
        return 0

    subscribers = get_subscribers_for_game(game_id, alert_type, conn)
    sent = 0
    for endpoint, keys in subscribers:
        if send_push(endpoint, keys, payload, conn):
            sent += 1

    if dedup_key:
        conn.execute("""
            INSERT OR IGNORE INTO notification_log (game_id, alert_type, message, recipients, dedup_key)
            VALUES (?, ?, ?, ?, ?)
        """, (payload.get('game_id') or game_id, alert_type,
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
