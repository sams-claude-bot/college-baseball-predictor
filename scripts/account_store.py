#!/usr/bin/env python3
"""Lightweight account/session storage for favorites + alert sync."""

from __future__ import annotations

import json
import secrets
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Iterable, Optional, Tuple

SESSION_COOKIE_NAME = 'dinger_session'
SESSION_MAX_AGE_SECONDS = 60 * 60 * 24 * 365  # 1 year
LINK_CODE_TTL_MINUTES = 15


def _utc_now_sql() -> str:
    return datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')


def _future_sql(minutes: int) -> str:
    return (datetime.utcnow() + timedelta(minutes=minutes)).strftime('%Y-%m-%d %H:%M:%S')


def _new_public_id() -> str:
    # Short, non-sensitive ID used for UI/debug display.
    return secrets.token_urlsafe(9).replace('-', '').replace('_', '')[:12]


def _new_session_token() -> str:
    return secrets.token_urlsafe(32)


def _new_link_code() -> str:
    alphabet = '23456789ABCDEFGHJKLMNPQRSTUVWXYZ'
    return ''.join(secrets.choice(alphabet) for _ in range(6))


def ensure_tables(conn: sqlite3.Connection) -> None:
    """Ensure account/session/favorites tables exist."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS user_accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            public_id TEXT UNIQUE NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS account_sessions (
            token TEXT PRIMARY KEY,
            account_id INTEGER NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            last_seen_at TEXT DEFAULT (datetime('now')),
            expires_at TEXT,
            FOREIGN KEY (account_id) REFERENCES user_accounts(id)
        );

        CREATE TABLE IF NOT EXISTS account_favorite_teams (
            account_id INTEGER NOT NULL,
            team_id TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (account_id, team_id),
            FOREIGN KEY (account_id) REFERENCES user_accounts(id)
        );

        CREATE TABLE IF NOT EXISTS account_favorite_games (
            account_id INTEGER NOT NULL,
            game_id TEXT NOT NULL,
            meta_json TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (account_id, game_id),
            FOREIGN KEY (account_id) REFERENCES user_accounts(id)
        );

        CREATE TABLE IF NOT EXISTS account_game_exclusions (
            account_id INTEGER NOT NULL,
            game_id TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (account_id, game_id),
            FOREIGN KEY (account_id) REFERENCES user_accounts(id)
        );

        CREATE TABLE IF NOT EXISTS account_link_codes (
            code TEXT PRIMARY KEY,
            account_id INTEGER NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            expires_at TEXT NOT NULL,
            used_at TEXT,
            used_by_account_id INTEGER,
            FOREIGN KEY (account_id) REFERENCES user_accounts(id),
            FOREIGN KEY (used_by_account_id) REFERENCES user_accounts(id)
        );

        CREATE INDEX IF NOT EXISTS idx_account_sessions_account
            ON account_sessions(account_id);
        CREATE INDEX IF NOT EXISTS idx_account_link_codes_expires
            ON account_link_codes(expires_at);
        """
    )
    conn.commit()


def cleanup_expired(conn: sqlite3.Connection):
    """Purge expired link codes and orphaned sessions. Best-effort."""
    try:
        conn.execute("DELETE FROM account_link_codes WHERE datetime(expires_at) < datetime('now')")
        conn.execute(
            "DELETE FROM account_sessions WHERE expires_at IS NOT NULL AND datetime(expires_at) < datetime('now')"
        )
        conn.commit()
    except Exception:
        pass  # Non-fatal


def _create_account(conn: sqlite3.Connection) -> Tuple[int, str]:
    for _ in range(10):
        public_id = _new_public_id()
        try:
            cur = conn.execute(
                "INSERT INTO user_accounts (public_id, created_at, updated_at) VALUES (?, ?, ?)",
                (public_id, _utc_now_sql(), _utc_now_sql()),
            )
            conn.commit()
            return int(cur.lastrowid), public_id
        except sqlite3.IntegrityError:
            continue
    raise RuntimeError('failed to generate unique account public_id')


def _create_session(conn: sqlite3.Connection, account_id: int) -> str:
    token = _new_session_token()
    conn.execute(
        """
        INSERT INTO account_sessions (token, account_id, created_at, last_seen_at, expires_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            token,
            account_id,
            _utc_now_sql(),
            _utc_now_sql(),
            (datetime.utcnow() + timedelta(seconds=SESSION_MAX_AGE_SECONDS)).strftime('%Y-%m-%d %H:%M:%S'),
        ),
    )
    conn.commit()
    return token


def _get_session_row(conn: sqlite3.Connection, session_token: Optional[str]):
    if not session_token:
        return None
    return conn.execute(
        """
        SELECT s.token, s.account_id, a.public_id
        FROM account_sessions s
        JOIN user_accounts a ON a.id = s.account_id
        WHERE s.token = ?
          AND (s.expires_at IS NULL OR datetime(s.expires_at) > datetime('now'))
        """,
        (session_token,),
    ).fetchone()


def ensure_account_session(
    conn: sqlite3.Connection,
    session_token: Optional[str],
) -> Tuple[int, str, str, bool]:
    """
    Ensure request has a valid account session.

    Returns (account_id, public_id, session_token, created_new)
    """
    ensure_tables(conn)

    row = _get_session_row(conn, session_token)
    if row:
        conn.execute(
            "UPDATE account_sessions SET last_seen_at = ? WHERE token = ?",
            (_utc_now_sql(), row['token']),
        )
        conn.commit()
        return int(row['account_id']), row['public_id'], row['token'], False

    account_id, public_id = _create_account(conn)
    new_token = _create_session(conn, account_id)
    return account_id, public_id, new_token, True


def get_account_id_for_session(conn: sqlite3.Connection, session_token: Optional[str]) -> Optional[int]:
    """Resolve account_id for a valid session token (or None)."""
    row = _get_session_row(conn, session_token)
    if not row:
        return None
    conn.execute(
        "UPDATE account_sessions SET last_seen_at = ? WHERE token = ?",
        (_utc_now_sql(), row['token']),
    )
    conn.commit()
    return int(row['account_id'])


def get_public_id(conn: sqlite3.Connection, account_id: int) -> Optional[str]:
    row = conn.execute(
        "SELECT public_id FROM user_accounts WHERE id = ?",
        (account_id,),
    ).fetchone()
    return row['public_id'] if row else None


def set_session_account(conn: sqlite3.Connection, session_token: str, account_id: int) -> None:
    conn.execute(
        "UPDATE account_sessions SET account_id = ?, last_seen_at = ? WHERE token = ?",
        (account_id, _utc_now_sql(), session_token),
    )
    conn.commit()


def _normalize_teams(team_ids: Optional[Iterable]) -> list[str]:
    clean = []
    if not team_ids:
        return clean
    seen = set()
    for tid in team_ids:
        if tid is None:
            continue
        text = str(tid).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        clean.append(text)
    return clean


def _normalize_games(games_input) -> Dict[str, dict]:
    out: Dict[str, dict] = {}

    if games_input is None:
        return out

    if isinstance(games_input, list):
        for item in games_input:
            if not isinstance(item, dict):
                continue
            gid = str(item.get('gameId') or item.get('game_id') or '').strip()
            if not gid:
                continue
            normalized = dict(item)
            normalized['gameId'] = gid
            out[gid] = normalized
        return out

    if isinstance(games_input, dict):
        for gid, raw_meta in games_input.items():
            game_id = str(gid or '').strip()
            if not game_id:
                continue
            meta = raw_meta if isinstance(raw_meta, dict) else {}
            normalized = dict(meta)
            normalized['gameId'] = game_id
            out[game_id] = normalized
        return out

    return out


def get_account_state(conn: sqlite3.Connection, account_id: int) -> dict:
    teams = [
        r['team_id']
        for r in conn.execute(
            "SELECT team_id FROM account_favorite_teams WHERE account_id = ? ORDER BY team_id",
            (account_id,),
        ).fetchall()
    ]

    games = {}
    rows = conn.execute(
        "SELECT game_id, meta_json FROM account_favorite_games WHERE account_id = ?",
        (account_id,),
    ).fetchall()
    for row in rows:
        game_id = row['game_id']
        meta = {}
        raw = row['meta_json']
        if raw:
            try:
                meta = json.loads(raw)
            except Exception:
                meta = {}
        if not isinstance(meta, dict):
            meta = {}
        meta['gameId'] = game_id
        games[game_id] = meta

    # Game exclusions (games unfollowed from team-follows)
    exclusions = [
        r['game_id']
        for r in conn.execute(
            "SELECT game_id FROM account_game_exclusions WHERE account_id = ? ORDER BY game_id",
            (account_id,),
        ).fetchall()
    ]

    return {'teams': teams, 'games': games, 'exclusions': exclusions}


def save_account_state(conn: sqlite3.Connection, account_id: int, team_ids, games_input, exclusions_input=None) -> dict:
    teams = _normalize_teams(team_ids)
    games = _normalize_games(games_input)
    exclusions = _normalize_teams(exclusions_input)  # reuse: list of game_id strings

    conn.execute("DELETE FROM account_favorite_teams WHERE account_id = ?", (account_id,))
    if teams:
        conn.executemany(
            "INSERT INTO account_favorite_teams (account_id, team_id, created_at) VALUES (?, ?, ?)",
            [(account_id, team_id, _utc_now_sql()) for team_id in teams],
        )

    conn.execute("DELETE FROM account_favorite_games WHERE account_id = ?", (account_id,))
    if games:
        now = _utc_now_sql()
        conn.executemany(
            """
            INSERT INTO account_favorite_games
                (account_id, game_id, meta_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (account_id, gid, json.dumps(meta), now, now)
                for gid, meta in games.items()
            ],
        )

    # Sync game exclusions (games unfollowed from team-follows)
    conn.execute("DELETE FROM account_game_exclusions WHERE account_id = ?", (account_id,))
    if exclusions:
        conn.executemany(
            "INSERT INTO account_game_exclusions (account_id, game_id, created_at) VALUES (?, ?, ?)",
            [(account_id, gid, _utc_now_sql()) for gid in exclusions],
        )

    conn.execute(
        "UPDATE user_accounts SET updated_at = ? WHERE id = ?",
        (_utc_now_sql(), account_id),
    )
    conn.commit()

    return {
        'teams_count': len(teams),
        'games_count': len(games),
        'exclusions_count': len(exclusions),
    }


def create_link_code(conn: sqlite3.Connection, account_id: int) -> Tuple[str, str]:
    ensure_tables(conn)
    expires_at = _future_sql(LINK_CODE_TTL_MINUTES)
    for _ in range(20):
        code = _new_link_code()
        try:
            conn.execute(
                """
                INSERT INTO account_link_codes (code, account_id, created_at, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (code, account_id, _utc_now_sql(), expires_at),
            )
            conn.commit()
            return code, expires_at
        except sqlite3.IntegrityError:
            continue
    raise RuntimeError('failed to generate unique link code')


def _push_subscriptions_exists(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='push_subscriptions'"
    ).fetchone()
    return row is not None


def _merge_accounts(conn: sqlite3.Connection, source_account_id: int, target_account_id: int) -> None:
    if source_account_id == target_account_id:
        return

    # Merge team follows (union).
    conn.execute(
        """
        INSERT OR IGNORE INTO account_favorite_teams (account_id, team_id, created_at)
        SELECT ?, team_id, ?
        FROM account_favorite_teams
        WHERE account_id = ?
        """,
        (target_account_id, _utc_now_sql(), source_account_id),
    )

    # Merge game follows (preserve target metadata if conflict).
    conn.execute(
        """
        INSERT OR IGNORE INTO account_favorite_games (account_id, game_id, meta_json, created_at, updated_at)
        SELECT ?, game_id, meta_json, ?, ?
        FROM account_favorite_games
        WHERE account_id = ?
        """,
        (target_account_id, _utc_now_sql(), _utc_now_sql(), source_account_id),
    )

    # Merge game exclusions.
    try:
        conn.execute(
            """
            INSERT OR IGNORE INTO account_game_exclusions (account_id, game_id, created_at)
            SELECT ?, game_id, ?
            FROM account_game_exclusions
            WHERE account_id = ?
            """,
            (target_account_id, _utc_now_sql(), source_account_id),
        )
    except sqlite3.OperationalError:
        pass  # Table may not exist on older DBs; non-fatal

    # If notification subscriptions are account-linked, move them to target.
    if _push_subscriptions_exists(conn):
        cols = [r[1] for r in conn.execute("PRAGMA table_info(push_subscriptions)").fetchall()]
        if 'account_id' in cols:
            conn.execute(
                "UPDATE push_subscriptions SET account_id = ? WHERE account_id = ?",
                (target_account_id, source_account_id),
            )

    # Clean up source account data (now merged into target).
    conn.execute("DELETE FROM account_favorite_teams WHERE account_id = ?", (source_account_id,))
    conn.execute("DELETE FROM account_favorite_games WHERE account_id = ?", (source_account_id,))
    try:
        conn.execute("DELETE FROM account_game_exclusions WHERE account_id = ?", (source_account_id,))
    except sqlite3.OperationalError:
        pass

    conn.execute(
        "UPDATE user_accounts SET updated_at = ? WHERE id IN (?, ?)",
        (_utc_now_sql(), source_account_id, target_account_id),
    )
    conn.commit()


def redeem_link_code(
    conn: sqlite3.Connection,
    code: str,
    current_account_id: Optional[int],
) -> Optional[int]:
    normalized = str(code or '').strip().upper()
    if not normalized:
        return None

    row = conn.execute(
        """
        SELECT code, account_id
        FROM account_link_codes
        WHERE code = ?
          AND used_at IS NULL
          AND datetime(expires_at) > datetime('now')
        """,
        (normalized,),
    ).fetchone()
    if not row:
        return None

    target_account_id = int(row['account_id'])

    if current_account_id and current_account_id != target_account_id:
        _merge_accounts(conn, current_account_id, target_account_id)

    conn.execute(
        """
        UPDATE account_link_codes
        SET used_at = ?, used_by_account_id = ?
        WHERE code = ?
        """,
        (_utc_now_sql(), current_account_id, normalized),
    )
    conn.commit()
    return target_account_id
