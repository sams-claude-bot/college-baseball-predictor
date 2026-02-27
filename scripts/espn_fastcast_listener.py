#!/usr/bin/env python3
"""
ESPN FastCast WebSocket Listener — real-time live score daemon.

Connects to ESPN's FastCast WebSocket, subscribes to college-baseball topics,
decodes score/status updates, and writes them to the DB via ScheduleGateway.
Also inserts events into the live_events table for SSE streaming.

Protocol:
  1. Discovery → websockethost endpoint returns {ip, securePort, token}
  2. Connect   → wss://{ip}:{securePort}/FastcastService/pubsub/profiles/12000?...
  3. Handshake → send {"op":"C"}, receive {"op":"C","rc":200,"sid":"..."}
  4. Subscribe → send {"op":"S","sid":"...","tc":"<topic>"} per topic
  5. Messages  → H=Checkpoint, R=Replay, P=Publish, B=Heartbeat

Usage:
    python3 scripts/espn_fastcast_listener.py
"""

import asyncio
import base64
import json
import logging
import signal
import sqlite3
import sys
import time
import urllib.request
import zlib
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / 'data' / 'baseball.db'
MAPPING_CACHE = PROJECT_ROOT / 'data' / 'espn_team_mapping.json'

DISCOVERY_URL = 'https://fastcast.semfs.engsvc.go.com/public/websockethost'
CHECKPOINT_BASE = 'https://fcast.espncdn.com/FastcastService/pubsub/profiles/12000'

TOPICS = [
    'scoreboard-baseball-college-baseball',
    'event-baseball-college-baseball',
]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger('fastcast')


# ── Payload Helpers ──────────────────────────────────────────────

def decode_payload(raw_pl):
    """Decode a FastCast payload.

    Payloads can arrive as:
      1. Plain base64+zlib compressed JSON (publish messages)
      2. A JSON string wrapping {ts, ~c, pl} where pl is the b64+zlib data (replay/checkpoint)
      3. A JSON string that's directly parseable (rare)
    """
    # If it's a string, first check if it's a JSON wrapper
    if isinstance(raw_pl, str):
        try:
            inner = json.loads(raw_pl)
            if isinstance(inner, dict) and 'pl' in inner:
                raw_pl = inner['pl']
            elif isinstance(inner, (list, dict)):
                return inner  # Already decoded JSON
        except (json.JSONDecodeError, ValueError):
            pass  # Not JSON, treat as base64

    if not isinstance(raw_pl, str):
        return raw_pl

    # Add base64 padding if needed
    padded = raw_pl + '=' * (-len(raw_pl) % 4)
    raw = base64.b64decode(padded)
    decompressed = zlib.decompress(raw, 15 + 32)  # auto-detect gzip/zlib
    return json.loads(decompressed)


def parse_event_id(path_key):
    """Extract ESPN event ID from a FastCast path key.

    Example: 's:1~l:14~e:401862650' → '401862650'
    """
    for part in path_key.split('~'):
        if part.startswith('e:'):
            return part[2:]
    return None


def parse_patch_path(path):
    """Parse a JSON Patch path from FastCast.

    Example: 's:1~l:14~e:401862650/competitions/0/competitors/1/score' →
             ('401862650', 'competitions/0/competitors/1/score')
    """
    parts = path.split('/', 1)
    event_key = parts[0]
    field_path = parts[1] if len(parts) > 1 else ''
    event_id = parse_event_id(event_key)
    return event_id, field_path


# ── Team Mapping ─────────────────────────────────────────────────

def load_espn_mapping():
    """Load ESPN team ID → our DB team ID mapping from cache."""
    if MAPPING_CACHE.exists():
        with open(MAPPING_CACHE) as f:
            return json.load(f)
    return {}


def build_espn_mapping_if_needed():
    """Build the mapping if cache is stale (>24h) or missing."""
    if MAPPING_CACHE.exists():
        import os
        age_hours = (time.time() - os.path.getmtime(MAPPING_CACHE)) / 3600
        if age_hours < 24:
            return load_espn_mapping()

    # Delegate to the existing builder in espn_live_scores
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        from espn_live_scores import build_espn_id_mapping
        mapping = build_espn_id_mapping(conn)
        conn.close()
        return mapping
    except Exception as e:
        log.warning('Could not build ESPN mapping: %s', e)
        return load_espn_mapping()


# ── Database Helpers ─────────────────────────────────────────────

def ensure_live_events_table(conn):
    """Create the live_events table if it doesn't exist."""
    conn.execute('''
        CREATE TABLE IF NOT EXISTS live_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            game_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            data_json TEXT NOT NULL
        )
    ''')
    conn.execute('''
        CREATE INDEX IF NOT EXISTS idx_live_events_id ON live_events(id)
    ''')
    conn.commit()


def cleanup_old_events(conn, hours=24):
    """Delete live_events older than *hours*."""
    conn.execute(
        "DELETE FROM live_events WHERE created_at < datetime('now', ?)",
        (f'-{hours} hours',),
    )
    conn.commit()


def insert_live_event(conn, game_id, event_type, data):
    """Insert a row into the live_events table."""
    conn.execute(
        'INSERT INTO live_events (game_id, event_type, data_json) VALUES (?, ?, ?)',
        (game_id, event_type, json.dumps(data)),
    )
    conn.commit()


# ── State Tracker ────────────────────────────────────────────────

class GameStateTracker:
    """Maintains in-memory state of live games and detects changes."""

    def __init__(self):
        self.espn_mapping = build_espn_mapping_if_needed()
        self.games = {}          # espn_event_id → dict of full event data
        self.prev_scores = {}    # game_id → (home_score, away_score)
        self.prev_status = {}    # game_id → status string
        self.event_teams = {}    # espn_event_id → (away_db_id, home_db_id, date)

    def refresh_mapping(self):
        self.espn_mapping = build_espn_mapping_if_needed()

    # -- Checkpoint ingestion -----------------------------------------

    def ingest_checkpoint(self, topic, checkpoint_data):
        """Ingest full checkpoint data and index all events."""
        events = self._extract_events(checkpoint_data)
        for espn_id, event in events.items():
            self.games[espn_id] = event
            self._index_event_teams(espn_id, event)
        log.info('Checkpoint for %s: indexed %d events', topic, len(events))

    def _extract_events(self, data):
        """Walk checkpoint data structure to find events keyed by ESPN ID.

        Checkpoint formats:
          - scoreboard: {events: [{uid: 's:1~l:14~e:401862650', ...}, ...]}
          - event: {sports: [{leagues: [{events: [...]}]}]}
          - patch-keyed: {'s:1~l:14~e:401862650': {...}, ...}
        """
        events = {}
        if isinstance(data, dict):
            # Check for events list (scoreboard format)
            if 'events' in data and isinstance(data['events'], list):
                for e in data['events']:
                    if isinstance(e, dict):
                        uid = e.get('uid', '')
                        eid = parse_event_id(uid)
                        if eid:
                            events[eid] = e
                        elif e.get('id'):
                            events[str(e['id'])] = e

            # Check for sports→leagues→events (event topic format)
            if 'sports' in data and isinstance(data['sports'], list):
                for sport in data['sports']:
                    for league in sport.get('leagues', []):
                        for e in league.get('events', []):
                            uid = e.get('uid', '')
                            eid = parse_event_id(uid)
                            if eid:
                                events[eid] = e
                            elif e.get('id'):
                                events[str(e['id'])] = e

            # Check for patch-keyed format (keys like 's:1~l:14~e:...')
            for key, val in data.items():
                eid = parse_event_id(key)
                if eid and isinstance(val, dict):
                    events[eid] = val

        return events

    def _index_event_teams(self, espn_id, event):
        """Map ESPN event → (away_db_id, home_db_id, date)."""
        if espn_id in self.event_teams:
            return
        try:
            comp = event.get('competitions', [{}])[0]
            competitors = comp.get('competitors', [])
            away = next((c for c in competitors if c.get('homeAway') == 'away'), None)
            home = next((c for c in competitors if c.get('homeAway') == 'home'), None)
            if not away or not home:
                return
            away_espn = str(away['team']['id']) if isinstance(away.get('team'), dict) else str(away.get('id', ''))
            home_espn = str(home['team']['id']) if isinstance(home.get('team'), dict) else str(home.get('id', ''))
            away_db = self.espn_mapping.get(away_espn)
            home_db = self.espn_mapping.get(home_espn)
            if not away_db or not home_db:
                return
            # Date from event
            date_str = event.get('date', '')[:10]  # 'YYYY-MM-DD...'
            if not date_str:
                date_str = datetime.utcnow().strftime('%Y-%m-%d')
            self.event_teams[espn_id] = (away_db, home_db, date_str)
        except (KeyError, IndexError, TypeError):
            pass

    # -- Patch application --------------------------------------------

    def apply_patches(self, patches):
        """Apply a list of JSON Patch operations and return detected changes.

        Returns a list of dicts describing what changed.
        """
        changes = []
        touched_events = set()

        for patch in patches:
            op = patch.get('op')
            path = patch.get('path', '')
            value = patch.get('value')

            event_id, field_path = parse_patch_path(path)
            if not event_id:
                continue

            touched_events.add(event_id)

            # Apply to in-memory state (best-effort nested set)
            if event_id in self.games and op in ('replace', 'add'):
                _nested_set(self.games[event_id], field_path, value)

        # After applying all patches, check each touched event for changes
        for eid in touched_events:
            change = self._detect_changes(eid)
            if change:
                changes.append(change)

        return changes

    def _detect_changes(self, espn_id):
        """Check if an event's score/status changed vs last known state."""
        event = self.games.get(espn_id)
        if not event:
            return None

        team_info = self.event_teams.get(espn_id)
        if not team_info:
            self._index_event_teams(espn_id, event)
            team_info = self.event_teams.get(espn_id)
            if not team_info:
                return None

        away_db, home_db, date_str = team_info
        game_id = f'{date_str}_{away_db}_{home_db}'

        # Extract current state from event
        comp = event.get('competitions', [{}])[0]
        competitors = comp.get('competitors', [])
        away = next((c for c in competitors if c.get('homeAway') == 'away'), None)
        home = next((c for c in competitors if c.get('homeAway') == 'home'), None)
        if not away or not home:
            return None

        try:
            away_score = int(away.get('score', 0)) if away.get('score') is not None else None
            home_score = int(home.get('score', 0)) if home.get('score') is not None else None
        except (ValueError, TypeError):
            away_score = home_score = None

        status_obj = comp.get('status', {})
        status_type = status_obj.get('type', {})
        status_name = status_type.get('name', '')
        inning_text = status_type.get('shortDetail', '') or status_type.get('detail', '')
        innings = status_obj.get('period')

        if status_name == 'STATUS_FINAL':
            db_status = 'final'
        elif status_name == 'STATUS_IN_PROGRESS':
            # Guard: don't mark future games as in-progress
            from datetime import datetime
            game_date = game.get('date', '') if game else ''
            today = datetime.now().strftime('%Y-%m-%d')
            if game_date and str(game_date) > today:
                db_status = 'scheduled'
            else:
                db_status = 'in-progress'
        elif status_name in ('STATUS_POSTPONED', 'STATUS_CANCELED', 'STATUS_DELAYED'):
            db_status = 'postponed'
        else:
            db_status = 'scheduled'

        # Extended fields
        away_hits = _safe_int(away.get('hits'))
        home_hits = _safe_int(home.get('hits'))
        away_errors = _safe_int(away.get('errors'))
        home_errors = _safe_int(home.get('errors'))

        linescore = None
        home_ls = home.get('linescores', [])
        away_ls = away.get('linescores', [])
        if home_ls or away_ls:
            linescore = json.dumps({
                'home': [int(l.get('value', 0)) for l in home_ls if isinstance(l, dict)],
                'away': [int(l.get('value', 0)) for l in away_ls if isinstance(l, dict)],
            })

        situation = None
        sit = comp.get('situation')
        if sit and db_status == 'in-progress':
            situation = {
                'outs': sit.get('outs', 0),
                'balls': sit.get('balls', 0),
                'strikes': sit.get('strikes', 0),
                'onFirst': sit.get('onFirst', False),
                'onSecond': sit.get('onSecond', False),
                'onThird': sit.get('onThird', False),
                'batter': _extract_athlete(sit.get('batter')),
                'pitcher': _extract_athlete(sit.get('pitcher')),
            }

        # Determine what changed
        prev_score = self.prev_scores.get(game_id)
        prev_st = self.prev_status.get(game_id)

        score_changed = prev_score != (home_score, away_score)
        status_changed = prev_st != db_status
        is_live = db_status == 'in-progress'

        if not score_changed and not status_changed and not is_live:
            return None

        # Update tracking
        self.prev_scores[game_id] = (home_score, away_score)
        self.prev_status[game_id] = db_status

        # Determine event type
        if status_changed and db_status == 'final':
            event_type = 'game_final'
        elif score_changed:
            event_type = 'score_update'
        elif status_changed:
            event_type = 'status_change'
        else:
            event_type = 'situation_update'

        return {
            'game_id': game_id,
            'event_type': event_type,
            'away_db_id': away_db,
            'home_db_id': home_db,
            'away_score': away_score,
            'home_score': home_score,
            'away_hits': away_hits,
            'home_hits': home_hits,
            'away_errors': away_errors,
            'home_errors': home_errors,
            'inning_text': inning_text,
            'innings': innings,
            'status': db_status,
            'linescore': linescore,
            'situation': situation,
        }


def _safe_int(val):
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _extract_athlete(data):
    if not data:
        return None
    if isinstance(data, dict):
        athlete = data.get('athlete', {})
        if isinstance(athlete, dict):
            return athlete.get('shortName') or athlete.get('displayName')
    return None


def _nested_set(obj, path, value):
    """Set a value in a nested dict/list structure by path string."""
    if not path:
        return
    keys = path.split('/')
    cur = obj
    for k in keys[:-1]:
        if isinstance(cur, dict):
            cur = cur.setdefault(k, {})
        elif isinstance(cur, list):
            try:
                cur = cur[int(k)]
            except (IndexError, ValueError):
                return
        else:
            return
    last = keys[-1]
    if isinstance(cur, dict):
        cur[last] = value
    elif isinstance(cur, list):
        try:
            cur[int(last)] = value
        except (IndexError, ValueError):
            pass


# ── DB Writer ────────────────────────────────────────────────────

def write_change_to_db(change):
    """Write a detected change to the database."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row

    try:
        from schedule_gateway import ScheduleGateway
        gw = ScheduleGateway(conn)

        game_id = change['game_id']
        db_status = change['status']
        home_score = change['home_score']
        away_score = change['away_score']

        # Check game exists
        row = conn.execute('SELECT id, status FROM games WHERE id = ?', (game_id,)).fetchone()
        if not row:
            # Try ScheduleGateway dedup
            row = gw.find_existing_game(
                change['game_id'].split('_')[0],  # date
                change['away_db_id'],
                change['home_db_id'],
            )
            if row:
                game_id = row['id']
            else:
                log.debug('Game %s not found in DB, skipping', game_id)
                return

        if db_status == 'final' and home_score is not None and away_score is not None:
            gw.finalize_game(game_id, home_score, away_score)
        elif db_status == 'in-progress' and home_score is not None and away_score is not None:
            gw.update_live_score(game_id, home_score, away_score,
                                 change['inning_text'], innings=change.get('innings'))

        # Extended fields — merge situation_json to preserve sb_* fields
        new_situation = json.dumps(change['situation']) if change['situation'] else None
        if new_situation:
            existing_row = conn.execute(
                'SELECT situation_json FROM games WHERE id = ?', (game_id,)
            ).fetchone()
            if existing_row and existing_row[0]:
                try:
                    existing = json.loads(existing_row[0])
                    new_sit = json.loads(new_situation)
                    for k, v in existing.items():
                        if k.startswith('sb_') and k not in new_sit:
                            new_sit[k] = v
                    new_situation = json.dumps(new_sit)
                except (json.JSONDecodeError, TypeError):
                    pass

        conn.execute('''
            UPDATE games
               SET home_hits = ?, away_hits = ?,
                   home_errors = ?, away_errors = ?,
                   linescore_json = COALESCE(?, linescore_json),
                   situation_json = ?
             WHERE id = ?
        ''', (change['home_hits'], change['away_hits'],
              change['home_errors'], change['away_errors'],
              change['linescore'], new_situation,
              game_id))
        conn.commit()

        # Insert live event for SSE
        ensure_live_events_table(conn)
        event_data = {
            'game_id': game_id,
            'home_score': home_score,
            'away_score': away_score,
            'home_hits': change['home_hits'],
            'away_hits': change['away_hits'],
            'home_errors': change['home_errors'],
            'away_errors': change['away_errors'],
            'inning_text': change['inning_text'],
            'status': db_status,
            'situation': change['situation'],
        }
        insert_live_event(conn, game_id, change['event_type'], event_data)

        log.info('%s %s | %s %d - %d %s | %s',
                 change['event_type'].upper(), game_id,
                 change['away_db_id'], away_score or 0,
                 home_score or 0, change['home_db_id'],
                 change['inning_text'])

    except Exception as e:
        log.error('Error writing change to DB: %s', e)
    finally:
        conn.close()


# ── WebSocket Listener ───────────────────────────────────────────

async def discover_ws():
    """Discover the FastCast WebSocket endpoint."""
    loop = asyncio.get_event_loop()
    def _fetch():
        with urllib.request.urlopen(DISCOVERY_URL, timeout=10) as resp:
            return json.load(resp)
    data = await loop.run_in_executor(None, _fetch)
    ip = data['ip']
    port = data['securePort']
    token = data['token']
    url = f'wss://{ip}:{port}/FastcastService/pubsub/profiles/12000?TrafficManager-Token={token}'
    return url


async def fetch_checkpoint(topic, mid):
    """Fetch checkpoint snapshot from CDN."""
    url = f'{CHECKPOINT_BASE}/topic/{topic}/message/{mid}/checkpoint'
    loop = asyncio.get_event_loop()
    def _fetch():
        with urllib.request.urlopen(url, timeout=15) as resp:
            return json.load(resp)
    data = await loop.run_in_executor(None, _fetch)
    # Decode the payload — may be compressed or raw JSON
    pl = data.get('pl')
    if pl:
        try:
            return decode_payload(pl)
        except Exception:
            # Payload might be uncompressed JSON already inside the response
            return data
    return data


async def listen_forever():
    """Main listener loop with reconnect logic."""
    import websockets

    tracker = GameStateTracker()
    backoff = 1

    # Ensure live_events table exists
    conn = sqlite3.connect(DB_PATH, timeout=30)
    ensure_live_events_table(conn)
    cleanup_old_events(conn)
    conn.close()

    while True:
        try:
            ws_url = await discover_ws()
            log.info('Connecting to FastCast: %s', ws_url[:80] + '...')

            async with websockets.connect(ws_url, ping_interval=None,
                                          close_timeout=10) as ws:
                # Handshake
                await ws.send(json.dumps({'op': 'C'}))
                resp = json.loads(await ws.recv())
                if resp.get('rc') != 200:
                    log.error('Handshake failed: %s', resp)
                    continue
                sid = resp['sid']
                log.info('Connected, sid=%s', sid)
                backoff = 1  # Reset on success

                # Subscribe to topics
                for topic in TOPICS:
                    await ws.send(json.dumps({'op': 'S', 'sid': sid, 'tc': topic}))
                    log.info('Subscribed to %s', topic)

                # Message loop
                async for raw in ws:
                    msg = json.loads(raw)
                    op = msg.get('op')
                    log.debug('MSG op=%s tc=%s', op, msg.get('tc', '-'))

                    if op == 'B':
                        # Heartbeat — respond
                        await ws.send(json.dumps({'op': 'B', 'sid': sid}))

                    elif op == 'H':
                        # Checkpoint — fetch full state
                        topic = msg.get('tc', '')
                        mid = msg.get('mid', 0)
                        try:
                            cp_data = await fetch_checkpoint(topic, mid)
                            tracker.ingest_checkpoint(topic, cp_data)
                        except Exception as e:
                            log.warning('Checkpoint fetch error for %s: %s', topic, e)

                    elif op in ('P', 'R'):
                        # Publish or Replay — decode patches
                        pl = msg.get('pl')
                        if not pl:
                            continue
                        try:
                            patches = decode_payload(pl)
                        except Exception as e:
                            log.warning('Payload decode error: %s', e)
                            continue

                        if not isinstance(patches, list):
                            continue

                        changes = tracker.apply_patches(patches)
                        for change in changes:
                            await asyncio.get_event_loop().run_in_executor(
                                None, write_change_to_db, change)

                    elif op == 'C':
                        pass  # Already handled above

                    else:
                        log.debug('Unknown op: %s', op)

        except Exception as e:
            # Log close details if available
            close_code = getattr(e, 'code', None) or getattr(e, 'rcvd', None)
            log.warning('Connection lost: %s (type=%s, close=%s) — reconnecting in %ds',
                        e, type(e).__name__, close_code, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)

            # Refresh mapping on reconnect
            tracker.refresh_mapping()


def main():
    log.info('ESPN FastCast listener starting')

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Graceful shutdown
    def _shutdown(sig, frame):
        log.info('Shutting down (signal %s)', sig)
        loop.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        loop.run_until_complete(listen_forever())
    except KeyboardInterrupt:
        log.info('Interrupted')
    finally:
        loop.close()


if __name__ == '__main__':
    main()
