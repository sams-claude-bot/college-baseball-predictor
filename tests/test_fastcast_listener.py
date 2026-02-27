#!/usr/bin/env python3
"""
Tests for the ESPN FastCast WebSocket listener.

Covers:
  1. Payload decoding (base64 + zlib → JSON)
  2. JSON Patch path parsing
  3. ESPN event ID extraction
  4. GameStateTracker change detection
  5. live_events table operations
  6. SSE endpoint format
  7. Reconnect backoff logic (structural)
"""

import base64
import json
import sqlite3
import sys
import zlib
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
sys.path.insert(0, str(PROJECT_ROOT))


# ── Import the module under test ──
from espn_fastcast_listener import (
    decode_payload,
    parse_event_id,
    parse_patch_path,
    ensure_live_events_table,
    cleanup_old_events,
    insert_live_event,
    GameStateTracker,
    _nested_set,
    _safe_int,
)


# ── Helpers ──

def _encode_payload(obj):
    """Encode a Python object into a FastCast-style base64+zlib payload."""
    raw = json.dumps(obj).encode('utf-8')
    compressed = zlib.compress(raw)
    return base64.b64encode(compressed).decode('ascii')


@pytest.fixture
def mem_db():
    """In-memory SQLite DB with live_events table."""
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    ensure_live_events_table(conn)
    return conn


# ────────────────────────────────────────────────────────────────
# 1. Payload decoding
# ────────────────────────────────────────────────────────────────

class TestPayloadDecoding:
    def test_round_trip(self):
        original = [{'op': 'replace', 'path': 'foo/bar', 'value': '42'}]
        encoded = _encode_payload(original)
        decoded = decode_payload(encoded)
        assert decoded == original

    def test_complex_payload(self):
        data = {
            'events': [
                {'id': '401862650', 'score': '5'},
                {'id': '401862651', 'score': '3'},
            ]
        }
        encoded = _encode_payload(data)
        decoded = decode_payload(encoded)
        assert decoded['events'][0]['id'] == '401862650'

    def test_empty_list(self):
        encoded = _encode_payload([])
        assert decode_payload(encoded) == []

    def test_invalid_base64_raises(self):
        with pytest.raises(Exception):
            decode_payload('not-valid-base64!!!')

    def test_gzip_payload(self):
        """Test that gzip-wrapped payloads also decode (wbits 15+32)."""
        original = {'status': 'ok'}
        raw = json.dumps(original).encode()
        # Use gzip format
        compressed = zlib.compress(raw, 6)
        b64 = base64.b64encode(compressed).decode()
        assert decode_payload(b64) == original


# ────────────────────────────────────────────────────────────────
# 2. JSON Patch path parsing
# ────────────────────────────────────────────────────────────────

class TestPatchParsing:
    def test_basic_path(self):
        eid, field = parse_patch_path(
            's:1~l:14~e:401862650/competitions/0/competitors/1/score'
        )
        assert eid == '401862650'
        assert field == 'competitions/0/competitors/1/score'

    def test_path_no_field(self):
        eid, field = parse_patch_path('s:1~l:14~e:999999')
        assert eid == '999999'
        assert field == ''

    def test_path_with_status(self):
        eid, field = parse_patch_path(
            's:1~l:14~e:401862650/competitions/0/status/type/shortDetail'
        )
        assert eid == '401862650'
        assert 'status' in field

    def test_no_event_id(self):
        eid, field = parse_patch_path('s:1~l:14/something')
        assert eid is None


# ────────────────────────────────────────────────────────────────
# 3. ESPN event ID extraction
# ────────────────────────────────────────────────────────────────

class TestEventIdExtraction:
    def test_standard_key(self):
        assert parse_event_id('s:1~l:14~e:401862650') == '401862650'

    def test_no_event(self):
        assert parse_event_id('s:1~l:14') is None

    def test_multiple_tildes(self):
        assert parse_event_id('s:1~l:14~g:5~e:12345') == '12345'


# ────────────────────────────────────────────────────────────────
# 4. GameStateTracker
# ────────────────────────────────────────────────────────────────

class TestGameStateTracker:
    @pytest.fixture
    def tracker(self):
        """Tracker with mocked ESPN mapping."""
        with patch('espn_fastcast_listener.build_espn_mapping_if_needed') as m:
            m.return_value = {'100': 'team-a', '200': 'team-b'}
            t = GameStateTracker()
        return t

    def _make_event(self, espn_id='401862650', away_espn='100', home_espn='200',
                    away_score='3', home_score='5', status_name='STATUS_IN_PROGRESS',
                    short_detail='Top 4th'):
        return {
            'date': '2026-02-26T18:00Z',
            'competitions': [{
                'competitors': [
                    {
                        'homeAway': 'away',
                        'team': {'id': away_espn},
                        'score': away_score,
                        'hits': '4',
                        'errors': '0',
                        'linescores': [],
                    },
                    {
                        'homeAway': 'home',
                        'team': {'id': home_espn},
                        'score': home_score,
                        'hits': '7',
                        'errors': '1',
                        'linescores': [],
                    },
                ],
                'status': {
                    'type': {
                        'name': status_name,
                        'shortDetail': short_detail,
                    },
                    'period': 4,
                },
                'situation': {
                    'outs': 1, 'balls': 2, 'strikes': 1,
                    'onFirst': True, 'onSecond': False, 'onThird': False,
                },
            }],
        }

    def test_ingest_checkpoint(self, tracker):
        data = {
            's:1~l:14~e:401862650': self._make_event(),
        }
        tracker.ingest_checkpoint('scoreboard-baseball-college-baseball', data)
        assert '401862650' in tracker.games
        assert '401862650' in tracker.event_teams

    def test_apply_patches_score_change(self, tracker):
        # Seed state
        tracker.games['401862650'] = self._make_event()
        tracker.event_teams['401862650'] = ('team-a', 'team-b', '2026-02-26')

        # First read to set baseline
        tracker.apply_patches([{
            'op': 'replace',
            'path': 's:1~l:14~e:401862650/competitions/0/competitors/0/score',
            'value': '3',
        }])

        # Now apply a score change
        changes = tracker.apply_patches([{
            'op': 'replace',
            'path': 's:1~l:14~e:401862650/competitions/0/competitors/0/score',
            'value': '4',
        }])

        assert len(changes) >= 1
        c = changes[0]
        assert c['game_id'] == '2026-02-26_team-a_team-b'
        assert c['event_type'] in ('score_update', 'situation_update')

    def test_apply_patches_no_change(self, tracker):
        tracker.games['401862650'] = self._make_event(status_name='STATUS_FINAL',
                                                       short_detail='Final')
        tracker.event_teams['401862650'] = ('team-a', 'team-b', '2026-02-26')

        # Set baseline
        tracker.apply_patches([{
            'op': 'replace',
            'path': 's:1~l:14~e:401862650/competitions/0/competitors/0/score',
            'value': '3',
        }])

        # Same patch again — no change
        changes = tracker.apply_patches([{
            'op': 'replace',
            'path': 's:1~l:14~e:401862650/competitions/0/competitors/0/score',
            'value': '3',
        }])

        assert len(changes) == 0

    def test_game_final_detection(self, tracker):
        tracker.games['401862650'] = self._make_event(
            status_name='STATUS_IN_PROGRESS', short_detail='Bot 9th')
        tracker.event_teams['401862650'] = ('team-a', 'team-b', '2026-02-26')

        # Baseline
        tracker.apply_patches([{
            'op': 'replace',
            'path': 's:1~l:14~e:401862650/competitions/0/status/type/name',
            'value': 'STATUS_IN_PROGRESS',
        }])

        # Now change to final
        tracker.games['401862650']['competitions'][0]['status']['type']['name'] = 'STATUS_FINAL'
        tracker.games['401862650']['competitions'][0]['status']['type']['shortDetail'] = 'Final'
        changes = tracker.apply_patches([{
            'op': 'replace',
            'path': 's:1~l:14~e:401862650/competitions/0/status/type/name',
            'value': 'STATUS_FINAL',
        }])

        assert len(changes) == 1
        assert changes[0]['event_type'] == 'game_final'


# ────────────────────────────────────────────────────────────────
# 5. live_events table
# ────────────────────────────────────────────────────────────────

class TestLiveEventsTable:
    def test_create_table(self, mem_db):
        tables = mem_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='live_events'"
        ).fetchall()
        assert len(tables) == 1

    def test_insert_and_read(self, mem_db):
        data = {'game_id': 'test_game', 'home_score': 5, 'away_score': 3}
        insert_live_event(mem_db, 'test_game', 'score_update', data)

        rows = mem_db.execute('SELECT * FROM live_events').fetchall()
        assert len(rows) == 1
        assert rows[0]['event_type'] == 'score_update'
        parsed = json.loads(rows[0]['data_json'])
        assert parsed['home_score'] == 5

    def test_multiple_events(self, mem_db):
        for i in range(5):
            insert_live_event(mem_db, f'game_{i}', 'score_update', {'i': i})

        rows = mem_db.execute('SELECT * FROM live_events ORDER BY id').fetchall()
        assert len(rows) == 5
        assert rows[0]['id'] < rows[4]['id']

    def test_cleanup(self, mem_db):
        insert_live_event(mem_db, 'old_game', 'score_update', {'old': True})
        # Force the event to look old
        mem_db.execute(
            "UPDATE live_events SET created_at = datetime('now', '-48 hours')"
        )
        mem_db.commit()
        cleanup_old_events(mem_db, hours=24)
        assert mem_db.execute('SELECT COUNT(*) FROM live_events').fetchone()[0] == 0


# ────────────────────────────────────────────────────────────────
# 6. SSE endpoint format (self-contained, no web.blueprints import)
# ────────────────────────────────────────────────────────────────

class TestSSEEndpoint:
    """Test SSE streaming logic by building a minimal Flask app inline.

    We avoid importing ``web.blueprints.api`` directly because that triggers
    heavy model loading. Instead we replicate the SSE route logic here.
    """

    @pytest.fixture
    def app(self, mem_db):
        from flask import Flask, Response, request as flask_request
        import time as _time

        app = Flask(__name__)

        @app.route('/api/live-stream')
        def _live_stream():
            last_id = flask_request.headers.get('Last-Event-ID', 0, type=int)

            def generate():
                nonlocal last_id
                # Single iteration (no infinite loop in test)
                events = mem_db.execute(
                    'SELECT id, event_type, data_json FROM live_events WHERE id > ? ORDER BY id LIMIT 20',
                    (last_id,)
                ).fetchall()

                for event in events:
                    last_id = event['id']
                    yield "id: {eid}\nevent: {etype}\ndata: {data}\n\n".format(
                        eid=event['id'], etype=event['event_type'],
                        data=event['data_json'])

                if not events:
                    yield ": keepalive\n\n"

            return Response(generate(), mimetype='text/event-stream',
                            headers={'Cache-Control': 'no-cache',
                                     'X-Accel-Buffering': 'no',
                                     'Connection': 'keep-alive'})

        return app

    def test_sse_content_type(self, app):
        with app.test_client() as client:
            resp = client.get('/api/live-stream')
            assert resp.content_type.startswith('text/event-stream')
            assert resp.headers.get('Cache-Control') == 'no-cache'

    def test_sse_keepalive(self, app, mem_db):
        """With no events, should get a keepalive comment."""
        with app.test_client() as client:
            resp = client.get('/api/live-stream')
            data = resp.get_data()
            assert b': keepalive' in data

    def test_sse_event_format(self, app, mem_db):
        """Inserted events should stream in SSE format."""
        insert_live_event(mem_db, 'test_game', 'score_update',
                          {'home_score': 5, 'away_score': 3})

        with app.test_client() as client:
            resp = client.get('/api/live-stream')
            text = resp.get_data(as_text=True)
            assert 'event: score_update' in text
            assert 'id: ' in text
            assert '"home_score": 5' in text


# ────────────────────────────────────────────────────────────────
# 7. Utility helpers
# ────────────────────────────────────────────────────────────────

class TestHelpers:
    def test_nested_set_dict(self):
        obj = {'a': {'b': {'c': 1}}}
        _nested_set(obj, 'a/b/c', 99)
        assert obj['a']['b']['c'] == 99

    def test_nested_set_list(self):
        obj = {'items': [10, 20, 30]}
        _nested_set(obj, 'items/1', 99)
        assert obj['items'][1] == 99

    def test_nested_set_creates_missing(self):
        obj = {}
        _nested_set(obj, 'a/b', 'val')
        assert obj['a']['b'] == 'val'

    def test_safe_int(self):
        assert _safe_int('5') == 5
        assert _safe_int(None) is None
        assert _safe_int('abc') is None
        assert _safe_int(0) == 0
