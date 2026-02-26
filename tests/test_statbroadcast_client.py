#!/usr/bin/env python3
"""
Tests for StatBroadcast client — decode, parse, and API integration.
Uses saved fixtures for offline testing; live tests marked with skip.
"""

import os
import re
import sys
import urllib.error
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from statbroadcast_client import (
    sb_decode, sb_encode_params, parse_situation,
    StatBroadcastClient, _first_match,
)

FIXTURES = PROJECT_ROOT / 'tests' / 'fixtures'


# ---------------------------------------------------------------------------
# Codec tests
# ---------------------------------------------------------------------------

class TestCodec:
    def test_sb_decode_roundtrip(self):
        """ROT13(base64(text)) -> text"""
        import codecs, base64
        original = "<div>Hello World! Score: 5-3</div>"
        encoded = codecs.encode(
            base64.b64encode(original.encode()).decode(),
            'rot_13'
        )
        assert sb_decode(encoded) == original

    def test_sb_decode_unicode(self):
        """Handles UTF-8 characters."""
        import codecs, base64
        original = "José García — 2-for-4, 1 HR"
        encoded = codecs.encode(
            base64.b64encode(original.encode('utf-8')).decode(),
            'rot_13'
        )
        assert sb_decode(encoded) == original

    def test_sb_decode_empty(self):
        """Empty string base64 is valid."""
        import codecs, base64
        encoded = codecs.encode(
            base64.b64encode(b'').decode(),
            'rot_13'
        )
        assert sb_decode(encoded) == ''

    def test_sb_encode_params(self):
        """Params are base64-encoded."""
        import base64
        params = "event=123&type=statbroadcast"
        result = sb_encode_params(params)
        assert base64.b64decode(result).decode() == params

    def test_sb_encode_params_special_chars(self):
        """Handles special characters in params."""
        params = "xml=lbst/652739.xml&xsl=baseball/sb.bsgame.views.broadcast.xsl"
        result = sb_encode_params(params)
        import base64
        assert base64.b64decode(result).decode() == params


# ---------------------------------------------------------------------------
# Fixture-based parsing tests
# ---------------------------------------------------------------------------

class TestParseFixtures:
    @pytest.fixture
    def stats_html(self):
        """Load saved decoded stats HTML fixture."""
        path = FIXTURES / 'sb_stats_652739_decoded.html'
        if not path.exists():
            pytest.skip("Stats fixture not available")
        return path.read_text()

    @pytest.fixture
    def event_xml(self):
        """Load saved decoded event XML fixture."""
        path = FIXTURES / 'sb_event_652739_decoded.xml'
        if not path.exists():
            pytest.skip("Event fixture not available")
        return path.read_text()

    @pytest.fixture
    def raw_stats(self):
        """Load saved raw (encoded) stats response."""
        path = FIXTURES / 'sb_stats_652739_raw.txt'
        if not path.exists():
            pytest.skip("Raw stats fixture not available")
        return path.read_text()

    @pytest.fixture
    def raw_event(self):
        """Load saved raw (encoded) event response."""
        path = FIXTURES / 'sb_event_652739_raw.txt'
        if not path.exists():
            pytest.skip("Raw event fixture not available")
        return path.read_text()

    def test_decode_raw_stats(self, raw_stats, stats_html):
        """Raw fixture decodes to expected HTML."""
        decoded = sb_decode(raw_stats)
        # Should contain HTML with game data
        assert '<div' in decoded
        assert 'sb-teamscore' in decoded

    def test_decode_raw_event(self, raw_event, event_xml):
        """Raw fixture decodes to expected XML."""
        decoded = sb_decode(raw_event)
        assert 'BCSResponse' in decoded or 'event' in decoded.lower()

    def test_parse_teams(self, stats_html):
        """Extracts team names."""
        sit = parse_situation(stats_html)
        assert sit.get('visitor') is not None
        assert sit.get('home') is not None
        assert 'BYU' in (sit.get('visitor') or '')
        assert 'Washington State' in (sit.get('home') or '')

    def test_parse_scores(self, stats_html):
        """Extracts numeric scores."""
        sit = parse_situation(stats_html)
        assert isinstance(sit.get('visitor_score'), int)
        assert isinstance(sit.get('home_score'), int)
        assert sit['visitor_score'] >= 0
        assert sit['home_score'] >= 0

    def test_parse_inning(self, stats_html):
        """Extracts inning information."""
        sit = parse_situation(stats_html)
        assert sit.get('inning') is not None
        assert isinstance(sit['inning'], int)
        assert 1 <= sit['inning'] <= 20
        assert sit.get('inning_half') in ('top', 'bottom', 'middle', None)

    def test_parse_outs(self, stats_html):
        """Extracts outs count."""
        sit = parse_situation(stats_html)
        # Outs should be present for in-progress games
        if sit.get('outs') is not None:
            assert sit['outs'] in (0, 1, 2)

    def test_parse_count(self, stats_html):
        """Extracts ball-strike count."""
        sit = parse_situation(stats_html)
        if sit.get('count') is not None:
            assert re.match(r'\d-\d', sit['count'])
            assert 0 <= sit.get('balls', 0) <= 3
            assert 0 <= sit.get('strikes', 0) <= 2

    def test_parse_batter(self, stats_html):
        """Extracts current batter info."""
        sit = parse_situation(stats_html)
        # Should have at least a name for in-progress games
        assert sit.get('batter_name') is not None
        assert len(sit['batter_name']) > 1

    def test_parse_pitcher(self, stats_html):
        """Extracts current pitcher info."""
        sit = parse_situation(stats_html)
        assert sit.get('pitcher_name') is not None
        assert len(sit['pitcher_name']) > 1

    def test_parse_title(self, stats_html):
        """Extracts game title (e.g., 'BYU 1, WSU 4 - T8th')."""
        sit = parse_situation(stats_html)
        assert sit.get('title') is not None
        assert 'BYU' in sit['title'] or 'WSU' in sit['title']


# ---------------------------------------------------------------------------
# Synthetic HTML parsing tests
# ---------------------------------------------------------------------------

class TestParseSynthetic:
    """Test parser with crafted HTML snippets."""

    def test_parse_scores_basic(self):
        html = '''
        <span class="sb-teamscore">5</span>
        <span class="sb-teamscore">3</span>
        '''
        sit = parse_situation(html)
        assert sit.get('visitor_score') == 5
        assert sit.get('home_score') == 3

    def test_parse_outs_zero(self):
        html = '<div>OUTS</div><i class="sbicon">0</i>'
        sit = parse_situation(html)
        assert sit.get('outs') == 0

    def test_parse_outs_two(self):
        html = '<div>OUTS</div><i class="sbicon font-size-300">2</i>'
        sit = parse_situation(html)
        assert sit.get('outs') == 2

    def test_parse_count_full(self):
        html = '<div class="font-size-125">3-2</div>'
        sit = parse_situation(html)
        assert sit['count'] == '3-2'
        assert sit['balls'] == 3
        assert sit['strikes'] == 2

    def test_parse_batter_with_position(self):
        html = 'At Bat for BYU: #29 Erickson,Ridge [C]</div>'
        sit = parse_situation(html)
        assert sit['batter_name'] == 'Erickson,Ridge'
        assert sit['batter_number'] == '29'
        assert sit['batter_position'] == 'C'

    def test_parse_batter_no_position(self):
        html = 'At Bat for WSU: #07 Smith,John</div>'
        sit = parse_situation(html)
        assert sit['batter_name'] == 'Smith,John'
        assert sit['batter_number'] == '07'

    def test_parse_pitcher_standard(self):
        html = '''<div class="card-header card-title">
        <div class="sb-team-logo HLogo"></div>Pitching For WSU: #54 Haider, Rylan</div>'''
        sit = parse_situation(html)
        assert sit['pitcher_name'] == 'Haider, Rylan'
        assert sit['pitcher_number'] == '54'

    def test_parse_on_mound_fallback(self):
        html = '''<span class="text-muted mr-1">On Mound:</span>
        <span class="larger font-weight-bold">Johnson,Ashton</span>'''
        sit = parse_situation(html)
        assert sit['pitcher_name'] == 'Johnson,Ashton'

    def test_parse_at_bat_fallback(self):
        html = '''<span class="text-muted mr-1">At Bat:</span>
        <span class="larger font-weight-bold">Erickson,Ridge</span>'''
        sit = parse_situation(html)
        assert sit['batter_name'] == 'Erickson,Ridge'

    def test_parse_inning_top(self):
        html = '<div class="font-size-125 mb-1">Top 8th</div>'
        sit = parse_situation(html)
        assert sit['inning'] == 8
        assert sit['inning_half'] == 'top'

    def test_parse_inning_bottom(self):
        html = '<div class="sb-bsgame-thisinning">Bot 3rd</div>'
        sit = parse_situation(html)
        assert sit['inning'] == 3
        assert sit['inning_half'] == 'bottom'

    def test_parse_empty_html(self):
        """Empty HTML returns empty dict, no crash."""
        sit = parse_situation('')
        assert isinstance(sit, dict)

    def test_parse_no_game_data(self):
        """Random HTML doesn't crash."""
        sit = parse_situation('<html><body>Hello world</body></html>')
        assert isinstance(sit, dict)


# ---------------------------------------------------------------------------
# Client tests (mocked HTTP)
# ---------------------------------------------------------------------------

class TestClient:
    def test_get_event_info_mocked(self):
        """Mocked event info request."""
        import codecs, base64

        fake_xml = '''<BCSResponse><event id="123" groupid="test" 
            homename="Team A" visitorname="Team B" 
            completed="0" archived="0">
            <title><![CDATA[Team B vs. Team A]]></title>
            <dbdate>2026-02-26</dbdate>
            <sport>bsgame</sport>
            <xmlfile><![CDATA[test/123.xml]]></xmlfile>
            <venue><![CDATA[Test Field]]></venue>
            <location><![CDATA[Test City, ST]]></location>
        </event></BCSResponse>'''

        encoded = codecs.encode(
            base64.b64encode(fake_xml.encode()).decode(), 'rot_13'
        )

        mock_resp = MagicMock()
        mock_resp.read.return_value = encoded.encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        client = StatBroadcastClient()
        with patch('urllib.request.urlopen', return_value=mock_resp):
            info = client.get_event_info(123)

        assert info is not None
        assert info['home'] == 'Team A'
        assert info['visitor'] == 'Team B'
        assert info['date'] == '2026-02-26'
        assert info['sport'] == 'bsgame'
        assert info['completed'] is False
        assert info['group_id'] == 'test'
        assert info['xml_file'] == 'test/123.xml'

    def test_get_event_info_404(self):
        """Returns None for 404."""
        client = StatBroadcastClient()
        with patch('urllib.request.urlopen',
                   side_effect=urllib.error.HTTPError(
                       'url', 404, 'Not Found', {}, None)):
            info = client.get_event_info(999999)
        assert info is None

    def test_get_live_stats_304(self):
        """Returns None HTML on 304 (no changes)."""
        client = StatBroadcastClient()
        headers = MagicMock()
        headers.get.return_value = '12345'
        err = urllib.error.HTTPError('url', 304, 'Not Modified', headers, None)
        with patch('urllib.request.urlopen', side_effect=err):
            html, ft = client.get_live_stats(123, 'test/123.xml', filetime=100)
        assert html is None
        assert ft == 12345


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_first_match_found(self):
        assert _first_match(r'hello (\w+)', 'hello world') == 'world'

    def test_first_match_not_found(self):
        assert _first_match(r'xyz (\w+)', 'hello world') is None

    def test_first_match_strips(self):
        assert _first_match(r'>(\s*test\s*)<', '> test <') == 'test'


# ---------------------------------------------------------------------------
# Live integration tests (skip in CI)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    os.environ.get('CI') == 'true' or os.environ.get('SKIP_LIVE') == '1',
    reason="Live network test"
)
class TestLiveIntegration:
    """These tests hit the real StatBroadcast API. Run manually."""

    def test_event_info_real(self):
        client = StatBroadcastClient()
        info = client.get_event_info(652739)
        assert info is not None
        assert info['sport'] == 'bsgame'
        assert 'BYU' in (info['visitor'] or '')

    def test_live_stats_real(self):
        client = StatBroadcastClient()
        info = client.get_event_info(652739)
        assert info is not None
        html, ft = client.get_live_stats(652739, info['xml_file'])
        assert html is not None
        assert '<div' in html
