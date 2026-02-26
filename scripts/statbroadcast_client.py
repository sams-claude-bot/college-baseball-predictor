#!/usr/bin/env python3
"""
StatBroadcast Client — decode and fetch live stat feeds.

StatBroadcast provides rich live game data for college baseball including
outs, count, batter/pitcher, lineup, and play-by-play. Their API responses
are encoded with ROT13 + base64.

Usage:
    # As a library
    from statbroadcast_client import StatBroadcastClient
    client = StatBroadcastClient()
    info = client.get_event_info(652739)
    stats, filetime = client.get_live_stats(652739, 'lbst/652739.xml')
    situation = client.parse_situation(stats)

    # CLI test
    python3 scripts/statbroadcast_client.py 652739
"""

import codecs
import base64
import json
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

PROJECT_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Codec helpers
# ---------------------------------------------------------------------------

def sb_decode(response_text: str) -> str:
    """Decode StatBroadcast response: ROT13 → base64 decode → UTF-8 text."""
    rot13 = codecs.decode(response_text, 'rot_13')
    return base64.b64decode(rot13).decode('utf-8')


def sb_encode_params(params: str) -> str:
    """Base64-encode request parameters for StatBroadcast API."""
    return base64.b64encode(params.encode()).decode()


# ---------------------------------------------------------------------------
# HTML parsing helpers
# ---------------------------------------------------------------------------

def _first_match(pattern: str, text: str, group: int = 1, flags: int = 0) -> Optional[str]:
    """Return first regex match group or None."""
    m = re.search(pattern, text, flags)
    return m.group(group).strip() if m else None


def parse_situation(html: str) -> Dict[str, Any]:
    """
    Parse live game situation from decoded StatBroadcast HTML.

    Returns dict with keys:
        visitor, home, visitor_score, home_score, inning, inning_half,
        outs, count, batter_name, batter_number, pitcher_name,
        pitcher_number, on_first, on_second, on_third
    """
    result = {}

    # Team names
    result['visitor'] = _first_match(
        r'sb-teamnameV[^>]*>(?:<[^>]*>)*\s*([A-Z][A-Za-z\s.&\'-]+)', html, flags=re.DOTALL
    )
    result['home'] = _first_match(
        r'sb-teamnameH[^>]*>(?:<[^>]*>)*\s*([A-Z][A-Za-z\s.&\'-]+)', html, flags=re.DOTALL
    )
    if result.get('visitor'):
        result['visitor'] = result['visitor'].strip()
    if result.get('home'):
        result['home'] = result['home'].strip()

    # Scores (first two sb-teamscore spans: visitor then home)
    scores = re.findall(r'sb-teamscore[^>]*>(\d+)', html)
    if len(scores) >= 2:
        result['visitor_score'] = int(scores[0])
        result['home_score'] = int(scores[1])

    # Inning from status bar (e.g., "Top 8th", "Bot 3rd", "Mid 5th")
    inning_raw = _first_match(
        r'font-size-125[^>]*>([^<]*(?:st|nd|rd|th))', html
    )
    if inning_raw:
        result['inning_display'] = inning_raw
        # Parse half and number
        low = inning_raw.lower()
        if 'top' in low or 't' == low[0]:
            result['inning_half'] = 'top'
        elif 'bot' in low or 'b' == low[0]:
            result['inning_half'] = 'bottom'
        elif 'mid' in low or 'm' == low[0]:
            result['inning_half'] = 'middle'
        num = re.search(r'(\d+)', inning_raw)
        if num:
            result['inning'] = int(num.group(1))

    # Also check the thisinning div
    if 'inning' not in result:
        thisinning = _first_match(r'sb-bsgame-thisinning[^>]*>([^<]+)', html)
        if thisinning:
            result['inning_display'] = thisinning
            low = thisinning.lower()
            if 'top' in low:
                result['inning_half'] = 'top'
            elif 'bot' in low:
                result['inning_half'] = 'bottom'
            num = re.search(r'(\d+)', thisinning)
            if num:
                result['inning'] = int(num.group(1))

    # Outs — StatBroadcast structure:
    #   OUTS</span>
    #   <span class="no-access"><i class="sbicon ...noaccess">ZZ</i></span>  (icon font, ignore)
    #   <span class="d-inline d-sm-none">2</span>  (actual number, mobile fallback)
    outs_match = re.search(
        r'OUTS</span>.*?<span[^>]*d-inline[^>]*>(\d)</span>',
        html, re.DOTALL
    )
    if not outs_match:
        # Fallback: any single digit in a span near OUTS text
        outs_match = re.search(
            r'OUTS.*?<span[^>]*>(\d)</span>',
            html, re.DOTALL
        )
    if outs_match:
        val = int(outs_match.group(1))
        if 0 <= val <= 2:
            result['outs'] = val

    # Count (ball-strike) — "0-1", "2-2", etc.
    count = _first_match(r'font-size-125[^>]*>(\d-\d)</div>', html)
    if count:
        result['count'] = count
        parts = count.split('-')
        if len(parts) == 2:
            result['balls'] = int(parts[0])
            result['strikes'] = int(parts[1])

    # Current batter — "At Bat for {team}: #{num} {Name} [{pos}]"
    batter_match = re.search(
        r'At Bat[^:]*:\s*#(\d+)\s*([^<\[]+?)(?:\s*\[([^\]]+)\])?\s*<',
        html
    )
    if batter_match:
        result['batter_number'] = batter_match.group(1)
        result['batter_name'] = batter_match.group(2).strip()
        if batter_match.group(3):
            result['batter_position'] = batter_match.group(3).strip()

    # Current pitcher — "Pitching For {team}: #{num} {Name}"
    pitcher_match = re.search(
        r'Pitching\s*(?:<[^>]*>)*\s*(?:For\s+\w+)?\s*:\s*#(\d+)\s*([^<]+)',
        html, re.IGNORECASE
    )
    if pitcher_match:
        result['pitcher_number'] = pitcher_match.group(1)
        result['pitcher_name'] = pitcher_match.group(2).strip().rstrip(',')

    # If pitcher not found in header, check "On Mound" status bar
    if 'pitcher_name' not in result:
        mound = _first_match(
            r'On Mound[^:]*:</span>\s*<span[^>]*>([^<]+)', html
        )
        if mound:
            result['pitcher_name'] = mound.strip()

    # If batter not found in header, check "At Bat" status bar
    if 'batter_name' not in result:
        atbat = _first_match(
            r'At Bat[^:]*:</span>\s*<span[^>]*>([^<]+)', html
        )
        if atbat:
            result['batter_name'] = atbat.strip()

    # Base runners from "Runners On Base" table
    # Format: <tr><td>1B</td><td>PlayerName</td>...</tr>
    runners_section = re.search(
        r'Runners On Base.*?</table>', html, re.DOTALL
    )
    if runners_section:
        runner_rows = re.findall(
            r'<tr[^>]*>\s*<td[^>]*>\s*(1B|2B|3B)\s*</td>\s*<td[^>]*>\s*([^<]+)',
            runners_section.group(0)
        )
        for base, name in runner_rows:
            name = name.strip()
            if base == '1B':
                result['on_first'] = True
                result['runner_first'] = name
            elif base == '2B':
                result['on_second'] = True
                result['runner_second'] = name
            elif base == '3B':
                result['on_third'] = True
                result['runner_third'] = name
    # Default to False if not found
    result.setdefault('on_first', False)
    result.setdefault('on_second', False)
    result.setdefault('on_third', False)

    # Line score — R/H/E per team
    # Extract from the line score table if present
    linescore_rows = re.findall(
        r'<tr[^>]*>.*?sb-teamname[VH][^>]*>.*?</tr>',
        html, re.DOTALL
    )
    if linescore_rows:
        for row in linescore_rows:
            tds = re.findall(r'<td[^>]*>([^<]*)</td>', row)
            if 'sb-teamnameV' in row and len(tds) >= 3:
                # Last few cells: R, H, E
                try:
                    result['visitor_hits'] = int(tds[-5]) if len(tds) >= 5 else None
                    result['visitor_errors'] = int(tds[-4]) if len(tds) >= 4 else None
                except (ValueError, IndexError):
                    pass
            elif 'sb-teamnameH' in row and len(tds) >= 3:
                try:
                    result['home_hits'] = int(tds[-5]) if len(tds) >= 5 else None
                    result['home_errors'] = int(tds[-4]) if len(tds) >= 4 else None
                except (ValueError, IndexError):
                    pass

    # Game status from title script
    title_match = re.search(r'cscore\s*=\s*"([^"]*)"', html)
    if title_match:
        result['title'] = title_match.group(1).strip()

    return result


# ---------------------------------------------------------------------------
# Client class
# ---------------------------------------------------------------------------

class StatBroadcastClient:
    """HTTP client for StatBroadcast API."""

    BASE_URL = "https://stats.statbroadcast.com/interface/webservice"
    REFERER = "https://stats.statbroadcast.com/broadcast/"
    USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
    TIMEOUT = 10

    def __init__(self):
        pass

    def _request(self, url: str, timeout: int = None) -> urllib.request.Request:
        """Build a request with standard headers."""
        return urllib.request.Request(url, headers={
            'User-Agent': self.USER_AGENT,
            'Referer': self.REFERER,
        })

    def get_event_info(self, event_id: int) -> Optional[Dict[str, Any]]:
        """
        Fetch event metadata (teams, date, sport, status).

        Returns dict with: event_id, home, visitor, date, sport, completed,
        archived, group_id, xml_file, title, venue, location
        """
        data = sb_encode_params("type=statbroadcast")
        url = f"{self.BASE_URL}/event/{event_id}?data={data}"
        req = self._request(url)

        try:
            with urllib.request.urlopen(req, timeout=self.TIMEOUT) as resp:
                decoded = sb_decode(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise
        except Exception:
            return None

        # Parse XML-ish response
        def _attr(name):
            m = re.search(rf'{name}="([^"]*)"', decoded)
            return m.group(1) if m else None

        def _tag(name):
            m = re.search(rf'<{name}>(.*?)</{name}>', decoded, re.DOTALL)
            if m:
                # Strip CDATA
                val = re.sub(r'<!\[CDATA\[(.*?)\]\]>', r'\1', m.group(1))
                return val.strip()
            return None

        return {
            'event_id': event_id,
            'home': _attr('homename'),
            'visitor': _attr('visitorname'),
            'date': _tag('dbdate'),
            'sport': _tag('sport'),
            'completed': _attr('completed') == '1',
            'archived': _attr('archived') == '1',
            'group_id': _attr('groupid'),
            'xml_file': _tag('xmlfile'),
            'title': _tag('title'),
            'venue': _tag('venue'),
            'location': _tag('location'),
            'home_image': _tag('homeimage'),
            'visitor_image': _tag('visitorimage'),
        }

    def get_live_stats(
        self,
        event_id: int,
        xml_file: str,
        filetime: int = 0,
        sport: str = 'bsgame',
        force_refresh: bool = False,
    ) -> Tuple[Optional[str], int]:
        """
        Fetch live stats HTML for a game.

        Args:
            event_id: StatBroadcast event ID
            xml_file: XML file path from event info (e.g., 'lbst/652739.xml')
            filetime: Last known filetime (0 or -1 for full refresh)
            sport: Sport code (bsgame for baseball)
            force_refresh: If True, set filetime=-1 and start=true

        Returns:
            (decoded_html, new_filetime) — html is None on 304 (no change) or error
        """
        params = (
            f"event={event_id}"
            f"&xml={xml_file}"
            f"&xsl=baseball/sb.{sport}.views.broadcast.xsl"
            f"&sport={sport}"
            f"&filetime={-1 if force_refresh else filetime}"
            f"&type=statbroadcast"
        )
        if filetime <= 0 or force_refresh:
            params += "&start=true"

        data = sb_encode_params(params)
        url = f"{self.BASE_URL}/stats?data={data}"
        req = self._request(url)

        try:
            with urllib.request.urlopen(req, timeout=self.TIMEOUT) as resp:
                new_filetime = int(resp.headers.get('Filetime', '0') or '0')

                if resp.status == 304 or resp.status == 204:
                    return None, new_filetime

                raw = resp.read().decode()
                if not raw.strip():
                    return None, new_filetime

                return sb_decode(raw), new_filetime

        except urllib.error.HTTPError as e:
            if e.code in (304, 204):
                new_ft = int(e.headers.get('Filetime', '0') or '0')
                return None, new_ft
            raise
        except Exception:
            return None, filetime

    def get_situation(self, event_id: int, xml_file: str,
                      filetime: int = 0) -> Tuple[Optional[Dict], int]:
        """
        Convenience: fetch stats and parse situation in one call.

        Returns (situation_dict, new_filetime).
        situation_dict is None if no changes (304) or error.
        """
        html, new_ft = self.get_live_stats(event_id, xml_file, filetime)
        if html is None:
            return None, new_ft
        return parse_situation(html), new_ft


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """CLI: test fetching a StatBroadcast event."""
    if len(sys.argv) < 2:
        print("Usage: python3 statbroadcast_client.py <event_id> [xml_file]")
        print("Example: python3 statbroadcast_client.py 652739")
        sys.exit(1)

    event_id = int(sys.argv[1])
    client = StatBroadcastClient()

    # Get event info first
    print(f"Fetching event {event_id}...")
    info = client.get_event_info(event_id)
    if not info:
        print(f"Event {event_id} not found")
        sys.exit(1)

    print(f"  {info['visitor']} @ {info['home']}")
    print(f"  Date: {info['date']}")
    print(f"  Sport: {info['sport']}")
    print(f"  Completed: {info['completed']}")
    print(f"  XML: {info['xml_file']}")
    print()

    xml_file = sys.argv[2] if len(sys.argv) > 2 else info['xml_file']
    if not xml_file:
        print("No XML file available")
        sys.exit(1)

    # Get live stats
    print("Fetching live stats...")
    sit, ft = client.get_situation(event_id, xml_file)
    if sit is None:
        print("No stats available (game may not have started)")
        sys.exit(0)

    print(f"  Score: {sit.get('visitor', '?')} {sit.get('visitor_score', '?')}"
          f" - {sit.get('home', '?')} {sit.get('home_score', '?')}")
    print(f"  Inning: {sit.get('inning_display', '?')}")
    print(f"  Outs: {sit.get('outs', '?')}")
    print(f"  Count: {sit.get('count', '?')}")
    print(f"  Batter: {sit.get('batter_name', '?')}"
          f" #{sit.get('batter_number', '?')}")
    print(f"  Pitcher: {sit.get('pitcher_name', '?')}"
          f" #{sit.get('pitcher_number', '?')}")
    print(f"  Title: {sit.get('title', '?')}")
    print(f"  Filetime: {ft}")


if __name__ == '__main__':
    main()
