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

    # Base runners — two sources:
    #
    # 1. Icon font in base-indicator div: <i class="sbicon ...">N</i>
    #    The sbicon font maps characters to diamond states:
    #      0 = empty, 1 = 1st, 2 = 2nd, 3 = 3rd,
    #      4 = 1st+2nd, 5 = 2nd+3rd, 6 = 1st+3rd, 7 = loaded
    #
    # 2. "Runners On Base" table (sometimes empty between plays)

    ICON_BASES = {
        '0': (False, False, False),  # empty
        '1': (True, False, False),   # 1st
        '2': (False, True, False),   # 2nd
        '3': (False, False, True),   # 3rd
        '4': (True, True, False),    # 1st + 2nd
        '5': (False, True, True),    # 2nd + 3rd
        '6': (True, False, True),    # 1st + 3rd
        '7': (True, True, True),     # loaded
    }

    bases_found = False

    # Method 1: Parse base-indicator icon font (most reliable)
    # There are two indicators (one per team half). Use the one that's
    # NOT inside a "noaccess" div, or the last one that changed.
    base_icons = re.findall(
        r'base-indicator.*?<i[^>]*sbicon[^>]*>([0-7])</i>',
        html, re.DOTALL
    )
    if base_icons:
        # Use the last (most recent/current half-inning) icon
        icon = base_icons[-1]
        if icon in ICON_BASES:
            first, second, third = ICON_BASES[icon]
            result['on_first'] = first
            result['on_second'] = second
            result['on_third'] = third
            bases_found = True

    # Method 2: Fallback to "Runners On Base" table
    if not bases_found:
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
            if runner_rows:
                bases_found = True

    # Default to False if nothing found
    result.setdefault('on_first', False)
    result.setdefault('on_second', False)
    result.setdefault('on_third', False)

    # Line score — R/H/E + per-inning scoring
    # Structure: <td>caret</td><td>VLogo TEAM</td><td>Inn1</td>...<td class="border-right"></td><td>R</td><td>H</td><td>E</td><td>LOB</td>
    # Visitor row has "VLogo", home has "HLogo"
    linescore_rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL)
    visitor_linescore_found = False
    home_linescore_found = False
    for row in linescore_rows:
        is_visitor = 'VLogo' in row
        is_home = 'HLogo' in row
        if not is_visitor and not is_home:
            continue
        # Only use the FIRST linescore row per team (skip standings/record tables)
        if is_visitor and visitor_linescore_found:
            continue
        if is_home and home_linescore_found:
            continue

        # Split on border-right separator to isolate innings from R/H/E/LOB
        parts = re.split(r'<td[^>]*border-right[^>]*>[^<]*</td>', row)
        if len(parts) >= 2:
            innings_part = parts[0]
            summary_part = parts[1]

            # Extract inning scores
            # Cells with inner HTML (caret icon, team logo) are NOT matched
            # by [^<]* so inn_cells is mostly pure inning scores.
            # Filter out empties (from caret/spacer cells without inner HTML).
            inn_cells = re.findall(r'<td[^>]*>([^<]*)</td>', innings_part)
            inning_scores = []
            for s in inn_cells:
                if not s.strip():
                    continue
                s = s.strip()
                if s == '' or s == '-' or s == 'X':
                    inning_scores.append(s)
                else:
                    try:
                        inning_scores.append(int(s))
                    except ValueError:
                        inning_scores.append(s)

            # Extract R/H/E/LOB from summary cells
            sum_cells = re.findall(r'<td[^>]*>([^<]*)</td>', summary_part)
            # R, H, E, LOB
            r_val = h_val = e_val = None
            try:
                if len(sum_cells) >= 1:
                    r_val = int(sum_cells[0])
                if len(sum_cells) >= 2:
                    h_val = int(sum_cells[1])
                if len(sum_cells) >= 3:
                    e_val = int(sum_cells[2])
            except (ValueError, IndexError):
                pass

            prefix = 'visitor' if is_visitor else 'home'
            if inning_scores:
                result[prefix + '_innings'] = inning_scores
            if h_val is not None:
                result[prefix + '_hits'] = h_val
            if e_val is not None:
                result[prefix + '_errors'] = e_val
            if is_visitor:
                visitor_linescore_found = True
            else:
                home_linescore_found = True
        else:
            # Fallback: no border-right separator found
            all_cells = re.findall(r'<td[^>]*>([^<]*)</td>', row)
            if len(all_cells) >= 5:
                prefix = 'visitor' if is_visitor else 'home'
                try:
                    result[prefix + '_hits'] = int(all_cells[-3])
                    result[prefix + '_errors'] = int(all_cells[-2])
                except (ValueError, IndexError):
                    pass
            if is_visitor:
                visitor_linescore_found = True
            else:
                home_linescore_found = True

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

    def get_play_by_play(self, event_id, xml_file):
        """
        Fetch the play-by-play view for a game.

        Returns parsed play-by-play list (see parse_plays()), or None on error.
        """
        params = (
            "event={event_id}"
            "&xml={xml_file}"
            "&xsl=baseball/sb.bsgame.views.pxp.xsl"
            "&sport=bsgame"
            "&filetime=1"
            "&type=statbroadcast"
            "&start=true"
        ).format(event_id=event_id, xml_file=xml_file)

        data = sb_encode_params(params)
        url = "{}/stats?data={}".format(self.BASE_URL, data)
        req = self._request(url)

        try:
            with urllib.request.urlopen(req, timeout=self.TIMEOUT) as resp:
                raw = resp.read().decode()
                if not raw.strip():
                    return None
                html = sb_decode(raw)
                return parse_plays(html)
        except Exception:
            return None


def parse_plays(html):
    """
    Parse StatBroadcast play-by-play HTML into structured data.

    Returns a list of inning dicts:
    [
        {
            'inning': 1,
            'half': 'top',       # 'top' or 'bottom'
            'label': 'Top of the 1st - TEAM Batting',
            'plays': [
                {
                    'text': 'J. Smith singled to left field.',
                    'batter': 'J. Smith',
                    'pitcher': 'T. Jones',
                    'outs_after': 0,
                    'scoring': '1B 7',
                    'is_scoring': False,
                    'bases': '1B',  # runner indicator if present
                },
                ...
            ],
            'summary': '0 Runs, 1 Hits, 0 Errors, 1 LOB'
        },
        ...
    ]
    """
    innings = []
    current_inning = None

    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL)

    for row in rows:
        cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
        if not cells:
            continue

        # Join all cell text for pattern matching
        row_text = ' '.join(re.sub(r'<[^>]+>', ' ', c).strip() for c in cells)
        row_text = re.sub(r'\s+', ' ', row_text).strip()

        # Check for inning header: "Top of the 1st - TEAM Batting" or "Bottom of the 3rd"
        inning_match = re.search(
            r'(Top|Bottom)\s+of\s+the\s+(\d+)\w*\s*[-–]\s*(\w[\w\s]*?)\s*Batting',
            row_text, re.IGNORECASE
        )
        if inning_match:
            half = 'top' if inning_match.group(1).lower() == 'top' else 'bottom'
            inning_num = int(inning_match.group(2))
            team = inning_match.group(3).strip()
            current_inning = {
                'inning': inning_num,
                'half': half,
                'label': '{} of the {} - {} Batting'.format(
                    inning_match.group(1), inning_match.group(2) + _ordinal(inning_num),
                    team
                ),
                'team': team,
                'plays': [],
                'summary': '',
            }
            innings.append(current_inning)
            continue

        # Check for inning summary: "TEAM Inning Summary: 0 Runs, 1 Hits, ..."
        summary_match = re.search(
            r'Inning Summary:\s*(.*)', row_text
        )
        if summary_match and current_inning:
            current_inning['summary'] = summary_match.group(1).strip()
            continue

        # Check for play row — has play text, batter, pitcher, outs columns
        if not current_inning:
            continue

        # Play rows have substantial text content and typically 4+ cells
        # Look for text that describes a baseball play
        play_patterns = (
            r'(singled|doubled|tripled|walked|struck out|grounded out|'
            r'flied out|homered|hit by pitch|reached|popped|lined|'
            r'fouled out|scored|stole|advanced|sac |sacrifice|'
            r'fielder|out at|caught stealing|picked off|wild pitch|'
            r'passed ball|balk|error|pinch hit|to [a-z]+ for)'
        )
        if not re.search(play_patterns, row_text, re.IGNORECASE):
            continue

        # Extract play details from cells
        # Typical structure: [bases_indicator, play_text, scoring_dec, batter, pitcher, outs]
        clean_cells = []
        for c in cells:
            clean = re.sub(r'<[^>]+>', '', c).strip()
            clean_cells.append(clean)

        play = {
            'text': '',
            'batter': '',
            'pitcher': '',
            'outs_after': None,
            'scoring': '',
            'is_scoring': False,
            'bases': '',
        }

        # Find the longest cell — that's likely the play description
        play_text = max(clean_cells, key=len) if clean_cells else ''
        play['text'] = play_text

        # Check for base indicator (1B, 2B, 3B in early cells)
        for c in clean_cells[:2]:
            if c in ('1B', '2B', '3B', 'HR'):
                play['bases'] = c
                break

        # Extract batter/pitcher from named cells or positional
        # They're usually the 2nd-to-last and 3rd-to-last cells
        if len(clean_cells) >= 4:
            # Outs is typically last
            try:
                play['outs_after'] = int(clean_cells[-1])
            except (ValueError, IndexError):
                pass
            # Pitcher is second to last
            play['pitcher'] = clean_cells[-2] if len(clean_cells) >= 2 else ''
            # Batter is third to last
            play['batter'] = clean_cells[-3] if len(clean_cells) >= 3 else ''
            # Scoring decision is fourth to last (if enough cells)
            if len(clean_cells) >= 5:
                play['scoring'] = clean_cells[-4] if clean_cells[-4] != play_text else ''

        # Detect scoring plays
        play['is_scoring'] = bool(re.search(r'scored|RBI|home run|homered', play_text, re.IGNORECASE))

        # Clean up: remove play text from batter/pitcher if accidentally captured
        if play['batter'] == play_text:
            play['batter'] = ''
        if play['pitcher'] == play_text:
            play['pitcher'] = ''

        current_inning['plays'].append(play)

    return innings


def _ordinal(n):
    """Return ordinal suffix for a number."""
    if 11 <= n % 100 <= 13:
        return 'th'
    return {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')


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
