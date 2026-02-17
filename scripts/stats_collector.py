#!/usr/bin/env python3
"""
Unified P4 Stats Collector - Collects batting and pitching stats from SIDEARM Sports sites.

Parses embedded Nuxt 3 JSON payload from SIDEARM-powered athletics sites.
Falls back to Playwright browser automation for JS-rendered sites (WMT Digital, etc.)

Usage:
    python3 scripts/stats_collector.py --all                    # All P4 teams
    python3 scripts/stats_collector.py --conference SEC         # One conference
    python3 scripts/stats_collector.py --team alabama           # Single team
    python3 scripts/stats_collector.py --resume                 # Resume from progress
    python3 scripts/stats_collector.py --dry-run                # Test without DB writes
    python3 scripts/stats_collector.py --recalc                 # Recalculate derived stats only
"""

import argparse
import json
import sys
import re
import time
import sqlite3
import subprocess
import logging
from pathlib import Path
from datetime import datetime

# Setup
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / 'data'
DB_PATH = DATA_DIR / 'baseball.db'
TEAM_URLS_FILE = DATA_DIR / 'p4_team_urls.json'
PROGRESS_FILE = DATA_DIR / 'stats_collector_progress.json'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger('stats_collector')

REQUEST_DELAY = 2
BROWSER_DELAY = 2  # Extra delay between browser requests to be polite

# ESPN team ID mapping (for future use when ESPN adds college baseball stats)
ESPN_IDS_FILE = DATA_DIR / 'espn_team_ids.json'

# Teams that don't have baseball programs (included in P4 conferences but no team)
NO_BASEBALL = {'colorado', 'iowa-state', 'smu', 'syracuse'}

# Teams known to require browser automation (WMT Digital / JS-rendered)
# These skip the SIDEARM parser and go straight to Playwright
BROWSER_REQUIRED = {
    # SEC
    'arkansas', 'auburn', 'kentucky', 'lsu', 'south-carolina', 'vanderbilt',
    # Big Ten
    'illinois', 'iowa', 'maryland', 'nebraska', 'penn-state', 'purdue',
    # ACC
    'california', 'clemson', 'georgia-tech', 'miami-fl', 'notre-dame', 
    'stanford', 'virginia', 'virginia-tech',
    # Big 12
    'arizona', 'arizona-state', 'byu', 'cincinnati', 'kansas-state', 'ucf',
}


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def load_team_urls():
    urls = {}
    # Load P4 URLs
    if TEAM_URLS_FILE.exists():
        with open(TEAM_URLS_FILE) as f:
            urls.update(json.load(f).get('teams', {}))
    # Load extended D1 URLs
    d1_file = DATA_DIR / 'd1_team_urls.json'
    if d1_file.exists():
        with open(d1_file) as f:
            urls.update(json.load(f).get('teams', {}))
    return urls


def get_teams_for_args(args):
    db = get_db()
    if args.team:
        teams = db.execute("SELECT id FROM teams WHERE id = ?", (args.team,)).fetchall()
    elif args.conference:
        teams = db.execute("SELECT id FROM teams WHERE conference = ?", (args.conference,)).fetchall()
    elif args.d1:
        teams = db.execute(
            "SELECT id FROM teams WHERE conference != '' AND conference IS NOT NULL ORDER BY conference, id"
        ).fetchall()
    else:
        teams = db.execute(
            "SELECT id FROM teams WHERE conference IN ('SEC','Big 12','Big Ten','ACC') ORDER BY conference, id"
        ).fetchall()
    db.close()
    return [t['id'] for t in teams]


def fetch_page(url, follow_redirects=True):
    import urllib.request
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode('utf-8', errors='ignore')
    except Exception as e:
        log.error(f"Fetch failed for {url}: {e}")
        return ""


def parse_sidearm_nuxt3(html):
    """Parse stats from SIDEARM Nuxt 3 payload."""
    # Find script containing the stats data
    idx = html.find('individualHittingStats')
    if idx < 0:
        idx = html.find('cumulativeStats')
    if idx < 0:
        return None, None

    start = html.rfind('<script', 0, idx)
    end = html.find('</script>', idx)
    if start < 0 or end < 0:
        return None, None

    script_content = html[html.find('>', start) + 1:end]

    # Parse JSON (may have trailing JS)
    data = None
    for trim in range(0, 300):
        try:
            data = json.loads(script_content[:len(script_content) - trim] if trim else script_content)
            break
        except json.JSONDecodeError:
            pass

    if not data or not isinstance(data, list):
        return None, None

    def resolve(val, depth=0):
        if depth > 150:
            return val
        if isinstance(val, int) and not isinstance(val, bool):
            if 0 <= val < len(data):
                return resolve(data[val], depth + 1)
            return val
        if isinstance(val, list):
            if len(val) == 2 and isinstance(val[0], str) and val[0] in (
                'ShallowReactive', 'Reactive', 'ShallowRef', 'Ref', 'Set'
            ):
                return resolve(val[1], depth + 1)
            return [resolve(item, depth + 1) for item in val]
        if isinstance(val, dict):
            return {k: resolve(v, depth + 1) for k, v in val.items()}
        return val

    for i, item in enumerate(data):
        if isinstance(item, dict) and 'individualHittingStats' in item:
            batting = resolve(item.get('individualHittingStats'))
            pitching = resolve(item.get('individualPitchingStats'))
            return (
                batting if isinstance(batting, list) else None,
                pitching if isinstance(pitching, list) else []
            )

    return None, None


def normalize_player_name(name):
    """Convert 'Last, First' to 'First Last' format."""
    if not name:
        return name
    if ',' in name:
        parts = name.split(',', 1)
        return f"{parts[1].strip()} {parts[0].strip()}"
    return name.strip()


def extract_player_batting(raw_stat, team_id):
    if not isinstance(raw_stat, dict):
        return None
    name = normalize_player_name(raw_stat.get('playerName', ''))
    if not name or name in ('Totals', 'Opponents') or raw_stat.get('isAFooterStat'):
        return None

    def si(val, default=0):
        if val is None or val == '' or val == '—':
            return default
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    return {
        'team_id': team_id,
        'name': name,
        'number': si(raw_stat.get('playerUniform')),
        'games': si(raw_stat.get('gamesPlayed')),
        'at_bats': si(raw_stat.get('atBats')),
        'runs': si(raw_stat.get('runs')),
        'hits': si(raw_stat.get('hits')),
        'doubles': si(raw_stat.get('doubles')),
        'triples': si(raw_stat.get('triples')),
        'home_runs': si(raw_stat.get('homeRuns')),
        'rbi': si(raw_stat.get('runsBattedIn')),
        'walks': si(raw_stat.get('walks')),
        'strikeouts': si(raw_stat.get('strikeouts')),
        'stolen_bases': si(raw_stat.get('stolenBases')),
        'caught_stealing': si(raw_stat.get('caughtStealing')),
    }


def extract_player_pitching(raw_stat, team_id):
    if not isinstance(raw_stat, dict):
        return None
    name = normalize_player_name(raw_stat.get('playerName', ''))
    if not name or name in ('Totals', 'Opponents') or raw_stat.get('isAFooterStat'):
        return None

    def si(val, default=0):
        if val is None or val == '' or val == '—':
            return default
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    def sf(val, default=0.0):
        if val is None or val == '' or val == '—':
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    return {
        'team_id': team_id,
        'name': name,
        'number': si(raw_stat.get('playerUniform')),
        'wins': si(raw_stat.get('wins')),
        'losses': si(raw_stat.get('losses')),
        'games_pitched': si(raw_stat.get('appearances')),
        'games_started': si(raw_stat.get('gamesStarted')),
        'saves': si(raw_stat.get('saves')),
        'innings_pitched': sf(raw_stat.get('inningsPitched')),
        'hits_allowed': si(raw_stat.get('hitsAllowed')),
        'runs_allowed': si(raw_stat.get('runsAllowed')),
        'earned_runs': si(raw_stat.get('earnedRunsAllowed')),
        'walks_allowed': si(raw_stat.get('walksAllowed')),
        'strikeouts_pitched': si(raw_stat.get('strikeouts')),
    }


def convert_ip_to_decimal(ip):
    """Convert baseball IP notation to decimal (6.1 -> 6.333, 6.2 -> 6.667)."""
    if ip == 0:
        return 0.0
    whole = int(ip)
    frac = round(ip - whole, 1)
    if abs(frac - 0.2) < 0.05:
        return whole + 2/3
    elif abs(frac - 0.1) < 0.05:
        return whole + 1/3
    return float(whole) + frac


def calculate_derived_stats(player):
    """Calculate batting_avg, obp, slg, ops, era, whip, k_per_9, bb_per_9."""
    ab = player.get('at_bats', 0) or 0
    h = player.get('hits', 0) or 0
    bb = player.get('walks', 0) or 0
    doubles = player.get('doubles', 0) or 0
    triples = player.get('triples', 0) or 0
    hr = player.get('home_runs', 0) or 0

    batting_avg = h / ab if ab > 0 else 0.0
    obp_denom = ab + bb
    obp = (h + bb) / obp_denom if obp_denom > 0 else 0.0
    tb = (h - doubles - triples - hr) + 2 * doubles + 3 * triples + 4 * hr
    slg = tb / ab if ab > 0 else 0.0
    ops = obp + slg

    ip = player.get('innings_pitched', 0) or 0
    er = player.get('earned_runs', 0) or 0
    ha = player.get('hits_allowed', 0) or 0
    wa = player.get('walks_allowed', 0) or 0
    k = player.get('strikeouts_pitched', 0) or 0
    ip_dec = convert_ip_to_decimal(ip)

    era = (er * 9) / ip_dec if ip_dec > 0 else 0.0
    whip = (wa + ha) / ip_dec if ip_dec > 0 else 0.0
    k_per_9 = (k * 9) / ip_dec if ip_dec > 0 else 0.0
    bb_per_9 = (wa * 9) / ip_dec if ip_dec > 0 else 0.0

    player['batting_avg'] = round(batting_avg, 3)
    player['obp'] = round(obp, 3)
    player['slg'] = round(slg, 3)
    player['ops'] = round(ops, 3)
    player['era'] = round(era, 2)
    player['whip'] = round(whip, 2)
    player['k_per_9'] = round(k_per_9, 2)
    player['bb_per_9'] = round(bb_per_9, 2)
    return player


PITCHING_DEFAULTS = {
    'wins': 0, 'losses': 0, 'games_pitched': 0, 'games_started': 0,
    'saves': 0, 'innings_pitched': 0, 'hits_allowed': 0, 'runs_allowed': 0,
    'earned_runs': 0, 'walks_allowed': 0, 'strikeouts_pitched': 0,
}


def upsert_player_stats(db, player):
    db.execute("""
        INSERT INTO player_stats (
            team_id, name, number, games, at_bats, runs, hits, doubles, triples,
            home_runs, rbi, walks, strikeouts, stolen_bases, caught_stealing,
            batting_avg, obp, slg, ops,
            wins, losses, era, games_pitched, games_started, saves,
            innings_pitched, hits_allowed, runs_allowed, earned_runs,
            walks_allowed, strikeouts_pitched, whip, k_per_9, bb_per_9,
            updated_at
        ) VALUES (
            :team_id, :name, :number, :games, :at_bats, :runs, :hits, :doubles, :triples,
            :home_runs, :rbi, :walks, :strikeouts, :stolen_bases, :caught_stealing,
            :batting_avg, :obp, :slg, :ops,
            :wins, :losses, :era, :games_pitched, :games_started, :saves,
            :innings_pitched, :hits_allowed, :runs_allowed, :earned_runs,
            :walks_allowed, :strikeouts_pitched, :whip, :k_per_9, :bb_per_9,
            CURRENT_TIMESTAMP
        )
        ON CONFLICT(team_id, name) DO UPDATE SET
            number=:number, games=:games, at_bats=:at_bats, runs=:runs,
            hits=:hits, doubles=:doubles, triples=:triples, home_runs=:home_runs,
            rbi=:rbi, walks=:walks, strikeouts=:strikeouts, stolen_bases=:stolen_bases,
            caught_stealing=:caught_stealing, batting_avg=:batting_avg, obp=:obp,
            slg=:slg, ops=:ops, wins=:wins, losses=:losses, era=:era,
            games_pitched=:games_pitched, games_started=:games_started, saves=:saves,
            innings_pitched=:innings_pitched, hits_allowed=:hits_allowed,
            runs_allowed=:runs_allowed, earned_runs=:earned_runs,
            walks_allowed=:walks_allowed, strikeouts_pitched=:strikeouts_pitched,
            whip=:whip, k_per_9=:k_per_9, bb_per_9=:bb_per_9,
            updated_at=CURRENT_TIMESTAMP
    """, player)


def _safe_int(val, default=0):
    """Safely convert to int, handling dashes and empty strings."""
    if val is None or val == '' or val == '—' or val == '-':
        return default
    try:
        # Handle percentage strings
        if isinstance(val, str):
            val = val.replace('%', '').strip()
        return int(float(val))
    except (ValueError, TypeError):
        return default


def _safe_float(val, default=0.0):
    """Safely convert to float."""
    if val is None or val == '' or val == '—' or val == '-':
        return default
    try:
        if isinstance(val, str):
            val = val.replace('%', '').strip()
        return float(val)
    except (ValueError, TypeError):
        return default


def parse_table_headers(headers):
    """Parse table headers and return column mapping."""
    header_map = {}
    normalized_headers = []
    
    for i, h in enumerate(headers):
        h_lower = h.lower().strip()
        normalized_headers.append(h_lower)
        
        # Batting columns
        if h_lower in ('name', 'player'):
            header_map['name'] = i
        elif h_lower in ('#', 'no', 'no.', 'number'):
            header_map['number'] = i
        elif h_lower in ('avg', 'ba', 'batting avg'):
            header_map['avg'] = i
        elif h_lower in ('gp', 'g', 'games'):
            header_map['games'] = i
        elif h_lower in ('ab', 'at bats'):
            header_map['at_bats'] = i
        elif h_lower in ('r', 'runs'):
            header_map['runs'] = i
        elif h_lower in ('h', 'hits'):
            header_map['hits'] = i
        elif h_lower in ('2b', 'doubles'):
            header_map['doubles'] = i
        elif h_lower in ('3b', 'triples'):
            header_map['triples'] = i
        elif h_lower in ('hr', 'home runs'):
            header_map['home_runs'] = i
        elif h_lower in ('rbi',):
            header_map['rbi'] = i
        elif h_lower in ('bb', 'walks'):
            header_map['walks'] = i
        elif h_lower in ('so', 'k', 'strikeouts'):
            header_map['strikeouts'] = i
        elif h_lower in ('sb', 'stolen bases'):
            header_map['stolen_bases'] = i
        elif h_lower in ('cs', 'caught stealing'):
            header_map['caught_stealing'] = i
        
        # Pitching columns
        elif h_lower in ('era',):
            header_map['era'] = i
        elif h_lower in ('w', 'wins'):
            header_map['wins'] = i
        elif h_lower in ('l', 'losses'):
            header_map['losses'] = i
        elif h_lower in ('app', 'appearances'):
            header_map['appearances'] = i
        elif h_lower in ('gs', 'games started'):
            header_map['games_started'] = i
        elif h_lower in ('sv', 'saves'):
            header_map['saves'] = i
        elif h_lower in ('ip', 'innings', 'innings pitched'):
            header_map['innings_pitched'] = i
        elif h_lower in ('ha', 'hits allowed'):
            header_map['hits_allowed'] = i
        elif h_lower in ('ra', 'runs allowed'):
            header_map['runs_allowed'] = i
        elif h_lower in ('er', 'earned runs'):
            header_map['earned_runs'] = i
        elif h_lower in ('bba', 'walks allowed', 'bb'):
            header_map['walks_allowed'] = i
        elif h_lower in ('so', 'k', 'strikeouts') and 'era' in header_map:
            # Pitching strikeouts if we already have ERA
            header_map['strikeouts_pitched'] = i
    
    return header_map, normalized_headers


def parse_batting_from_table(rows, header_map, team_id):
    """Parse batting stats from table rows using header mapping."""
    players = []
    
    for row in rows:
        # Skip totals/opponents rows
        if not row or len(row) <= max(header_map.values()):
            continue
        
        name_idx = header_map.get('name', 0)
        name = row[name_idx].strip() if name_idx < len(row) else ''
        
        if not name or name.lower() in ('totals', 'opponents', 'team', ''):
            continue
        
        # Normalize name
        name = normalize_player_name(name)
        
        player = {
            'team_id': team_id,
            'name': name,
            'number': _safe_int(row[header_map['number']]) if 'number' in header_map and header_map['number'] < len(row) else 0,
            'games': _safe_int(row[header_map['games']]) if 'games' in header_map and header_map['games'] < len(row) else 0,
            'at_bats': _safe_int(row[header_map['at_bats']]) if 'at_bats' in header_map and header_map['at_bats'] < len(row) else 0,
            'runs': _safe_int(row[header_map['runs']]) if 'runs' in header_map and header_map['runs'] < len(row) else 0,
            'hits': _safe_int(row[header_map['hits']]) if 'hits' in header_map and header_map['hits'] < len(row) else 0,
            'doubles': _safe_int(row[header_map['doubles']]) if 'doubles' in header_map and header_map['doubles'] < len(row) else 0,
            'triples': _safe_int(row[header_map['triples']]) if 'triples' in header_map and header_map['triples'] < len(row) else 0,
            'home_runs': _safe_int(row[header_map['home_runs']]) if 'home_runs' in header_map and header_map['home_runs'] < len(row) else 0,
            'rbi': _safe_int(row[header_map['rbi']]) if 'rbi' in header_map and header_map['rbi'] < len(row) else 0,
            'walks': _safe_int(row[header_map['walks']]) if 'walks' in header_map and header_map['walks'] < len(row) else 0,
            'strikeouts': _safe_int(row[header_map['strikeouts']]) if 'strikeouts' in header_map and header_map['strikeouts'] < len(row) else 0,
            'stolen_bases': _safe_int(row[header_map['stolen_bases']]) if 'stolen_bases' in header_map and header_map['stolen_bases'] < len(row) else 0,
            'caught_stealing': _safe_int(row[header_map['caught_stealing']]) if 'caught_stealing' in header_map and header_map['caught_stealing'] < len(row) else 0,
        }
        
        players.append(player)
    
    return players


def parse_pitching_from_table(rows, header_map, team_id):
    """Parse pitching stats from table rows using header mapping."""
    players = []
    
    for row in rows:
        if not row or len(row) <= max(header_map.values()):
            continue
        
        name_idx = header_map.get('name', 0)
        name = row[name_idx].strip() if name_idx < len(row) else ''
        
        if not name or name.lower() in ('totals', 'opponents', 'team', ''):
            continue
        
        name = normalize_player_name(name)
        
        player = {
            'name': name,
            'number': _safe_int(row[header_map['number']]) if 'number' in header_map and header_map['number'] < len(row) else 0,
            'wins': _safe_int(row[header_map['wins']]) if 'wins' in header_map and header_map['wins'] < len(row) else 0,
            'losses': _safe_int(row[header_map['losses']]) if 'losses' in header_map and header_map['losses'] < len(row) else 0,
            'games_pitched': _safe_int(row[header_map['appearances']]) if 'appearances' in header_map and header_map['appearances'] < len(row) else 0,
            'games_started': _safe_int(row[header_map['games_started']]) if 'games_started' in header_map and header_map['games_started'] < len(row) else 0,
            'saves': _safe_int(row[header_map['saves']]) if 'saves' in header_map and header_map['saves'] < len(row) else 0,
            'innings_pitched': _safe_float(row[header_map['innings_pitched']]) if 'innings_pitched' in header_map and header_map['innings_pitched'] < len(row) else 0,
            'hits_allowed': _safe_int(row[header_map['hits_allowed']]) if 'hits_allowed' in header_map and header_map['hits_allowed'] < len(row) else 0,
            'runs_allowed': _safe_int(row[header_map['runs_allowed']]) if 'runs_allowed' in header_map and header_map['runs_allowed'] < len(row) else 0,
            'earned_runs': _safe_int(row[header_map['earned_runs']]) if 'earned_runs' in header_map and header_map['earned_runs'] < len(row) else 0,
            'walks_allowed': _safe_int(row[header_map['walks_allowed']]) if 'walks_allowed' in header_map and header_map['walks_allowed'] < len(row) else 0,
            'strikeouts_pitched': _safe_int(row[header_map['strikeouts_pitched']]) if 'strikeouts_pitched' in header_map and header_map['strikeouts_pitched'] < len(row) else 0,
        }
        
        players.append(player)
    
    return players


def collect_stats_with_browser(team_id, url, db, dry_run=False):
    """
    Use Playwright to render JS-heavy stats pages and extract data.
    Works for WMT Digital sites and other JS-rendered pages.
    """
    log.info(f"  Using browser automation for {team_id}...")
    
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        log.error("  Playwright not installed. Run: pip install playwright && playwright install chromium")
        return None
    
    batting_players = []
    pitching_players = []
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            page = context.new_page()
            
            # Try multiple URL patterns
            base = url.rsplit('/stats', 1)[0] + '/stats'
            urls_to_try = [url]
            
            # Add variants without year suffix for SIDEARM sites
            if url.endswith('/2026'):
                urls_to_try.extend([url.replace('/2026', '/2025'), base])
            elif url.endswith('/2025'):
                urls_to_try.extend([url.replace('/2025', '/2026'), base])
            else:
                urls_to_try.extend([base + '/2026', base + '/2025'])
            
            page_loaded = False
            final_url = None
            
            for try_url in urls_to_try:
                try:
                    log.info(f"    Trying {try_url}")
                    response = page.goto(try_url, wait_until='domcontentloaded', timeout=20000)
                    
                    if response and response.status == 200:
                        # Wait for tables to potentially render
                        time.sleep(2)
                        
                        # Try to wait for table or stats content
                        try:
                            page.wait_for_selector('table, .stats-table, [class*="stats"], wmt-stats-iframe', timeout=8000)
                        except:
                            pass
                        
                        # Check if we have stats content
                        content = page.content()
                        if ('individualHittingStats' in content or 
                            '<table' in content.lower() or
                            'wmt_stats2_iframe_url' in content or
                            'wmt-stats-iframe' in content):
                            page_loaded = True
                            final_url = try_url
                            break
                except Exception as e:
                    log.debug(f"    URL {try_url} failed: {e}")
                    continue
            
            if not page_loaded:
                log.error(f"  Browser: Failed to load any URL for {team_id}")
                browser.close()
                return None
            
            log.info(f"    Loaded {final_url}")
            
            # Get the page content
            html = page.content()
            
            # First try SIDEARM Nuxt parsing (some sites have JS but still embed data)
            batting_raw, pitching_raw = parse_sidearm_nuxt3(html)
            
            if batting_raw:
                log.info(f"    Found SIDEARM data via browser")
                browser.close()
                # Process using existing SIDEARM flow
                return _process_sidearm_data(batting_raw, pitching_raw, team_id, db, dry_run)
            
            # Otherwise, try to extract from rendered tables
            log.info(f"    Parsing rendered tables...")
            
            # Look for stat tables in the DOM
            tables = page.query_selector_all('table')
            
            batting_table_found = False
            pitching_table_found = False
            
            for table in tables:
                try:
                    # Get headers
                    headers = []
                    header_cells = table.query_selector_all('th')
                    if not header_cells:
                        header_cells = table.query_selector_all('thead td')
                    
                    for th in header_cells:
                        headers.append(th.inner_text().strip())
                    
                    if not headers or len(headers) < 5:
                        continue
                    
                    header_map, normalized = parse_table_headers(headers)
                    
                    # Determine if this is batting or pitching
                    is_pitching = 'era' in header_map or 'innings_pitched' in header_map
                    is_batting = 'at_bats' in header_map or 'avg' in header_map
                    
                    # Get rows
                    rows = []
                    row_elements = table.query_selector_all('tbody tr')
                    for row_el in row_elements:
                        cells = row_el.query_selector_all('td')
                        row = [c.inner_text().strip() for c in cells]
                        if row and len(row) >= 3:
                            rows.append(row)
                    
                    if is_batting and not batting_table_found and rows:
                        batting_players = parse_batting_from_table(rows, header_map, team_id)
                        if batting_players:
                            batting_table_found = True
                            log.info(f"    Found batting table with {len(batting_players)} players")
                    
                    if is_pitching and not pitching_table_found and rows:
                        pitching_players = parse_pitching_from_table(rows, header_map, team_id)
                        if pitching_players:
                            pitching_table_found = True
                            log.info(f"    Found pitching table with {len(pitching_players)} pitchers")
                
                except Exception as e:
                    log.debug(f"    Table parsing error: {e}")
                    continue
            
            # Check for WMT iframe (component or JSON data)
            if not batting_table_found:
                wmt_url = None
                log.info(f"    No tables found, checking for WMT iframe...")
                
                # Try to find WMT iframe component
                wmt_iframe = page.query_selector('wmt-stats-iframe')
                if wmt_iframe:
                    wmt_path = wmt_iframe.get_attribute('path')
                    if wmt_path:
                        wmt_url = f"https://wmt.games{wmt_path}"
                        log.info(f"    Found WMT component with path: {wmt_path}")
                
                # Also check for WMT URL in page JSON (SIDEARM sites embed this)
                if not wmt_url:
                    wmt_match = re.search(r'wmt_stats2_iframe_url["\s:]+(["\'])?(https://wmt\.games/[^"\'<>\s]+)', html)
                    if wmt_match:
                        wmt_url = wmt_match.group(2)
                        log.info(f"    Found WMT URL in JSON: {wmt_url}")
                
                if wmt_url:
                    log.info(f"    Found WMT iframe, navigating to {wmt_url}")
                    
                    try:
                        page.goto(wmt_url, wait_until='networkidle', timeout=15000)
                        time.sleep(3)  # Wait for JS rendering
                        
                        # Re-try table extraction
                        tables = page.query_selector_all('table')
                        for table in tables:
                            try:
                                headers = [th.inner_text().strip() for th in table.query_selector_all('th')]
                                if not headers or len(headers) < 5:
                                    continue
                                
                                header_map, _ = parse_table_headers(headers)
                                is_pitching = 'era' in header_map or 'innings_pitched' in header_map
                                is_batting = 'at_bats' in header_map or 'avg' in header_map
                                
                                rows = []
                                for row_el in table.query_selector_all('tbody tr'):
                                    row = [c.inner_text().strip() for c in row_el.query_selector_all('td')]
                                    if row and len(row) >= 3:
                                        rows.append(row)
                                
                                if is_batting and not batting_table_found and rows:
                                    batting_players = parse_batting_from_table(rows, header_map, team_id)
                                    if batting_players:
                                        batting_table_found = True
                                
                                if is_pitching and not pitching_table_found and rows:
                                    pitching_players = parse_pitching_from_table(rows, header_map, team_id)
                                    if pitching_players:
                                        pitching_table_found = True
                            
                            except Exception as e:
                                continue
                    except Exception as e:
                        log.warning(f"    WMT navigation failed: {e}")
            
            browser.close()
            
            if not batting_players:
                log.error(f"  Browser: No batting data found for {team_id}")
                return None
            
            # Merge batting and pitching
            pitching_by_name = {p['name']: p for p in pitching_players}
            
            batting_count = 0
            for bp in batting_players:
                for k, v in PITCHING_DEFAULTS.items():
                    bp.setdefault(k, v)
                
                if bp['name'] in pitching_by_name:
                    ps = pitching_by_name.pop(bp['name'])
                    for k in PITCHING_DEFAULTS:
                        bp[k] = ps.get(k, 0)
                
                calculate_derived_stats(bp)
                if not dry_run:
                    upsert_player_stats(db, bp)
                batting_count += 1
            
            # Pitch-only players
            pitching_only = 0
            for name, ps in pitching_by_name.items():
                player = {
                    'team_id': team_id, 'name': name, 'number': ps.get('number', 0),
                    'games': 0, 'at_bats': 0, 'runs': 0, 'hits': 0, 'doubles': 0,
                    'triples': 0, 'home_runs': 0, 'rbi': 0, 'walks': 0, 'strikeouts': 0,
                    'stolen_bases': 0, 'caught_stealing': 0,
                }
                for k in PITCHING_DEFAULTS:
                    player[k] = ps.get(k, 0)
                calculate_derived_stats(player)
                if not dry_run:
                    upsert_player_stats(db, player)
                pitching_only += 1
            
            if not dry_run:
                db.commit()
            
            log.info(f"  {team_id} (browser): {batting_count} batters, {len(pitching_players)} pitchers ({pitching_only} pitch-only)")
            return {'batting': batting_count, 'pitching': len(pitching_players), 'status': 'ok', 'source': 'browser'}
    
    except Exception as e:
        log.error(f"  Browser error for {team_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


def _process_sidearm_data(batting_raw, pitching_raw, team_id, db, dry_run):
    """Process SIDEARM Nuxt data and upsert to DB."""
    # Build name->pitching lookup
    pitching_by_name = {}
    if pitching_raw:
        for praw in pitching_raw:
            ps = extract_player_pitching(praw, team_id)
            if ps:
                pitching_by_name[ps['name']] = ps

    batting_count = 0
    for raw in batting_raw:
        player = extract_player_batting(raw, team_id)
        if not player:
            continue

        # Merge pitching if this player also pitches
        for k, v in PITCHING_DEFAULTS.items():
            player.setdefault(k, v)
        if player['name'] in pitching_by_name:
            ps = pitching_by_name.pop(player['name'])
            for k in PITCHING_DEFAULTS:
                player[k] = ps[k]

        calculate_derived_stats(player)
        if not dry_run:
            upsert_player_stats(db, player)
        batting_count += 1

    # Pitchers who didn't bat
    pitching_only = 0
    for name, ps in pitching_by_name.items():
        player = {
            'team_id': team_id, 'name': name, 'number': ps['number'],
            'games': 0, 'at_bats': 0, 'runs': 0, 'hits': 0, 'doubles': 0,
            'triples': 0, 'home_runs': 0, 'rbi': 0, 'walks': 0, 'strikeouts': 0,
            'stolen_bases': 0, 'caught_stealing': 0,
        }
        for k in PITCHING_DEFAULTS:
            player[k] = ps[k]
        calculate_derived_stats(player)
        if not dry_run:
            upsert_player_stats(db, player)
        pitching_only += 1

    if not dry_run:
        db.commit()

    total_pitchers = len([p for p in (pitching_raw or [])
                         if isinstance(p, dict) and not p.get('isAFooterStat')
                         and p.get('playerName') not in ('Totals', 'Opponents', None)])

    log.info(f"  {team_id}: {batting_count} batters, {total_pitchers} pitchers ({pitching_only} pitch-only)")
    return {'batting': batting_count, 'pitching': total_pitchers, 'status': 'ok', 'source': 'sidearm'}


def collect_team_stats(team_id, url, db, dry_run=False):
    """Collect batting + pitching stats for one team."""
    log.info(f"  Fetching {team_id}...")
    
    # For teams known to require browser, skip SIDEARM and go straight to Playwright
    if team_id in BROWSER_REQUIRED:
        result = collect_stats_with_browser(team_id, url, db, dry_run)
        if result:
            return result
        # If browser fails, try SIDEARM as last resort
        log.info(f"    Browser failed, trying SIDEARM fallback...")

    # Try multiple URL patterns
    base = url.rsplit('/stats', 1)[0] + '/stats'
    urls_to_try = [url]
    if url.endswith('/2026'):
        urls_to_try.extend([url.replace('/2026', '/2025'), base])
    elif url.endswith('/2025'):
        urls_to_try.extend([url.replace('/2025', '/2026'), base])
    else:
        urls_to_try.extend([base + '/2026', base + '/2025'])

    batting_raw = None
    pitching_raw = None

    for try_url in urls_to_try:
        html = fetch_page(try_url)
        if not html or len(html) < 1000:
            continue
        if 'individualHittingStats' not in html and 'cumulativeStats' not in html:
            continue

        batting_raw, pitching_raw = parse_sidearm_nuxt3(html)
        if batting_raw:
            log.info(f"  Found data at {try_url}")
            break

    if not batting_raw:
        # SIDEARM failed, try browser fallback
        if team_id not in BROWSER_REQUIRED:
            log.info(f"    SIDEARM failed, trying browser fallback...")
            result = collect_stats_with_browser(team_id, url, db, dry_run)
            if result:
                return result
        
        log.error(f"  FAILED: No parseable stats for {team_id}")
        return {'batting': 0, 'pitching': 0, 'status': 'failed'}

    return _process_sidearm_data(batting_raw, pitching_raw, team_id, db, dry_run)


def recalculate_all_stats(db):
    """Recalculate all derived stats from raw counts."""
    log.info("Recalculating all derived stats...")
    players = db.execute("SELECT * FROM player_stats").fetchall()
    count = 0
    for p in players:
        player = dict(p)
        calculate_derived_stats(player)
        db.execute("""
            UPDATE player_stats SET
                batting_avg=?, obp=?, slg=?, ops=?,
                era=?, whip=?, k_per_9=?, bb_per_9=?,
                updated_at=CURRENT_TIMESTAMP
            WHERE id=?
        """, (player['batting_avg'], player['obp'], player['slg'], player['ops'],
              player['era'], player['whip'], player['k_per_9'], player['bb_per_9'],
              player['id']))
        count += 1
    db.commit()
    log.info(f"Recalculated stats for {count} players")


def load_progress():
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE) as f:
                return json.load(f)
        except:
            pass
    return {'completed': [], 'failed': [], 'last_run': None}


def save_progress(progress):
    progress['last_run'] = datetime.now().isoformat()
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Collect P4 baseball stats')
    parser.add_argument('--all', action='store_true', help='All P4 teams')
    parser.add_argument('--d1', action='store_true', help='All D1 teams with conference assignments')
    parser.add_argument('--conference', help='Conference (e.g., SEC)')
    parser.add_argument('--team', help='Single team ID')
    parser.add_argument('--resume', action='store_true', help='Resume from progress')
    parser.add_argument('--dry-run', action='store_true', help='No DB writes')
    parser.add_argument('--recalc', action='store_true', help='Recalculate derived stats')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    if not any([args.all, args.conference, args.team, args.recalc]):
        parser.print_help()
        sys.exit(1)

    db = get_db()

    if args.recalc:
        recalculate_all_stats(db)
        db.close()
        return

    team_urls = load_team_urls()
    team_ids = get_teams_for_args(args)

    if not team_ids:
        log.error("No teams found")
        sys.exit(1)

    progress = load_progress()
    if args.resume:
        team_ids = [t for t in team_ids if t not in progress['completed']]
        log.info(f"Resuming: {len(progress['completed'])} done, {len(team_ids)} remaining")
    else:
        progress = {'completed': [], 'failed': [], 'last_run': None}

    log.info(f"Collecting stats for {len(team_ids)} teams")
    start_time = time.time()
    results = {'ok': 0, 'failed': 0, 'total_batting': 0, 'total_pitching': 0}

    for i, team_id in enumerate(team_ids, 1):
        # Skip teams without baseball programs
        if team_id in NO_BASEBALL:
            log.info(f"[{i}/{len(team_ids)}] {team_id} - SKIPPED (no baseball program)")
            continue

        url = team_urls.get(team_id)
        if not url:
            log.warning(f"No URL for {team_id}")
            continue

        log.info(f"[{i}/{len(team_ids)}] {team_id}")

        try:
            import signal
            
            def _timeout_handler(signum, frame):
                raise TimeoutError(f"Team {team_id} timed out after 90s")
            
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(90)  # 90 second per-team timeout (browser takes longer)
            
            try:
                result = collect_team_stats(team_id, url, db, dry_run=args.dry_run)
            finally:
                signal.alarm(0)  # Cancel alarm
                signal.signal(signal.SIGALRM, old_handler)
            
            if result['status'] == 'ok':
                results['ok'] += 1
                results['total_batting'] += result['batting']
                results['total_pitching'] += result['pitching']
                progress['completed'].append(team_id)
                if 'source' in result:
                    progress.setdefault('sources', {})[team_id] = result['source']
            else:
                results['failed'] += 1
                progress['failed'].append(team_id)
        except (TimeoutError, Exception) as e:
            log.error(f"Exception for {team_id}: {e}")
            results['failed'] += 1
            progress['failed'].append(team_id)

        save_progress(progress)
        
        # Add delay between requests
        if i < len(team_ids):
            delay = BROWSER_DELAY if team_id in BROWSER_REQUIRED else REQUEST_DELAY
            time.sleep(delay)

    elapsed = time.time() - start_time
    log.info(f"\n{'='*50}")
    log.info(f"RESULTS")
    log.info(f"Time: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    log.info(f"Teams OK: {results['ok']}/{len(team_ids)}")
    log.info(f"Teams Failed: {results['failed']}")
    log.info(f"Total Batters: {results['total_batting']}")
    log.info(f"Total Pitchers: {results['total_pitching']}")
    if progress['failed']:
        log.info(f"Failed: {', '.join(progress['failed'])}")

    db.close()


if __name__ == '__main__':
    main()
