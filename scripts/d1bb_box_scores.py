#!/usr/bin/env python3
"""
D1Baseball Box Score Scraper

Scrapes box scores from D1Baseball.com -> SIDEARM athletics sites.
Collects per-game batting and pitching stats for all D1 games.

Usage:
    python3 scripts/d1bb_box_scores.py --date 2026-02-13
    python3 scripts/d1bb_box_scores.py --yesterday
"""

import argparse
import json
import sys
import time
import sqlite3
import re
import logging
from pathlib import Path
from datetime import datetime, timedelta

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("Error: playwright not installed. Run: pip install playwright && playwright install chromium")
    sys.exit(1)

PROJECT_DIR = Path(__file__).parent.parent
DB_PATH = PROJECT_DIR / 'data' / 'baseball.db'

# Force unbuffered output
class FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s', 
    datefmt='%H:%M:%S',
    handlers=[FlushHandler(sys.stdout)]
)
log = logging.getLogger('d1bb_box_scores')

# Team name mapping from D1Baseball/SIDEARM display names to our DB IDs
TEAM_MAPPING = {
    # SEC
    'alabama': 'alabama',
    'arkansas': 'arkansas',
    'auburn': 'auburn',
    'florida': 'florida',
    'georgia': 'georgia',
    'kentucky': 'kentucky',
    'lsu': 'lsu',
    'mississippi state': 'mississippi-state',
    'missouri': 'missouri',
    'oklahoma': 'oklahoma',
    'ole miss': 'ole-miss',
    'south carolina': 'south-carolina',
    'tennessee': 'tennessee',
    'texas': 'texas',
    'texas a&m': 'texas-am',
    'vanderbilt': 'vanderbilt',
    
    # ACC
    'boston college': 'boston-college',
    'clemson': 'clemson',
    'duke': 'duke',
    'florida state': 'florida-state',
    'georgia tech': 'georgia-tech',
    'louisville': 'louisville',
    'miami': 'miami-fl',
    'nc state': 'nc-state',
    'north carolina': 'north-carolina',
    'notre dame': 'notre-dame',
    'pittsburgh': 'pittsburgh',
    'virginia': 'virginia',
    'virginia tech': 'virginia-tech',
    'wake forest': 'wake-forest',
    'california': 'california',
    'stanford': 'stanford',
    
    # Big 12
    'arizona': 'arizona',
    'arizona state': 'arizona-state',
    'baylor': 'baylor',
    'byu': 'byu',
    'cincinnati': 'cincinnati',
    'colorado': 'colorado',
    'houston': 'houston',
    'iowa state': 'iowa-state',
    'kansas': 'kansas',
    'kansas state': 'kansas-state',
    'oklahoma state': 'oklahoma-state',
    'tcu': 'tcu',
    'texas tech': 'texas-tech',
    'ucf': 'ucf',
    'utah': 'utah',
    'west virginia': 'west-virginia',
    
    # Big Ten
    'illinois': 'illinois',
    'indiana': 'indiana',
    'iowa': 'iowa',
    'maryland': 'maryland',
    'michigan': 'michigan',
    'michigan state': 'michigan-state',
    'minnesota': 'minnesota',
    'nebraska': 'nebraska',
    'northwestern': 'northwestern',
    'ohio state': 'ohio-state',
    'oregon': 'oregon',
    'oregon state': 'oregon-state',
    'penn state': 'penn-state',
    'purdue': 'purdue',
    'rutgers': 'rutgers',
    'ucla': 'ucla',
    'usc': 'usc',
    'southern california': 'usc',
    'washington': 'washington',
    'wisconsin': 'wisconsin',
    
    # Sun Belt
    'appalachian state': 'appalachian-state',
    'coastal carolina': 'coastal-carolina',
    'georgia southern': 'georgia-southern',
    'georgia state': 'georgia-state',
    'james madison': 'james-madison',
    'louisiana': 'louisiana',
    'marshall': 'marshall',
    'old dominion': 'old-dominion',
    'south alabama': 'south-alabama',
    'southern miss': 'southern-miss',
    'texas state': 'texas-state',
    'troy': 'troy',
    'ul monroe': 'ul-monroe',
    'little rock': 'little-rock',
    'arkansas state': 'arkansas-state',
    
    # American
    'charlotte': 'charlotte',
    'east carolina': 'east-carolina',
    'florida atlantic': 'florida-atlantic',
    'memphis': 'memphis',
    'rice': 'rice',
    'south florida': 'south-florida',
    'tulane': 'tulane',
    'uab': 'uab',
    'utsa': 'utsa',
    'wichita state': 'wichita-state',
    
    # Others
    'air force': 'air-force',
    'army': 'army',
    'army west point': 'army',
    'army-west-point': 'army',
    'ball state': 'ball-state',
    'bowling green': 'bowling-green',
    'bryant': 'bryant',
    'butler': 'butler',
    'cal poly': 'cal-poly',
    'cal state fullerton': 'cal-state-fullerton',
    'cal state northridge': 'cal-state-northridge',
    'campbell': 'campbell',
    'central michigan': 'central-michigan',
    'college of charleston': 'college-of-charleston',
    'connecticut': 'connecticut',
    'creighton': 'creighton',
    'dallas baptist': 'dallas-baptist',
    'davidson': 'davidson',
    'dayton': 'dayton',
    'delaware': 'delaware',
    'elon': 'elon',
    'evansville': 'evansville',
    'fairfield': 'fairfield',
    'florida gulf coast': 'fgcu',
    'fgcu': 'fgcu',
    'fordham': 'fordham',
    'fresno state': 'fresno-state',
    'gardner-webb': 'gardner-webb',
    'george mason': 'george-mason',
    'george washington': 'george-washington',
    'georgetown': 'georgetown',
    'gonzaga': 'gonzaga',
    'grambling': 'grambling-state',
    'grand canyon': 'grand-canyon',
    'hawaii': 'hawaii',
    'high point': 'high-point',
    'hofstra': 'hofstra',
    'holy cross': 'holy-cross',
    'houston christian': 'houston-christian',
    'incarnate word': 'incarnate-word',
    'iona': 'iona',
    'jacksonville': 'jacksonville',
    'jacksonville state': 'jacksonville-state',
    'kennesaw state': 'kennesaw-state',
    'kent state': 'kent-state',
    'la salle': 'la-salle',
    'lamar': 'lamar',
    'lehigh': 'lehigh',
    'liberty': 'liberty',
    'lipscomb': 'lipscomb',
    'long beach state': 'long-beach-state',
    'longwood': 'longwood',
    'louisiana tech': 'louisiana-tech',
    'loyola marymount': 'loyola-marymount',
    'maine': 'maine',
    'marist': 'marist',
    'mcneese': 'mcneese',
    'mercer': 'mercer',
    'miami (oh)': 'miami-oh',
    'middle tennessee': 'middle-tennessee',
    'milwaukee': 'milwaukee',
    'missouri state': 'missouri-state',
    'monmouth': 'monmouth',
    'morehead state': 'morehead-state',
    'murray state': 'murray-state',
    'navy': 'navy',
    'nevada': 'nevada',
    'new jersey tech': 'njit',
    'njit': 'njit',
    'new mexico': 'new-mexico',
    'new mexico state': 'new-mexico-state',
    'new orleans': 'new-orleans',
    'nicholls': 'nicholls',
    'north alabama': 'north-alabama',
    'north dakota state': 'north-dakota-state',
    'north florida': 'north-florida',
    'northern colorado': 'northern-colorado',
    'northern illinois': 'northern-illinois',
    'omaha': 'omaha',
    'oral roberts': 'oral-roberts',
    'pacific': 'pacific',
    'pepperdine': 'pepperdine',
    'portland': 'portland',
    'prairie view': 'prairie-view',
    'presbyterian': 'presbyterian',
    'queens (nc)': 'queens',
    'queens': 'queens',
    'radford': 'radford',
    'rhode island': 'rhode-island',
    'richmond': 'richmond',
    'rider': 'rider',
    'sacramento state': 'sacramento-state',
    'sacred heart': 'sacred-heart',
    'saint joseph\'s': 'saint-josephs',
    'saint louis': 'saint-louis',
    'saint mary\'s': 'saint-marys',
    'saint mary\'s (ca)': 'saint-marys',
    'sam houston state': 'sam-houston',
    'sam houston': 'sam-houston',
    'samford': 'samford',
    'san diego': 'san-diego',
    'san diego state': 'san-diego-state',
    'san francisco': 'san-francisco',
    'san jose state': 'san-jose-state',
    'santa clara': 'santa-clara',
    'seattle': 'seattle',
    'seton hall': 'seton-hall',
    'siena': 'siena',
    'siu edwardsville': 'siue',
    'siue': 'siue',
    'south dakota state': 'south-dakota-state',
    'southeast missouri': 'southeast-missouri',
    'southeastern louisiana': 'southeastern-louisiana',
    'southern': 'southern',
    'southern illinois': 'southern-illinois',
    'southern indiana': 'southern-indiana',
    'st. bonaventure': 'st-bonaventure',
    'st. john\'s': 'st-johns',
    'stetson': 'stetson',
    'stony brook': 'stony-brook',
    'tarleton state': 'tarleton-state',
    'tennessee tech': 'tennessee-tech',
    'tennessee-martin': 'ut-martin',
    'ut martin': 'ut-martin',
    'the citadel': 'citadel',
    'citadel': 'citadel',
    'toledo': 'toledo',
    'towson': 'towson',
    'uc davis': 'uc-davis',
    'uc irvine': 'uc-irvine',
    'uc riverside': 'uc-riverside',
    'uc san diego': 'uc-san-diego',
    'uc santa barbara': 'uc-santa-barbara',
    'umass': 'umass',
    'umass lowell': 'umass-lowell',
    'umes': 'maryland-eastern-shore',
    'unc asheville': 'unc-asheville',
    'unc greensboro': 'unc-greensboro',
    'unc wilmington': 'unc-wilmington',
    'unlv': 'unlv',
    'usc upstate': 'usc-upstate',
    'ut rio grande valley': 'utrgv',
    'utrgv': 'utrgv',
    'utah tech': 'utah-tech',
    'utah valley': 'utah-valley',
    'valparaiso': 'valparaiso',
    'vcu': 'vcu',
    'villanova': 'villanova',
    'wagner': 'wagner',
    'western carolina': 'western-carolina',
    'western illinois': 'western-illinois',
    'western kentucky': 'western-kentucky',
    'western michigan': 'western-michigan',
    'william & mary': 'william-mary',
    'william and mary': 'william-mary',
    'winthrop': 'winthrop',
    'wofford': 'wofford',
    'wright state': 'wright-state',
    'xavier': 'xavier',
    'youngstown state': 'youngstown-state',
    'akron': 'akron',
    'alabama state': 'alabama-state',
    'alcorn state': 'alcorn-state',
    'austin peay': 'austin-peay',
    'bellarmine': 'bellarmine',
    'bucknell': 'bucknell',
    'california baptist': 'california-baptist',
    'central arkansas': 'central-arkansas',
    'charleston southern': 'charleston-southern',
    'coppin state': 'coppin-state',
    'csu bakersfield': 'csu-bakersfield',
    'eastern kentucky': 'eastern-kentucky',
    'florida a&m': 'florida-am',
    'florida international': 'fiu',
    'fiu': 'fiu',
    'jackson state': 'jackson-state',
    'texas a&m-corpus christi': 'texas-am-corpus-christi',
    'texas southern': 'texas-southern',
    'sfa': 'stephen-f-austin',
    'stephen f. austin': 'stephen-f-austin',
    'semo': 'southeast-missouri',
    'vmi': 'vmi',
    'etsu': 'etsu',
    'nccu': 'north-carolina-central',
    'a&m-corpus christi': 'texas-am-corpus-christi',
    'corpus christi': 'texas-am-corpus-christi',
    'indiana state': 'indiana-state',
    'indiana st': 'indiana-state',
    'ttu': 'texas-tech',
    'vu': 'vanderbilt',
    'unc': 'north-carolina',
    'osu': 'ohio-state',
    'msu': 'mississippi-state',
    'lsu tigers': 'lsu',
    'fsu': 'florida-state',
    'usc trojans': 'usc',
    'liu': 'liu',
    'uapb': 'arkansas-pine-bluff',
    'arkansas-pine bluff': 'arkansas-pine-bluff',
    'northern kentucky': 'northern-kentucky',
    'eastern illinois': 'eastern-illinois',
    'southern university': 'southern',
    'grambling state': 'grambling-state',
    'southeast missouri state': 'southeast-missouri',
    'abilene christian': 'abilene-christian',
}

# Request delay between box score pages (seconds)
BOX_SCORE_DELAY = 15


def get_db():
    """Get database connection."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def normalize_team_name(name):
    """Normalize team name for matching."""
    if not name:
        return None
    name = name.lower().strip()
    # Remove rank prefix (e.g., "21Wake Forest" -> "wake forest")
    name = re.sub(r'^\d+\s*', '', name)
    # Clean up
    name = name.strip()
    
    # Try direct mapping
    if name in TEAM_MAPPING:
        return TEAM_MAPPING[name]
    
    # Try without special characters
    name_clean = re.sub(r'[^\w\s]', '', name)
    if name_clean in TEAM_MAPPING:
        return TEAM_MAPPING[name_clean]
    
    # Try slug-style
    slug = name.replace(' ', '-').replace('_', '-')
    slug = re.sub(r'[^\w-]', '', slug)
    return slug


def get_box_score_urls(page, date_str):
    """
    Get all box score URLs from D1Baseball scores page.
    Returns tuple: (list of (url, type) tuples, dict of game_info from page).
    type is 'sidearm' or 'statbroadcast'.
    game_info dict maps box_score_url -> {'away_team': name, 'home_team': name, 'away_score': int, 'home_score': int}
    """
    url = f"https://d1baseball.com/scores/?date={date_str}"
    log.info(f"Navigating to {url}")
    
    try:
        page.goto(url, timeout=60000, wait_until='domcontentloaded')
        page.wait_for_timeout(5000)  # Wait for JS to render
    except Exception as e:
        log.error(f"Failed to navigate to D1Baseball: {e}")
        return [], {}
    
    # Extract box score URLs via JavaScript
    try:
        urls = page.evaluate("""
            () => {
                return Array.from(document.querySelectorAll('a'))
                    .filter(a => a.textContent.includes('Box Score'))
                    .map(a => a.href);
            }
        """)
    except Exception as e:
        log.error(f"Failed to extract box score URLs: {e}")
        return [], {}
    
    # Extract game info (R/H/E) from page - simple approach
    game_info = {}
    try:
        # Get all game containers and extract team/score info
        game_data = page.evaluate("""
            () => {
                const games = [];
                // Find game score cards
                const cards = document.querySelectorAll('.score-card, [class*="game-card"], article');
                cards.forEach(card => {
                    const boxLink = card.querySelector('a[href*="boxscore"], a[href*="Box"]');
                    if (!boxLink) return;
                    
                    const url = boxLink.href;
                    const text = card.innerText;
                    
                    // Look for team-score patterns in text
                    const lines = text.split(/[\\n\\r]+/).map(l => l.trim()).filter(l => l);
                    const teamScores = [];
                    
                    for (const line of lines) {
                        // Match: "TeamName  5" or "TeamName 12" at end of line
                        const match = line.match(/^([A-Za-z][A-Za-z0-9\\s\\.\\-\\'&]+?)\\s+(\\d{1,2})\\s*$/);
                        if (match) {
                            teamScores.push({ team: match[1].trim(), score: parseInt(match[2]) });
                        }
                    }
                    
                    if (teamScores.length >= 2) {
                        games.push({
                            url: url,
                            away_team: teamScores[0].team,
                            away_score: teamScores[0].score,
                            home_team: teamScores[1].team,
                            home_score: teamScores[1].score
                        });
                    }
                });
                return games;
            }
        """)
        
        for g in game_data:
            game_info[g['url']] = {
                'away_team': g['away_team'],
                'home_team': g['home_team'],
                'away_score': g['away_score'],
                'home_score': g['home_score']
            }
    except Exception as e:
        log.warning(f"Could not extract R/H/E from D1Baseball page: {e}")
        # Non-critical - continue without game info
    
    # Categorize and dedupe
    box_score_urls = []
    seen = set()
    sidearm_count = 0
    statbroadcast_count = 0
    
    for url in urls:
        # Skip duplicates
        if url in seen:
            continue
        seen.add(url)
        
        # Categorize by URL
        if 'statbroadcast.com' in url:
            box_score_urls.append((url, 'statbroadcast'))
            statbroadcast_count += 1
        else:
            box_score_urls.append((url, 'sidearm'))
            sidearm_count += 1
    
    log.info(f"Found {len(box_score_urls)} unique box score URLs: {sidearm_count} SIDEARM, {statbroadcast_count} StatBroadcast")
    return box_score_urls, game_info


def parse_box_score_text(text):
    """
    Parse box score text from SIDEARM page.
    
    Format quirks:
    - Team name on its own line (e.g., "WAKE FOREST")
    - "Player" on next line (header start)
    - Player data has name on one line, stats (starting with \t) on next line
    
    Returns dict with teams, scores, batting and pitching stats.
    """
    result = {
        'away_team': None,
        'home_team': None,
        'away_score': None,
        'home_score': None,
        'date': None,
        'batting': {},  # team_id -> list of player stats
        'pitching': {},  # team_id -> list of player stats
    }
    
    lines = text.split('\n')
    
    # Find the "vs" line to get team names
    vs_pattern = re.compile(r'^(.+?)\s+vs\s+(.+?)$', re.IGNORECASE)
    for line in lines:
        match = vs_pattern.match(line.strip())
        if match:
            result['away_team'] = match.group(1).strip()
            result['home_team'] = match.group(2).strip()
            break
    
    # Find date
    date_pattern = re.compile(r'(\d{1,2}/\d{1,2}/\d{4})')
    for line in lines:
        match = date_pattern.search(line)
        if match:
            result['date'] = match.group(1)
            break
    
    # Parse batting and pitching tables
    current_team = None
    current_section = None  # 'batting' or 'pitching'
    pending_player = None  # Player name waiting for stats line
    
    i = 0
    while i < len(lines):
        raw_line = lines[i]  # Keep original for tab detection
        line = raw_line.strip()
        
        # Check for team name + "Player" pattern (start of stats section)
        if line and not line.startswith('TEAM') and not line.startswith('BATTING') and not line.startswith('BASERUNNING') and not line.startswith('FIELDING'):
            # Look for: TeamName -> "Player" -> next line determines batting vs pitching
            if i + 2 < len(lines) and lines[i + 1].strip() == 'Player':
                # Check what comes after "Player" (skip tabs/empty lines)
                j = i + 2
                while j < len(lines) and (lines[j].strip() == '' or lines[j].strip() == '\t'):
                    j += 1
                
                if j < len(lines):
                    indicator = lines[j].strip()
                    if indicator == 'Pos' or indicator.startswith('Pos'):
                        # This is batting section
                        current_team = normalize_team_name(line)
                        current_section = 'batting'
                        pending_player = None
                        if current_team:
                            result['batting'][current_team] = []
                        # Skip to after "LOB" header
                        while i < len(lines) and 'LOB' not in lines[i]:
                            i += 1
                        i += 1
                        continue
                    elif indicator == 'IP' or indicator.startswith('IP'):
                        # This is pitching section
                        current_team = normalize_team_name(line)
                        current_section = 'pitching'
                        pending_player = None
                        if current_team:
                            result['pitching'][current_team] = []
                        # Skip to after "NP" header
                        while i < len(lines) and 'NP' not in lines[i]:
                            i += 1
                        i += 1
                        continue
        
        # End of section markers
        if line in ('BATTING', 'BASERUNNING', 'FIELDING') or line.startswith('Win:') or line.startswith('Loss:'):
            current_section = None
            pending_player = None
            i += 1
            continue
        
        if line.startswith('Totals'):
            pending_player = None
            i += 1
            continue
        
        # Parse batting: name line -> stats line (starts with \t)
        if current_section == 'batting' and current_team:
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # If we have a pending player and this line starts with tab = stats
            if pending_player and raw_line.startswith('\t'):
                parts = raw_line.split('\t')
                # parts[0] is empty (before first tab), so: [empty, Pos, AB, R, H, RBI, BB, IBB, SO, LOB]
                if len(parts) >= 10:
                    try:
                        pos = parts[1].strip()
                        ab = safe_int(parts[2])
                        r = safe_int(parts[3])
                        h = safe_int(parts[4])
                        rbi = safe_int(parts[5])
                        bb = safe_int(parts[6])
                        ibb = safe_int(parts[7])
                        so = safe_int(parts[8])
                        lob = safe_int(parts[9])
                        
                        # Skip pitchers in lineup (all empty stats)
                        if ab > 0 or bb > 0 or r > 0:
                            stat = {
                                'player_name': pending_player,
                                'position': pos,
                                'ab': ab, 'r': r, 'h': h, 'rbi': rbi,
                                'bb': bb, 'ibb': ibb, 'so': so, 'lob': lob,
                            }
                            result['batting'][current_team].append(stat)
                    except (ValueError, IndexError):
                        pass
                pending_player = None
                i += 1
                continue
            
            # Otherwise this is a player name line
            if not raw_line.startswith('\t') and line not in ('BATTING', 'BASERUNNING', 'FIELDING', 'Totals'):
                pending_player = line
                i += 1
                continue
        
        # Parse pitching: name line -> stats line (starts with number or \t)
        if current_section == 'pitching' and current_team:
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # If we have pending player and this line has stats
            if pending_player:
                # Pitching stats line starts with IP (like "2.1" or "4.0")
                # or sometimes starts with \t
                parts = raw_line.split('\t')
                
                # Check if first part looks like IP
                first = parts[0].strip()
                if first and (first[0].isdigit() or first == ''):
                    try:
                        # If starts with tab, shift indices
                        offset = 1 if first == '' else 0
                        ip = safe_float(parts[0 + offset])
                        h = safe_int(parts[1 + offset])
                        r = safe_int(parts[2 + offset])
                        er = safe_int(parts[3 + offset])
                        bb = safe_int(parts[4 + offset])
                        so = safe_int(parts[5 + offset])
                        wp = safe_int(parts[6 + offset]) if len(parts) > 6 + offset else 0
                        bk = safe_int(parts[7 + offset]) if len(parts) > 7 + offset else 0
                        hbp = safe_int(parts[8 + offset]) if len(parts) > 8 + offset else 0
                        ibb_p = safe_int(parts[9 + offset]) if len(parts) > 9 + offset else 0
                        ab = safe_int(parts[10 + offset]) if len(parts) > 10 + offset else 0
                        bf = safe_int(parts[11 + offset]) if len(parts) > 11 + offset else 0
                        fo = safe_int(parts[12 + offset]) if len(parts) > 12 + offset else 0
                        go = safe_int(parts[13 + offset]) if len(parts) > 13 + offset else 0
                        np = safe_int(parts[14 + offset]) if len(parts) > 14 + offset else 0
                        
                        stat = {
                            'player_name': pending_player,
                            'ip': ip, 'h': h, 'r': r, 'er': er,
                            'bb': bb, 'so': so, 'wp': wp, 'bk': bk,
                            'hbp': hbp, 'ibb': ibb_p, 'ab': ab, 'bf': bf,
                            'fo': fo, 'go': go, 'np': np,
                        }
                        result['pitching'][current_team].append(stat)
                    except (ValueError, IndexError):
                        pass
                    pending_player = None
                    i += 1
                    continue
            
            # Otherwise check if this is a player name line
            if not line.startswith('\t') and not line[0].isdigit() and line not in ('Totals',):
                pending_player = line.strip()
                i += 1
                continue
        
        i += 1
    
    return result


def safe_int(val, default=0):
    """Safely convert to int."""
    if not val or val == '-' or str(val).strip() == '':
        return default
    try:
        return int(str(val).strip())
    except (ValueError, TypeError):
        return default


def safe_float(val, default=0.0):
    """Safely convert to float."""
    if not val or val == '-' or str(val).strip() == '':
        return default
    try:
        return float(str(val).strip())
    except (ValueError, TypeError):
        return default


def find_game_id(db, team1_id, team2_id, date_str):
    """
    Find game_id in our database matching these teams and date.
    Returns game_id or None.
    """
    # Convert date format if needed (M/D/YYYY -> YYYY-MM-DD)
    if '/' in date_str:
        parts = date_str.split('/')
        if len(parts) == 3:
            month, day, year = parts
            date_str = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    
    cursor = db.cursor()
    
    # Try both orders (home/away)
    cursor.execute("""
        SELECT id FROM games 
        WHERE date = ? 
        AND ((home_team_id = ? AND away_team_id = ?) OR (home_team_id = ? AND away_team_id = ?))
    """, (date_str, team1_id, team2_id, team2_id, team1_id))
    
    row = cursor.fetchone()
    return row['id'] if row else None


def scrape_box_score(page, url, retry_on_fail=True):
    """
    Scrape a single SIDEARM box score page.
    Returns parsed box score data or None on failure.
    """
    log.info(f"  Scraping SIDEARM: {url[:80]}...")
    
    def attempt_scrape(wait_time):
        try:
            page.goto(url, timeout=45000, wait_until='domcontentloaded')
            page.wait_for_timeout(wait_time)
        except Exception as e:
            log.warning(f"    Failed to navigate: {e}")
            return None
        
        # Get page text
        try:
            text = page.evaluate("() => document.body.innerText")
        except Exception as e:
            log.warning(f"    Failed to extract page text: {e}")
            return None
        
        if not text or len(text) < 500:
            log.warning(f"    Page text too short ({len(text) if text else 0} chars)")
            return None
        
        # Parse the text
        box_score = parse_box_score_text(text)
        
        # Validate we got something
        batting_count = sum(len(stats) for stats in box_score['batting'].values())
        pitching_count = sum(len(stats) for stats in box_score['pitching'].values())
        
        if batting_count == 0 and pitching_count == 0:
            return None
        
        return box_score
    
    # First attempt with 3 second wait
    result = attempt_scrape(3000)
    
    if result:
        batting_count = sum(len(stats) for stats in result['batting'].values())
        pitching_count = sum(len(stats) for stats in result['pitching'].values())
        log.info(f"    Parsed: {len(result['batting'])} teams batting ({batting_count} players), {len(result['pitching'])} teams pitching ({pitching_count} players)")
        return result
    
    # Retry with longer wait if first attempt failed
    if retry_on_fail:
        log.info(f"    Retrying with 8 second wait...")
        result = attempt_scrape(8000)
        if result:
            batting_count = sum(len(stats) for stats in result['batting'].values())
            pitching_count = sum(len(stats) for stats in result['pitching'].values())
            log.info(f"    Parsed (retry): {len(result['batting'])} teams batting ({batting_count} players), {len(result['pitching'])} teams pitching ({pitching_count} players)")
            return result
    
    log.warning(f"    No stats parsed from SIDEARM page (failed after retry)")
    return None


def scrape_statbroadcast_box_score(page, url):
    """
    Scrape a StatBroadcast box score page using the full box score view.
    
    StatBroadcast pages require clicking "Home Stats" and "Visitor Stats" buttons
    to access full batting and pitching stats with all columns (AB, R, H, RBI, BB, K, etc.).
    
    Returns parsed box score data in same format as scrape_box_score() or None on failure.
    """
    log.info(f"  Scraping StatBroadcast: {url[:80]}...")
    
    result = {
        'away_team': None,
        'home_team': None,
        'away_score': None,
        'home_score': None,
        'date': None,
        'batting': {},
        'pitching': {},
    }
    
    try:
        page.goto(url, timeout=45000, wait_until='domcontentloaded')
        
        # Wait for page to fully load
        try:
            page.wait_for_selector('button:has-text("Home Stats")', timeout=10000)
        except:
            log.warning(f"    Home Stats button not found")
            pass
        
        # Give extra time for dynamic content
        page.wait_for_timeout(3000)
        
    except Exception as e:
        log.warning(f"    Failed to navigate: {e}")
        return None
    
    # Parse team names and scores from title: "AWAY #, HOME # - Final" or "AWAY vs. HOME"
    try:
        title = page.title()
        # Try "Texas Tech 3, Vanderbilt 13 - Final" format
        title_match = re.match(r'([A-Za-z\s\'\.\-&#]+?)\s*(\d+),\s*([A-Za-z\s\'\.\-&#]+?)\s*(\d+)\s*-\s*(?:Final|F)', title, re.IGNORECASE)
        if title_match:
            away_name = title_match.group(1).strip()
            result['away_score'] = int(title_match.group(2))
            home_name = title_match.group(3).strip()
            result['home_score'] = int(title_match.group(4))
            
            result['away_team'] = normalize_team_name(away_name)
            result['home_team'] = normalize_team_name(home_name)
            log.info(f"    Title: {away_name} {result['away_score']} @ {home_name} {result['home_score']}")
        else:
            # Try "Texas Tech vs. Vanderbilt" format (for in-progress games)
            vs_match = re.match(r'([A-Za-z\s\'\.\-&#]+?)\s+vs\.?\s+([A-Za-z\s\'\.\-&#]+)', title, re.IGNORECASE)
            if vs_match:
                away_name = vs_match.group(1).strip()
                home_name = vs_match.group(2).strip()
                result['away_team'] = normalize_team_name(away_name)
                result['home_team'] = normalize_team_name(home_name)
                log.info(f"    Title: {away_name} @ {home_name}")
    except Exception as e:
        log.warning(f"    Failed to parse title: {e}")
    
    # Click "Home Stats" button to get home team stats
    try:
        home_button = page.locator('button:has-text("Home Stats")')
        if home_button.count() > 0:
            home_button.click()
            page.wait_for_timeout(1500)  # Wait for view to load
            
            # Parse home team box score
            home_batting, home_pitching, home_team_name = _parse_statbroadcast_stats_view(page)
            
            # Use team name from table header if we didn't get it from title
            if home_team_name:
                home_team_id = normalize_team_name(home_team_name)
                if home_team_id:
                    result['home_team'] = home_team_id
            
            if result['home_team']:
                result['batting'][result['home_team']] = home_batting
                result['pitching'][result['home_team']] = home_pitching
    except Exception as e:
        log.warning(f"    Failed to parse Home Stats: {e}")
    
    # Click "Visitor Stats" button to get visitor team stats
    try:
        visitor_button = page.locator('button:has-text("Visitor Stats")')
        if visitor_button.count() > 0:
            visitor_button.click()
            page.wait_for_timeout(1500)  # Wait for view to load
            
            # Parse visitor team box score
            away_batting, away_pitching, away_team_name = _parse_statbroadcast_stats_view(page)
            
            # Use team name from table header if we didn't get it from title
            if away_team_name:
                away_team_id = normalize_team_name(away_team_name)
                if away_team_id:
                    result['away_team'] = away_team_id
            
            if result['away_team']:
                result['batting'][result['away_team']] = away_batting
                result['pitching'][result['away_team']] = away_pitching
    except Exception as e:
        log.warning(f"    Failed to parse Visitor Stats: {e}")
    
    # Try to get scores from line score if not in title
    if result['away_score'] is None or result['home_score'] is None:
        try:
            # Look for line score table
            page_text = page.inner_text('body')
            # Match pattern like "TTU ... 3 6 2" and "VU ... 13 12 2" at end of line score
            lines = page_text.split('\n')
            for line in lines:
                # Look for R H E L pattern at end
                parts = line.strip().split()
                if len(parts) >= 4:
                    # Check if last 4 parts are numbers (R, H, E, L)
                    try:
                        nums = [int(p) for p in parts[-4:]]
                        # First number is runs
                        if result['away_score'] is None:
                            result['away_score'] = nums[0]
                        elif result['home_score'] is None:
                            result['home_score'] = nums[0]
                    except ValueError:
                        pass
        except:
            pass
    
    # Validate we got something
    batting_count = sum(len(stats) for stats in result['batting'].values())
    pitching_count = sum(len(stats) for stats in result['pitching'].values())
    
    if batting_count == 0 and pitching_count == 0:
        log.warning(f"    No stats parsed from StatBroadcast page")
        return None
    
    log.info(f"    Parsed: {len(result['batting'])} teams batting ({batting_count} players), {len(result['pitching'])} teams pitching ({pitching_count} players)")
    return result


def _parse_statbroadcast_stats_view(page):
    """
    Parse batting and pitching stats from the current StatBroadcast stats view.
    
    Returns: (batting_list, pitching_list, team_name)
    """
    batting = []
    pitching = []
    team_name = None
    
    try:
        # Get all tables on the page
        tables = page.locator('table').all()
        
        for table in tables:
            try:
                # Get header row to identify table type
                header_row = table.locator('tr').first
                header_text = header_row.inner_text().strip()
                
                # Check if this is a batting table (has AB, R, H, RBI columns)
                if 'AB' in header_text and 'RBI' in header_text and 'PLAYER' in header_text:
                    # Extract team name from "XXX Box Score" label above table
                    try:
                        # The table is preceded by a div with team name
                        prev_sibling = table.locator('xpath=./preceding-sibling::*[1]')
                        if prev_sibling.count() > 0:
                            label_text = prev_sibling.inner_text()
                            # Extract team name from "Vanderbilt Box Score" or "Texas Tech Box Score"
                            box_match = re.match(r'(.+?)\s+Box Score', label_text.strip())
                            if box_match:
                                team_name = box_match.group(1).strip()
                    except:
                        pass
                    
                    # Parse batting rows
                    rows = table.locator('tbody tr').all()
                    for row in rows:
                        try:
                            cells = row.locator('td').all()
                            if len(cells) < 10:
                                continue
                            
                            # Get cell texts
                            cell_texts = [c.inner_text().strip() for c in cells]
                            
                            # Skip "Team Batting Totals" row
                            if 'Totals' in cell_texts[0] or 'Team' in cell_texts[0]:
                                continue
                            
                            # Expected columns: POS, #, PLAYER, AB, R, H, RBI, 2B, 3B, HR, BB, K, ...
                            # Find column indices from header
                            pos = cell_texts[0] if len(cell_texts) > 0 else ''
                            player_name = cell_texts[2] if len(cell_texts) > 2 else ''
                            
                            # Skip pitchers in batting lineup (pos = 'p' and all stats are '-')
                            if pos.lower() == 'p' and cell_texts[3] == '-':
                                continue
                            
                            # Parse stats (handle '-' as 0)
                            ab = safe_int(cell_texts[3]) if len(cell_texts) > 3 else 0
                            r = safe_int(cell_texts[4]) if len(cell_texts) > 4 else 0
                            h = safe_int(cell_texts[5]) if len(cell_texts) > 5 else 0
                            rbi = safe_int(cell_texts[6]) if len(cell_texts) > 6 else 0
                            bb = safe_int(cell_texts[10]) if len(cell_texts) > 10 else 0
                            so = safe_int(cell_texts[11]) if len(cell_texts) > 11 else 0
                            
                            # Skip empty players
                            if not player_name or (ab == 0 and r == 0 and h == 0 and bb == 0):
                                continue
                            
                            stat = {
                                'player_name': player_name,
                                'position': pos,
                                'ab': ab,
                                'r': r,
                                'h': h,
                                'rbi': rbi,
                                'bb': bb,
                                'ibb': 0,
                                'so': so,
                                'lob': 0,
                            }
                            batting.append(stat)
                        except Exception as e:
                            continue
                
                # Check if this is a pitching table (has IP, H, R, ER, BB, K columns)
                elif 'IP' in header_text and 'ER' in header_text and 'Player' in header_text:
                    # Parse pitching rows
                    rows = table.locator('tbody tr').all()
                    for row in rows:
                        try:
                            cells = row.locator('td').all()
                            if len(cells) < 8:
                                continue
                            
                            cell_texts = [c.inner_text().strip() for c in cells]
                            
                            # Skip "Team Pitching Totals" row
                            if 'Totals' in cell_texts[0] or 'Team' in cell_texts[0]:
                                continue
                            
                            # Expected columns: #, Player, Dec, IP, H, R, ER, BB, K, ...
                            player_name = cell_texts[1] if len(cell_texts) > 1 else ''
                            
                            # Parse stats
                            ip = safe_float(cell_texts[3]) if len(cell_texts) > 3 else 0.0
                            h = safe_int(cell_texts[4]) if len(cell_texts) > 4 else 0
                            r = safe_int(cell_texts[5]) if len(cell_texts) > 5 else 0
                            er = safe_int(cell_texts[6]) if len(cell_texts) > 6 else 0
                            bb = safe_int(cell_texts[7]) if len(cell_texts) > 7 else 0
                            so = safe_int(cell_texts[8]) if len(cell_texts) > 8 else 0
                            
                            # Get pitch count if available (TP column, varies by page)
                            np = 0
                            for i, t in enumerate(cell_texts):
                                if i > 15 and t.isdigit() and int(t) > 10:
                                    np = int(t)
                                    break
                            
                            # Skip empty pitchers
                            if not player_name:
                                continue
                            
                            stat = {
                                'player_name': player_name,
                                'ip': ip,
                                'h': h,
                                'r': r,
                                'er': er,
                                'bb': bb,
                                'so': so,
                                'wp': 0,
                                'bk': 0,
                                'hbp': 0,
                                'ibb': 0,
                                'ab': 0,
                                'bf': 0,
                                'fo': 0,
                                'go': 0,
                                'np': np,
                            }
                            pitching.append(stat)
                        except Exception as e:
                            continue
            except:
                continue
    except Exception as e:
        log.warning(f"    Failed to parse stats tables: {e}")
    
    return batting, pitching, team_name


def store_box_score(db, game_id, box_score):
    """
    Store box score data in database.
    Returns (batting_count, pitching_count).
    """
    cursor = db.cursor()
    batting_count = 0
    pitching_count = 0
    
    # Store batting stats
    for team_id, players in box_score['batting'].items():
        for stat in players:
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO game_batting_stats (
                        game_id, team_id, player_name, position,
                        at_bats, runs, hits, rbi, home_runs, walks, strikeouts,
                        stolen_bases, batting_avg, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, 0, NULL, CURRENT_TIMESTAMP)
                """, (
                    game_id, team_id, stat['player_name'], stat.get('position'),
                    stat['ab'], stat['r'], stat['h'], stat['rbi'],
                    stat['bb'], stat['so']
                ))
                batting_count += 1
            except sqlite3.Error as e:
                log.warning(f"      Failed to insert batting: {stat['player_name']} - {e}")
    
    # Store pitching stats
    for team_id, players in box_score['pitching'].items():
        for stat in players:
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO game_pitching_stats (
                        game_id, team_id, player_name,
                        innings_pitched, hits_allowed, runs_allowed, earned_runs,
                        walks, strikeouts, home_runs_allowed, pitches, strikes,
                        era, win, loss, save, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, 0, NULL, 0, 0, 0, CURRENT_TIMESTAMP)
                """, (
                    game_id, team_id, stat['player_name'],
                    stat['ip'], stat['h'], stat['r'], stat['er'],
                    stat['bb'], stat['so'], stat.get('np', 0)
                ))
                pitching_count += 1
            except sqlite3.Error as e:
                log.warning(f"      Failed to insert pitching: {stat['player_name']} - {e}")
    
    db.commit()
    return batting_count, pitching_count


def update_game_scores(db, game_id, home_score, away_score):
    """Update game scores in the games table (fallback when full box score fails)."""
    try:
        cursor = db.cursor()
        cursor.execute("""
            UPDATE games SET home_score = ?, away_score = ?
            WHERE id = ? AND (home_score IS NULL OR away_score IS NULL)
        """, (home_score, away_score, game_id))
        db.commit()
        return cursor.rowcount > 0
    except Exception as e:
        log.warning(f"    Failed to update game scores: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Scrape D1Baseball box scores')
    parser.add_argument('--date', help='Date (YYYY-MM-DD)')
    parser.add_argument('--yesterday', action='store_true')
    parser.add_argument('--headless', action='store_true', default=True, help='Run browser in headless mode')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of games (0=all)')
    args = parser.parse_args()
    
    if args.yesterday:
        date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    elif args.date:
        date = args.date
    else:
        parser.print_help()
        sys.exit(1)
    
    log.info(f"D1Baseball Box Score Scraper - {date}")
    log.info("=" * 50)
    
    # Launch browser
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=args.headless,
            args=['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage']
        )
        context = browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080}
        )
        page = context.new_page()
        
        # Get box score URLs and R/H/E info from D1Baseball
        url_list, game_info = get_box_score_urls(page, date)
        if not url_list:
            log.error("No box score URLs found")
            browser.close()
            sys.exit(1)
        
        # Apply limit if specified
        if args.limit > 0:
            url_list = url_list[:args.limit]
            log.info(f"Limited to {args.limit} games")
        
        db = get_db()
        
        # Stats tracking
        total_urls = len(url_list)
        success_count = 0
        failed_count = 0
        no_match_count = 0
        fallback_count = 0
        total_batting = 0
        total_pitching = 0
        failed_urls = []  # Track failed URLs for debugging
        
        # Track by type
        sidearm_success = 0
        sidearm_fail = 0
        statbroadcast_success = 0
        statbroadcast_fail = 0
        
        for i, (url, url_type) in enumerate(url_list, 1):
            log.info(f"[{i}/{total_urls}] Processing {url_type.upper()} box score...")
            
            # Scrape the page based on type
            if url_type == 'statbroadcast':
                box_score = scrape_statbroadcast_box_score(page, url)
            else:
                box_score = scrape_box_score(page, url, retry_on_fail=True)
            
            if not box_score:
                failed_count += 1
                failed_urls.append((url, url_type))
                
                if url_type == 'statbroadcast':
                    statbroadcast_fail += 1
                else:
                    sidearm_fail += 1
                
                # Try to use R/H/E fallback from D1Baseball page
                if url in game_info:
                    info = game_info[url]
                    away_team_id = normalize_team_name(info['away_team'])
                    home_team_id = normalize_team_name(info['home_team'])
                    
                    if away_team_id and home_team_id:
                        game_id = find_game_id(db, home_team_id, away_team_id, date)
                        if game_id and info.get('home_score') is not None and info.get('away_score') is not None:
                            if update_game_scores(db, game_id, info['home_score'], info['away_score']):
                                log.info(f"    Fallback: Updated scores for game {game_id} ({info['away_score']}-{info['home_score']})")
                                fallback_count += 1
                
                if i < total_urls:
                    time.sleep(BOX_SCORE_DELAY)
                continue
            
            # Try to find matching game in our DB
            team_ids = list(box_score['batting'].keys()) + list(box_score['pitching'].keys())
            team_ids = list(set([t for t in team_ids if t]))
            
            if len(team_ids) < 2:
                log.warning(f"    Could not identify both teams")
                no_match_count += 1
                if i < total_urls:
                    time.sleep(BOX_SCORE_DELAY)
                continue
            
            # Use the date from the box score or the requested date
            game_date = box_score.get('date') or date
            
            game_id = find_game_id(db, team_ids[0], team_ids[1], game_date)
            if not game_id:
                # Try with more team combinations
                found = False
                for t1 in team_ids:
                    for t2 in team_ids:
                        if t1 != t2:
                            game_id = find_game_id(db, t1, t2, game_date)
                            if game_id:
                                found = True
                                break
                    if found:
                        break
            
            if not game_id:
                log.warning(f"    No matching game found for {team_ids} on {game_date}")
                no_match_count += 1
                if i < total_urls:
                    time.sleep(BOX_SCORE_DELAY)
                continue
            
            # Store the stats
            batting_count, pitching_count = store_box_score(db, game_id, box_score)
            total_batting += batting_count
            total_pitching += pitching_count
            success_count += 1
            
            if url_type == 'statbroadcast':
                statbroadcast_success += 1
            else:
                sidearm_success += 1
            
            log.info(f"    Stored: {batting_count} batting + {pitching_count} pitching for game {game_id}")
            
            # Delay between requests
            if i < total_urls:
                time.sleep(BOX_SCORE_DELAY)
        
        browser.close()
        db.close()
        
        # Summary
        log.info("")
        log.info("=" * 50)
        log.info("SUMMARY")
        log.info("=" * 50)
        log.info(f"Box score URLs found: {total_urls}")
        log.info(f"Successfully parsed: {success_count}")
        log.info(f"  - SIDEARM: {sidearm_success} success, {sidearm_fail} failed")
        log.info(f"  - StatBroadcast: {statbroadcast_success} success, {statbroadcast_fail} failed")
        log.info(f"Failed to parse: {failed_count}")
        log.info(f"No game match in DB: {no_match_count}")
        log.info(f"Fallback scores updated: {fallback_count}")
        log.info(f"Total batting stats: {total_batting}")
        log.info(f"Total pitching stats: {total_pitching}")
        
        # Log failed URLs for debugging
        if failed_urls:
            log.info("")
            log.info("Failed URLs:")
            for url, url_type in failed_urls[:10]:  # Limit to first 10
                log.info(f"  [{url_type}] {url[:80]}")


if __name__ == '__main__':
    main()
