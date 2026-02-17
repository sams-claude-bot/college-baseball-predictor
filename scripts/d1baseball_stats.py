#!/usr/bin/env python3
"""
D1Baseball.com Stats Scraper - Pulls player batting and pitching stats for any team.

Usage:
    python3 scripts/d1baseball_stats.py arkansas
    python3 scripts/d1baseball_stats.py --all          # All holdout teams
    python3 scripts/d1baseball_stats.py --slugs arkansas kentucky vanderbilt
"""

import argparse
import re
import sqlite3
import sys
import time
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

PROJECT_DIR = Path(__file__).parent.parent
DB_PATH = PROJECT_DIR / 'data' / 'baseball.db'

# Import database-backed team resolver
sys.path.insert(0, str(PROJECT_DIR / 'scripts'))
from team_resolver import resolve_team as db_resolve_team, add_alias

# Holdout teams that need stats
# D1Baseball slugs for holdout teams (smu/syracuse have no baseball program)
HOLDOUT_SLUGS = ['gatech', 'kentucky', 'vandy']

# D1Baseball slug -> DB team_id mapping (when slug differs from DB id)
# Note: Most mappings are now in the database (team_aliases table)
SLUG_TO_TEAM_ID = {
    'gatech': 'georgia-tech',
    'vandy': 'vanderbilt',
}

# DB team_id -> D1Baseball slug (for all P4 teams)
TEAM_ID_TO_D1BB = None  # Loaded lazily from config/d1bb_slugs.json

def _load_slug_map():
    global TEAM_ID_TO_D1BB
    if TEAM_ID_TO_D1BB is not None:
        return
    slugs_file = PROJECT_DIR / 'config' / 'd1bb_slugs.json'
    if slugs_file.exists():
        import json
        data = json.loads(slugs_file.read_text())
        TEAM_ID_TO_D1BB = data.get('team_id_to_d1bb_slug', {})
        # Also populate reverse mapping
        for team_id, d1bb_slug in TEAM_ID_TO_D1BB.items():
            if d1bb_slug != team_id and d1bb_slug not in SLUG_TO_TEAM_ID:
                SLUG_TO_TEAM_ID[d1bb_slug] = team_id
    else:
        TEAM_ID_TO_D1BB = {}

def get_all_d1_slugs():
    """Get D1Baseball slugs for ALL D1 teams."""
    _load_slug_map()
    return list(TEAM_ID_TO_D1BB.values())

P4_CONFERENCES = {'SEC', 'ACC', 'Big 12', 'Big Ten'}

def get_p4_slugs(conference=None):
    """Get D1Baseball slugs for P4 conference teams."""
    _load_slug_map()
    slugs_file = PROJECT_DIR / 'config' / 'd1bb_slugs.json'
    if not slugs_file.exists():
        return HOLDOUT_SLUGS
    import json
    data = json.loads(slugs_file.read_text())
    conferences = data.get('conferences', {})
    if conference:
        team_ids = conferences.get(conference, [])
    else:
        team_ids = [tid for conf, teams in conferences.items() 
                    for tid in teams if conf in P4_CONFERENCES]
    return [TEAM_ID_TO_D1BB.get(tid, tid) for tid in team_ids]

CLASS_MAP = {'FR': 'FR', 'So.': 'SO', 'SO': 'SO', 'Jr.': 'JR', 'JR': 'JR', 
             'Sr.': 'SR', 'SR': 'SR', 'GR': 'GR', 'Gr.': 'GR', 'RS': 'RS',
             'Fr.': 'FR', 'R-Fr.': 'FR', 'R-So.': 'SO', 'R-Jr.': 'JR', 'R-Sr.': 'SR'}


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def resolve_team_id(db, slug):
    """Find the team_id in the DB for a given slug."""
    _load_slug_map()
    
    # Check explicit mapping first (for D1BB-specific slugs)
    if slug in SLUG_TO_TEAM_ID:
        return SLUG_TO_TEAM_ID[slug]
    
    # Use database resolver (checks team_aliases table)
    result = db_resolve_team(slug)
    if result:
        return result
    
    # Try direct match in teams table
    row = db.execute("SELECT id FROM teams WHERE id = ?", (slug,)).fetchone()
    if row:
        return row['id']
    
    # Try name match
    name_guess = slug.replace('-', ' ').title()
    row = db.execute("SELECT id FROM teams WHERE name LIKE ?", (f'%{name_guess}%',)).fetchone()
    if row:
        return row['id']
    
    return None


def fetch_stats_page(slug):
    url = f'https://d1baseball.com/team/{slug}/stats/'
    req = Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
    })
    with urlopen(req, timeout=20) as resp:
        return resp.read().decode('utf-8', errors='ignore')


def safe_int(val, default=0):
    if val is None or val == '' or val == '—' or val == '-':
        return default
    try:
        return int(float(str(val).strip()))
    except (ValueError, TypeError):
        return default


def safe_float(val, default=0.0):
    if val is None or val == '' or val == '—' or val == '-':
        return default
    try:
        return float(str(val).strip())
    except (ValueError, TypeError):
        return default


def parse_ip(val):
    """Parse innings pitched like '5.0', '4.1', '4.2' (where .1 = 1/3, .2 = 2/3)."""
    s = str(val).strip()
    if '.' in s:
        parts = s.split('.')
        whole = int(parts[0])
        frac = int(parts[1])
        return whole + frac / 3.0
    return float(s)


def parse_with_bs4(html):
    """Parse batting and pitching tables using BeautifulSoup."""
    soup = BeautifulSoup(html, 'html.parser')
    
    batting = []
    pitching = []
    
    tables = soup.find_all('table')
    
    # We want the STANDARD batting (has 'POS' and 'BA' columns) and
    # STANDARD pitching (has 'W', 'L', 'ERA', 'IP' columns).
    # Skip advanced/batted ball tables.
    batting_table = None
    pitching_table = None
    
    for table in tables:
        thead = table.find('thead')
        if not thead:
            continue
        headers = [th.get_text(strip=True) for th in thead.find_all(['th', 'td'])]
        headers_set = set(h.upper() for h in headers)
        
        if 'POS' in headers_set and 'BA' in headers_set and not batting_table:
            batting_table = (table, headers)
        elif 'ERA' in headers_set and 'IP' in headers_set and 'W' in headers_set and not pitching_table:
            pitching_table = (table, headers)
    
    for table, headers in [(batting_table, 'bat'), (pitching_table, 'pit')]:
        if table is None:
            continue
        tbl, hdrs = table if isinstance(table, tuple) else (table, headers)
        # Unpack - table is already a tuple
        pass
    
    def extract_rows(tbl_tuple):
        if not tbl_tuple:
            return []
        tbl, hdrs = tbl_tuple
        rows = []
        tbody = tbl.find('tbody')
        if not tbody:
            return rows
        for tr in tbody.find_all('tr'):
            cells = [c.get_text(strip=True) for c in tr.find_all(['td', 'th'])]
            if len(cells) < len(hdrs):
                continue
            row = dict(zip(hdrs, cells))
            name = row.get('Player', '')
            if not name or name.lower() in ('totals', 'opponents'):
                continue
            rows.append(row)
        return rows
    
    batting = extract_rows(batting_table)
    pitching = extract_rows(pitching_table)
    
    return batting, pitching


def parse_from_text(html):
    """Fallback: parse from the rendered text using regex on the readability-style output."""
    # This parses the markdown-style output from web_fetch
    # But since we have raw HTML and BS4, this is backup
    
    # Extract text content
    from html.parser import HTMLParser
    
    class TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self.text = []
        def handle_data(self, data):
            self.text.append(data)
    
    extractor = TextExtractor()
    extractor.feed(html)
    text = '\n'.join(extractor.text)
    
    # Split into batting and pitching sections
    batting = []
    pitching = []
    
    # Find sections by looking for header patterns
    batting_headers = ['Player', 'Team', 'Class', 'POS', 'BA', 'OBP', 'SLG', 'OPS', 'GP', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'HBP', 'BB', 'K', 'SB', 'CS']
    pitching_headers = ['Player', 'Team', 'Class', 'W', 'L', 'ERA', 'APP', 'GS', 'CG', 'SHO', 'SV', 'IP', 'H', 'R', 'ER', 'BB', 'K', 'HBP', 'BA']
    
    return batting, pitching


def build_player_batting(row, team_id):
    """Convert a batting row dict to our DB format."""
    name = row.get('Player', '').strip()
    if not name:
        return None
    
    return {
        'name': name,
        'team_id': team_id,
        'position': row.get('POS', None),
        'year': CLASS_MAP.get(row.get('Class', '').strip(), None),
        'games': safe_int(row.get('GP')),
        'at_bats': safe_int(row.get('AB')),
        'runs': safe_int(row.get('R')),
        'hits': safe_int(row.get('H')),
        'doubles': safe_int(row.get('2B')),
        'triples': safe_int(row.get('3B')),
        'home_runs': safe_int(row.get('HR')),
        'rbi': safe_int(row.get('RBI')),
        'walks': safe_int(row.get('BB')),
        'strikeouts': safe_int(row.get('K')),
        'stolen_bases': safe_int(row.get('SB')),
        'caught_stealing': safe_int(row.get('CS')),
        'batting_avg': safe_float(row.get('BA')),
        'obp': safe_float(row.get('OBP')),
        'slg': safe_float(row.get('SLG')),
        'ops': safe_float(row.get('OPS')),
    }


def build_player_pitching(row, team_id):
    """Convert a pitching row dict to our DB format."""
    name = row.get('Player', '').strip()
    if not name:
        return None
    
    ip_raw = row.get('IP', '0')
    ip = parse_ip(ip_raw)
    h = safe_int(row.get('H'))
    bb = safe_int(row.get('BB'))
    k = safe_int(row.get('K'))
    gs = safe_int(row.get('GS'))
    sv = safe_int(row.get('SV'))
    
    whip = (bb + h) / ip if ip > 0 else 0.0
    k_per_9 = (k * 9) / ip if ip > 0 else 0.0
    bb_per_9 = (bb * 9) / ip if ip > 0 else 0.0
    
    return {
        'name': name,
        'team_id': team_id,
        'year': CLASS_MAP.get(row.get('Class', '').strip(), None),
        'wins': safe_int(row.get('W')),
        'losses': safe_int(row.get('L')),
        'era': safe_float(row.get('ERA')),
        'games_pitched': safe_int(row.get('APP')),
        'games_started': gs,
        'saves': sv,
        'innings_pitched': round(ip, 2),
        'hits_allowed': h,
        'runs_allowed': safe_int(row.get('R')),
        'earned_runs': safe_int(row.get('ER')),
        'walks_allowed': bb,
        'strikeouts_pitched': k,
        'whip': round(whip, 3),
        'k_per_9': round(k_per_9, 2),
        'bb_per_9': round(bb_per_9, 2),
        'is_starter': 1 if gs > 0 else 0,
        'is_closer': 1 if sv > 0 else 0,
    }


def merge_players(batting_list, pitching_list):
    """Merge batting and pitching data for players who appear in both."""
    players = {}
    
    # Defaults for all fields
    defaults = {
        'number': None, 'position': None, 'year': None,
        'games': 0, 'at_bats': 0, 'runs': 0, 'hits': 0, 'doubles': 0, 'triples': 0,
        'home_runs': 0, 'rbi': 0, 'walks': 0, 'strikeouts': 0, 'stolen_bases': 0,
        'caught_stealing': 0, 'batting_avg': 0.0, 'obp': 0.0, 'slg': 0.0, 'ops': 0.0,
        'wins': 0, 'losses': 0, 'era': 0.0, 'games_pitched': 0, 'games_started': 0,
        'saves': 0, 'innings_pitched': 0.0, 'hits_allowed': 0, 'runs_allowed': 0,
        'earned_runs': 0, 'walks_allowed': 0, 'strikeouts_pitched': 0,
        'whip': 0.0, 'k_per_9': 0.0, 'bb_per_9': 0.0, 'is_starter': 0, 'is_closer': 0,
    }
    
    for b in batting_list:
        if not b:
            continue
        key = (b['team_id'], b['name'])
        p = dict(defaults)
        p.update({k: v for k, v in b.items() if v is not None})
        players[key] = p
    
    for pit in pitching_list:
        if not pit:
            continue
        key = (pit['team_id'], pit['name'])
        if key in players:
            # Merge pitching into existing batting record
            for k, v in pit.items():
                if k in ('name', 'team_id'):
                    continue
                if k == 'year' and players[key].get('year'):
                    continue  # Keep batting year
                players[key][k] = v
        else:
            p = dict(defaults)
            p.update({k: v for k, v in pit.items() if v is not None})
            players[key] = p
    
    return list(players.values())


def insert_snapshot(db, player, snapshot_date):
    """Insert a snapshot row for historical tracking (standard stats only)."""
    db.execute("""
        INSERT OR REPLACE INTO player_stats_snapshots (
            snapshot_date, team_id, player_name, player_class, position,
            games, at_bats, runs, hits, doubles, triples, home_runs,
            rbi, walks, strikeouts, stolen_bases, caught_stealing,
            batting_avg, obp, slg, ops,
            wins, losses, era, appearances, games_started, saves,
            innings_pitched, hits_allowed, runs_allowed, earned_runs,
            walks_allowed, strikeouts_pitched, whip
        ) VALUES (
            :snapshot_date, :team_id, :name, :year, :position,
            :games, :at_bats, :runs, :hits, :doubles, :triples, :home_runs,
            :rbi, :walks, :strikeouts, :stolen_bases, :caught_stealing,
            :batting_avg, :obp, :slg, :ops,
            :wins, :losses, :era, :games_pitched, :games_started, :saves,
            :innings_pitched, :hits_allowed, :runs_allowed, :earned_runs,
            :walks_allowed, :strikeouts_pitched, :whip
        )
    """, {**player, 'snapshot_date': snapshot_date})


def upsert_player(db, player):
    db.execute("""
        INSERT INTO player_stats (
            team_id, name, number, position, year,
            games, at_bats, runs, hits, doubles, triples,
            home_runs, rbi, walks, strikeouts, stolen_bases, caught_stealing,
            batting_avg, obp, slg, ops,
            wins, losses, era, games_pitched, games_started, saves,
            innings_pitched, hits_allowed, runs_allowed, earned_runs,
            walks_allowed, strikeouts_pitched, whip, k_per_9, bb_per_9,
            is_starter, is_closer, updated_at
        ) VALUES (
            :team_id, :name, :number, :position, :year,
            :games, :at_bats, :runs, :hits, :doubles, :triples,
            :home_runs, :rbi, :walks, :strikeouts, :stolen_bases, :caught_stealing,
            :batting_avg, :obp, :slg, :ops,
            :wins, :losses, :era, :games_pitched, :games_started, :saves,
            :innings_pitched, :hits_allowed, :runs_allowed, :earned_runs,
            :walks_allowed, :strikeouts_pitched, :whip, :k_per_9, :bb_per_9,
            :is_starter, :is_closer, CURRENT_TIMESTAMP
        )
        ON CONFLICT(team_id, name) DO UPDATE SET
            number=:number, position=:position, year=:year,
            games=:games, at_bats=:at_bats, runs=:runs,
            hits=:hits, doubles=:doubles, triples=:triples, home_runs=:home_runs,
            rbi=:rbi, walks=:walks, strikeouts=:strikeouts, stolen_bases=:stolen_bases,
            caught_stealing=:caught_stealing, batting_avg=:batting_avg, obp=:obp,
            slg=:slg, ops=:ops, wins=:wins, losses=:losses, era=:era,
            games_pitched=:games_pitched, games_started=:games_started, saves=:saves,
            innings_pitched=:innings_pitched, hits_allowed=:hits_allowed,
            runs_allowed=:runs_allowed, earned_runs=:earned_runs,
            walks_allowed=:walks_allowed, strikeouts_pitched=:strikeouts_pitched,
            whip=:whip, k_per_9=:k_per_9, bb_per_9=:bb_per_9,
            is_starter=:is_starter, is_closer=:is_closer,
            updated_at=CURRENT_TIMESTAMP
    """, player)


def scrape_team(slug, db, dry_run=False):
    team_id = resolve_team_id(db, slug)
    if not team_id:
        print(f"  ERROR: No team_id found in DB for slug '{slug}'")
        return 0
    
    print(f"  Fetching {slug} (team_id={team_id})...")
    html = fetch_stats_page(slug)
    
    if not html:
        print(f"  ERROR: Empty response for {slug}")
        return 0
    
    if HAS_BS4:
        batting_rows, pitching_rows = parse_with_bs4(html)
    else:
        batting_rows, pitching_rows = parse_from_text(html)
    
    print(f"  Parsed: {len(batting_rows)} batters, {len(pitching_rows)} pitchers")
    
    batting = [build_player_batting(r, team_id) for r in batting_rows]
    pitching = [build_player_pitching(r, team_id) for r in pitching_rows]
    
    players = merge_players(batting, pitching)
    
    if dry_run:
        for p in players:
            print(f"    {p['name']:25s} pos={str(p.get('position') or '?'):4s} yr={str(p.get('year') or '?'):2s} "
                  f"BA={p['batting_avg']:.3f} HR={p['home_runs']} "
                  f"ERA={p['era']:.2f} IP={p['innings_pitched']:.1f}")
        return len(players)
    
    today = date.today().isoformat()
    for p in players:
        upsert_player(db, p)
        insert_snapshot(db, p, today)
    db.commit()
    
    print(f"  Upserted {len(players)} players for {slug} (+ snapshots for {today})")
    return len(players)


def main():
    parser = argparse.ArgumentParser(description='Scrape D1Baseball.com stats')
    parser.add_argument('slugs', nargs='*', help='Team slug(s)')
    parser.add_argument('--all', action='store_true', help='Scrape all holdout teams (legacy)')
    parser.add_argument('--all-d1', action='store_true', help='Scrape ALL D1 teams on D1Baseball (~311 teams)')
    parser.add_argument('--p4', action='store_true', help='Scrape all P4 conference teams')
    parser.add_argument('--conference', type=str, help='Scrape a specific conference (SEC, ACC, Big 12, Big Ten, etc.)')
    parser.add_argument('--dry-run', action='store_true', help='Print without DB writes')
    parser.add_argument('--delay', type=float, default=4.0, help='Seconds between requests (default: 4)')
    args = parser.parse_args()
    
    if args.all_d1:
        _load_slug_map()
        slugs = get_all_d1_slugs()
    elif args.p4:
        _load_slug_map()
        slugs = get_p4_slugs(conference=args.conference)
    elif args.conference:
        _load_slug_map()
        slugs = get_p4_slugs(conference=args.conference)
    elif args.all:
        slugs = HOLDOUT_SLUGS
    elif args.slugs:
        slugs = args.slugs
    else:
        parser.print_help()
        sys.exit(1)
    
    db = get_db()
    total = 0
    
    for slug in slugs:
        print(f"\n=== {slug} ===")
        try:
            count = scrape_team(slug, db, dry_run=args.dry_run)
            total += count
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        if len(slugs) > 1:
            time.sleep(args.delay)  # Rate limit — be polite to D1Baseball
    
    db.close()
    print(f"\n{'='*40}")
    print(f"Total: {total} players across {len(slugs)} teams")


if __name__ == '__main__':
    main()
