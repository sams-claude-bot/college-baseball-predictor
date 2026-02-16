#!/usr/bin/env python3
"""
Unified P4 Stats Collector - Collects batting and pitching stats from SIDEARM Sports sites.

Parses embedded Nuxt 3 JSON payload from SIDEARM-powered athletics sites.

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

# ESPN team ID mapping (for future use when ESPN adds college baseball stats)
ESPN_IDS_FILE = DATA_DIR / 'espn_team_ids.json'

# WMT iframe URLs for WordPress-based sites
WMT_URLS = {
    'miami-fl': 'https://wmt.games/miamihurricanes/stats/season/614661',
}

# Teams that don't have baseball programs (included in P4 conferences but no team)
NO_BASEBALL = {'colorado', 'iowa-state', 'smu', 'syracuse'}


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def load_team_urls():
    with open(TEAM_URLS_FILE) as f:
        return json.load(f).get('teams', {})


def get_teams_for_args(args):
    db = get_db()
    if args.team:
        teams = db.execute("SELECT id FROM teams WHERE id = ?", (args.team,)).fetchall()
    elif args.conference:
        teams = db.execute("SELECT id FROM teams WHERE conference = ?", (args.conference,)).fetchall()
    else:
        teams = db.execute(
            "SELECT id FROM teams WHERE conference IN ('SEC','Big 12','Big Ten','ACC') ORDER BY conference, id"
        ).fetchall()
    db.close()
    return [t['id'] for t in teams]


def fetch_page(url, follow_redirects=True):
    cmd = ['curl', '-s', '--max-time', '15']
    if follow_redirects:
        cmd.append('-L')
    cmd.append(url)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return result.stdout
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


def collect_wmt_stats(team_id, wmt_url, db, dry_run=False):
    """Collect stats from WMT (Web Management Tool) iframe used by WordPress sites."""
    log.info(f"  Trying WMT scraper for {team_id}...")
    
    try:
        import subprocess as sp
        from wmt_scraper import parse_wmt_batting_text, parse_wmt_pitching_text
        
        # Fetch batting page
        batting_url = wmt_url
        if '?' not in batting_url:
            batting_url += '?overall=Batting'
        
        # WMT requires a browser - try curl first as a quick check
        result = sp.run(
            ['curl', '-s', '--max-time', '15', '-L',
             '-H', 'User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
             batting_url],
            capture_output=True, text=True, timeout=20
        )
        
        batting_html = result.stdout
        pitching_html = ""
        
        # If curl got data with stats
        if 'AVG' in batting_html and 'NAME' in batting_html:
            # Extract text content (simple HTML strip)
            import re
            text = re.sub(r'<[^>]+>', '\n', batting_html)
            batting_players = parse_wmt_batting_text(text)
        else:
            log.warning(f"  WMT batting page needs browser (JS-rendered)")
            return None
        
        # Fetch pitching
        pitching_url = wmt_url.split('?')[0] + '?overall=Pitching'
        result = sp.run(
            ['curl', '-s', '--max-time', '15', '-L',
             '-H', 'User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
             pitching_url],
            capture_output=True, text=True, timeout=20
        )
        
        pitching_html = result.stdout
        pitching_players = []
        if 'ERA' in pitching_html and 'NAME' in pitching_html:
            text = re.sub(r'<[^>]+>', '\n', pitching_html)
            pitching_players = parse_wmt_pitching_text(text)
        
        if not batting_players:
            log.error(f"  WMT: No batting data parsed for {team_id}")
            return None
        
        # Build pitching lookup
        pitching_by_name = {p['name']: p for p in pitching_players}
        
        batting_count = 0
        for bp in batting_players:
            player = {
                'team_id': team_id,
                **bp,
            }
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
        
        # Pitch-only players
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
        
        total_pitchers = len(pitching_players)
        log.info(f"  {team_id} (WMT): {batting_count} batters, {total_pitchers} pitchers ({pitching_only} pitch-only)")
        return {'batting': batting_count, 'pitching': total_pitchers, 'status': 'ok', 'source': 'wmt'}
        
    except Exception as e:
        log.error(f"  WMT scraper failed for {team_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


def collect_team_stats(team_id, url, db, dry_run=False):
    """Collect batting + pitching stats for one team."""
    log.info(f"  Fetching {team_id}...")

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
        log.error(f"  FAILED: No parseable stats for {team_id}")
        return {'batting': 0, 'pitching': 0, 'status': 'failed'}

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
            # Try SIDEARM Nuxt parsing first
            result = collect_team_stats(team_id, url, db, dry_run=args.dry_run)
            
            # If SIDEARM failed, try WMT fallback
            if result['status'] != 'ok' and team_id in WMT_URLS:
                log.info(f"  SIDEARM failed, trying WMT fallback...")
                result = collect_wmt_stats(team_id, WMT_URLS[team_id], db, dry_run=args.dry_run)
                if result is None:
                    result = {'status': 'failed', 'batting': 0, 'pitching': 0}
            
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
        except Exception as e:
            log.error(f"Exception for {team_id}: {e}")
            import traceback
            traceback.print_exc()
            results['failed'] += 1
            progress['failed'].append(team_id)

        save_progress(progress)
        if i < len(team_ids):
            time.sleep(REQUEST_DELAY)

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
