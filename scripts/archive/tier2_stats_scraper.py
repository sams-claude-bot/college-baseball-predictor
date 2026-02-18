#!/usr/bin/env python3
"""
Tier 2 Stats Scraper - Collects batting and pitching stats from SIDEARM sites.
Uses 15 second delays between fetches for politeness.

Usage:
    python3 scripts/tier2_stats_scraper.py --conference "Sun Belt"
    python3 scripts/tier2_stats_scraper.py --conference AAC
    python3 scripts/tier2_stats_scraper.py --conference A-10
"""

import argparse
import json
import sys
import re
import time
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
import urllib.request

# Setup
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / 'data'
DB_PATH = DATA_DIR / 'baseball.db'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger('tier2_stats')

REQUEST_DELAY = 15  # 15 second delay as requested


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def load_team_urls():
    urls = {}
    d1_file = DATA_DIR / 'd1_team_urls.json'
    if d1_file.exists():
        with open(d1_file) as f:
            urls.update(json.load(f).get('teams', {}))
    p4_file = DATA_DIR / 'p4_team_urls.json'
    if p4_file.exists():
        with open(p4_file) as f:
            urls.update(json.load(f).get('teams', {}))
    return urls


def fetch_page(url):
    """Fetch a page with error handling and redirect support."""
    import subprocess
    try:
        # Use curl for better redirect handling
        result = subprocess.run(
            ['curl', '-s', '-L', '--max-time', '30',
             '-H', 'User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
             url],
            capture_output=True, text=True, timeout=35
        )
        return result.stdout
    except Exception as e:
        log.error(f"Fetch failed for {url}: {e}")
        return ""


def parse_sidearm_nuxt3(html):
    """Parse stats from SIDEARM Nuxt 3 payload."""
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
    """Insert or update player stats."""
    # Check if player exists
    existing = db.execute(
        "SELECT id FROM player_stats WHERE team_id = ? AND name = ?",
        (player['team_id'], player['name'])
    ).fetchone()
    
    if existing:
        # Update existing row
        db.execute("""
            UPDATE player_stats SET
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
            WHERE team_id=:team_id AND name=:name
        """, player)
    else:
        # Insert new row
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
        """, player)


def collect_team_stats(team_id, url, db):
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
        urls_to_try.extend([base + '/2026', base + '/2025', base])

    batting_raw = None
    pitching_raw = None

    for i, try_url in enumerate(urls_to_try):
        if i > 0:
            time.sleep(2)  # Small delay between URL attempts
        log.info(f"    Trying: {try_url}")
        html = fetch_page(try_url)
        if not html or len(html) < 1000:
            continue
        if 'individualHittingStats' not in html and 'cumulativeStats' not in html:
            continue

        batting_raw, pitching_raw = parse_sidearm_nuxt3(html)
        if batting_raw:
            log.info(f"    ✓ Found data!")
            break

    if not batting_raw:
        log.error(f"  ✗ FAILED: No parseable stats for {team_id}")
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
        upsert_player_stats(db, player)
        pitching_only += 1

    total_pitchers = len([p for p in (pitching_raw or [])
                         if isinstance(p, dict) and not p.get('isAFooterStat')
                         and p.get('playerName') not in ('Totals', 'Opponents', None)])

    log.info(f"  ✓ {team_id}: {batting_count} batters, {total_pitchers} pitchers ({pitching_only} pitch-only)")
    return {'batting': batting_count, 'pitching': total_pitchers, 'status': 'ok'}


def main():
    parser = argparse.ArgumentParser(description='Collect Tier 2 baseball stats')
    parser.add_argument('--conference', required=True, help='Conference name (e.g., "Sun Belt")')
    args = parser.parse_args()

    db = get_db()
    team_urls = load_team_urls()

    # Get teams for this conference
    teams = db.execute(
        "SELECT id, name FROM teams WHERE conference = ? ORDER BY name",
        (args.conference,)
    ).fetchall()

    if not teams:
        log.error(f"No teams found for conference: {args.conference}")
        sys.exit(1)

    log.info(f"Collecting stats for {len(teams)} teams in {args.conference}")
    log.info(f"Using {REQUEST_DELAY}s delay between requests")
    start_time = time.time()
    
    results = {'ok': 0, 'failed': 0, 'total_batting': 0, 'total_pitching': 0}
    failed_teams = []

    for i, team in enumerate(teams, 1):
        team_id = team['id']
        url = team_urls.get(team_id)
        
        if not url:
            log.warning(f"[{i}/{len(teams)}] {team_id} - No URL found, skipping")
            results['failed'] += 1
            failed_teams.append(team_id)
            continue

        log.info(f"[{i}/{len(teams)}] {team['name']} ({team_id})")

        try:
            result = collect_team_stats(team_id, url, db)
            
            if result['status'] == 'ok':
                results['ok'] += 1
                results['total_batting'] += result['batting']
                results['total_pitching'] += result['pitching']
            else:
                results['failed'] += 1
                failed_teams.append(team_id)
        except Exception as e:
            log.error(f"Exception for {team_id}: {e}")
            results['failed'] += 1
            failed_teams.append(team_id)

        # Delay between requests (skip on last team)
        if i < len(teams):
            log.info(f"  Waiting {REQUEST_DELAY}s...")
            time.sleep(REQUEST_DELAY)

    # Commit after conference
    db.commit()
    log.info(f"Committed {args.conference} to database")

    elapsed = time.time() - start_time
    log.info(f"\n{'='*50}")
    log.info(f"RESULTS for {args.conference}")
    log.info(f"Time: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    log.info(f"Teams OK: {results['ok']}/{len(teams)}")
    log.info(f"Teams Failed: {results['failed']}")
    log.info(f"Total Batters: {results['total_batting']}")
    log.info(f"Total Pitchers: {results['total_pitching']}")
    if failed_teams:
        log.info(f"Failed teams: {', '.join(failed_teams)}")

    db.close()
    
    # Return exit code based on success
    sys.exit(0 if results['failed'] == 0 else 1)


if __name__ == '__main__':
    main()
