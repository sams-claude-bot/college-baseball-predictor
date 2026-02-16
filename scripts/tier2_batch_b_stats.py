#!/usr/bin/env python3
"""
Tier 2 Batch B Stats Collector - CAA, WCC, MVC, Big East, C-USA, ASUN
15-second delays between all web fetches. Skip failures, don't retry.
"""

import json
import time
import sqlite3
import re
from pathlib import Path
from datetime import datetime
import urllib.request

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / 'data'
DB_PATH = DATA_DIR / 'baseball.db'
PROGRESS_FILE = DATA_DIR / 'batch_b_progress.json'

REQUEST_DELAY = 15  # 15 seconds between ALL fetches

CONFERENCES = ['CAA', 'WCC', 'MVC', 'Big East', 'C-USA', 'ASUN']

def load_progress():
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE) as f:
                return json.load(f)
        except:
            pass
    return {'completed': [], 'failed': []}

def save_progress(progress):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f)

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def get_db():
    conn = sqlite3.connect(str(DB_PATH), timeout=60)  # Wait up to 60s for lock
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
    conn.execute("PRAGMA busy_timeout=60000")  # 60 second busy timeout
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

def load_espn_ids():
    espn_file = DATA_DIR / 'espn_team_ids.json'
    if espn_file.exists():
        with open(espn_file) as f:
            return json.load(f)
    return {}

def fetch_page(url):
    import subprocess
    try:
        result = subprocess.run(
            ['curl', '-s', '--max-time', '20', '-L',
             '-H', 'User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
             url],
            capture_output=True, text=True, timeout=25
        )
        return result.stdout
    except Exception as e:
        log(f"  Fetch error: {e}")
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

def normalize_name(name):
    if not name:
        return name
    if ',' in name:
        parts = name.split(',', 1)
        return f"{parts[1].strip()} {parts[0].strip()}"
    return name.strip()

def si(val, default=0):
    if val is None or val == '' or val == '—':
        return default
    try:
        return int(val)
    except:
        return default

def sf(val, default=0.0):
    if val is None or val == '' or val == '—':
        return default
    try:
        return float(val)
    except:
        return default

def extract_batting(raw, team_id):
    if not isinstance(raw, dict):
        return None
    name = normalize_name(raw.get('playerName', ''))
    if not name or name in ('Totals', 'Opponents') or raw.get('isAFooterStat'):
        return None
    return {
        'team_id': team_id,
        'name': name,
        'number': si(raw.get('playerUniform')),
        'games': si(raw.get('gamesPlayed')),
        'at_bats': si(raw.get('atBats')),
        'runs': si(raw.get('runs')),
        'hits': si(raw.get('hits')),
        'doubles': si(raw.get('doubles')),
        'triples': si(raw.get('triples')),
        'home_runs': si(raw.get('homeRuns')),
        'rbi': si(raw.get('runsBattedIn')),
        'walks': si(raw.get('walks')),
        'strikeouts': si(raw.get('strikeouts')),
        'stolen_bases': si(raw.get('stolenBases')),
        'caught_stealing': si(raw.get('caughtStealing')),
    }

def extract_pitching(raw, team_id):
    if not isinstance(raw, dict):
        return None
    name = normalize_name(raw.get('playerName', ''))
    if not name or name in ('Totals', 'Opponents') or raw.get('isAFooterStat'):
        return None
    return {
        'team_id': team_id,
        'name': name,
        'number': si(raw.get('playerUniform')),
        'wins': si(raw.get('wins')),
        'losses': si(raw.get('losses')),
        'games_pitched': si(raw.get('appearances')),
        'games_started': si(raw.get('gamesStarted')),
        'saves': si(raw.get('saves')),
        'innings_pitched': sf(raw.get('inningsPitched')),
        'hits_allowed': si(raw.get('hitsAllowed')),
        'runs_allowed': si(raw.get('runsAllowed')),
        'earned_runs': si(raw.get('earnedRunsAllowed')),
        'walks_allowed': si(raw.get('walksAllowed')),
        'strikeouts_pitched': si(raw.get('strikeouts')),
    }

def convert_ip_to_decimal(ip):
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

def upsert_player_stats(db, player, retries=5):
    for attempt in range(retries):
        try:
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
            return  # Success
        except sqlite3.OperationalError as e:
            if 'locked' in str(e) and attempt < retries - 1:
                log(f"    DB locked, retry {attempt + 1}/{retries}...")
                time.sleep(2)
            else:
                raise

def collect_team_stats(team_id, url, db):
    """Collect stats for one team. Returns (batters, pitchers) or None on failure."""
    log(f"  Fetching {team_id}...")
    
    # Try multiple URL patterns
    urls_to_try = [url]
    base = url.rsplit('/stats', 1)[0] + '/stats' if '/stats' in url else url
    if url.endswith('/2026'):
        urls_to_try.append(url.replace('/2026', ''))
    else:
        urls_to_try.append(base + '/2026')
    
    batting_raw = None
    pitching_raw = None
    
    for try_url in urls_to_try:
        html = fetch_page(try_url)
        time.sleep(REQUEST_DELAY)
        
        if not html or len(html) < 1000:
            continue
        if 'individualHittingStats' not in html and 'cumulativeStats' not in html:
            continue
            
        batting_raw, pitching_raw = parse_sidearm_nuxt3(html)
        if batting_raw:
            log(f"  Found data at {try_url}")
            break
    
    if not batting_raw:
        log(f"  FAILED: No parseable stats for {team_id}")
        return None
    
    # Build pitching lookup
    pitching_by_name = {}
    if pitching_raw:
        for praw in pitching_raw:
            ps = extract_pitching(praw, team_id)
            if ps:
                pitching_by_name[ps['name']] = ps
    
    batting_count = 0
    for raw in batting_raw:
        player = extract_batting(raw, team_id)
        if not player:
            continue
        
        # Merge pitching if exists
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
    
    # Commit after each team to avoid losing progress
    db.commit()
    
    total_pitchers = len([p for p in (pitching_raw or [])
                         if isinstance(p, dict) and not p.get('isAFooterStat')
                         and p.get('playerName') not in ('Totals', 'Opponents', None)])
    
    log(f"  {team_id}: {batting_count} batters, {total_pitchers} pitchers ({pitching_only} pitch-only)")
    return (batting_count, total_pitchers)

def main():
    log("=" * 60)
    log("TIER 2 BATCH B STATS COLLECTOR")
    log(f"Conferences: {', '.join(CONFERENCES)}")
    log("=" * 60)
    
    db = get_db()
    team_urls = load_team_urls()
    progress = load_progress()
    
    # Get teams for our conferences
    teams_by_conf = {}
    for conf in CONFERENCES:
        rows = db.execute("SELECT id, name FROM teams WHERE conference = ? ORDER BY name", (conf,)).fetchall()
        teams_by_conf[conf] = [(r['id'], r['name']) for r in rows]
        log(f"{conf}: {len(teams_by_conf[conf])} teams")
    
    total_teams = sum(len(t) for t in teams_by_conf.values())
    log(f"Total: {total_teams} teams")
    log(f"Already completed: {len(progress['completed'])}")
    log("")
    
    results = {
        'successful': [],
        'failed': [],
        'total_batters': 0,
        'total_pitchers': 0,
    }
    
    for conf in CONFERENCES:
        log(f"\n{'='*40}")
        log(f"CONFERENCE: {conf}")
        log(f"{'='*40}")
        
        conf_success = 0
        conf_fail = 0
        
        for team_id, team_name in teams_by_conf[conf]:
            # Skip if already completed
            if team_id in progress['completed']:
                log(f"  {team_id}: Already done, skipping")
                continue
            
            url = team_urls.get(team_id)
            if not url:
                log(f"  {team_id}: No URL, skipping")
                results['failed'].append((team_id, 'no_url'))
                progress['failed'].append(team_id)
                save_progress(progress)
                conf_fail += 1
                continue
            
            result = collect_team_stats(team_id, url, db)
            if result:
                batters, pitchers = result
                results['successful'].append((team_id, batters, pitchers))
                results['total_batters'] += batters
                results['total_pitchers'] += pitchers
                progress['completed'].append(team_id)
                conf_success += 1
            else:
                results['failed'].append((team_id, 'scrape_failed'))
                progress['failed'].append(team_id)
                conf_fail += 1
            
            save_progress(progress)
        
        log(f"\n{conf} complete: {conf_success} success, {conf_fail} failed")
    
    db.close()
    
    # Final report
    log("\n" + "=" * 60)
    log("FINAL REPORT")
    log("=" * 60)
    log(f"Teams successful: {len(results['successful'])}")
    log(f"Teams failed: {len(results['failed'])}")
    log(f"Total batters with stats: {results['total_batters']}")
    log(f"Total pitchers with stats: {results['total_pitchers']}")
    
    if results['failed']:
        log("\nFailed teams:")
        for team_id, reason in results['failed']:
            log(f"  - {team_id}: {reason}")
    
    log("\n" + "=" * 60)

if __name__ == '__main__':
    main()
