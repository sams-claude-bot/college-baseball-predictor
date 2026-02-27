#!/usr/bin/env python3
"""
ESPN Live Scores Updater

Fetches live scores from ESPN's public API and updates the games table.
Designed to run frequently (every 1 minute) during game hours.

Usage:
    python3 scripts/espn_live_scores.py           # Update today's games
    python3 scripts/espn_live_scores.py 2026-02-25 # Update specific date
"""

import sys
import json
import sqlite3
import urllib.request
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
DB_PATH = PROJECT_ROOT / 'data' / 'baseball.db'

ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/baseball/college-baseball/scoreboard"


def _extract_athlete(data):
    """Extract athlete name from ESPN situation data."""
    if not data:
        return None
    if isinstance(data, dict):
        athlete = data.get('athlete', {})
        if isinstance(athlete, dict):
            return athlete.get('shortName') or athlete.get('displayName')
    return None


ESPN_TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/baseball/college-baseball/teams?limit=500"

# Cache file for ESPN ID -> our DB ID mapping
MAPPING_CACHE = PROJECT_ROOT / 'data' / 'espn_team_mapping.json'


def build_espn_id_mapping(conn):
    """Build mapping from ESPN team ID -> our DB team ID.
    Uses ESPN teams API + fuzzy matching. Cached to disk."""
    
    # Check cache (refresh daily)
    if MAPPING_CACHE.exists():
        import os
        age_hours = (datetime.now().timestamp() - os.path.getmtime(MAPPING_CACHE)) / 3600
        if age_hours < 24:
            with open(MAPPING_CACHE) as f:
                return json.load(f)
    
    # Fetch all ESPN teams
    try:
        with urllib.request.urlopen(ESPN_TEAMS_URL, timeout=15) as resp:
            data = json.load(resp)
    except Exception as e:
        print(f"Error fetching ESPN teams: {e}", file=sys.stderr)
        # Fall back to cache if available
        if MAPPING_CACHE.exists():
            with open(MAPPING_CACHE) as f:
                return json.load(f)
        return {}
    
    espn_teams = {}
    for t in data['sports'][0]['leagues'][0]['teams']:
        team = t['team']
        espn_teams[team['id']] = {
            'name': team.get('displayName', ''),
            'short': team.get('shortDisplayName', ''),
        }
    
    # Load our teams
    our_teams = conn.execute('SELECT id, name FROM teams').fetchall()
    db_by_name = {}
    for team in our_teams:
        db_by_name[team['name'].lower()] = team['id']
    
    # Manual overrides for tricky names (ESPN display name -> our ID)
    manual = {
        'Miami Hurricanes': 'miami-fl',
        'Miami (OH) RedHawks': 'miami-ohio',
        'Ole Miss Rebels': 'ole-miss',
        'LSU Tigers': 'lsu',
        'UCF Knights': 'ucf',
        'UConn Huskies': 'uconn',
        'UNLV Rebels': 'unlv',
        'BYU Cougars': 'byu',
        'NC State Wolfpack': 'nc-state',
        'SMU Mustangs': 'smu',
        'TCU Horned Frogs': 'tcu',
        'UAB Blazers': 'uab',
        'VCU Rams': 'vcu',
        'UIC Flames': 'uic',
        'FIU Panthers': 'florida-international',
        'Florida International Panthers': 'florida-international',
        'UTSA Roadrunners': 'utsa',
        'UTEP Miners': 'utep',
        'App State Mountaineers': 'appalachian-state',
        'SE Louisiana Lions': 'southeastern-louisiana',
        'UC Santa Barbara Gauchos': 'uc-santa-barbara',
        'UMass Minutemen': 'massachusetts',
        'Massachusetts Minutemen': 'massachusetts',
        'Pennsylvania Quakers': 'pennsylvania',
        'UAlbany Great Danes': 'albany',
        'Seattle U Redhawks': 'seattle',
        "Hawai'i Rainbow Warriors": 'hawaii',
        'McNeese Cowboys': 'mcneese',
        'Nicholls Colonels': 'nicholls',
        'Sam Houston Bearkats': 'sam-houston',
        'UT Martin Skyhawks': 'tennessee-martin',
        'Little Rock Trojans': 'little-rock',
        'UL Monroe Warhawks': 'louisiana-monroe',
        'Grambling Tigers': 'grambling-state',
        'Southern Jaguars': 'southern',
        'Houston Christian Huskies': 'houston-christian',
        'South Carolina Upstate Spartans': 'south-carolina-upstate',
        'North Dakota Fighting Hawks': 'north-dakota-state',
        'South Dakota Coyotes': 'south-dakota-state',
        'UT Rio Grande Valley Vaqueros': 'utrgv',
        'UTRGV Vaqueros': 'utrgv',
        'Texas A&M-Corpus Christi Islanders': 'texas-aandm-corpus-christi',
        'Texas A&M-CC Islanders': 'texas-aandm-corpus-christi',
        'St. Thomas-Minnesota Tommies': 'st-thomas-minnesota',
        'Maryland Eastern Shore Hawks': 'maryland-eastern-shore',
        'NJIT Highlanders': 'njit',
        'SIU Edwardsville Cougars': 'siu-edwardsville',
        'UT Arlington Mavericks': 'ut-arlington',
        'UMBC Retrievers': 'umbc',
        'UMass Lowell River Hawks': 'umass-lowell',
        'UNC Wilmington Seahawks': 'unc-wilmington',
        'UNC Greensboro Spartans': 'unc-greensboro',
        'UNC Asheville Bulldogs': 'unc-asheville',
        'North Carolina A&T Aggies': 'north-carolina-at',
        "Louisiana Ragin' Cajuns": 'louisiana',
        'Southeast Missouri State Redhawks': 'southeast-missouri',
    }
    
    # Build the mapping: ESPN ID (str) -> our DB team ID
    mapping = {}
    
    for eid, einfo in espn_teams.items():
        ename = einfo['name']
        
        # 1. Manual override
        if ename in manual:
            mapping[str(eid)] = manual[ename]
            continue
        
        # 2. Direct name match
        if ename.lower() in db_by_name:
            mapping[str(eid)] = db_by_name[ename.lower()]
            continue
        
        # 3. Short name match
        if einfo['short'].lower() in db_by_name:
            mapping[str(eid)] = db_by_name[einfo['short'].lower()]
            continue
        
        # 4. Name containment (our name appears in ESPN name)
        for db_name, db_id in db_by_name.items():
            if db_name in ename.lower() and len(db_name) > 3:
                mapping[str(eid)] = db_id
                break
    
    # Save cache
    MAPPING_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(MAPPING_CACHE, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"ESPN mapping: {len(mapping)}/{len(espn_teams)} teams matched")
    return mapping


def fetch_espn_scores(date_str):
    """Fetch scores from ESPN API for a given date."""
    espn_date = date_str.replace('-', '')
    url = f"{ESPN_SCOREBOARD_URL}?dates={espn_date}&limit=200"
    
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return json.load(resp)
    except Exception as e:
        print(f"Error fetching ESPN API: {e}", file=sys.stderr)
        return None


def espn_status_to_db(status_type, game_date=None):
    """Convert ESPN status to our DB status.
    
    If game_date is in the future, reject 'in-progress' (ESPN sometimes
    prematurely marks upcoming games as live).
    """
    from datetime import datetime
    name = status_type.get('name', '')
    if name == 'STATUS_FINAL':
        return 'final'
    elif name == 'STATUS_IN_PROGRESS':
        # Guard: don't mark future games as in-progress
        if game_date:
            try:
                today = datetime.now().strftime('%Y-%m-%d')
                if str(game_date) > today:
                    return 'scheduled'
            except (ValueError, TypeError):
                pass
        return 'in-progress'
    elif name == 'STATUS_SCHEDULED':
        return 'scheduled'
    elif name in ('STATUS_POSTPONED', 'STATUS_CANCELED', 'STATUS_DELAYED'):
        return 'postponed'
    return 'scheduled'


def update_scores(date_str=None):
    """Main update function."""
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')
    
    data = fetch_espn_scores(date_str)
    if not data:
        return 0
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    espn_mapping = build_espn_id_mapping(conn)
    
    # Get our games for this date
    our_games = {}
    rows = conn.execute(
        'SELECT id, home_team_id, away_team_id, status, home_score, away_score FROM games WHERE date = ?',
        (date_str,)
    ).fetchall()
    for r in rows:
        key = (r['away_team_id'], r['home_team_id'])
        our_games[key] = dict(r)
    
    updated = 0
    unmatched_espn = []
    
    for event in data.get('events', []):
        comp = event['competitions'][0]
        status = comp['status']
        
        # Get teams
        competitors = comp['competitors']
        away_espn = [c for c in competitors if c['homeAway'] == 'away']
        home_espn = [c for c in competitors if c['homeAway'] == 'home']
        
        if not away_espn or not home_espn:
            continue
        
        away_espn = away_espn[0]
        home_espn = home_espn[0]
        
        away_espn_id = str(away_espn['team']['id'])
        home_espn_id = str(home_espn['team']['id'])
        away_name = away_espn['team']['displayName']
        home_name = home_espn['team']['displayName']
        
        # Match to our DB via ESPN ID mapping
        away_id = espn_mapping.get(away_espn_id)
        home_id = espn_mapping.get(home_espn_id)
        
        if not away_id or not home_id:
            if status['type']['name'] != 'STATUS_SCHEDULED':
                unmatched_espn.append(f"{away_name} @ {home_name} (espn:{away_espn_id}/{home_espn_id})")
            continue
        
        key = (away_id, home_id)
        if key not in our_games:
            continue
        
        game = our_games[key]
        db_status = espn_status_to_db(status['type'], game_date=game.get('date'))
        
        # Get scores
        away_score = int(away_espn.get('score', 0)) if away_espn.get('score') else None
        home_score = int(home_espn.get('score', 0)) if home_espn.get('score') else None
        
        # Get hits and errors
        away_hits = int(away_espn.get('hits', 0)) if away_espn.get('hits') is not None else None
        home_hits = int(home_espn.get('hits', 0)) if home_espn.get('hits') is not None else None
        away_errors = int(away_espn.get('errors', 0)) if away_espn.get('errors') is not None else None
        home_errors = int(home_espn.get('errors', 0)) if home_espn.get('errors') is not None else None
        
        # Get linescore (inning-by-inning)
        linescore = None
        home_ls = home_espn.get('linescores', [])
        away_ls = away_espn.get('linescores', [])
        if home_ls or away_ls:
            linescore = json.dumps({
                'home': [int(l.get('value', 0)) for l in home_ls],
                'away': [int(l.get('value', 0)) for l in away_ls],
            })
        
        # Get inning info
        inning_text = status['type'].get('detail', '')
        if not inning_text:
            inning_text = status['type'].get('shortDetail', '')
        
        innings = status.get('period', None)
        
        # Get live situation (runners, count, pitcher, batter, last play)
        situation = None
        sit = comp.get('situation')
        if sit and db_status == 'in-progress':
            situation = json.dumps({
                'outs': sit.get('outs', 0),
                'balls': sit.get('balls', 0),
                'strikes': sit.get('strikes', 0),
                'onFirst': sit.get('onFirst', False),
                'onSecond': sit.get('onSecond', False),
                'onThird': sit.get('onThird', False),
                'batter': _extract_athlete(sit.get('batter')),
                'pitcher': _extract_athlete(sit.get('pitcher')),
                'lastPlay': sit.get('lastPlay', {}).get('text', '') if isinstance(sit.get('lastPlay'), dict) else '',
                'dueUp': [_extract_athlete(a) for a in sit.get('dueUp', [])] if sit.get('dueUp') else [],
            })
        
        # Determine winner
        winner_id = None
        if db_status == 'final' and home_score is not None and away_score is not None:
            if home_score > away_score:
                winner_id = home_id
            elif away_score > home_score:
                winner_id = away_id
        
        # Check if anything changed (score, hits, situation all count)
        score_changed = (game['home_score'] != home_score or game['away_score'] != away_score)
        status_changed = (game['status'] != db_status)
        
        if not score_changed and not status_changed and db_status != 'in-progress':
            continue
        
        # Update the game with full data
        from schedule_gateway import ScheduleGateway
        gw = ScheduleGateway(conn)
        
        if db_status == 'final' and home_score is not None and away_score is not None:
            gw.finalize_game(game['id'], home_score, away_score)
        elif db_status == 'in-progress' and home_score is not None and away_score is not None:
            gw.update_live_score(game['id'], home_score, away_score, inning_text, innings=innings)
        else:
            conn.execute('''
                UPDATE games 
                SET status = ?, home_score = ?, away_score = ?, 
                    winner_id = ?, inning_text = ?, innings = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (db_status, home_score, away_score, winner_id, inning_text, innings, game['id']))
        
        # Always update extended fields (hits, errors, linescore, situation)
        # Merge situation_json to preserve StatBroadcast sb_* fields
        merged_situation = situation
        if situation:
            existing_sit = conn.execute(
                'SELECT situation_json FROM games WHERE id = ?', (game['id'],)
            ).fetchone()
            if existing_sit and existing_sit[0]:
                try:
                    existing = json.loads(existing_sit[0])
                    new_sit = json.loads(situation)
                    # Preserve sb_* fields from StatBroadcast
                    for k, v in existing.items():
                        if k.startswith('sb_') and k not in new_sit:
                            new_sit[k] = v
                    merged_situation = json.dumps(new_sit)
                except (json.JSONDecodeError, TypeError):
                    pass

        conn.execute('''
            UPDATE games 
            SET home_hits = ?, away_hits = ?, 
                home_errors = ?, away_errors = ?,
                linescore_json = ?,
                situation_json = ?
            WHERE id = ?
        ''', (home_hits, away_hits, home_errors, away_errors,
              linescore, merged_situation, game['id']))
        
        updated += 1
        rhe = f"R:{away_score}-{home_score}"
        if away_hits is not None:
            rhe += f" H:{away_hits}-{home_hits} E:{away_errors}-{home_errors}"
        print(f"Updated: {away_id} @ {home_id} ({db_status}, {inning_text}) {rhe}")
    
    conn.commit()
    conn.close()
    
    if unmatched_espn:
        print(f"\nUnmatched ESPN games ({len(unmatched_espn)}):", file=sys.stderr)
        for u in unmatched_espn[:10]:
            print(f"  {u}", file=sys.stderr)
    
    print(f"\nTotal updated: {updated}")
    return updated


if __name__ == '__main__':
    date = sys.argv[1] if len(sys.argv) > 1 else None
    update_scores(date)
