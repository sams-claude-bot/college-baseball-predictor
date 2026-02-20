#!/usr/bin/env python3
"""
Verify a team's schedule against D1Baseball.

Compares our database games to D1Baseball's team schedule page and reports
mismatches: extra games, missing games, wrong dates, wrong opponents, wrong home/away.

Usage:
    python3 scripts/verify_team_schedule.py omaha
    python3 scripts/verify_team_schedule.py omaha --fix        # Auto-delete extra DB games
    python3 scripts/verify_team_schedule.py omaha --verbose    # Show all games including OK
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'scripts'))
from database import get_connection
from team_resolver import resolve_team as db_resolve_team

D1BB_BASE = "https://d1baseball.com"
SLUGS_FILE = PROJECT_DIR / 'config' / 'd1bb_slugs.json'


def load_d1bb_slugs() -> dict:
    """Load team_id -> d1bb_slug mapping."""
    if SLUGS_FILE.exists():
        data = json.loads(SLUGS_FILE.read_text())
        return data.get('team_id_to_d1bb_slug', {})
    return {}


def load_reverse_slug_map() -> dict:
    """Load d1bb_slug -> team_id mapping."""
    fwd = load_d1bb_slugs()
    return {v: k for k, v in fwd.items()}


def get_d1bb_slug(team_id: str) -> str:
    """Get D1Baseball slug for a team."""
    slugs = load_d1bb_slugs()
    return slugs.get(team_id)


def fetch_d1bb_schedule(d1bb_slug: str) -> list:
    """Fetch and parse a team's schedule from D1Baseball.
    
    Returns list of dicts with: date, opponent_slug, is_home, result, scores
    """
    import requests
    
    url = f"{D1BB_BASE}/team/{d1bb_slug}/schedule/"
    resp = requests.get(url, timeout=15, headers={
        'User-Agent': 'Mozilla/5.0 (compatible; BaseballBot/1.0)'
    })
    resp.raise_for_status()
    html = resp.text
    
    games = []
    reverse_slugs = load_reverse_slug_map()
    seen = set()
    
    # Use only the full-team-schedule table to avoid duplicates from mobile view
    table_match = re.search(r'full-team-schedule.*?</table>', html, re.DOTALL)
    if not table_match:
        print("⚠️  Could not find full-team-schedule table on page")
        return games
    table_html = table_match.group()
    
    # D1BB structure: date link, then @ or vs between date and team link
    # Pattern: date=YYYYMMDD ... (@ or vs) ... /team/SLUG/
    # The @ character only appears when the team is away
    for m in re.finditer(
        r'date=(\d{8})(.*?)/team/([^/]+)/schedule/',
        table_html, re.DOTALL
    ):
        date_raw, middle, opp_slug = m.groups()
        date_str = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:8]}"
        
        # Skip if opponent is the team itself
        if opp_slug == d1bb_slug:
            continue
        
        # Dedupe by (date, opponent)
        key = (date_str, opp_slug)
        if key in seen:
            continue
        seen.add(key)
        
        # Determine home/away from the text between date link and team link
        # "@" in the middle section means away; "vs" means home/neutral
        is_home = '@' not in middle
        
        # Check for result (W/L + score) after the team link
        # Look ahead in the table HTML for score info
        after_pos = m.end()
        after_block = table_html[after_pos:after_pos + 500]
        after_text = re.sub(r'<[^>]+>', ' ', after_block)
        
        result = None
        score_match = re.search(r'([WL])\s+(\d+)\s*-\s*(\d+)', after_text)
        if score_match:
            result = {
                'outcome': score_match.group(1),
                'team_score': int(score_match.group(2)),
                'opp_score': int(score_match.group(3)),
            }
        
        # Resolve opponent to our DB team_id
        opp_team_id = reverse_slugs.get(opp_slug, opp_slug)
        
        games.append({
            'date': date_str,
            'opponent_d1bb_slug': opp_slug,
            'opponent_team_id': opp_team_id,
            'is_home': is_home,
            'result': result,
        })
    
    return games


def get_db_schedule(team_id: str) -> list:
    """Get team's games from our database."""
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        SELECT id, date, home_team_id, away_team_id, status, home_score, away_score
        FROM games
        WHERE home_team_id = ? OR away_team_id = ?
        ORDER BY date
    ''', (team_id, team_id))
    games = [dict(r) for r in c.fetchall()]
    conn.close()
    return games


def verify(team_slug: str, fix: bool = False, verbose: bool = False):
    """Verify team schedule against D1Baseball."""
    
    d1bb_slug = get_d1bb_slug(team_slug)
    if not d1bb_slug:
        print(f"❌ No D1Baseball slug found for '{team_slug}'. Check d1bb_slugs.json.")
        return False
    
    print(f"🔍 Verifying: {team_slug} (D1Baseball: {d1bb_slug})")
    print(f"   Source: {D1BB_BASE}/team/{d1bb_slug}/schedule/")
    print()
    
    d1bb_games = fetch_d1bb_schedule(d1bb_slug)
    db_games = get_db_schedule(team_slug)
    
    print(f"   D1Baseball games: {len(d1bb_games)} | DB games: {len(db_games)}")
    print()
    
    # Build D1BB lookup: (date, opponent) -> game info
    d1bb_by_date_opp = {}
    for g in d1bb_games:
        key = (g['date'], g['opponent_team_id'])
        d1bb_by_date_opp[key] = g
    
    # Build DB lookup
    db_by_date_opp = {}
    for g in db_games:
        opp = g['away_team_id'] if g['home_team_id'] == team_slug else g['home_team_id']
        key = (g['date'], opp)
        db_by_date_opp[key] = g
    
    issues = []
    ok_count = 0
    
    # Check each DB game against D1BB
    for g in db_games:
        opp = g['away_team_id'] if g['home_team_id'] == team_slug else g['home_team_id']
        key = (g['date'], opp)
        
        if key in d1bb_by_date_opp:
            d1g = d1bb_by_date_opp[key]
            
            # Check home/away
            db_is_home = (g['home_team_id'] == team_slug)
            if db_is_home != d1g['is_home']:
                issues.append({
                    'type': 'wrong_home_away',
                    'date': g['date'],
                    'game_id': g['id'],
                    'opponent': opp,
                    'db_home': db_is_home,
                    'd1bb_home': d1g['is_home'],
                })
            else:
                ok_count += 1
                if verbose:
                    ha = "vs" if db_is_home else "@"
                    print(f"  ✅ {g['date']}  {ha} {opp}")
        else:
            issues.append({
                'type': 'extra_in_db',
                'date': g['date'],
                'game_id': g['id'],
                'opponent': opp,
                'matchup': f"{g['away_team_id']} @ {g['home_team_id']}",
            })
    
    # Check D1BB games missing from DB
    for g in d1bb_games:
        key = (g['date'], g['opponent_team_id'])
        if key not in db_by_date_opp:
            ha = "vs" if g['is_home'] else "@"
            issues.append({
                'type': 'missing_in_db',
                'date': g['date'],
                'opponent': g['opponent_team_id'],
                'd1bb_slug': g['opponent_d1bb_slug'],
                'is_home': g['is_home'],
            })
    
    # Report
    print(f"📊 Results: {ok_count} games OK, {len(issues)} issues found")
    print()
    
    if not issues:
        print("✅ Schedule is clean!")
        return True
    
    # Group by issue type
    extra = [i for i in issues if i['type'] == 'extra_in_db']
    missing = [i for i in issues if i['type'] == 'missing_in_db']
    wrong_ha = [i for i in issues if i['type'] == 'wrong_home_away']
    
    if extra:
        print(f"🗑️  Extra games in DB (not on D1Baseball): {len(extra)}")
        for i in extra:
            print(f"   {i['date']}  {i['matchup']}  (id: {i['game_id']})")
    
    if missing:
        print(f"➕ Missing from DB (on D1Baseball): {len(missing)}")
        for i in missing:
            ha = "vs" if i['is_home'] else "@"
            print(f"   {i['date']}  {ha} {i['opponent']} (d1bb: {i['d1bb_slug']})")
    
    if wrong_ha:
        print(f"🔄 Wrong home/away: {len(wrong_ha)}")
        for i in wrong_ha:
            print(f"   {i['date']}  vs {i['opponent']}  DB={'home' if i['db_home'] else 'away'} D1BB={'home' if i['d1bb_home'] else 'away'}  (id: {i['game_id']})")
    
    if fix:
        print()
        conn = get_connection()
        cur = conn.cursor()
        fixed = 0
        
        for i in extra:
            game_id = i['game_id']
            cur.execute('DELETE FROM games WHERE id = ?', (game_id,))
            cur.execute('DELETE FROM model_predictions WHERE game_id = ?', (game_id,))
            cur.execute('DELETE FROM totals_predictions WHERE game_id = ?', (game_id,))
            print(f"   🗑️  Deleted {game_id}")
            fixed += 1
        
        # Insert missing games
        for i in missing:
            opp = i['opponent']
            date = i['date']
            if i['is_home']:
                home_id, away_id = team_slug, opp
            else:
                home_id, away_id = opp, team_slug
            game_id = f"{date}_{away_id}_{home_id}"
            
            # Check it doesn't already exist (could be ID format mismatch)
            cur.execute('SELECT id FROM games WHERE id = ?', (game_id,))
            if cur.fetchone():
                continue
            
            # Get score from D1Baseball if available
            d1g = d1bb_by_date_opp.get((date, opp))
            home_score = away_score = winner_id = None
            status = 'scheduled'
            if d1g and d1g.get('result'):
                r = d1g['result']
                if i['is_home']:
                    home_score = r['team_score']
                    away_score = r['opp_score']
                else:
                    home_score = r['opp_score']
                    away_score = r['team_score']
                winner_id = home_id if home_score > away_score else away_id
                status = 'final'
            
            cur.execute('''
                INSERT INTO games (id, date, home_team_id, away_team_id, home_score, away_score, winner_id, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (game_id, date, home_id, away_id, home_score, away_score, winner_id, status))
            score_str = f" ({home_score}-{away_score})" if home_score is not None else ""
            print(f"   ➕ Added {game_id}{score_str}")
            fixed += 1
        
        # Backfill scores on existing games missing them
        scores_filled = 0
        for g in db_games:
            if g['status'] == 'final' or g['home_score'] is not None:
                continue
            opp = g['away_team_id'] if g['home_team_id'] == team_slug else g['home_team_id']
            key = (g['date'], opp)
            d1g = d1bb_by_date_opp.get(key)
            if d1g and d1g.get('result'):
                r = d1g['result']
                db_is_home = (g['home_team_id'] == team_slug)
                if db_is_home:
                    hs, aws = r['team_score'], r['opp_score']
                else:
                    hs, aws = r['opp_score'], r['team_score']
                winner = g['home_team_id'] if hs > aws else g['away_team_id']
                cur.execute('''
                    UPDATE games SET home_score=?, away_score=?, winner_id=?, status='final'
                    WHERE id=?
                ''', (hs, aws, winner, g['id']))
                print(f"   📝 Score: {g['id']} → {hs}-{aws}")
                scores_filled += 1
        
        conn.commit()
        conn.close()
        print(f"\n🔧 Fixed {fixed} issues, filled {scores_filled} scores.")
    
    return False


def main():
    parser = argparse.ArgumentParser(description='Verify team schedule against D1Baseball')
    parser.add_argument('team', help='Team slug (e.g., omaha, mississippi-state)')
    parser.add_argument('--fix', action='store_true', help='Auto-delete extra DB games')
    parser.add_argument('--verbose', action='store_true', help='Show all games including OK')
    args = parser.parse_args()
    
    verify(args.team, fix=args.fix, verbose=args.verbose)


if __name__ == '__main__':
    main()
