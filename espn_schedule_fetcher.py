#!/usr/bin/env python3

import sqlite3
import json
import re
from datetime import datetime
import time

def get_team_games_from_db(team_id):
    """Get all games for a team from our database"""
    conn = sqlite3.connect('data/baseball.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, date, home_team_id, away_team_id, venue, status
        FROM games 
        WHERE home_team_id = ? OR away_team_id = ?
        ORDER BY date
    """, (team_id, team_id))
    
    games = cursor.fetchall()
    conn.close()
    
    return games

def get_espn_team_id_mapping():
    """Common ESPN team ID mappings for major schools"""
    return {
        'alabama': '333',
        'arkansas': '8', 
        'auburn': '2',
        'florida': '57',
        'georgia': '61',
        'kentucky': '96',
        'lsu': '99',
        'mississippi-state': '344',
        'missouri': '142',
        'oklahoma': '201',
        'ole-miss': '145',
        'south-carolina': '2579',
        'tennessee': '2633',
        'texas': '251',
        'texas-am': '245',
        'vanderbilt': '238',
        'florida-state': '52',
        'georgia-tech': '59',
        'duke': '150',
        'north-carolina': '153',
        'virginia': '258',
        'virginia-tech': '259',
        'clemson': '228',
        'miami': '2390',
        'notre-dame': '87',
        'louisville': '97'
    }

def try_espn_schedules(team_id, team_name):
    """Try different ESPN URL patterns to find the team schedule"""
    espn_mapping = get_espn_team_id_mapping()
    
    # Try known mapping first
    if team_id in espn_mapping:
        espn_id = espn_mapping[team_id]
        espn_urls = [
            f"https://www.espn.com/college-sports/team/schedule/_/id/{espn_id}/season/2026",
            f"https://www.espn.com/college-sports/team/_/id/{espn_id}",
        ]
        return espn_urls
    
    # Try to construct URLs based on team name
    name_variants = [
        team_name.lower().replace(' ', '-').replace('&', 'and'),
        team_name.lower().replace(' state', '-state').replace(' ', '-'),
        team_name.lower().replace('university of ', '').replace(' ', '-'),
    ]
    
    # These would be guesses - we'd need to validate them
    return []

def verify_single_team(team_id, team_name, team_nickname):
    """Verify a single team's schedule - this will be called by main script"""
    print(f"\nVerifying: {team_name} ({team_id})")
    
    # Get our games
    db_games = get_team_games_from_db(team_id)
    print(f"  Database games: {len(db_games)}")
    
    if len(db_games) == 0:
        return {
            'status': 'no_games_in_db',
            'team_id': team_id,
            'team_name': team_name,
            'db_games_count': 0
        }
    
    # Try to find ESPN URLs
    espn_urls = try_espn_schedules(team_id, team_name)
    
    return {
        'status': 'ready_for_espn_fetch',
        'team_id': team_id,
        'team_name': team_name,
        'db_games_count': len(db_games),
        'espn_urls': espn_urls,
        'sample_games': db_games[:3]  # First 3 games for reference
    }

def get_sec_teams():
    """Get SEC teams for verification"""
    conn = sqlite3.connect('data/baseball.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, name, nickname 
        FROM teams 
        WHERE conference = 'SEC'
        ORDER BY name
    """)
    
    teams = cursor.fetchall()
    conn.close()
    
    return teams

if __name__ == "__main__":
    print("=== SEC TEAM SCHEDULE VERIFICATION PREP ===")
    
    sec_teams = get_sec_teams()
    print(f"Found {len(sec_teams)} SEC teams")
    
    verification_prep = {}
    
    for team_id, team_name, team_nickname in sec_teams:
        result = verify_single_team(team_id, team_name, team_nickname)
        verification_prep[team_id] = result
    
    # Save prep results
    with open('data/sec_verification_prep.json', 'w') as f:
        json.dump(verification_prep, f, indent=2)
    
    print(f"\nPreparation complete. Results saved to sec_verification_prep.json")
    
    # Summary
    statuses = {}
    for team_id, result in verification_prep.items():
        status = result['status']
        statuses[status] = statuses.get(status, 0) + 1
    
    print("\nStatus Summary:")
    for status, count in statuses.items():
        print(f"  {status}: {count} teams")
    
    # Show teams ready for ESPN fetch
    ready_teams = [r for r in verification_prep.values() if r['status'] == 'ready_for_espn_fetch']
    print(f"\nTeams ready for ESPN verification: {len(ready_teams)}")
    for team in ready_teams:
        print(f"  {team['team_name']}: {team['db_games_count']} games, {len(team['espn_urls'])} ESPN URLs to try")