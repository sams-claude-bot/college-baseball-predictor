#!/usr/bin/env python3

import sqlite3
import requests
from bs4 import BeautifulSoup
import time
import json
import re
from datetime import datetime
from urllib.parse import urljoin, urlparse

def get_all_teams():
    """Get all teams from database"""
    conn = sqlite3.connect('data/baseball.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, name, nickname, conference 
        FROM teams 
        ORDER BY conference, name
    """)
    
    teams = cursor.fetchall()
    conn.close()
    
    return teams

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

def find_espn_team_page(team_name, team_nickname):
    """Try to find ESPN team page for a given team"""
    # ESPN URL patterns to try
    base_patterns = [
        "https://www.espn.com/college-sports/team/_/id/{}/{}",
        "https://www.espn.com/college-sports/team/schedule/_/id/{}/{}"
    ]
    
    # Common transformations for team names
    search_names = []
    
    # Try exact name
    search_names.append(f"{team_name} {team_nickname}".lower())
    search_names.append(team_name.lower())
    search_names.append(team_nickname.lower())
    
    # Try with common substitutions
    name_variations = [
        team_name.lower().replace(' ', '-').replace('&', 'and'),
        team_name.lower().replace(' state', '-state').replace(' ', '-'),
        team_name.lower().replace('university of ', '').replace(' ', '-'),
        team_nickname.lower().replace(' ', '-')
    ]
    
    search_names.extend(name_variations)
    
    print(f"Searching ESPN for: {team_name} {team_nickname}")
    print(f"  Trying variations: {search_names[:3]}...")
    
    # For now, return placeholder - we'll implement ESPN scraping carefully
    return None, None

def verify_team_schedule(team_id, team_name, team_nickname):
    """Verify a single team's schedule against ESPN"""
    print(f"\n=== Verifying {team_name} {team_nickname} ({team_id}) ===")
    
    # Get our database games
    db_games = get_team_games_from_db(team_id)
    print(f"Database games: {len(db_games)}")
    
    # Try to find ESPN page
    espn_url, espn_games = find_espn_team_page(team_name, team_nickname)
    
    if not espn_url:
        print(f"  ⚠️  Could not find ESPN page for {team_name}")
        return {'status': 'espn_not_found', 'db_games': len(db_games)}
    
    # Compare schedules (placeholder)
    return {'status': 'placeholder', 'db_games': len(db_games)}

def priority_conferences():
    """Return conferences in priority order for verification"""
    return [
        'sec',      # SEC - highest priority
        'acc',      # ACC 
        'big12',    # Big 12
        'big-ten',  # Big Ten
        'pac12',    # Pac-12
        'aac',      # American Athletic
        'mwc',      # Mountain West
        'c-usa',    # Conference USA
    ]

def run_schedule_verification(limit_teams=None, target_conference=None):
    """Run schedule verification for all or selected teams"""
    print("=== ESPN SCHEDULE VERIFICATION STARTED ===")
    start_time = datetime.now()
    
    teams = get_all_teams()
    print(f"Total teams in database: {len(teams)}")
    
    if target_conference:
        teams = [t for t in teams if t[3] == target_conference]
        print(f"Filtered to {target_conference}: {len(teams)} teams")
    
    if limit_teams:
        teams = teams[:limit_teams]
        print(f"Limited to first {limit_teams} teams")
    
    verification_results = {}
    
    # Group teams by conference for organized processing
    teams_by_conf = {}
    for team in teams:
        conf = team[3] or 'no-conference'
        if conf not in teams_by_conf:
            teams_by_conf[conf] = []
        teams_by_conf[conf].append(team)
    
    # Process conferences in priority order
    priority_confs = priority_conferences()
    processed_conferences = set()
    
    # First, process priority conferences
    for conf in priority_confs:
        if conf in teams_by_conf and conf not in processed_conferences:
            print(f"\n{'='*20} {conf.upper()} CONFERENCE {'='*20}")
            for team in teams_by_conf[conf]:
                team_id, name, nickname, conference = team
                result = verify_team_schedule(team_id, name, nickname)
                verification_results[team_id] = result
                
                # Rate limiting
                time.sleep(0.5)
            processed_conferences.add(conf)
    
    # Then process remaining conferences
    for conf, conf_teams in teams_by_conf.items():
        if conf not in processed_conferences:
            print(f"\n{'='*20} {conf.upper()} CONFERENCE {'='*20}")
            for team in conf_teams:
                team_id, name, nickname, conference = team
                result = verify_team_schedule(team_id, name, nickname)
                verification_results[team_id] = result
                
                # Rate limiting
                time.sleep(0.5)
    
    # Save results
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    summary = {
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration_seconds': duration,
        'teams_processed': len(verification_results),
        'results': verification_results
    }
    
    with open('data/espn_verification_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== VERIFICATION COMPLETE ===")
    print(f"Processed {len(verification_results)} teams in {duration:.1f} seconds")
    print("Results saved to data/espn_verification_results.json")
    
    return summary

if __name__ == "__main__":
    # Start with SEC conference as a test
    print("Starting with SEC conference for initial verification...")
    results = run_schedule_verification(target_conference='sec')
    
    # Print summary
    statuses = {}
    for team_id, result in results['results'].items():
        status = result['status']
        statuses[status] = statuses.get(status, 0) + 1
    
    print(f"\nSummary of verification statuses:")
    for status, count in statuses.items():
        print(f"  {status}: {count} teams")