#!/usr/bin/env python3

import json
import sqlite3
import re
from datetime import datetime
import time

class ESPNScheduleVerifier:
    def __init__(self):
        self.verification_results = {}
        self.discrepancies = []
        
    def get_team_info(self, team_id):
        """Get team info from database"""
        conn = sqlite3.connect('data/baseball.db')
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, name, nickname FROM teams WHERE id = ?", (team_id,))
        team = cursor.fetchone()
        conn.close()
        
        return team
        
    def get_team_games_from_db(self, team_id):
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
    
    def get_espn_team_urls(self, team_id):
        """Get ESPN URLs to try for a team"""
        espn_mapping = {
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
            'vanderbilt': '238'
        }
        
        if team_id in espn_mapping:
            espn_id = espn_mapping[team_id]
            return [
                f"https://www.espn.com/college-sports/team/schedule/_/id/{espn_id}",
                f"https://www.espn.com/college-sports/team/schedule/_/id/{espn_id}/season/2026"
            ]
        return []
    
    def verify_team_schedule(self, team_id):
        """Verify a single team's schedule against ESPN"""
        team_info = self.get_team_info(team_id)
        if not team_info:
            return None
            
        team_id, team_name, team_nickname = team_info
        print(f"\n=== Verifying {team_name} ({team_id}) ===")
        
        # Get our database games
        db_games = self.get_team_games_from_db(team_id)
        print(f"Database games: {len(db_games)}")
        
        # Get ESPN URLs to try
        espn_urls = self.get_espn_team_urls(team_id)
        
        if not espn_urls:
            print(f"  ⚠️  No ESPN URLs configured for {team_name}")
            return {
                'status': 'no_espn_urls',
                'db_games_count': len(db_games)
            }
        
        # Try each ESPN URL
        espn_schedule_found = False
        espn_games = []
        
        # This is where we would call the web_fetch tool
        # For now, I'll create a placeholder that shows the process
        
        print(f"  ESPN URLs to try: {len(espn_urls)}")
        for url in espn_urls:
            print(f"    {url}")
        
        # Placeholder for actual ESPN fetching
        result = {
            'status': 'prepared_for_fetch',
            'team_id': team_id,
            'team_name': team_name,
            'db_games_count': len(db_games),
            'espn_urls': espn_urls,
            'sample_db_games': [
                {
                    'id': game[0],
                    'date': game[1], 
                    'home': game[2],
                    'away': game[3],
                    'venue': game[4]
                } for game in db_games[:5]
            ]
        }
        
        return result
    
    def run_verification_for_conference(self, conference='SEC', limit=None):
        """Run verification for all teams in a conference"""
        conn = sqlite3.connect('data/baseball.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, name, nickname 
            FROM teams 
            WHERE conference = ?
            ORDER BY name
        """, (conference,))
        
        teams = cursor.fetchall()
        conn.close()
        
        if limit:
            teams = teams[:limit]
        
        print(f"=== VERIFYING {conference} CONFERENCE ({len(teams)} teams) ===")
        
        for team_id, team_name, team_nickname in teams:
            result = self.verify_team_schedule(team_id)
            if result:
                self.verification_results[team_id] = result
            
            # Small delay to be respectful
            time.sleep(0.5)
        
        return self.verification_results

def show_team_for_manual_verification(team_id):
    """Show a single team's info for manual ESPN verification"""
    verifier = ESPNScheduleVerifier()
    result = verifier.verify_team_schedule(team_id)
    
    if result and 'espn_urls' in result:
        print(f"\n{'='*60}")
        print(f"MANUAL ESPN VERIFICATION FOR {result['team_name'].upper()}")
        print(f"{'='*60}")
        print(f"Team ID: {result['team_id']}")
        print(f"Games in database: {result['db_games_count']}")
        print(f"\nESPN URLs to check:")
        for url in result['espn_urls']:
            print(f"  {url}")
        
        print(f"\nSample games from database:")
        for i, game in enumerate(result['sample_db_games'], 1):
            home_away = "@ " + game['home'] if game['home'] != result['team_id'] else "vs " + game['away']
            print(f"  {i}. {game['date']} - {home_away} ({game['venue'] or 'No venue'})")
    
    return result

if __name__ == "__main__":
    # Start with a single team for testing - Mississippi State
    print("Testing ESPN verification with Mississippi State...")
    result = show_team_for_manual_verification('mississippi-state')
    
    print("\n" + "="*60)
    print("NEXT STEPS FOR FULL VERIFICATION:")
    print("="*60)
    print("1. Test the ESPN URLs manually")
    print("2. Use web_fetch tool to get schedules")
    print("3. Parse the ESPN schedule data") 
    print("4. Compare with database games")
    print("5. Report discrepancies")