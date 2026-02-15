#!/usr/bin/env python3

import json
import sqlite3
import re
from datetime import datetime

class P4StatsProcessor:
    def __init__(self):
        self.progress_file = '/home/sam/college-baseball-predictor/data/stats_scraper_progress.json'
        self.teams_file = '/home/sam/college-baseball-predictor/data/p4_team_urls.json'
        self.db_file = '/home/sam/college-baseball-predictor/data/baseball.db'
        
    def load_team_urls(self):
        """Load team URLs"""
        with open(self.teams_file, 'r') as f:
            return json.load(f)
    
    def load_progress(self):
        """Load current progress"""
        try:
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self.create_initial_progress()
    
    def save_progress(self, progress_data):
        """Save progress to file"""
        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
    
    def get_next_team(self):
        """Get the next team to process"""
        progress = self.load_progress()
        team_urls = self.load_team_urls()
        
        completed_teams = progress['progress'].get('completed_teams', [])
        failed_teams = progress['progress'].get('teams_failed', [])
        
        # Process teams in conference order
        for conference in progress['team_mapping']:
            for team_id in progress['team_mapping'][conference]:
                if team_id not in completed_teams and team_id not in failed_teams and team_id in team_urls['teams']:
                    return team_id, team_urls['teams'][team_id], conference
        
        return None, None, None
    
    def clean_name(self, name):
        """Remove asterisk and clean up player name"""
        return name.replace("* ", "").strip()
    
    def parse_compound_field(self, field_str, separator=' - '):
        """Parse compound fields like GP-GS or SB-ATT"""
        if separator in field_str:
            parts = field_str.split(separator)
            try:
                return int(parts[0].strip()), int(parts[1].strip())
            except ValueError:
                return 0, 0
        try:
            return int(field_str.strip()), 0
        except ValueError:
            return 0, 0
    
    def safe_float(self, value, default=0.0):
        """Safely convert to float"""
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def safe_int(self, value, default=0):
        """Safely convert to int"""
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
    
    def parse_batting_stats_from_snapshot(self, snapshot_text):
        """Extract batting stats from browser snapshot text"""
        batting_players = []
        
        # Look for batting table data in snapshot
        lines = snapshot_text.split('\n')
        in_batting_section = False
        
        for line in lines:
            line = line.strip()
            
            # Look for batting table rows - they usually have player names and stats
            if 'cell "' in line and any(x in line for x in ['.', 'AVG', 'OPS']):
                # Try to extract stats from table cells
                # This is a simplified parser - in a real implementation we'd need more sophisticated parsing
                continue
        
        return batting_players
    
    def parse_pitching_stats_from_snapshot(self, snapshot_text):
        """Extract pitching stats from browser snapshot text"""
        pitching_players = []
        
        # Similar logic for pitching stats
        lines = snapshot_text.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Look for pitching table rows
            if 'cell "' in line and any(x in line for x in ['ERA', 'WHIP']):
                continue
        
        return pitching_players
    
    def insert_team_stats_from_data(self, team_id, batting_data, pitching_data):
        """Insert parsed stats into database"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        # Clear existing team data
        cursor.execute("DELETE FROM player_stats WHERE team_id = ?", (team_id,))
        
        batting_count = len(batting_data)
        pitching_count = len(pitching_data)
        
        # Insert batting stats
        for player_data in batting_data:
            cursor.execute("""
                INSERT OR REPLACE INTO player_stats (
                    team_id, name, number, games, at_bats, runs, hits, doubles, triples,
                    home_runs, rbi, walks, strikeouts, stolen_bases, caught_stealing,
                    batting_avg, obp, slg, ops, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (team_id, *player_data, datetime.now().isoformat()))
        
        # Insert pitching stats
        for player_data in pitching_data:
            cursor.execute("""
                INSERT OR IGNORE INTO player_stats (team_id, name, number) VALUES (?, ?, ?)
            """, (team_id, player_data[0], player_data[1]))
            
            cursor.execute("""
                UPDATE player_stats SET
                    wins = ?, losses = ?, era = ?, games_pitched = ?, games_started = ?,
                    saves = ?, innings_pitched = ?, hits_allowed = ?, runs_allowed = ?,
                    earned_runs = ?, walks_allowed = ?, strikeouts_pitched = ?, whip = ?,
                    updated_at = ?
                WHERE team_id = ? AND name = ?
            """, (*player_data[2:], datetime.now().isoformat(), team_id, player_data[0]))
        
        conn.commit()
        conn.close()
        
        return batting_count, pitching_count
    
    def mark_team_completed(self, team_id, batting_count, pitching_count):
        """Mark a team as completed in progress"""
        progress = self.load_progress()
        
        completed_teams = progress['progress'].get('completed_teams', [])
        completed_teams.append(team_id)
        
        progress['progress']['completed_teams'] = completed_teams
        progress['progress']['teams_completed'] = len(completed_teams)
        progress['progress']['batting_stats_scraped'] += batting_count
        progress['progress']['pitching_stats_scraped'] += pitching_count
        progress['progress']['current_team'] = None
        
        self.save_progress(progress)
        return progress
    
    def mark_team_failed(self, team_id):
        """Mark a team as failed in progress"""
        progress = self.load_progress()
        
        failed_teams = progress['progress'].get('teams_failed', [])
        failed_teams.append(team_id)
        progress['progress']['teams_failed'] = failed_teams
        progress['progress']['current_team'] = None
        
        self.save_progress(progress)
        return progress
    
    def get_progress_summary(self):
        """Get current progress summary"""
        progress = self.load_progress()
        p = progress['progress']
        
        completed_count = len(p.get('completed_teams', []))
        failed_count = len(p.get('teams_failed', []))
        batting_stats = p.get('batting_stats_scraped', 0)
        pitching_stats = p.get('pitching_stats_scraped', 0)
        
        return {
            'completed': completed_count,
            'failed': failed_count,
            'total': 67,
            'batting_stats': batting_stats,
            'pitching_stats': pitching_stats,
            'remaining': 67 - completed_count - failed_count
        }

def main():
    processor = P4StatsProcessor()
    
    # Get next team to process
    team_id, url, conference = processor.get_next_team()
    
    if team_id:
        print(f"Next team to process: {team_id} ({conference})")
        print(f"URL: {url}")
        
        # Show progress
        summary = processor.get_progress_summary()
        print(f"\nProgress: {summary['completed']}/{summary['total']} completed, {summary['failed']} failed")
        print(f"Stats collected: {summary['batting_stats']} batting, {summary['pitching_stats']} pitching")
        
    else:
        print("All teams processed!")
        summary = processor.get_progress_summary()
        print(f"Final stats: {summary['completed']}/{summary['total']} completed")
        print(f"Failed teams: {summary['failed']}")
        print(f"Total stats: {summary['batting_stats']} batting, {summary['pitching_stats']} pitching")

if __name__ == "__main__":
    main()