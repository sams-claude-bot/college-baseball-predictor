#!/usr/bin/env python3

import sys
import json

def mark_team_failed(team_id):
    progress_file = '/home/sam/college-baseball-predictor/data/stats_scraper_progress.json'
    
    with open(progress_file, 'r') as f:
        progress = json.load(f)
    
    failed_teams = progress['progress'].get('teams_failed', [])
    if team_id not in failed_teams:
        failed_teams.append(team_id)
        progress['progress']['teams_failed'] = failed_teams
        progress['progress']['current_team'] = None
    
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)
    
    print(f"Marked {team_id} as failed")
    print(f"Total failed teams: {len(failed_teams)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        mark_team_failed(sys.argv[1])
    else:
        mark_team_failed("arkansas")