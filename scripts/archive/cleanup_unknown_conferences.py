#!/usr/bin/env python3
"""
Clean up teams with unknown conferences using web search.

This script searches for conference information for teams that have NULL, empty, 
or 'Unknown' conference values and updates the database accordingly.
"""

import sqlite3
import time
import json
import logging
from pathlib import Path
import sys
import os

# Add the parent directory to the path so we can import from scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import web_search from the main OpenClaw environment - we'll use subprocess for this
import subprocess

PROJECT_DIR = Path(__file__).parent.parent
DB_PATH = PROJECT_DIR / 'data' / 'baseball.db'
PROGRESS_FILE = PROJECT_DIR / 'data' / 'conference_cleanup_progress.json'

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger('conf_cleanup')

# Conference mappings for common abbreviations
CONFERENCE_MAPPINGS = {
    'Southeastern Conference': 'SEC',
    'Atlantic Coast Conference': 'ACC', 
    'Big Ten Conference': 'Big Ten',
    'Big 12 Conference': 'Big 12',
    'American Athletic Conference': 'AAC',
    'Sun Belt Conference': 'Sun Belt',
    'Conference USA': 'C-USA',
    'Mountain West Conference': 'MWC',
    'West Coast Conference': 'WCC',
    'Big East Conference': 'Big East',
    'Atlantic 10 Conference': 'A-10',
    'Colonial Athletic Association': 'CAA',
    'Missouri Valley Conference': 'MVC',
    'Southern Conference': 'SoCon',
    'ASUN Conference': 'ASUN',
    'Southland Conference': 'Southland',
    'Ohio Valley Conference': 'OVC',
    'Patriot League': 'Patriot',
    'Ivy League': 'Ivy',
    'Metro Atlantic Athletic Conference': 'MAAC',
    'Northeast Conference': 'NEC',
    'Horizon League': 'Horizon',
    'Big West Conference': 'Big West',
    'Summit League': 'Summit',
    'Mid-Eastern Athletic Conference': 'MEAC',
    'Southwestern Athletic Conference': 'SWAC',
    'Western Athletic Conference': 'WAC',
    'Mid-American Conference': 'MAC',
    'America East Conference': 'America East',
    'Big South Conference': 'Big South',
}

# D1 Conference list for verification
D1_CONFERENCES = {
    'SEC', 'ACC', 'Big Ten', 'Big 12', 'AAC', 'Sun Belt', 'C-USA', 'MWC', 
    'WCC', 'Big East', 'A-10', 'CAA', 'MVC', 'SoCon', 'ASUN', 'Southland',
    'OVC', 'Patriot', 'Ivy', 'MAAC', 'NEC', 'Horizon', 'Big West', 'Summit',
    'MEAC', 'SWAC', 'WAC', 'MAC', 'America East', 'Big South', 'Independent'
}


def load_progress():
    """Load progress from JSON file if it exists."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_progress(progress):
    """Save progress to JSON file."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def search_team_conference(team_name, team_nickname=""):
    """
    Search for a team's conference using web search.
    Returns tuple: (conference, athletics_url)
    """
    search_queries = [
        f"{team_name} baseball conference 2024",
        f"{team_name} {team_nickname} baseball conference",
        f"{team_name} athletics conference",
        f"{team_name} college baseball"
    ]
    
    for query in search_queries:
        try:
            log.info(f"  Searching: {query}")
            
            # Use subprocess to call the web_search function from OpenClaw
            result = subprocess.run([
                'python3', '-c', f'''
import sys
sys.path.append("/home/sam/.openclaw/skills")
from web_search import web_search
result = web_search("{query}", count=5)
print("SEARCH_RESULTS_START")
for item in result:
    print(f"TITLE: {{item.get('title', '')}}") 
    print(f"URL: {{item.get('url', '')}}")
    print(f"SNIPPET: {{item.get('snippet', '')}}")
    print("---")
print("SEARCH_RESULTS_END")
'''
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                output = result.stdout
                return parse_search_results(output, team_name)
            else:
                log.warning(f"Search failed for {query}: {result.stderr}")
                
        except Exception as e:
            log.warning(f"Search error for {query}: {e}")
            
        time.sleep(2)  # Small delay between queries
    
    return None, None


def parse_search_results(search_output, team_name):
    """Parse search results to extract conference and athletics URL."""
    if "SEARCH_RESULTS_START" not in search_output:
        return None, None
        
    lines = search_output.split('\n')
    start_idx = next((i for i, line in enumerate(lines) if "SEARCH_RESULTS_START" in line), -1)
    end_idx = next((i for i, line in enumerate(lines) if "SEARCH_RESULTS_END" in line), len(lines))
    
    if start_idx == -1:
        return None, None
        
    results = lines[start_idx+1:end_idx]
    
    conference = None
    athletics_url = None
    
    current_title = ""
    current_url = ""
    current_snippet = ""
    
    for line in results:
        line = line.strip()
        if line.startswith("TITLE: "):
            current_title = line[7:]
        elif line.startswith("URL: "):
            current_url = line[5:]
        elif line.startswith("SNIPPET: "):
            current_snippet = line[9:]
        elif line == "---":
            # Process this result
            conf = extract_conference_from_text(current_title + " " + current_snippet)
            if conf and not conference:
                conference = conf
                
            if current_url and ("athletics" in current_url.lower() or team_name.lower().replace(' ', '') in current_url.lower()):
                if not athletics_url:
                    athletics_url = current_url
                    
            # Reset for next result
            current_title = current_url = current_snippet = ""
    
    return conference, athletics_url


def extract_conference_from_text(text):
    """Extract conference name from text snippet."""
    text_lower = text.lower()
    
    # Look for explicit conference mentions
    for full_name, abbrev in CONFERENCE_MAPPINGS.items():
        if full_name.lower() in text_lower:
            return abbrev
            
    # Look for abbreviated forms
    for abbrev in D1_CONFERENCES:
        if abbrev.lower() in text_lower and abbrev != 'Independent':
            return abbrev
    
    # Look for specific patterns
    patterns = [
        'naia', 'd2', 'division ii', 'division 2', 'd3', 'division iii', 'division 3',
        'juco', 'junior college', 'community college', 'division i', 'division 1', 'd1'
    ]
    
    for pattern in patterns:
        if pattern in text_lower:
            if pattern in ['naia', 'd2', 'division ii', 'division 2', 'd3', 'division iii', 'division 3', 'juco', 'junior college', 'community college']:
                return 'Non-D1'
    
    return None


def get_unknown_teams():
    """Get all teams with unknown conferences."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    
    teams = conn.execute("""
        SELECT id, name, nickname 
        FROM teams 
        WHERE conference IS NULL OR conference = '' OR conference = 'Unknown'
        ORDER BY id
    """).fetchall()
    
    conn.close()
    return teams


def update_team_conference(team_id, conference, athletics_url=None):
    """Update a team's conference and optionally athletics URL."""
    conn = sqlite3.connect(str(DB_PATH))
    
    if athletics_url:
        conn.execute("""
            UPDATE teams 
            SET conference = ?, athletics_url = ? 
            WHERE id = ?
        """, (conference, athletics_url, team_id))
    else:
        conn.execute("""
            UPDATE teams 
            SET conference = ? 
            WHERE id = ?
        """, (conference, team_id))
    
    conn.commit()
    conn.close()


def batch_update_teams(updates):
    """Batch update teams with conference information."""
    conn = sqlite3.connect(str(DB_PATH))
    
    for team_id, conference, athletics_url in updates:
        if athletics_url:
            conn.execute("""
                UPDATE teams 
                SET conference = ?, athletics_url = ? 
                WHERE id = ?
            """, (conference, athletics_url, team_id))
        else:
            conn.execute("""
                UPDATE teams 
                SET conference = ? 
                WHERE id = ?
            """, (conference, team_id))
    
    conn.commit()
    conn.close()
    log.info(f"Batch updated {len(updates)} teams")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Clean up unknown team conferences")
    parser.add_argument('--start', type=int, default=0, help='Start index')
    parser.add_argument('--count', type=int, help='Number of teams to process')
    parser.add_argument('--resume', action='store_true', help='Resume from progress file')
    args = parser.parse_args()
    
    teams = get_unknown_teams()
    progress = load_progress() if args.resume else {}
    
    start_idx = args.start
    if args.resume and progress:
        processed_teams = set(progress.get('processed', []))
        teams = [t for t in teams if t['id'] not in processed_teams]
        log.info(f"Resuming - {len(processed_teams)} already processed, {len(teams)} remaining")
    
    if args.count:
        teams = teams[start_idx:start_idx + args.count]
    else:
        teams = teams[start_idx:]
    
    log.info(f"Processing {len(teams)} teams starting from index {start_idx}")
    
    batch_updates = []
    processed = set(progress.get('processed', []))
    
    try:
        for i, team in enumerate(teams):
            log.info(f"[{i+1}/{len(teams)}] Processing: {team['name']} ({team['id']})")
            
            conference, athletics_url = search_team_conference(team['name'], team['nickname'] or "")
            
            if conference:
                log.info(f"  Found: {conference}" + (f" | URL: {athletics_url}" if athletics_url else ""))
                batch_updates.append((team['id'], conference, athletics_url))
                processed.add(team['id'])
            else:
                log.info(f"  No conference found, marking as Unknown for manual review")
                batch_updates.append((team['id'], 'Unknown', None))
                processed.add(team['id'])
            
            # Save progress periodically
            if len(batch_updates) >= 5:
                batch_update_teams(batch_updates)
                progress['processed'] = list(processed)
                save_progress(progress)
                batch_updates = []
                log.info(f"Progress saved - {len(processed)} teams processed")
            
            # Delay between searches
            delay = 8  # 8 second delay as requested
            log.info(f"  Waiting {delay}s before next search...")
            time.sleep(delay)
    
    except KeyboardInterrupt:
        log.info("Interrupted by user")
    finally:
        # Process any remaining batch updates
        if batch_updates:
            batch_update_teams(batch_updates)
            progress['processed'] = list(processed)
            save_progress(progress)
        
        log.info(f"Completed processing. Total processed: {len(processed)}")


if __name__ == '__main__':
    main()