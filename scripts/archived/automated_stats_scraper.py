#!/usr/bin/env python3

import json
import sqlite3
import time
import re
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException

class P4StatsCollector:
    def __init__(self):
        self.progress_file = '/home/sam/college-baseball-predictor/data/stats_scraper_progress.json'
        self.teams_file = '/home/sam/college-baseball-predictor/data/p4_team_urls.json'
        self.db_file = '/home/sam/college-baseball-predictor/data/baseball.db'
        
        # Chrome options for headless operation
        self.chrome_options = Options()
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-dev-shm-usage')
        self.chrome_options.add_argument('--disable-gpu')
        self.chrome_options.add_argument('--window-size=1920,1080')
        # Keep it non-headless for debugging initially
        # self.chrome_options.add_argument('--headless')
        
        self.driver = None
        self.wait = None
        
    def setup_driver(self):
        """Initialize the Chrome driver"""
        try:
            self.driver = webdriver.Chrome(options=self.chrome_options)
            self.wait = WebDriverWait(self.driver, 10)
            return True
        except Exception as e:
            print(f"Failed to initialize driver: {e}")
            return False
    
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
    
    def create_initial_progress(self):
        """Create initial progress structure"""
        return {
            "test_run_started": datetime.now().isoformat(),
            "total_teams": 67,
            "conferences": {"SEC": 16, "Big Ten": 18, "ACC": 17, "Big 12": 16},
            "progress": {
                "current_conference": "SEC",
                "teams_completed": 0,
                "teams_failed": [],
                "batting_stats_scraped": 0,
                "pitching_stats_scraped": 0,
                "current_team": None,
                "completed_teams": []
            },
            "team_mapping": {
                "SEC": ["alabama", "arkansas", "auburn", "florida", "georgia", "kentucky", "lsu", "mississippi-state", "missouri", "oklahoma", "ole-miss", "south-carolina", "tennessee", "texas", "texas-am", "vanderbilt"],
                "Big Ten": ["illinois", "indiana", "iowa", "maryland", "michigan", "michigan-state", "minnesota", "nebraska", "northwestern", "ohio-state", "penn-state", "purdue", "rutgers", "ucla", "usc", "washington", "wisconsin", "oregon"],
                "ACC": ["boston-college", "california", "clemson", "duke", "florida-state", "georgia-tech", "louisville", "miami-fl", "nc-state", "north-carolina", "notre-dame", "pittsburgh", "smu", "stanford", "syracuse", "virginia", "wake-forest"],
                "Big 12": ["arizona", "arizona-state", "baylor", "byu", "cincinnati", "colorado", "houston", "kansas", "kansas-state", "oklahoma-state", "tcu", "texas-tech", "ucf", "utah", "west-virginia", "iowa-state"]
            }
        }
    
    def load_team_urls(self):
        """Load team URLs"""
        with open(self.teams_file, 'r') as f:
            return json.load(f)
    
    def clean_name(self, name):
        """Remove asterisk and clean up player name"""
        return name.replace("* ", "").strip()
    
    def parse_compound_field(self, field_str, separator=' - '):
        """Parse compound fields like GP-GS or SB-ATT"""
        if separator in field_str:
            parts = field_str.split(separator)
            return int(parts[0].strip()), int(parts[1].strip())
        return int(field_str.strip()), 0
    
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
    
    def scrape_team_batting_stats(self, team_id, url):
        """Scrape batting stats for a team"""
        batting_data = []
        try:
            print(f"Loading {team_id} batting stats from {url}")
            self.driver.get(url)
            
            # Wait for page to load
            time.sleep(4)
            
            # Find batting table - look for table with batting data
            try:
                # Look for the stats table - it usually has headers like Player, AVG, OPS, etc.
                table = self.wait.until(EC.presence_of_element_located((By.XPATH, "//table[.//th[contains(text(), 'AVG')] and .//th[contains(text(), 'OPS')]]")))
                
                # Get all data rows (excluding header and total rows)
                rows = table.find_elements(By.XPATH, ".//tbody/tr[not(contains(translate(td[2]/text(), 'TOTAL', 'total'), 'total')) and not(contains(translate(td[2]/text(), 'OPPONENTS', 'opponents'), 'opponents'))]")
                
                for row in rows:
                    try:
                        cells = row.find_elements(By.TAG_NAME, "td")
                        if len(cells) >= 20:  # Ensure we have enough columns
                            row_data = [cell.text.strip() for cell in cells]
                            if row_data[1] and row_data[1] not in ['Total', 'Opponents']:  # Skip summary rows
                                batting_data.append(row_data)
                    except Exception as e:
                        print(f"Error parsing batting row: {e}")
                        continue
                        
            except TimeoutException:
                print(f"Could not find batting table for {team_id}")
                return []
                
        except Exception as e:
            print(f"Error scraping batting stats for {team_id}: {e}")
            return []
        
        return batting_data
    
    def scrape_team_pitching_stats(self, team_id):
        """Switch to pitching stats and scrape them"""
        pitching_data = []
        try:
            # Find and click the pitching dropdown
            print(f"Switching to pitching stats for {team_id}")
            
            # Look for dropdown with "Batting" and switch to "Pitching"
            dropdown = self.wait.until(EC.element_to_be_clickable((By.XPATH, "//select[contains(@class, 'form-control') or contains(@aria-label, 'Menu')]//option[contains(text(), 'Batting')]/..")))
            
            select = Select(dropdown)
            select.select_by_visible_text("Pitching")
            
            # Wait for pitching table to load
            time.sleep(3)
            
            # Find pitching table
            table = self.wait.until(EC.presence_of_element_located((By.XPATH, "//table[.//th[contains(text(), 'ERA')] and .//th[contains(text(), 'WHIP')]]")))
            
            # Get all data rows
            rows = table.find_elements(By.XPATH, ".//tbody/tr[not(contains(translate(td[2]/text(), 'TOTAL', 'total'), 'total')) and not(contains(translate(td[2]/text(), 'OPPONENTS', 'opponents'), 'opponents'))]")
            
            for row in rows:
                try:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) >= 20:  # Ensure we have enough columns
                        row_data = [cell.text.strip() for cell in cells]
                        if row_data[1] and row_data[1] not in ['Total', 'Opponents']:
                            pitching_data.append(row_data)
                except Exception as e:
                    print(f"Error parsing pitching row: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error scraping pitching stats for {team_id}: {e}")
            return []
            
        return pitching_data
    
    def insert_team_stats(self, team_id, batting_data, pitching_data):
        """Insert team stats into database"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        # Clear existing team data
        cursor.execute("DELETE FROM player_stats WHERE team_id = ?", (team_id,))
        
        batting_count = 0
        pitching_count = 0
        
        # Process batting data
        for row in batting_data:
            try:
                if len(row) >= 21:  # Minimum columns needed
                    number = self.safe_int(row[0]) if row[0].isdigit() else None
                    name = self.clean_name(row[1])
                    avg = self.safe_float(row[2])
                    ops = self.safe_float(row[3])
                    gp_gs = row[4]
                    ab = self.safe_int(row[5])
                    r = self.safe_int(row[6])
                    h = self.safe_int(row[7])
                    doubles = self.safe_int(row[8])
                    triples = self.safe_int(row[9])
                    hr = self.safe_int(row[10])
                    rbi = self.safe_int(row[11])
                    # Skip TB column [12]
                    slg = self.safe_float(row[13])
                    bb = self.safe_int(row[14])
                    hbp = self.safe_int(row[15])
                    so = self.safe_int(row[16])
                    gdp = self.safe_int(row[17])
                    obp = self.safe_float(row[18])
                    # Skip SF, SH columns
                    sb_att = row[21] if len(row) > 21 else "0 - 0"
                    
                    # Parse compound fields
                    games, games_started = self.parse_compound_field(gp_gs)
                    stolen_bases, caught_stealing = self.parse_compound_field(sb_att)
                    
                    cursor.execute("""
                        INSERT OR REPLACE INTO player_stats (
                            team_id, name, number, games, at_bats, runs, hits, doubles, triples,
                            home_runs, rbi, walks, strikeouts, stolen_bases, caught_stealing,
                            batting_avg, obp, slg, ops, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        team_id, name, number, games, ab, r, h, doubles, triples,
                        hr, rbi, bb, so, stolen_bases, caught_stealing,
                        avg, obp, slg, ops, datetime.now().isoformat()
                    ))
                    batting_count += 1
                    
            except Exception as e:
                print(f"Error inserting batting row for {team_id}: {e}")
                continue
        
        # Process pitching data
        for row in pitching_data:
            try:
                if len(row) >= 20:
                    number = self.safe_int(row[0]) if row[0].isdigit() else None
                    name = self.clean_name(row[1])
                    era = self.safe_float(row[2])
                    whip = self.safe_float(row[3])
                    w_l = row[4]
                    app_gs = row[5]
                    # Skip CG column [6]
                    # Skip SHO column [7] 
                    sv = self.safe_int(row[8])
                    ip = self.safe_float(row[9])
                    h_allowed = self.safe_int(row[10])
                    r_allowed = self.safe_int(row[11])
                    er = self.safe_int(row[12])
                    bb_allowed = self.safe_int(row[13])
                    so_pitched = self.safe_int(row[14])
                    
                    # Parse compound fields
                    wins, losses = self.parse_compound_field(w_l)
                    appearances, games_started_pitch = self.parse_compound_field(app_gs)
                    
                    # Try to update existing record or create new one
                    cursor.execute("""
                        INSERT OR IGNORE INTO player_stats (team_id, name, number) VALUES (?, ?, ?)
                    """, (team_id, name, number))
                    
                    cursor.execute("""
                        UPDATE player_stats SET
                            wins = ?, losses = ?, era = ?, games_pitched = ?, games_started = ?,
                            saves = ?, innings_pitched = ?, hits_allowed = ?, runs_allowed = ?,
                            earned_runs = ?, walks_allowed = ?, strikeouts_pitched = ?, whip = ?,
                            updated_at = ?
                        WHERE team_id = ? AND name = ?
                    """, (
                        wins, losses, era, appearances, games_started_pitch,
                        sv, ip, h_allowed, r_allowed, er, bb_allowed, so_pitched, whip,
                        datetime.now().isoformat(), team_id, name
                    ))
                    pitching_count += 1
                    
            except Exception as e:
                print(f"Error inserting pitching row for {team_id}: {e}")
                continue
        
        conn.commit()
        conn.close()
        
        return batting_count, pitching_count
    
    def process_team(self, team_id, url):
        """Process a single team's stats"""
        print(f"\n=== Processing {team_id.upper()} ===")
        
        try:
            # Scrape batting stats
            batting_data = self.scrape_team_batting_stats(team_id, url)
            print(f"Found {len(batting_data)} batting records")
            
            # Scrape pitching stats
            pitching_data = self.scrape_team_pitching_stats(team_id)
            print(f"Found {len(pitching_data)} pitching records")
            
            # Insert into database
            batting_count, pitching_count = self.insert_team_stats(team_id, batting_data, pitching_data)
            
            print(f"Inserted: {batting_count} batting, {pitching_count} pitching records")
            
            return True, batting_count, pitching_count
            
        except Exception as e:
            print(f"Failed to process {team_id}: {e}")
            return False, 0, 0
    
    def run_full_collection(self):
        """Run the complete stats collection process"""
        if not self.setup_driver():
            print("Failed to setup driver. Exiting.")
            return
        
        try:
            progress = self.load_progress()
            teams_data = self.load_team_urls()
            team_urls = teams_data['teams']
            
            total_batting_stats = progress['progress'].get('batting_stats_scraped', 0)
            total_pitching_stats = progress['progress'].get('pitching_stats_scraped', 0)
            completed_teams = progress['progress'].get('completed_teams', [])
            failed_teams = progress['progress'].get('teams_failed', [])
            
            # Process teams in conference order
            for conference in progress['team_mapping']:
                print(f"\n{'='*50}")
                print(f"PROCESSING {conference} CONFERENCE")
                print(f"{'='*50}")
                
                for team_id in progress['team_mapping'][conference]:
                    if team_id in completed_teams:
                        print(f"Skipping {team_id} (already completed)")
                        continue
                        
                    if team_id not in team_urls:
                        print(f"No URL found for {team_id}")
                        failed_teams.append(team_id)
                        continue
                    
                    # Update current team
                    progress['progress']['current_team'] = team_id
                    self.save_progress(progress)
                    
                    # Process the team
                    success, batting_count, pitching_count = self.process_team(team_id, team_urls[team_id])
                    
                    if success:
                        completed_teams.append(team_id)
                        total_batting_stats += batting_count
                        total_pitching_stats += pitching_count
                        progress['progress']['teams_completed'] = len(completed_teams)
                    else:
                        failed_teams.append(team_id)
                    
                    # Update progress
                    progress['progress']['completed_teams'] = completed_teams
                    progress['progress']['teams_failed'] = failed_teams
                    progress['progress']['batting_stats_scraped'] = total_batting_stats
                    progress['progress']['pitching_stats_scraped'] = total_pitching_stats
                    self.save_progress(progress)
                    
                    # Wait between teams (15 seconds as requested)
                    if team_id != progress['team_mapping'][conference][-1]:  # Don't wait after last team
                        print("Waiting 15 seconds before next team...")
                        time.sleep(15)
            
            # Final report
            print(f"\n{'='*50}")
            print("FINAL REPORT")
            print(f"{'='*50}")
            print(f"Teams successfully scraped: {len(completed_teams)}/67")
            print(f"Total batting stats updated: {total_batting_stats}")
            print(f"Total pitching stats updated: {total_pitching_stats}")
            
            if failed_teams:
                print(f"\nFailed teams ({len(failed_teams)}):")
                for team in failed_teams:
                    print(f"  - {team}")
            
            # Sample some data to verify
            print(f"\n=== Sample Data Verification ===")
            self.sample_verification()
            
        finally:
            if self.driver:
                self.driver.quit()
    
    def sample_verification(self):
        """Sample some data to verify it looks right"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        # Get a few sample records
        cursor.execute("""
            SELECT team_id, name, batting_avg, ops, era, whip 
            FROM player_stats 
            WHERE batting_avg > 0 OR era > 0 
            ORDER BY team_id, name 
            LIMIT 10
        """)
        
        print("Sample records:")
        for row in cursor.fetchall():
            team, name, avg, ops, era, whip = row
            print(f"  {team}: {name} - AVG:{avg:.3f} OPS:{ops:.3f} ERA:{era:.2f} WHIP:{whip:.2f}")
        
        conn.close()

if __name__ == "__main__":
    collector = P4StatsCollector()
    collector.run_full_collection()