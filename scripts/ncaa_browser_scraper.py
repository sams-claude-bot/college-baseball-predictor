#!/usr/bin/env python3
"""
NCAA Team Stats Scraper — uses openclaw browser (CDP) to bypass Akamai.

Scrapes stats.ncaa.org via the managed browser, extracts table data via JS,
saves JSON, and loads into DB via ncaa_stats_scraper.py.

Usage:
    python3 scripts/ncaa_browser_scraper.py --seasons 2021,2022,2023,2024,2025
    python3 scripts/ncaa_browser_scraper.py --seasons 2025 --stats era,batting_avg,obp
    python3 scripts/ncaa_browser_scraper.py --seasons 2025 --stats all
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_utils import ScriptRunner

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / 'data'

# NCAA stat codes: stat_seq → (name, ranking_period for final stats)
# ranking_period varies by season — use "Final Statistics" option
STAT_CODES = {
    'era':          211,
    'batting_avg':  210,
    'fielding_pct': 212,
    'scoring':      213,
    'slugging':     321,
    'hr_per_game':  323,
    'obp':          504,
    'k_per_9':      425,
    'bb_per_9':     509,
    'whip':         597,
    'k_bb_ratio':   591,
    'dp_per_game':  328,
    'sb_per_game':  326,
}

# Season → academic_year mapping (2025 season = 2025 academic year)
# Final stats ranking_period codes (found by inspecting the dropdown)
SEASON_RANKING_PERIODS = {
    2025: 104,
    2024: 100,
    2023: 96,
    2022: 92,
    2021: 88,
}

# JS to extract all table rows after showing all entries
EXTRACT_TABLE_JS = """
() => {
    const rows = document.querySelectorAll('#rankings_table tbody tr, #stat_grid tbody tr');
    const data = [];
    for (const row of rows) {
        const cells = row.querySelectorAll('td');
        if (cells.length < 4) continue;  // Skip header/separator rows
        
        const rank = cells[0]?.textContent?.trim();
        if (rank === 'Reclassifying' || !rank) continue;
        
        const team = cells[1]?.textContent?.trim();
        const games = cells[2]?.textContent?.trim();
        const record = cells[3]?.textContent?.trim();
        // Last cell is always the stat value
        const value = cells[cells.length - 1]?.textContent?.trim();
        
        if (team && value) {
            data.push({
                rank: rank === '-' ? null : parseInt(rank) || null,
                team: team,
                games: parseInt(games) || null,
                record: record || null,
                value: parseFloat(value) || null
            });
        }
    }
    return data;
}
"""

# JS to change "show entries" dropdown to show all
SHOW_ALL_JS = """
(maxEntries) => {
    const select = document.querySelector('select[name="rankings_table_length"]') || document.querySelector('select[name="stat_grid_length"]');
    if (!select) return false;
    // Find the option with the highest value (shows all)
    const options = Array.from(select.options);
    const maxOpt = options.reduce((a, b) => 
        parseInt(a.value) > parseInt(b.value) ? a : b
    );
    select.value = maxOpt.value;
    select.dispatchEvent(new Event('change'));
    return parseInt(maxOpt.value);
}
"""


def scrape_stat(season, stat_name, stat_code, runner):
    """Scrape a single stat for a single season using playwright CDP."""
    import websocket
    
    ranking_period = SEASON_RANKING_PERIODS.get(season)
    if not ranking_period:
        runner.warn(f"No ranking period for season {season}, skipping")
        return None
    
    url = (f"https://stats.ncaa.org/rankings/national_ranking?"
           f"academic_year={season}&division=1&ranking_period={ranking_period}"
           f"&sport_code=MBA&stat_seq={stat_code}")
    
    runner.info(f"  Scraping {stat_name} for {season}: {url}")
    
    # Use subprocess to interact with openclaw browser via CLI
    # Navigate to URL
    nav_result = subprocess.run(
        ['openclaw', 'browser', 'navigate', '--profile', 'openclaw', '--url', url],
        capture_output=True, text=True, timeout=30
    )
    if nav_result.returncode != 0:
        runner.error(f"  Navigation failed: {nav_result.stderr}")
        return None
    
    # Wait for page to load
    time.sleep(3)
    
    # Show all entries
    show_result = subprocess.run(
        ['openclaw', 'browser', 'eval', '--profile', 'openclaw', '--js', SHOW_ALL_JS.strip()],
        capture_output=True, text=True, timeout=15
    )
    
    # Wait for table to re-render with all rows
    time.sleep(3)
    
    # Extract table data
    extract_result = subprocess.run(
        ['openclaw', 'browser', 'eval', '--profile', 'openclaw', '--js', EXTRACT_TABLE_JS.strip()],
        capture_output=True, text=True, timeout=15
    )
    
    if extract_result.returncode != 0:
        runner.error(f"  Extraction failed: {extract_result.stderr}")
        return None
    
    try:
        data = json.loads(extract_result.stdout)
        runner.info(f"  Got {len(data)} teams for {stat_name} ({season})")
        return data
    except json.JSONDecodeError:
        runner.error(f"  Failed to parse extracted data")
        return None


def scrape_stat_playwright(season, stat_name, stat_code, page, runner):
    """Scrape using an existing playwright page connected via CDP."""
    ranking_period = SEASON_RANKING_PERIODS.get(season)
    if not ranking_period:
        runner.warn(f"No ranking period for season {season}, skipping")
        return None
    
    url = (f"https://stats.ncaa.org/rankings/national_ranking?"
           f"academic_year={season}&division=1&ranking_period={ranking_period}"
           f"&sport_code=MBA&stat_seq={stat_code}")
    
    runner.info(f"  Scraping {stat_name} for {season}...")
    
    try:
        page.goto(url, timeout=30000, wait_until='networkidle')
    except Exception as e:
        runner.error(f"  Navigation failed: {e}")
        return None
    
    # Wait for table
    try:
        page.wait_for_selector('#rankings_table, #stat_grid', timeout=10000)
    except:
        runner.warn(f"  Table not found, checking for block page...")
        if 'Access Denied' in page.title():
            runner.error(f"  Blocked by Akamai!")
            return None
        return None
    
    # Show all entries
    try:
        page.evaluate(SHOW_ALL_JS)
        time.sleep(2)
    except Exception as e:
        runner.warn(f"  Could not show all entries: {e}")
    
    # Extract
    try:
        data = page.evaluate(EXTRACT_TABLE_JS)
        runner.info(f"  Got {len(data)} teams for {stat_name} ({season})")
        return data
    except Exception as e:
        runner.error(f"  Extraction failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='NCAA Stats Browser Scraper')
    parser.add_argument('--seasons', default='2021,2022,2023,2024,2025',
                       help='Comma-separated seasons (default: 2021-2025)')
    parser.add_argument('--stats', default='all',
                       help='Comma-separated stat names or "all" (default: all)')
    parser.add_argument('--delay', type=float, default=2.0,
                       help='Delay between requests in seconds (default: 2.0)')
    parser.add_argument('--method', choices=['cdp', 'cli'], default='cdp',
                       help='Scrape method: cdp (connect to openclaw browser) or cli')
    args = parser.parse_args()
    
    runner = ScriptRunner("ncaa_browser_scraper")
    
    seasons = [int(s.strip()) for s in args.seasons.split(',')]
    if args.stats == 'all':
        stats = list(STAT_CODES.keys())
    else:
        stats = [s.strip() for s in args.stats.split(',')]
        for s in stats:
            if s not in STAT_CODES:
                runner.error(f"Unknown stat: {s}. Available: {', '.join(STAT_CODES.keys())}")
                runner.finish()
    
    total_jobs = len(seasons) * len(stats)
    runner.info(f"Scraping {len(stats)} stats × {len(seasons)} seasons = {total_jobs} pages")
    
    # Connect to openclaw browser via CDP
    if args.method == 'cdp':
        from playwright.sync_api import sync_playwright
        
        # Get CDP endpoint from openclaw
        result = subprocess.run(
            ['openclaw', 'browser', 'status', '--profile', 'openclaw', '--json'],
            capture_output=True, text=True, timeout=10
        )
        
        # The openclaw browser exposes CDP — connect via playwright
        pw = sync_playwright().start()
        try:
            # Connect to the running browser
            # Try openclaw browser port, fallback to default
            cdp_port = 18800
            try:
                import subprocess as _sp
                status = _sp.run(['openclaw', 'browser', 'status', '--profile', 'openclaw'],
                               capture_output=True, text=True, timeout=5)
                for line in status.stdout.split('\n'):
                    if 'cdpPort' in line:
                        cdp_port = int(line.split(':')[1].strip())
            except:
                pass
            browser = pw.chromium.connect_over_cdp(f'http://127.0.0.1:{cdp_port}')
            context = browser.contexts[0]
            page = context.new_page()
            runner.info("Connected to openclaw browser via CDP")
        except Exception as e:
            runner.error(f"Could not connect to openclaw browser: {e}")
            runner.info("Make sure the openclaw browser is running (openclaw browser start)")
            runner.finish()
            return
    
    all_results = []
    success = 0
    failed = 0
    
    for season in seasons:
        for stat_name in stats:
            stat_code = STAT_CODES[stat_name]
            
            if args.method == 'cdp':
                data = scrape_stat_playwright(season, stat_name, stat_code, page, runner)
            else:
                data = scrape_stat(season, stat_name, stat_code, runner)
            
            if data and len(data) > 0:
                # Save individual file
                outfile = DATA_DIR / f'ncaa_{season}_{stat_name}.json'
                with open(outfile, 'w') as f:
                    json.dump(data, f, indent=2)
                
                # Accumulate for combined load
                all_results.append({
                    'stat_name': stat_name,
                    'season': season,
                    'teams': data
                })
                success += 1
            else:
                failed += 1
            
            time.sleep(args.delay)
    
    # Save combined results
    if all_results:
        combined_file = DATA_DIR / 'ncaa_stats_combined.json'
        with open(combined_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        runner.info(f"Saved combined results to {combined_file}")
        
        # Load into DB
        runner.info("Loading into database...")
        load_result = subprocess.run(
            ['python3', str(PROJECT_DIR / 'scripts' / 'ncaa_stats_scraper.py'), 
             'load', str(combined_file)],
            capture_output=True, text=True, timeout=60, cwd=str(PROJECT_DIR)
        )
        runner.info(load_result.stdout[-500:] if load_result.stdout else "No output")
        if load_result.returncode != 0:
            runner.warn(f"Loader stderr: {load_result.stderr[-300:]}")
    
    if args.method == 'cdp':
        page.close()
        browser.close()
        pw.stop()
    
    runner.add_stat("stats_scraped", success)
    runner.add_stat("stats_failed", failed)
    runner.add_stat("total_teams_scraped", sum(len(r['teams']) for r in all_results))
    runner.finish()


if __name__ == '__main__':
    main()
