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
    # Team stats (not individual!) — verify at stats.ncaa.org Team section
    'era':          211,
    'batting_avg':  210,
    'fielding_pct': 212,
    'scoring':      213,
    'slugging':     327,   # 327=team, 321=individual
    'hr_per_game':  323,
    'obp':          589,   # 589=team, 504=individual
    'k_per_9':      425,
    'bb_per_9':     509,
    'whip':         597,
    'k_bb_ratio':   591,
    'dp_per_game':  328,
    'sb_per_game':  326,
}

# Season → academic_year mapping (2025 season = 2025 academic year)
# Final stats ranking_period codes (found by inspecting the dropdown)
# "Final Statistics" ranking_period codes per season
# These change yearly — verify via the rp dropdown on stats.ncaa.org
# Use None for in-progress seasons — will auto-detect latest available period
SEASON_RANKING_PERIODS = {
    2026: None,  # In-progress season — auto-detect latest ranking period
    2025: 104,   # 06/22/2025-Final Statistics
    2024: 100,   # 06/24/2024-Final Statistics
    2023: 94,    # 06/26/2023-Final Statistics
    2022: 90,    # 06/26/2022-Final Statistics
    2021: 96,    # 06/30/2021-Final Statistics
}


def detect_latest_ranking_period(season, page=None, runner=None):
    """
    Auto-detect the latest ranking period for a season by reading the
    #rp dropdown on stats.ncaa.org. Returns the ranking_period int or None.
    
    Works with either a playwright page object or via openclaw CLI.
    """
    log = runner.info if runner else print
    warn = runner.warn if runner else print
    
    url = (f"https://stats.ncaa.org/rankings/change_sport_year_div?"
           f"sport_code=MBA&academic_year={season}&division=1")
    
    if page:
        # Playwright CDP mode
        try:
            page.goto(url, timeout=30000, wait_until='networkidle')
            page.wait_for_selector('select#rp', timeout=10000)
            
            rp_options = page.evaluate("""
                () => {
                    const sel = document.querySelector('select#rp');
                    if (!sel) return [];
                    return Array.from(sel.options).map(o => ({
                        value: parseFloat(o.value),
                        text: o.textContent.trim(),
                        selected: o.selected
                    }));
                }
            """)
        except Exception as e:
            warn(f"Failed to detect ranking period via playwright: {e}")
            return None
    else:
        # CLI mode (openclaw browser)
        try:
            nav_result = subprocess.run(
                ['openclaw', 'browser', 'navigate', '--profile', 'openclaw', '--url', url],
                capture_output=True, text=True, timeout=30
            )
            if nav_result.returncode != 0:
                warn(f"Navigation failed: {nav_result.stderr}")
                return None
            
            time.sleep(4)
            
            extract_result = subprocess.run(
                ['openclaw', 'browser', 'eval', '--profile', 'openclaw', '--js',
                 '() => { const sel = document.querySelector("select#rp"); '
                 'if (!sel) return []; '
                 'return Array.from(sel.options).map(o => ({value: parseFloat(o.value), '
                 'text: o.textContent.trim(), selected: o.selected})); }'],
                capture_output=True, text=True, timeout=15
            )
            if extract_result.returncode != 0:
                warn(f"Extraction failed: {extract_result.stderr}")
                return None
            
            rp_options = json.loads(extract_result.stdout)
        except Exception as e:
            warn(f"Failed to detect ranking period via CLI: {e}")
            return None
    
    if not rp_options:
        warn(f"No ranking period options found for {season}")
        return None
    
    # The first option (selected=true) is the latest/most recent
    # Pick the one with the highest value (most recent cumulative stats)
    # Filter out weekly ranges (text contains " - ") — prefer cumulative dates
    cumulative = [o for o in rp_options if ' - ' not in o.get('text', '')]
    if cumulative:
        best = max(cumulative, key=lambda o: o['value'])
    else:
        best = max(rp_options, key=lambda o: o['value'])
    
    rp = int(best['value'])
    log(f"Auto-detected ranking period for {season}: {rp} ({best.get('text', '?')})")
    return rp

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
() => {
    const select = document.querySelector('select[name="rankings_table_length"]') || document.querySelector('select[name="stat_grid_length"]');
    if (!select) return false;
    // DataTables uses -1 for "show all"
    const allOpt = Array.from(select.options).find(o => o.value === '-1');
    if (allOpt) {
        select.value = '-1';
    } else {
        // Fallback: pick the largest numeric option
        const options = Array.from(select.options);
        const maxOpt = options.reduce((a, b) => 
            parseInt(a.value) > parseInt(b.value) ? a : b
        );
        select.value = maxOpt.value;
    }
    // Use jQuery if available (DataTables listens on jQuery events)
    if (typeof jQuery !== 'undefined') {
        jQuery(select).trigger('change');
    } else {
        select.dispatchEvent(new Event('change', {bubbles: true}));
    }
    return select.value;
}
"""


def scrape_stat(season, stat_name, stat_code, runner, _rp_cache={}):
    """Scrape a single stat for a single season using openclaw CLI."""
    ranking_period = SEASON_RANKING_PERIODS.get(season, 'MISSING')
    if ranking_period == 'MISSING':
        runner.warn(f"Season {season} not configured, skipping")
        return None
    if ranking_period is None:
        # Auto-detect for in-progress season (cache across calls)
        if season not in _rp_cache:
            _rp_cache[season] = detect_latest_ranking_period(season, runner=runner)
        ranking_period = _rp_cache[season]
        if not ranking_period:
            runner.warn(f"Could not auto-detect ranking period for {season}, skipping")
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


def scrape_stat_playwright(season, stat_name, stat_code, page, runner, _rp_cache={}):
    """Scrape using an existing playwright page connected via CDP."""
    ranking_period = SEASON_RANKING_PERIODS.get(season, 'MISSING')
    if ranking_period == 'MISSING':
        runner.warn(f"Season {season} not configured, skipping")
        return None
    if ranking_period is None:
        # Auto-detect for in-progress season (cache across calls)
        if season not in _rp_cache:
            _rp_cache[season] = detect_latest_ranking_period(season, page=page, runner=runner)
        ranking_period = _rp_cache[season]
        if not ranking_period:
            runner.warn(f"Could not auto-detect ranking period for {season}, skipping")
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
        time.sleep(4)  # Give DataTables time to render all 300+ rows
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
    parser.add_argument('--seasons', default='2026',
                       help='Comma-separated seasons (default: 2026 current season)')
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
                # Probe the default openclaw port first
                probe = _sp.run(['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}',
                                f'http://127.0.0.1:18800/json/version'],
                               capture_output=True, text=True, timeout=5)
                if probe.stdout.strip() == '200':
                    cdp_port = 18800
                else:
                    # Fallback: try openclaw status
                    status = _sp.run(['openclaw', 'browser', 'status', '--profile', 'openclaw'],
                                   capture_output=True, text=True, timeout=5)
                    for line in status.stdout.split('\n'):
                        if 'cdpPort' in line:
                            port_candidate = int(line.split(':')[1].strip())
                            # Verify it's actually responding
                            probe2 = _sp.run(['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}',
                                            f'http://127.0.0.1:{port_candidate}/json/version'],
                                           capture_output=True, text=True, timeout=5)
                            if probe2.stdout.strip() == '200':
                                cdp_port = port_candidate
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
