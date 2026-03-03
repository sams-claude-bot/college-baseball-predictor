#!/usr/bin/env python3
"""
Headless FanDuel NCAA Baseball odds scraper — no AI needed.
Launches headless Chromium, scrapes the FD page, parses odds into JSON.

Usage: python3 scripts/fd_headless_scrape.py [raw_output_file]
Output: data/fd_odds_today.json

FanDuel button text patterns:
  "Moneyline, Team Name, +136 Odds"
  "Run Line, Team Name, 1.5, -136 Odds"
  "Total Runs, Team Name, OVER, Over 12.5, -108 Odds"
"""
import json
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
URL = "https://sportsbook.fanduel.com/baseball/ncaa---baseball"


def scrape_page_text(raw_file=None):
    """Launch headless browser and get page text."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(URL, timeout=25000)
        time.sleep(5)

        # Scroll to load all lazy-loaded content
        for i in range(15):
            page.evaluate(f'window.scrollTo(0, {(i + 1) * 800})')
            time.sleep(0.4)
        time.sleep(2)

        text = page.evaluate('() => document.body.innerText')
        browser.close()

    if raw_file:
        Path(raw_file).write_text(text)
        print(f"Raw text: {len(text)} chars -> {raw_file}")

    return text


def fix_int(s):
    """Parse integer with unicode minus signs."""
    return int(s.replace('\u2212', '-').replace('\u2013', '-'))


def parse_odds(text):
    """Parse FD page text into list of game dicts.
    
    FanDuel layout is different from DK — teams use mascot names
    (e.g., 'Nebraska Cornhuskers'). Format varies but typically:
    
    Time
    Away Team Mascot
    spread  away_spread_odds  away_ml
    Home Team Mascot  
    spread  home_spread_odds  home_ml
    O/U line  over_odds  under_odds
    """
    games = []
    lines = text.split('\n')

    # FD uses @ symbol or "at" between teams sometimes,
    # but more reliably, look for consecutive team lines with odds
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for "@" separator (FD format: "Away @ Home")
        if ' @ ' in line and not line.startswith('http'):
            parts = line.split(' @ ')
            if len(parts) == 2:
                away = parts[0].strip()
                home = parts[1].strip()
                
                # Collect numeric values after
                j = i + 1
                nums = []
                while j < len(lines) and j < i + 15:
                    val = lines[j].strip()
                    if ' @ ' in val or val.startswith('http'):
                        break
                    # Look for odds-like values
                    try:
                        cleaned = val.replace('\u2212', '-').replace('\u2013', '-')
                        if cleaned.startswith('+') or cleaned.startswith('-'):
                            nums.append(fix_int(cleaned))
                        elif '.' in cleaned:
                            nums.append(float(cleaned))
                    except (ValueError, IndexError):
                        pass
                    j += 1
                
                if len(nums) >= 2:
                    game = {'away': away, 'home': home}
                    # Try to identify ML values (larger absolute values, integers)
                    int_vals = [n for n in nums if isinstance(n, int)]
                    float_vals = [n for n in nums if isinstance(n, float)]
                    
                    if len(int_vals) >= 2:
                        game['away_ml'] = int_vals[0]
                        game['home_ml'] = int_vals[1]
                    if len(float_vals) >= 1:
                        # Could be spread or O/U
                        for fv in float_vals:
                            if 5 < fv < 25:  # likely O/U
                                game['over_under'] = fv
                            elif abs(fv) < 10:  # likely spread
                                game['spread'] = fv
                    
                    if 'away_ml' in game:
                        games.append(game)
        i += 1

    return games


def main():
    raw_file = sys.argv[1] if len(sys.argv) > 1 else None

    print(f"Scraping FanDuel NCAA Baseball odds...")
    text = scrape_page_text(raw_file)

    games = parse_odds(text)
    print(f"Parsed {len(games)} games:")
    for g in games:
        ml_str = f"{g.get('away_ml', '?'):+d}/{g.get('home_ml', '?'):+d}" if 'away_ml' in g else "?"
        ou_str = f"O/U: {g['over_under']}" if 'over_under' in g else "no O/U"
        print(f"  {g['away']:30s} @ {g['home']:30s}  ML: {ml_str:>10s}  {ou_str}")

    out_path = PROJECT_DIR / 'data' / 'fd_odds_today.json'
    out_path.write_text(json.dumps(games, indent=2))
    print(f"\nSaved {len(games)} games -> {out_path}")


if __name__ == '__main__':
    main()
