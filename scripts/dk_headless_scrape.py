#!/usr/bin/env python3
"""
Headless DK NCAA Baseball odds scraper — no AI needed.
Launches headless Chromium, scrapes the DK page, parses odds into JSON.

Usage: python3 scripts/dk_headless_scrape.py [raw_output_file]
Output: data/dk_odds_today.json
"""
import json
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
URL = "https://sportsbook.draftkings.com/leagues/baseball/ncaa-baseball"


def scrape_page_text(raw_file=None):
    """Launch headless browser and get page text."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(URL, timeout=25000)
        time.sleep(5)

        # Scroll to load all lazy-loaded content
        for i in range(12):
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
    """Parse DK page text into list of game dicts."""
    games = []
    lines = text.split('\n')

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == 'AT' and i > 0 and i + 1 < len(lines):
            away = lines[i - 1].strip()
            home = lines[i + 1].strip()

            skip = ('', 'GAME', 'Run Line', 'Total', 'Moneyline', 'College Baseball Odds')
            if away in skip or home in skip:
                i += 1
                continue

            # Collect numeric values after home team name
            j = i + 2
            nums = []
            while j < len(lines) and j < i + 20:
                val = lines[j].strip()
                if val.startswith('Today') or val.startswith('Tomorrow') or val == 'More Bets':
                    break
                if val not in ('', 'O', 'U', 'AT'):
                    nums.append(val)
                j += 1

            # Expected pattern (10 values):
            # away_spread, away_spread_odds, over_under, over_odds, away_ml
            # home_spread, home_spread_odds, over_under, under_odds, home_ml
            if len(nums) >= 10:
                try:
                    game = {
                        'away': away,
                        'home': home,
                        'spread': float(nums[0]),
                        'away_spread_odds': fix_int(nums[1]),
                        'over_under': float(nums[2]),
                        'over_odds': fix_int(nums[3]),
                        'away_ml': fix_int(nums[4]),
                        'home_spread_odds': fix_int(nums[6]),
                        'under_odds': fix_int(nums[8]),
                        'home_ml': fix_int(nums[9]),
                    }
                    games.append(game)
                except (ValueError, IndexError) as e:
                    print(f"  WARN: parse failed for {away} @ {home}: {e}")
            elif len(nums) >= 2:
                # Minimal: just moneylines (no spread/total)
                try:
                    game = {
                        'away': away,
                        'home': home,
                        'away_ml': fix_int(nums[0]),
                        'home_ml': fix_int(nums[1]),
                    }
                    games.append(game)
                except (ValueError, IndexError):
                    pass
        i += 1

    return games


def main():
    raw_file = sys.argv[1] if len(sys.argv) > 1 else None

    print(f"Scraping DK NCAA Baseball odds...")
    text = scrape_page_text(raw_file)

    games = parse_odds(text)
    print(f"Parsed {len(games)} games:")
    for g in games:
        ml_str = f"{g['away_ml']:+d}/{g['home_ml']:+d}"
        ou_str = f"O/U: {g['over_under']}" if 'over_under' in g else "no O/U"
        print(f"  {g['away']:25s} @ {g['home']:25s}  ML: {ml_str:>10s}  {ou_str}")

    out_path = PROJECT_DIR / 'data' / 'dk_odds_today.json'
    out_path.write_text(json.dumps(games, indent=2))
    print(f"\nSaved {len(games)} games -> {out_path}")


if __name__ == '__main__':
    main()
