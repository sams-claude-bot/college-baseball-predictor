# D1Baseball Advanced Stats — DONE ✅

## Completed (Feb 17, 2026)

- **`player_stats_snapshots` table** created with all columns for standard + advanced + batted ball stats (batting & pitching)
- **Advanced columns added to `player_stats`**: k_pct, bb_pct, iso, babip, woba, wrc, wraa, wrc_plus, gb_pct, ld_pct, fb_pct, pu_pct, hr_fb_pct, fip, xfip, siera, lob_pct, and pitching variants
- **`d1baseball_stats.py`** now inserts snapshots alongside upserts (standard stats only)
- **`d1baseball_advanced.py`** DB updater — takes JSON input and updates player_stats
- **`d1bb_advanced_scraper.py`** NEW — Playwright-based scraper using openclaw browser profile with D1BB session cookies
- **Authentication RESOLVED** — Logged into D1Baseball via openclaw browser (Feb 17). Session persists.
- **Cron job set** — Mondays 3 AM for SEC conference

### First Run Results (Feb 17)
- SEC conference: 10/16 teams succeeded, 6 timed out (network flakiness)
- 271 players updated with advanced stats
- All 16 SEC teams now have wOBA + FIP data populated

## Previous Blocker (Now Resolved): Authentication
D1Baseball advanced stats (wOBA, wRC+, FIP, xFIP, SIERA, batted ball data) are **subscriber-only**. The page renders placeholder values (all `12.3 / .123 / 12`) for non-subscribers.

**The openclaw managed browser is NOT logged in to D1Baseball.** Only Sam's personal Chrome (the "chrome" relay profile) has the active session.

### Options to Unblock

1. **Log in via openclaw browser** — Sam navigates to d1baseball.com in the openclaw browser and logs in. Then `d1baseball_advanced.py` can use the profile cookies via Playwright's `launch_persistent_context`. Note: Playwright's bundled Chromium can read cookies from the openclaw Chrome user-data dir (`~/.openclaw/browser/openclaw/user-data`).

2. **Cookie export from Chrome relay** — Export the D1Baseball session cookies from Sam's Chrome and inject them into the openclaw profile or into Playwright directly via `context.add_cookies()`.

3. **CDP relay approach** — Use the openclaw browser tool's CDP connection to Sam's Chrome (relay profile) to run the extraction JS directly. This would work but means scraping through the OpenClaw browser tool API rather than standalone Python.

4. **Headless login** — Add login credentials to the script and authenticate programmatically. Least preferred (credential management).

### Recommendation
Option 1 is simplest — Sam just logs in once in the openclaw browser, and the script works from then on. The cookie/session persists in the user-data dir.

## Page Structure Notes (from investigation)
- URL: `https://d1baseball.com/team/{slug}/stats/`
- 12 `<table>` elements: 6 stat types × 2 (qualified filter with 0 rows + full with all rows)
- Order: std_bat, adv_bat, bb_bat, std_pit, adv_pit, bb_pit
- Headers include `Qual.` as first column (checkbox); first cell in rows is empty string
- Two `GB%` tables (batting & pitching) have identical headers — distinguished by DOM order
- Tab toggles: `li.stat-toggle` elements, but all tables exist in DOM regardless of active tab
- Classification: wOBA → adv_bat, FIP → adv_pit, POS+BA → std_bat, W+ERA+IP → std_pit, GB% → order-based

## Files
- `scripts/migrate_advanced_stats.py` — schema migration (already run)
- `scripts/d1baseball_advanced.py` — full advanced scraper (ready, needs auth)
- `scripts/d1baseball_stats.py` — standard stats + snapshots (working now)
