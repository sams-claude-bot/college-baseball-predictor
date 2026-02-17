# Stats Collection Architecture

## Overview

Two-tier stats collection system:
1. **Nightly (basic stats)** — No login required, runs headlessly via BeautifulSoup
2. **Weekly (advanced stats)** — Requires D1Baseball subscription, uses Chrome relay browser

## Basic Stats (Nightly at 1 AM CST)

**What:** Standard batting (BA, HR, RBI, OBP, SLG) and pitching (W, L, ERA, IP, K, BB) stats.

**How:** `python3 scripts/d1baseball_stats.py --p4` scrapes public stats pages via HTTP + BeautifulSoup. No JavaScript rendering needed.

**Coverage:** All 65 P4 teams (SEC, ACC, Big 12, Big Ten minus teams without baseball programs).

**Rate limiting:** 4 seconds between team requests.

**Commands:**
```bash
# All P4 teams
python3 scripts/d1baseball_stats.py --p4

# Single conference
python3 scripts/d1baseball_stats.py --conference SEC

# Specific teams (D1Baseball slugs)
python3 scripts/d1baseball_stats.py clemson vandy lsu
```

## Advanced Stats (Weekly, Sundays)

**What:** Advanced batting (wOBA, wRC+, ISO, BABIP, K%, BB%), batted ball (GB%, LD%, FB%, HR/FB%), advanced pitching (FIP, xFIP, SIERA, LOB%), batted ball pitching.

**How:** D1Baseball paywalls advanced stats — requires authenticated browser session. OpenClaw sub-agents use Chrome relay (Sam's logged-in Chrome) to:
1. Navigate to each team's stats page
2. Execute JavaScript to click through stat tabs and extract table data
3. Save JSON to `/tmp/{team_id}_advanced.json`
4. Run `python3 scripts/d1baseball_advanced.py --file /tmp/{team_id}_advanced.json` to update DB

**Coverage:** All P4 teams, one sub-agent per conference (run sequentially since they share Chrome relay).

**Prerequisite:** Sam's Chrome must have the OpenClaw Browser Relay extension attached to a tab logged into D1Baseball.

### Tab Structure on D1Baseball Stats Pages
- 6 `li.stat-toggle` elements per page
- Index 0: Standard Batting, 1: Advanced Batting, 2: Batted Ball Batting
- Index 3: Standard Pitching, 4: Advanced Pitching, 5: Batted Ball Pitching
- Data lives in `.dataTables_scrollBody table` elements
- Tables 0-2 = batting section, tables 3-5 = pitching section

### Extraction JS Pattern
Click toggle → wait 1500ms → extract visible table from correct section offset → repeat for each tab.

### DB Updater Script
`scripts/d1baseball_advanced.py` handles all DB operations:
- Matches player names to existing `player_stats` records
- Updates advanced stat columns (wrc_plus, fip, xfip, siera, iso, babip, woba, etc.)
- Creates snapshots in `player_stats_snapshots`
- Supports `--file` (JSON path) or stdin input

## Slug Mapping

`config/d1bb_slugs.json` maps DB team IDs ↔ D1Baseball URL slugs. Key differences:
- `georgia-tech` → `gatech`
- `vanderbilt` → `vandy`
- `miami-fl` → `miamifl`
- `south-carolina` → `scarolina`
- `mississippi-state` → `missst`
- `north-carolina` → `unc`
- `northwestern` → `nwestern`
- (see full map in config file)

## Database Tables

- `player_stats` — Current stats (basic + advanced columns)
- `player_stats_snapshots` — Daily historical snapshots (71 columns)
- Advanced columns: `wrc_plus, fip, xfip, siera, iso, babip, woba, wraa, wrc, k_pct, bb_pct, gb_pct, ld_pct, fb_pct, pu_pct, hr_fb_pct` (plus `_pitch` suffixed variants for pitchers)
