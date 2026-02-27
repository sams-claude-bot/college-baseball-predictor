# StatBroadcast Integration

**Date:** 2026-02-26
**Status:** ✅ Fully deployed and live
**Author:** Clawd

## Overview

StatBroadcast provides live stat feeds for college baseball with **far richer data than ESPN** — outs, count, batter/pitcher, base runners, per-inning linescore, and play-by-play. Their encoding is trivially decoded (ROT13 + base64). Pure HTTP polling, no browser needed for live data.

## Architecture

```
┌──────────────────────┐     ┌────────────────────┐     ┌─────────────────┐
│  Season Scrape       │     │  SB Poller         │     │  Dashboard      │
│  (weekly, browser)   │     │  (20s, systemd)    │     │  (gunicorn:5000)│
│                      │     │                    │     │                 │
│  Browser loads SB    │────>│  Polls broadcast + │────>│  Game cards:    │
│  schedule pages for  │     │  PXP views         │     │  outs, count,   │
│  all 255 schools     │     │  Decodes ROT13+b64 │     │  diamond, P/AB  │
│  Extracts event IDs  │     │  Merges sb_* into  │     │                 │
│  Probes event API    │     │  situation_json     │     │  Game page:     │
│  Matches to games    │     │  Emits SSE events   │     │  linescore,     │
│                      │     │                    │     │  play-by-play   │
│  Pre-game (daily):   │     │  Detects completed  │     │                 │
│  Verifies + scans    │     │  games (Final)     │     │  SSE live updates│
└──────────────────────┘     └────────────────────┘     └─────────────────┘
         │                           │
         v                           v
  statbroadcast_events        games.situation_json
  sb_scrape_state.json        live_events table
  sb_group_ids.json
```

## Decoding

```python
import codecs, base64

def sb_decode(text):
    return base64.b64decode(codecs.decode(text, 'rot_13')).decode('utf-8')
```

## API Endpoints

### Event Metadata
```
GET /interface/webservice/event/{id}?data={base64("type=statbroadcast")}
```
Returns XML: team names, date, sport, completed flag, group ID, XML file path.

### Live Stats (Broadcast View)
```
GET /interface/webservice/stats?data={base64(params)}
```
Params: `event={id}&xml={xmlfile}&xsl=baseball/sb.bsgame.views.broadcast.xsl&sport=bsgame&filetime={ts}&type=statbroadcast&start=true`

Returns decoded HTML with: score, inning, outs, count, batter/pitcher, linescore (R/H/E per inning), base runners (icon font), lineup, field alignment.

Uses `Filetime` header for conditional polling (304 = no changes).

### Play-by-Play View
Same endpoint, XSL: `baseball/sb.bsgame.views.pxp.xsl`

Returns per-inning plays with batter, result, outs, scoring decisions.

### Other Views Available
- `baseball/sb.bsgame.views.box.xsl` — box score
- `baseball/sb.bsgame.views.scoring.xsl` — scoring summary
- `baseball/sb.bsgame.views.situational.xsl` — situational stats
- `baseball/sb.bsgame.views.pitching.xsl` — pitching stats
- `baseball/sb.bsgame.views.linescore.xsl` — linescore only

## Files

| File | Purpose |
|------|---------|
| `scripts/statbroadcast_client.py` | Core: decode, encode, fetch, parse_situation, parse_plays |
| `scripts/statbroadcast_discovery.py` | Event ID discovery (scan, group, SIDEARM), game matching |
| `scripts/statbroadcast_poller.py` | 20s daemon: polls broadcast + PXP, merges into DB, emits SSE |
| `scripts/statbroadcast_pregame.py` | Daily pre-game: broad scan + group scan + SIDEARM for new games |
| `scripts/statbroadcast_season_scrape.py` | Browser-based: loads SB schedule pages for all 255 schools |
| `scripts/sb_group_ids.json` | Team ID → SB group ID mapping (255 schools) |
| `data/sb_scrape_state.json` | Per-school last-scraped timestamps |
| `scripts/statbroadcast-poller.service` | Systemd unit (20s interval, auto-restart) |

## DB Schema

### `statbroadcast_events` table
```sql
sb_event_id INTEGER PRIMARY KEY,
game_id TEXT,                   -- FK to games.id
home_team TEXT,
visitor_team TEXT,
home_team_id TEXT,
visitor_team_id TEXT,
game_date TEXT,
group_id TEXT,
xml_file TEXT,
completed INTEGER DEFAULT 0,
discovered_at TIMESTAMP
```

### `games.situation_json` — sb_* fields
All StatBroadcast data is prefixed with `sb_` to avoid overwriting ESPN data:

| Field | Type | Source |
|-------|------|--------|
| `sb_outs` | int | Broadcast: mobile fallback span |
| `sb_count` | str | Broadcast: "1-2" |
| `sb_balls` | int | Parsed from count |
| `sb_strikes` | int | Parsed from count |
| `sb_batter` | str | Broadcast: "Smith,Trevor" |
| `sb_batter_position` | str | Broadcast: "1B", "CF" |
| `sb_pitcher` | str | Broadcast: "Johnson,Ashton" |
| `sb_on_first` | bool | Icon font glyph (0-7) |
| `sb_on_second` | bool | Icon font glyph |
| `sb_on_third` | bool | Icon font glyph |
| `sb_inning` | int | Inning number |
| `sb_inning_half` | str | "top" or "bottom" |
| `sb_inning_display` | str | "Bot 5th" |
| `sb_visitor_score` | int | Linescore R column |
| `sb_home_score` | int | Linescore R column |
| `sb_visitor_innings` | list | Per-inning runs [0,0,1,0,3] |
| `sb_home_innings` | list | Per-inning runs |
| `sb_visitor_hits` | int | Linescore H column |
| `sb_home_hits` | int | Linescore H column |
| `sb_visitor_errors` | int | Linescore E column |
| `sb_home_errors` | int | Linescore E column |
| `sb_plays` | list | Current inning play-by-play |
| `sb_updated_at` | str | ISO timestamp |

## Key Parsing Details

### Base Runners — Icon Font (sbicon)
The "Runners On Base" table is **unreliable** (empty between plays). The reliable source is the `base-indicator` div containing an sbicon font character:

```css
/* From sbicon.css */
.sbicon-runners-empty:before    {content: "\0030";}  /* "0" */
.sbicon-runners-first:before    {content: "\0031";}  /* "1" */
.sbicon-runners-second:before   {content: "\0032";}  /* "2" */
.sbicon-runners-third:before    {content: "\0033";}  /* "3" */
.sbicon-runners-first-second    {content: "\0034";}  /* "4" */
.sbicon-runners-second-third    {content: "\0035";}  /* "5" */
.sbicon-runners-first-third     {content: "\0036";}  /* "6" */
.sbicon-runners-loaded          {content: "\0037";}  /* "7" */
```

Parse: `<i class="sbicon">N</i>` where N is 0-7.

### Outs — Mobile Fallback
Desktop outs use icon font (`ZZ` junk behind `noaccess` paywall). The reliable source is the mobile fallback span:
```html
<span class="d-inline d-sm-none">2</span>
```

### Linescore — First Row Only
The HTML contains TWO rows with `VLogo`/`HLogo` — first is the linescore, second is a standings/record table. **Only use the first match** per team.

Inning cells are split from R/H/E by a `border-right` separator cell. Cells with inner HTML (caret icon, team logo) are automatically skipped by the `[^<]*` regex.

### `noaccess` Class
Some sections are behind a paywall (`noaccess` class on parent div). Affected: base diamond (icon still readable), defensive alignment, some stats. Outs mobile fallback is NOT behind noaccess.

## Event Discovery

### Three-phase approach:

1. **Season Scrape** (weekly, browser-based)
   - Loads `statbroadcast.com/events/statbroadcast.php?gid={schoolId}` for all 255 mapped schools
   - Uses `openclaw browser navigate` + `evaluate` (JS-rendered pages)
   - Extracts all event IDs, probes event API for metadata
   - Matches to games by team name + date
   - ~10s per school, ~40 min for full run
   - Tracks `last_scraped` per school in `sb_scrape_state.json`

2. **Daily Pre-game** (9:30 AM CST cron)
   - Verifies today's registered events still valid (dates haven't changed)
   - Runs broad ID scan + group scan for unmatched games
   - Announces results to #college-baseball

3. **Group Schedule Scan**
   - For each home team with a mapped group ID, scans SB schedule page
   - Falls back to SIDEARM school site scraping (only ≤20 unmatched)

### School sites only post SB links day-of/weekend — NOT full season
### SB's own schedule pages show the FULL season

## Cron Jobs

| Job | Schedule | ID | Purpose |
|-----|----------|-----|---------|
| SB Season Scrape | Sunday 3 AM CST | `c805616e` | Weekly refresh of all school event IDs |
| SB Pre-game Discovery | Daily 9:30 AM CST | `4d719179` | Verify + discover today's events |

## Systemd Services

| Service | Description |
|---------|-------------|
| `statbroadcast-poller` | 20s polling daemon, auto-restart on failure |
| `espn-fastcast` | WebSocket listener for ESPN score updates |
| `college-baseball-dashboard` | Gunicorn (workers=2, threads=4) on :5000 |

## Dashboard Integration

### Scores Page (game cards)
- Prefers `sb_*` data over ESPN for: outs, count, pitcher, batter, base runners
- Base diamond SVG with live fill updates via SSE
- Split sections: green "Live" (has SB data) vs gray "Live — ESPN only"
- JS `updateGameCard()` updates outs dots, count, pitcher, batter, diamond fills
- SSE listener for `sb_situation` events

### Game Detail Page
- Live situation panel: diamond, outs, count, pitcher/batter with position
- Per-inning linescore (falls back to SB when ESPN linescore unavailable)
- "Previously This Inning" play-by-play card with color-coded badges
- Scoring plays highlighted green, base hits in blue

### Page Caching
- Scores page cache **skipped** when any game is `in-progress` for that date
- Prevents stale base runner/outs data from being served

## Known Issues & Edge Cases

1. **SB returns 404 after game ends** — no backfill possible for missed games
2. **ESPN prematurely marks future games in-progress** — date guard added to reject
3. **Suspended games** — SJSU @ SMC suspended 4-4 after 12 innings; edge case, handle manually
4. **Zombie gunicorn processes** — `systemctl restart` can fail silently if old process holds port; use `fuser -k 5000/tcp` first
5. **FastCast listener drops** — reconnects OK but `rc:400 "Missing OpCode"` on heartbeat
6. **SIDEARM URL guessing** — fails for most schools, no team website URL mapping in DB

## Data Coverage

- **255 D1 schools** mapped in `sb_group_ids.json`
- SB covers most SIDEARM-powered athletic programs
- Small/NAIA/D2/D3 teams covered when playing a mapped larger team
- Not all games have SB coverage (neutral-site tournaments, ESPN-exclusive)

## ESPN Guard

Both `espn_live_scores.py` and `espn_fastcast_listener.py` have a date guard:
```python
if game_date > today:
    db_status = 'scheduled'  # Reject premature in-progress
```
