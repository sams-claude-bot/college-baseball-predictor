# Tier 2 Batch B - Final Report

**Agent:** tier2-batch-b  
**Date:** 2026-02-16  
**Conferences:** MVC, Big East, C-USA, ASUN

## Summary

### ✅ Completed

| Conference | Teams | Avg Games | Min Games | Teams w/Rosters |
|------------|-------|-----------|-----------|-----------------|
| ASUN | 10 | 65.5 | 58 | 8 |
| Big East | 8 | 59.4 | 52 | 5 |
| C-USA | 10 | 64.9 | 0* | 9 |

*C-USA min is 0 due to UTEP having no ESPN data

### ⚠️ MVC Still Needs Work

| Conference | Teams | Avg Games | Min Games | Teams w/Rosters |
|------------|-------|-----------|-----------|-----------------|
| MVC | 12 | 24.0 | 0 | 5 |

## Changes Made

### Big East Cleanup
- Removed **Marquette** and **Providence** - these schools don't have baseball programs
- Merged duplicate UConn entries (connecticut + uconn → uconn)
- Final: 8 teams with 52-67 games each

### Schedules Added
- **MVC:** 28 new games from ESPN
- **Big East:** 12 new games from ESPN  
- **C-USA:** 76 new games from ESPN
- **ASUN:** 63 new games from ESPN

### Player Rosters Added
- Total: **1,730 player entries** across 27 teams
- Basic info: name, number, position
- Batting/pitching stats NOT yet scraped

## Teams Still Below 40 Games

ESPN API has **no schedule data** for these teams:
- **Drake** (MVC) - 0 games
- **Northern Iowa** (MVC) - 0 games  
- **UTEP** (C-USA) - 0 games

These teams have limited data:
- **UALR** - 3 games
- **Valparaiso** - 11 games
- **Bradley** - 12 games
- **Southern Illinois** - 19 games
- **Illinois State** - 22 games
- **Indiana State** - 25 games
- **Belmont** - 28 games
- **Evansville** - 29 games

## Recommendations for Follow-up

1. **Manual scraping needed** for Drake, Northern Iowa, UTEP - their schedules are not in ESPN's API
2. **Athletics website scraping** for remaining MVC teams with <40 games
3. **Player stats** (batting/pitching) need separate scraping effort

## Files Changed
- `data/baseball.db` - schedules and rosters
- `data/espn_team_ids.json` - added evansville, ualr, middle-tennessee IDs

## Commits
1. "Tier 2: add MVC/Big East/C-USA/ASUN schedules"
2. "Tier 2: add player rosters for MVC/Big East/C-USA/ASUN"
