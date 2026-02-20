# Improvements & Technical Debt
*Created: 2026-02-19 by Clawd*

## Data Quality
- [ ] Audit for duplicate game IDs (e.g. `indiana_north-carolina_g2` vs `indiana_unc_g1` for same matchup) — may be more lurking
- [ ] Audit `winner_id` NULLs on final games — the UNC/Indiana fix exposed this can happen on re-inserted games
- [ ] Drop 8 empty legacy tables: `ncaa_individual_stats`, `ncaa_team_stats`, `player_boxscore_batting`, `player_boxscore_pitching`, `predictions`, `spread_predictions`, `team_stats`, `team_stats_snapshots`
- [ ] Clean up near-empty tables: `game_boxscores` (6 rows), `statbroadcast_boxscores` (2 rows) — dead test data

## Models
- [ ] Revisit v2/v3 neural net and gradient boosting models mid-season (parked — not enough data yet)
- [ ] Investigate neural net accuracy drop (88% → 74%) — possible overfitting as sample size grew
- [ ] Revisit spread model calibration — currently disabled in betting
- [ ] Integrate `nn_features_enhanced.py` into v2/v3 pipeline when ready

## Pipeline & Cron
- [x] ~~Archive `d1bb_advanced_scraper.py`~~ — redundant with `d1bb_scraper.py` (done Feb 19)
- [ ] Verify `verification_job.py` in system cron (runs from project root at 3 AM, not `scripts/`) — is it still needed or stale?
- [ ] Verify `scrape_all_lineups.sh` in system cron (Mon 6 AM) — undocumented, is it needed?
- [ ] DK odds scraper reliability — AI-in-the-loop works but fails silently sometimes
- [ ] Delete disabled OpenClaw cron jobs that were fully replaced by system cron (16 sitting disabled)

## Upcoming
- [ ] **March 16**: NCAA official RPI data drops — pull it in and populate `ncaa_rpi` / `ncaa_rank` columns in `team_rpi` table, then compare against Sam's RPI

## New Features
- [ ] Finish lineup/starter prediction system:
  1. [x] Fix `d1bb_lineups.py` to use config slugs (done Feb 19)
  2. [x] Run lineup scrape for all 407 teams (running Feb 19)
  3. [ ] Wire up `infer_starters.py` to predict DOW rotation slots per team
  4. [ ] Feed predicted starters into pitching model for better matchup predictions
  5. [ ] Add lineup scraping to a nightly/weekly cron once stable
- [ ] Starter predictions page on dashboard
- [ ] Verify `model_testing.html` and `model_trends.html` pages are wired up and working

## Housekeeping
- [ ] Commit doc updates from Feb 19 audit (MANIFEST.md, CONTEXT.md, MEMORY.md)
- [ ] `d1bb_full_schedule_overwrite.py` — useful for mid-season full schedule refresh, document when to use
- [ ] Review and update MEMORY.md with Feb 19 work (doc audit, lineup scraper, game fixes)
