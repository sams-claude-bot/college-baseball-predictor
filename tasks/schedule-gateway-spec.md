# Task: Build Schedule Gateway Module

## Objective
Create `scripts/schedule_gateway.py` — the **single write path** to the `games` table. All schedule scripts will route through this module instead of writing directly.

## Problem Statement
Currently 7+ scripts write to `games` with 3 different upsert strategies, causing:
- Ghost games (different ID generation per script)
- Duplicate games (each script's dedup misses what another created)
- Score conflicts between competing updaters
- Known bugs: arkansas/kansas substring match, miami-ohio/miami-fl collision, ghost gm3/gm4 games

## Architecture

### New Module: `scripts/schedule_gateway.py`

```python
class ScheduleGateway:
    """Single write path to the games table."""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.resolver = TeamResolver()  # from scripts/team_resolver.py
    
    # === ID Generation (ONE canonical implementation) ===
    
    def canonical_game_id(self, date: str, away_id: str, home_id: str, game_num: int = 1) -> str:
        """Generate canonical game ID. Format: YYYY-MM-DD_away-id_home-id[_gmN]
        game_num=1 has no suffix. game_num=2+ gets _gm2, _gm3 etc.
        Max game_num=2 (doubleheader limit)."""
    
    # === Team Resolution (delegates to TeamResolver) ===
    
    def resolve_team(self, name: str, slug: str = None, source: str = None) -> str | None:
        """Resolve a team name/slug to canonical team_id.
        Uses TeamResolver (alias table). No substring matching ever."""
    
    # === Game Writes (THE upsert) ===
    
    def upsert_game(self, date, away_id, home_id, game_num=1, 
                     time=None, home_score=None, away_score=None,
                     status=None, inning_text=None, source=None) -> str:
        """Insert or update a game. Returns action taken: 'created', 'updated', 'unchanged', 'replaced'.
        
        Dedup strategy (in order):
        1. Exact canonical ID match
        2. Legacy suffix variants (_g1/_gm1 for game 1, _g2 for game 2)  
        3. Swapped home/away ID match (same date, same teams, reversed)
        4. Fuzzy: same date + same two teams in any order (no suffix for game 1)
        
        Score update rules:
        - Never overwrite final scores with None
        - in-progress can update scores on existing scheduled/in-progress
        - final can update scores on anything
        - scheduled cannot overwrite in-progress or final
        
        Ghost replacement:
        - If matched game has different ID, migrate FK rows then replace
        """
    
    def finalize_game(self, game_id, home_score, away_score) -> bool:
        """Mark a game as final with scores. Computes winner_id. Clears inning_text."""
    
    def mark_postponed(self, game_id, reason=None) -> bool:
        """Mark game as postponed."""
    
    def mark_cancelled(self, game_id, reason=None) -> bool:
        """Mark game as cancelled."""
    
    def update_live_score(self, game_id, home_score, away_score, inning_text) -> bool:
        """Update in-progress game scores + inning text."""
    
    # === Ghost/Duplicate Management ===
    
    def find_existing_game(self, date, away_id, home_id, game_num=1):
        """Find existing game row matching this matchup (any ID variant)."""
    
    def migrate_fk_rows(self, old_game_id, new_game_id):
        """Migrate foreign key references from old to new game ID.
        Tables: model_predictions, betting_lines, game_weather, tracked_bets, 
        tracked_confident_bets, totals_predictions, game_predictions,
        pitching_matchups, game_batting_stats, game_pitching_stats, etc."""
```

## Key Design Rules

1. **No substring matching in team resolution.** The arkansas/kansas bug was caused by substring matching. Only exact alias lookups via TeamResolver.

2. **Game ID is deterministic.** Given (date, away_id, home_id, game_num), the ID is always the same. No per-script variations.

3. **Status hierarchy.** `final` > `in-progress` > `scheduled` > `postponed`/`cancelled`. A lower-status update cannot overwrite a higher-status game unless explicitly forced.

4. **FK migration on ghost replacement.** When replacing an ESPN ghost or mismatched ID, migrate all foreign key rows (predictions, betting lines, weather, etc.) before deleting the old game.

5. **Audit logging.** Every write should print a structured log line: `[ScheduleGateway] {action} {game_id} (source={source})` so cron logs are traceable.

6. **Max 2 games per matchup per day.** Reject game_num > 2 with a warning.

## Tests Required: `tests/test_schedule_gateway.py`

### Unit Tests (use in-memory SQLite)
1. `test_canonical_game_id` — format, suffix logic, max game_num=2 enforcement
2. `test_upsert_create_new_game` — basic insert
3. `test_upsert_update_scores` — update existing game with scores
4. `test_upsert_no_overwrite_final` — scheduled update can't overwrite final
5. `test_upsert_legacy_suffix_match` — finds _g1 when looking for unsuffixed game 1
6. `test_upsert_swapped_home_away` — finds game with teams in opposite order
7. `test_upsert_fuzzy_same_date_teams` — finds game by date+teams when ID differs
8. `test_upsert_ghost_replacement` — ESPN ghost gets replaced, FKs migrated
9. `test_status_hierarchy` — final > in-progress > scheduled, can't downgrade
10. `test_doubleheader_game_1_and_2` — two games same matchup same day
11. `test_reject_game_num_3` — game_num=3 rejected with warning
12. `test_finalize_game` — sets status=final, computes winner, clears inning_text
13. `test_mark_postponed` — status set, reason stored
14. `test_mark_cancelled` — status set
15. `test_update_live_score` — in-progress update with inning text
16. `test_fk_migration` — model_predictions, betting_lines rows migrate to new game ID
17. `test_resolve_team_no_substring` — "kansas" does NOT resolve to "arkansas"
18. `test_resolve_team_exact_alias` — "FIU" → "florida-international"

### Integration Test: Win/Loss Record Verification
19. `test_win_loss_records_match_d1baseball` — **THE critical test.**
    - Pick 5-10 teams (mix of SEC + mid-major): mississippi-state, auburn, florida, wichita-state, dallas-baptist, hofstra, omaha
    - Fetch their current schedule from D1Baseball (HTTP, using `verify_team_schedule.fetch_d1bb_schedule()`)
    - Count wins and losses from D1BB data
    - Count wins and losses from our DB (`games` table WHERE status='final')
    - Assert they match exactly
    - If mismatch, report which games differ (extra in DB, missing from DB, score disagreements)
    - This test should be runnable standalone: `pytest tests/test_schedule_gateway.py::test_win_loss_records_match_d1baseball -v`
    - Mark it with `@pytest.mark.integration` so it can be excluded from fast unit test runs (it makes HTTP calls)

## Files to Create/Modify

### Create:
- `scripts/schedule_gateway.py` — the gateway module
- `tests/test_schedule_gateway.py` — comprehensive tests

### Do NOT modify yet (Phase 2):
- Do NOT rewire existing scripts in this task
- Do NOT change cron jobs
- Do NOT touch d1bb_schedule.py, d1bb_team_sync.py, etc.
- This task is ONLY the gateway module + tests

## Acceptance Criteria
1. `pytest tests/test_schedule_gateway.py -v` — all 18 unit tests pass
2. `pytest tests/test_schedule_gateway.py -v -m integration` — win/loss records match D1Baseball for all test teams
3. Module is importable: `from scripts.schedule_gateway import ScheduleGateway`
4. No changes to existing scripts (this is additive only)

## Existing Code Reference
- Team resolver: `scripts/team_resolver.py` (172 lines, uses `team_aliases` table)
- Current upsert in d1bb_schedule.py: lines 354-440 (best ghost handling)
- Current upsert in d1bb_team_sync.py: lines 81-240 (best score reconciliation)
- FK tables list: see `_replace_espn_ghost()` in d1bb_schedule.py lines 245-280
- D1BB schedule fetcher: `scripts/verify_team_schedule.py` `fetch_d1bb_schedule()` (HTTP, no browser)
- DB schema for games: `CREATE TABLE games (id TEXT PRIMARY KEY, date TEXT, time TEXT, home_team_id TEXT, away_team_id TEXT, home_score INTEGER, away_score INTEGER, winner_id TEXT, status TEXT, inning_text TEXT, ...)`
- DB helper: `scripts/database.py` `get_connection()`

## Notes
- The integration test makes HTTP calls to d1baseball.com — add appropriate timeout/retry
- Use `scripts/database.py`'s `get_connection()` for DB access
- Import TeamResolver from `scripts/team_resolver.py` — don't reinvent resolution
- The ghost replacement logic in `d1bb_schedule.py` `_replace_espn_ghost()` is the most battle-tested FK migration — port it
