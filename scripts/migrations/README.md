# scripts/migrations/ — Manual DB Migration Scripts

These scripts are high-risk, one-time database migrations. They are retained for traceability and recovery reference.

## Safety rules

- Do not run casually.
- Do not include in cron/systemd automation.
- Read the specific script fully before execution.
- Back up `data/baseball.db` first.
- Run against the intended environment only (staging/test copy preferred first).

## Current contents

- `001_fix_game_ids.py`
- `002_merge_duplicate_teams.py`
- `003_cleanup_orphan_teams.py`

## Cleanup note

This README is documentation-only. No migration logic was changed in this pass.
