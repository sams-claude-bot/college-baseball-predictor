# Cleanup Progress — 2026-02-22 (Pass 2)

Safe cleanup continuation after commits `663e2d5` and `367a8ef`.

Scope of this pass: documentation sync + organization labeling only. No deletions. No runtime behavior changes. No cron command behavior changes.

## Exactly What Changed

### 1) Phase 1 docs/path validation + ownership wording

- Tightened canonical documentation ownership language in:
  - `README.md`
  - `CONTEXT.md`
  - `MANIFEST.md`
  - `docs/DASHBOARD.md`
- Added/updated explicit ownership boundaries:
  - `README.md` = overview/quickstart only
  - `CONTEXT.md` = canonical operational reference
  - `MANIFEST.md` = canonical path/classification inventory
  - `docs/DASHBOARD.md` = dashboard route/data dependency reference only

### 2) Cron reference reconciliation (docs only; no behavior changes)

- Updated `CONTEXT.md` cron section to reflect merged active step:
  - `01_schedule_and_finalize.sh` (active merged flow)
  - retained note that `01_schedule_sync.sh` / `01b_late_scores.sh` still exist as legacy/overlap scripts
- Added `full_train.sh` to the system-cron table in `CONTEXT.md`
- Updated `docs/DASHBOARD.md` references from `01_schedule_sync.sh` to `01_schedule_and_finalize.sh`
- Updated `docs/DASHBOARD.md` data-flow line to use `full_train.sh` name explicitly

### 3) Path-validation sweep fixes (docs/manifests)

- Fixed a stale/non-existent path reference in `MANIFEST.md`:
  - replaced the struck-through advanced-scraper path entry with archived historical path `scripts/archive/d1baseball_advanced.py`

### 4) Safe Phase 2 subset (no moves, no runtime changes)

- Added archive convention READMEs (no file moves):
  - `scripts/archive/README.md`
  - `scripts/archived/README.md`
- Added migration safety README:
  - `scripts/migrations/README.md`
- Added comment-only canonical/copy headers to service files:
  - `web/college-baseball-dashboard.service` (marked canonical repo source)
  - `config/baseball-dashboard.service` (marked legacy/reference copy)

### 5) Progress tracking updates

- Updated `docs/CLEANUP_CHECKLIST.md` checkboxes to reflect completed Phase 1 items and completed safe subset items (migrations README + service canonical/copy header + canonical service path choice).
- `Create archive index/readme with categories and provenance dates` remains unchecked (partial progress only; convention READMEs added but not a full categorized archive index yet).

## Lightweight Validation Run (This Pass)

### Commands used

- `rg -n "01_schedule_sync\\.sh|01_schedule_and_finalize\\.sh|full_train\\.sh" README.md CONTEXT.md MANIFEST.md docs/*.md`
- `diff -u config/baseball-dashboard.service web/college-baseball-dashboard.service`
- Custom Python backtick-path existence sweep across:
  - `README.md`
  - `CONTEXT.md`
  - `MANIFEST.md`
  - `docs/*.md`

### Results

- Cron references now show active merged script names in canonical docs (`CONTEXT.md`, `docs/DASHBOARD.md`) while retaining legacy overlap note in `CONTEXT.md`.
- Service file diff still shows expected configuration divergence; this pass only added comment headers (no unit behavior changes).
- Path sweep initially flagged:
  - `MANIFEST.md` stale path for advanced scraper (fixed in this pass)
  - historical/missing path references inside `docs/CLEANUP_AUDIT_2026-02-22.md` (expected; that document intentionally lists stale/missing paths as findings)
  - a pre-fix mention of the stale manifest path in this progress doc draft (updated before final validation)

## What Remains

### Phase 2 (safe, behavior-neutral) remaining

- Choose and document a single long-term archive namespace (`scripts/archive/` vs `scripts/archived/`)
- Build a fuller archive index/readme with categories + provenance dates
- Document grouping convention for experimental models/trainers (no moves yet or later safe moves with wrappers)
- Decide whether to physically consolidate service unit location (`config/` vs `web/`) after confirming operator workflow

### Later / higher risk (deferred)

- Any file moves, deletions, or DB cleanup
- Any cron command/path/order changes
- Any runtime/service execution changes
