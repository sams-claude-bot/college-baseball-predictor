# Cleanup Checklist

Conservative cleanup checklist for organization + docs sync. No deletions or runtime behavior changes unless explicitly marked and reviewed.

## Phase 1 (Low Risk): Docs + Classification

- [x] Confirm documentation ownership:
  - [x] `README.md` = overview/quickstart only
  - [x] `CONTEXT.md` = operational source of truth
  - [x] `MANIFEST.md` = path/classification inventory
  - [x] `docs/DASHBOARD.md` = dashboard routes/data dependencies only
- [x] Update `MANIFEST.md` paths that no longer exist (e.g., archived path mismatches)
- [x] Add status banner to `docs/stats-collection.md` (historical/superseded) or rewrite to current D1BB flow
- [x] Add status banner to `docs/P4_STATS_COLLECTION.md` (historical/superseded) or archive it
- [x] Add status banner to `docs/advanced-stats-TODO.md` (historical record) or archive it
- [x] Update `CONTEXT.md` cron section to clearly identify active vs legacy overlap in `cron/`
- [x] Update `README.md` links to point to canonical docs only
- [x] Add note documenting duplicate service files and canonical source choice (before moving files)

## Phase 2 (Medium Risk): Behavior-Neutral Reorganization

- [ ] Choose one archive namespace:
  - [ ] Keep `scripts/archive/` and migrate `scripts/archived/`
  - [ ] Or keep `scripts/archived/` and migrate `scripts/archive/`
- [ ] Create archive index/readme with categories and provenance dates
- [ ] Group experimental model/trainer files under a documented convention:
  - [ ] `models/lightgbm_model_v2.py`
  - [ ] `models/xgboost_model_v2.py`
  - [ ] `models/neural_model_v3.py`
  - [ ] `models/nn_features_enhanced.py`
  - [ ] `scripts/train_gradient_boosting_v2.py`
  - [ ] `scripts/train_neural_v3.py`
- [x] Add safety README in `scripts/migrations/` (manual DB migrations; do not run casually)
- [ ] Decide canonical service unit file path:
  - [ ] `config/baseball-dashboard.service`
  - [x] `web/college-baseball-dashboard.service`
- [x] Mark non-canonical service file as generated/copy/legacy (comment header)

## Phase 3 (High Risk, Defer): Prune Candidates After Verification

- [ ] Verify zero-byte root DB placeholders are unused:
  - [ ] `baseball.db`
  - [ ] `college_baseball.db`
- [ ] Build explicit prune proposal for confirmed-unused files (no deletion until approved)
- [ ] Validate no external references before any deletion:
  - [ ] `cron/`
  - [ ] systemd unit files
  - [ ] docs
  - [ ] tests
  - [ ] operator runbooks/manual commands
- [ ] Document rollback plan for any move/delete

## Evidence Refresh Commands (Repeat Before Changes)

- [ ] `rg --files scripts models web cron docs`
- [ ] `rg -n "train_gradient_boosting_v2|train_neural_v3|lightgbm_model_v2|xgboost_model_v2|neural_model_v3|nn_features_enhanced" .`
- [ ] `diff -u config/baseball-dashboard.service web/college-baseball-dashboard.service`
- [ ] `rg -n "scripts/d1baseball_stats.py|scripts/d1baseball_advanced.py|scripts/p4_stats_scraper.py" docs README.md CONTEXT.md MANIFEST.md`
