# MANIFEST.md — Code Classification
# ⚠️ CHECK THIS BEFORE ARCHIVING OR DELETING ANYTHING ⚠️

> Canonical inventory of active code, cron jobs, and services.
> Last updated: 2026-02-27 (D1B consolidation, FastCast removal, housekeeping)

## Legend
- 🔴 **CRITICAL** — App crashes if removed (web, models, core libs)
- 🟡 **CRON** — Called by scheduled cron jobs (breaking = silent data loss)
- 🟢 **ONE-SHOT** — Manual/utility scripts, safe to archive
- 🔵 **LIBRARY** — Imported by other scripts/models

---

## web/ — 🔴 CRITICAL (Dashboard)
| File | Role | Imported By |
|------|------|-------------|
| `web/app.py` | Flask dashboard | systemd service |

## models/ — 🔴 CRITICAL (Prediction Engine)
| File | Role | Imported By |
|------|------|-------------|
| `models/__init__.py` | Package init | everything |
| `models/compare_models.py` | Model registry, `MODELS` dict | `web/app.py`, `power_rankings.py` |
| `models/nn_features.py` | Feature computation (`FeatureComputer`) | training scripts, predictions |
| `models/predictor_db.py` | `Predictor` class for predict+track | `predict_and_track.py` |
| `models/ensemble_model.py` | Ensemble predictions | `add_game.py`, dashboard |
| `models/neural_model.py` | Neural net (BaseballNet) | training, ensemble |
| `models/nn_totals_model.py` | O/U totals model | training, predictions |
| `models/nn_spread_model.py` | Spread model | training, predictions |
| `models/nn_dow_totals_model.py` | Day-of-week totals | finetune |
| `models/elo_model.py` | Elo ratings | `update_elo.py`, `add_game.py` |
| `models/xgboost_model.py` | XGBoost model | `train_gradient_boosting.py` |
| `models/lightgbm_model.py` | LightGBM model | `train_gradient_boosting.py` |
| `models/base_model.py` | Base class | all models |
| `models/advanced_model.py` | Advanced stats model | ensemble |
| `models/conference_model.py` | Conference strength | ensemble |
| `models/log5_model.py` | Log5 model | ensemble |
| `models/momentum_model.py` | Momentum model | ensemble |
| `models/pitching_model.py` | Pitching model | ensemble |
| `models/poisson_model.py` | Poisson model | ensemble |
| `models/prior_model.py` | Prior/preseason model | ensemble |
| `models/pythagorean_model.py` | Pythagorean model | ensemble |
| `models/runs_ensemble.py` | Runs ensemble | ensemble |
| `models/weather_model.py` | Weather adjustments | ensemble |

### Experimental — On Hold (revisit mid-season with more data)
| File | Role |
|------|------|
| `models/neural_model_v3.py` | Neural v3 — new features/params |
| `models/lightgbm_model_v2.py` | LightGBM v2 |
| `models/xgboost_model_v2.py` | XGBoost v2 |
| `models/nn_features_enhanced.py` | Enhanced feature pipeline for v2/v3 |

## scripts/ — 🔵 LIBRARY (shared)
| File | Role | Imported By |
|------|------|-------------|
| `scripts/__init__.py` | Package init | everything |
| `scripts/database.py` | `get_connection()` + DB helpers | most scripts |
| `scripts/team_resolver.py` | Team name → ID resolution | DK scraper, SB scripts |
| `scripts/statbroadcast_client.py` | SB API client (ROT13+base64 decode, event info) | SB poller, season scrape, D1B discovery |
| `scripts/statbroadcast_discovery.py` | SB shared library: table schema, match_game, upsert, active events | SB poller, D1B discovery, season scrape, tests |
| `scripts/schedule_gateway.py` | **ScheduleGateway** — single write path to games table | d1b_team_sync, finalize_games, espn_live_scores |
| `scripts/verify_team_schedule.py` | D1B schedule verification + comparison | d1b_team_sync, finalize_games, tests |
| `scripts/run_utils.py` | ScriptRunner, logging utils | all cron bash scripts |

## Live Scoring Stack (3-tier)
| Tier | File | Method | Coverage | Schedule |
|------|------|--------|----------|----------|
| 1 (primary) | `scripts/statbroadcast_poller.py` | SB XML feeds, 20s poll | ~80% of games | systemd `statbroadcast-poller` |
| 2 (backup) | `scripts/espn_live_scores.py` | ESPN REST API | ~100 games | system cron, every 1 min 11-23 |
| 3 (catch-all) | `scripts/d1b_live_check.py` | D1B scores page status | all D1 games | system cron, every 30 min 11-23 |

## scripts/ — 🟡 CRON (scheduled, do NOT archive)

### Daily Pipeline (system cron, no AI)
| File | Schedule | Role |
|------|----------|------|
| `cron/00_schedule_sync.sh` | 12:30 AM | ESPN schedule sync |
| `cron/01_schedule_and_finalize.sh` | 5:00 AM | Finalize yesterday + D1B team sync |
| `cron/02_stats_scrape.sh` | 5:15 AM | D1B stats scrape (all teams) |
| `cron/03_derived_stats.sh` | 5:45 AM | Snapshots, pitching/batting quality |
| `cron/04_nightly_eval.sh` | 6:15 AM | Elo, verification, integrity |
| `cron/05_full_train.sh` | 7:00 AM | Train all models |
| `cron/06_morning_pipeline.sh` | 8:15 AM | Predict, bets, weather |
| `cron/d1b_game_times.sh` | 8:30/8:35 AM | D1B game time fill (today + tomorrow) |

### OpenClaw Cron (AI-powered)
| Job | Schedule | Role |
|-----|----------|------|
| D1B Pre-Game Discovery | 9:25 AM | D1B scrape → times + status + SB events |
| DK Odds Scrape | 8:00 AM | DraftKings browser scrape |
| FD Odds Scrape | 8:15 AM | FanDuel browser scrape |
| Pre-Game Odds Scheduler | 9:30 AM | Schedule pre-game odds refresh |
| SB Season Scrape | Sun 3:00 AM | Weekly SB event discovery |
| PEAR Ratings Fetch | Mon 7:00 AM | PEAR ratings |
| D1B Rankings | Mon 10:00 PM | Top 25 rankings |

### Cron Scripts (called by pipeline above)
| File | Role |
|------|------|
| `scripts/d1b_scraper.py` | D1B stats (basic + advanced) |
| `scripts/d1b_schedule.py` | D1B schedule/scores |
| `scripts/d1b_team_sync.py` | D1B team schedule sync |
| `scripts/finalize_games.py` | Finalize yesterday's games |
| `scripts/d1b_pregame_discovery.py` | D1B-powered pregame: times, status, SB events |
| `scripts/statbroadcast_season_scrape.py` | Weekly SB event discovery |
| `scripts/bet_selection_v2.py` | Best bet selection |
| `scripts/record_daily_bets.py` | Evaluate bet results |
| `scripts/update_elo.py` | Elo rating updates |
| `scripts/aggregate_team_stats.py` | Team stat aggregation |
| `scripts/predict_and_track.py` | Predictions + accuracy |
| `scripts/train_all_models.py` | Weekly model training |
| `scripts/power_rankings.py` | Weekly power rankings |
| `scripts/rankings.py` | D1B Top 25 |
| `scripts/weather.py` | Weather data |
| `scripts/backup_db.py` | Database backup |
| `scripts/verification_check.py` | Data verification |
| `scripts/dk_odds_scraper.py` | DK odds loading |
| `scripts/fd_odds_scraper.py` | FanDuel odds loading |
| `scripts/snapshot_stats.py` | Daily stat snapshots |
| `scripts/compute_pitching_quality.py` | Pitching quality tables |
| `scripts/compute_batting_quality.py` | Batting quality tables |
| `scripts/fetch_pear_ratings.py` | PEAR ratings |

### Cron-Adjacent (called by cron scripts, not directly scheduled)
| File | Role | Called By |
|------|------|-----------|
| `scripts/train_neural_v2.py` | Neural v2 training | `train_all_models.py` |
| `scripts/train_gradient_boosting.py` | XGB/LGB training | `train_all_models.py` |
| `scripts/finetune_weekly.py` | Weekly finetune | `train_all_models.py` |
| `scripts/compute_historical_features.py` | Historical features | training pipeline |

## scripts/ — 🟢 ONE-SHOT (safe to archive)
| File | Role |
|------|------|
| `scripts/add_game.py` | Manual game entry tool |
| `scripts/infer_starters.py` | Infer probable starters |
| `scripts/migrate_doubleheader_suffixes.py` | One-time game ID migration |
| `scripts/d1b_full_schedule_overwrite.py` | Full season schedule overwrite |
| `scripts/d1b_lineups.py` | D1B lineup scraper |
| `scripts/build_sb_mapping.py` | Build SB school→group_id mapping |
| `scripts/check_team_records.py` | Record verification |
| `scripts/backfill_missing_games.py` | Backfill missing games |
| `scripts/scrape_all_lineups.sh` | Batch lineup scrape |
| `scripts/train_gradient_boosting_v2.py` | XGB/LGB v2 (experimental) |
| `scripts/train_neural_v3.py` | Neural v3 (experimental) |
| `scripts/ncaa_stats_scrape.sh` | NCAA stats scraper |

## scripts/archive/ — 123 archived scripts
Old/replaced scripts. Includes FastCast listener, SB pregame scan, and legacy scrapers.

---

## Config — 🔴 DO NOT DELETE
| File | Role |
|------|------|
| `config/d1bb_slugs.json` | Team ID → D1B slug mapping (311 teams) |
| `data/d1b_slug_mapping.json` | D1B page slug → team ID (314 mappings) |
| `data/espn_team_mapping.json` | ESPN ID → team ID mapping |
| `scripts/sb_group_ids.json` | Team → SB group ID (216 teams) |

## Data — 🔴 DO NOT DELETE
| File | Role |
|------|------|
| `data/baseball.db` | The database (SQLite, WAL mode) |
| `data/debug_flags.json` | Debug page flags |

---

*Update this file when adding/removing scripts or cron jobs.*
