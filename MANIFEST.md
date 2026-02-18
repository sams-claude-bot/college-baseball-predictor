# MANIFEST.md — Code Classification
# ⚠️ CHECK THIS BEFORE ARCHIVING OR DELETING ANYTHING ⚠️

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

## scripts/ — 🔵 LIBRARY (shared)
| File | Role | Imported By |
|------|------|-------------|
| `scripts/__init__.py` | Package init | everything |
| `scripts/database.py` | `get_connection()` + DB helpers | most scripts |
| `scripts/team_resolver.py` | Team name → ID resolution | DK scraper cron |

## scripts/ — 🟡 CRON (scheduled jobs, do NOT archive)
| File | Cron Job | Schedule |
|------|----------|----------|
| `scripts/d1bb_scraper.py` | Nightly D1 Stats + Schedule | 1 AM daily |
| `scripts/d1bb_schedule.py` | Score updates (15min) + nightly schedule | */15 12-23 + 1 AM |
| `scripts/d1bb_box_scores.py` | Nightly Scores & Box Scores | 2 AM daily |
| `scripts/d1bb_advanced_scraper.py` | Advanced stats (SEC) | Mondays 3 AM |
| `scripts/bet_selection_v2.py` | Record daily best bets | 9 AM + pre-game |
| `scripts/record_daily_bets.py` | Evaluate bet results | 2 AM nightly |
| `scripts/update_elo.py` | Elo rating updates | 2 AM nightly |
| `scripts/aggregate_team_stats.py` | Team stat aggregation | 2 AM nightly |
| `scripts/predict_and_track.py` | Predictions + accuracy | 2 AM nightly + Sunday 10 PM |
| `scripts/train_all_models.py` | Weekly model training | Sunday 9:30 PM |
| `scripts/power_rankings.py` | Weekly power rankings | Monday 12 PM |
| `scripts/rankings.py` | D1BB Top 25 rankings | Monday 10 PM |
| `scripts/weather.py` | Weather data fetch | 8 AM daily |
| `scripts/backup_db.py` | Database backup | 2 AM nightly |
| `scripts/verification_check.py` | Data verification | post-collection |

## scripts/ — 🟡 CRON-ADJACENT (called by cron scripts, not directly scheduled)
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
| `scripts/build_pitching_infrastructure.py` | Populate players + pitcher_game_log tables |
| `scripts/infer_starters.py` | Infer probable starters for pitching_matchups |

## scripts/archive/ — Already archived (78 scripts)
Old/replaced scripts. Safe to ignore.

---

## Config files — 🔴 DO NOT DELETE
- `config/d1bb_slugs.json` — Team ID → D1BB slug mapping (311 teams)
- `config/` — All config files are critical

## Data — 🔴 DO NOT DELETE
- `data/baseball.db` — The database
- `data/debug_flags.json` — Debug page flags
- `data/bug_reports.json` — Bug reports

---

*Last updated: 2026-02-18 by Clawd*
*Update this file when adding/removing scripts or cron jobs.*
