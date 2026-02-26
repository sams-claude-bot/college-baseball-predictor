#!/usr/bin/env python3
"""
Predict games and track model performance.

Usage:
    python predict_and_track.py predict [DATE]       # Generate predictions for games
    python predict_and_track.py predict --refresh-existing          # Smart refresh (today+1 + changed games)
    python predict_and_track.py predict --refresh-existing --refresh-all  # Full refresh window
    python predict_and_track.py evaluate [DATE]      # Grade predictions against results
    python predict_and_track.py accuracy             # Show overall model accuracy
    python predict_and_track.py validate [--fix]     # Check for missing predictions
    python predict_and_track.py validate DATE --fix  # Fix missing predictions for DATE
"""

import sys
import sqlite3
import math
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.predictor_db import Predictor
from models.nn_totals_slim import SlimTotalsModel
from scripts.database import get_connection
from scripts.run_utils import ScriptRunner

MODEL_NAMES = ['pythagorean', 'elo', 'log5', 'advanced', 'pitching', 'conference', 'prior', 'poisson', 'neural', 'xgboost', 'lightgbm', 'pear', 'quality', 'ensemble', 'meta_ensemble']


def _load_calibration_params(cur):
    """Load calibration parameters, supporting both Platt and isotonic methods."""
    import json as _json
    try:
        cur.execute("SELECT model_name, a, b, n_samples, method, isotonic_json FROM model_calibration")
        params = {}
        for r in cur.fetchall():
            name = r[0]
            method = r[4] if r[4] else "platt"
            if method == "isotonic" and r[5]:
                iso_data = _json.loads(r[5])
                params[name] = {
                    "method": "isotonic",
                    "x": iso_data["x"],
                    "y": iso_data["y"],
                    "n": int(r[3]),
                }
            else:
                params[name] = {
                    "method": "platt",
                    "a": float(r[1]),
                    "b": float(r[2]),
                    "n": int(r[3]),
                }
        return params
    except Exception:
        return {}


def _apply_calibration(p, params):
    """Apply calibration (Platt or isotonic) to a raw probability."""
    if params is None:
        return p
    n = params.get("n", 0)
    if n < 120:
        return p

    method = params.get("method", "platt")

    if method == "isotonic":
        return _apply_isotonic(p, params)
    else:
        return _apply_platt(p, params)


def _apply_platt(p, params):
    a = params["a"]
    b = params["b"]
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    x = math.log(p / (1 - p))
    z = a * x + b
    calibrated = 1.0 / (1.0 + math.exp(-z))
    return min(max(calibrated, 0.001), 0.999)


def _apply_isotonic(p, params):
    """Apply isotonic calibration using stored breakpoints via linear interpolation."""
    x_thresh = params["x"]
    y_thresh = params["y"]
    p = float(p)
    if len(x_thresh) == 0:
        return p
    if p <= x_thresh[0]:
        return max(y_thresh[0], 0.001)
    if p >= x_thresh[-1]:
        return min(y_thresh[-1], 0.999)
    # Binary search for interval then linear interpolation
    lo, hi = 0, len(x_thresh) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if x_thresh[mid] <= p:
            lo = mid
        else:
            hi = mid
    # Linear interpolation between x_thresh[lo] and x_thresh[hi]
    x0, x1 = x_thresh[lo], x_thresh[hi]
    y0, y1 = y_thresh[lo], y_thresh[hi]
    if x1 == x0:
        calibrated = y0
    else:
        t = (p - x0) / (x1 - x0)
        calibrated = y0 + t * (y1 - y0)
    return min(max(calibrated, 0.001), 0.999)


def predict_games(date=None, days=3, runner=None, refresh_existing=False, refresh_all=False):
    """Generate and store predictions for upcoming games.
    
    Args:
        date: Specific date (YYYY-MM-DD) to predict. If None, predicts next `days` days.
        days: Number of days to look ahead when date is None (default: 3).
        runner: Optional ScriptRunner instance for logging.
    """
    from datetime import timedelta
    
    conn = get_connection()
    cur = conn.cursor()
    calibration_params = _load_calibration_params(cur)

    # Ensure raw_home_prob column exists
    try:
        cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='model_predictions'")
        schema_row = cur.fetchone()
        if schema_row and "raw_home_prob" not in schema_row[0]:
            cur.execute("ALTER TABLE model_predictions ADD COLUMN raw_home_prob REAL")
    except Exception:
        pass

    if date:
        date_start = date
        date_end = date
    else:
        today = datetime.now()
        date_start = today.strftime("%Y-%m-%d")
        # Smart default for refresh-existing: today + 1 day
        if refresh_existing and not refresh_all:
            date_end = (today + timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            date_end = (today + timedelta(days=days-1)).strftime("%Y-%m-%d")
    
    if refresh_existing:
        if refresh_all:
            # Full brute-force refresh across date window
            cur.execute('''
                SELECT DISTINCT g.id, g.home_team_id, g.away_team_id, h.name, a.name
                FROM games g
                JOIN teams h ON g.home_team_id = h.id
                JOIN teams a ON g.away_team_id = a.id
                WHERE g.date BETWEEN ? AND ?
                  AND g.status IN ('scheduled', 'in-progress')
            ''', (date_start, date_end))
        else:
            # Smart refresh: today+1 window plus games changed since last prediction
            cur.execute('''
                SELECT DISTINCT g.id, g.home_team_id, g.away_team_id, h.name, a.name
                FROM games g
                JOIN teams h ON g.home_team_id = h.id
                JOIN teams a ON g.away_team_id = a.id
                WHERE g.status IN ('scheduled', 'in-progress')
                  AND (
                        g.date BETWEEN ? AND ?
                        OR EXISTS (
                            SELECT 1
                            FROM model_predictions mp
                            WHERE mp.game_id = g.id
                              AND datetime(mp.predicted_at) < datetime(g.updated_at)
                        )
                        OR g.id NOT IN (SELECT DISTINCT game_id FROM model_predictions)
                  )
            ''', (date_start, date_end))
    else:
        # Get games that are missing predictions from ANY model
        cur.execute('''
            SELECT DISTINCT g.id, g.home_team_id, g.away_team_id, h.name, a.name
            FROM games g
            JOIN teams h ON g.home_team_id = h.id
            JOIN teams a ON g.away_team_id = a.id
            WHERE g.date BETWEEN ? AND ?
            AND g.status = 'scheduled'
            AND (
                -- Games with no predictions at all
                g.id NOT IN (SELECT DISTINCT game_id FROM model_predictions)
                -- OR games missing predictions from some models
                OR g.id IN (
                    SELECT game_id FROM (
                        SELECT g2.id as game_id, COUNT(DISTINCT mp.model_name) as model_count
                        FROM games g2
                        LEFT JOIN model_predictions mp ON g2.id = mp.game_id
                        WHERE g2.date BETWEEN ? AND ?
                        GROUP BY g2.id
                        HAVING model_count < ?
                    )
                )
            )
        ''', (date_start, date_end, date_start, date_end, len(MODEL_NAMES)))
    
    games = cur.fetchall()
    date_label = date_start if date_start == date_end else f"{date_start} to {date_end}"
    if runner:
        runner.info(f"Found {len(games)} games needing predictions for {date_label}")
    else:
        print(f"Found {len(games)} games needing predictions for {date_label}")
    
    # Initialize predictors for each model
    predictors = {name: Predictor(model=name) for name in MODEL_NAMES if name != 'meta_ensemble'}
    
    # Initialize meta-ensemble (runs after other models)
    from models.meta_ensemble import MetaEnsemble
    meta_ensemble = MetaEnsemble()
    
    # Initialize NN slim totals model
    nn_slim_totals = SlimTotalsModel()
    
    predictions_made = 0
    models_run = 0
    for game_id, home_id, away_id, home_name, away_name in games:
        if runner:
            runner.info(f"{away_name} @ {home_name}:")
        else:
            print(f"\n{away_name} @ {home_name}:")
        
        # Check which models already have predictions for this game
        cur.execute('SELECT model_name FROM model_predictions WHERE game_id = ?', (game_id,))
        existing_models = {row[0] for row in cur.fetchall()}

        for model_name, predictor in predictors.items():
            if (not refresh_existing) and model_name in existing_models:
                continue
            try:
                # Pass team IDs (not names) to avoid home/away confusion
                result = predictor.predict_game(home_id, away_id)
                home_prob = result.get('home_win_probability', 0.5)
                home_runs = result.get('projected_home_runs', result.get('predicted_home_runs'))
                away_runs = result.get('projected_away_runs', result.get('predicted_away_runs'))
                
                # Sanity check: verify the model understood home/away correctly
                returned_home = result.get('home_team', '').lower().replace(' ', '-')
                if returned_home and returned_home != home_id:
                    # Model returned teams in wrong order — flip probabilities
                    if runner:
                        runner.warn(f"  {model_name:12}: HOME/AWAY MISMATCH - expected {home_id}, got {returned_home}. Flipping.")
                    home_prob = 1 - home_prob
                    home_runs, away_runs = away_runs, home_runs
                
                raw_prob = home_prob
                home_prob = _apply_calibration(home_prob, calibration_params.get(model_name))

                cur.execute('''
                    INSERT INTO model_predictions
                    (game_id, model_name, predicted_home_prob, predicted_home_runs, predicted_away_runs, raw_home_prob)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(game_id, model_name) DO UPDATE SET
                        predicted_home_prob = excluded.predicted_home_prob,
                        predicted_home_runs = excluded.predicted_home_runs,
                        predicted_away_runs = excluded.predicted_away_runs,
                        raw_home_prob = excluded.raw_home_prob,
                        predicted_at = CURRENT_TIMESTAMP
                ''', (game_id, model_name, home_prob, home_runs, away_runs, raw_prob))
                
                if runner:
                    runs_str = f"{home_runs:.1f}-{away_runs:.1f}" if (home_runs is not None and away_runs is not None) else "—"
                    runner.info(f"  {model_name:12}: {home_prob*100:5.1f}% {home_name} | {runs_str}")
                else:
                    runs_str = f"{home_runs:.1f}-{away_runs:.1f}" if (home_runs is not None and away_runs is not None) else "—"
                    print(f"  {model_name:12}: {home_prob*100:5.1f}% {home_name} | {runs_str}")
                predictions_made += 1
                models_run += 1
            except Exception as e:
                if runner:
                    runner.warn(f"  {model_name:12}: ERROR - {e}")
                else:
                    print(f"  {model_name:12}: ERROR - {e}")
        
        # Meta-ensemble prediction (after all other models have run)
        if (refresh_existing or 'meta_ensemble' not in existing_models):
            try:
                conn.commit()  # ensure other predictions are visible
                meta_prob = meta_ensemble.predict(game_id=game_id)
                cur.execute('''
                    INSERT INTO model_predictions 
                    (game_id, model_name, predicted_home_prob)
                    VALUES (?, 'meta_ensemble', ?)
                    ON CONFLICT(game_id, model_name) DO UPDATE SET
                        predicted_home_prob = excluded.predicted_home_prob,
                        predicted_at = CURRENT_TIMESTAMP
                ''', (game_id, meta_prob))
                if runner:
                    runner.info(f"  {'meta_ensemble':12}: {meta_prob*100:5.1f}% {home_name}")
                else:
                    print(f"  {'meta_ensemble':12}: {meta_prob*100:5.1f}% {home_name}")
                predictions_made += 1
                models_run += 1
            except Exception as e:
                if runner:
                    runner.warn(f"  {'meta_ensemble':12}: ERROR - {e}")
                else:
                    print(f"  {'meta_ensemble':12}: ERROR - {e}")

        # Totals predictions: runs ensemble + per-component models + nn_totals
        # Runs for ALL games regardless of DK lines
        cur.execute('''
            SELECT over_under FROM betting_lines 
            WHERE home_team_id = ? AND away_team_id = ? AND over_under IS NOT NULL AND over_under > 0
            ORDER BY captured_at DESC LIMIT 1
        ''', (home_id, away_id))
        dk_row = cur.fetchone()
        dk_line = dk_row[0] if dk_row else None
        
        # Check if we already have totals for this game
        cur.execute('SELECT model_name FROM totals_predictions WHERE game_id = ?', (game_id,))
        existing_totals = {row[0] for row in cur.fetchall()}
        
        if 'runs_ensemble' not in existing_totals:
            # Runs ensemble + per-component model totals
            try:
                from models.runs_ensemble import predict as runs_predict, get_model_projections
                runs_result = runs_predict(home_id, away_id, total_line=dk_line)
                
                # Store ensemble totals prediction
                ens_total = runs_result.get('projected_total', 0)
                ou_data = runs_result.get('over_under', {})
                over_prob = ou_data.get('over_prob')
                under_prob = ou_data.get('under_prob')
                # Use DK line if available, otherwise store without O/U prediction
                if dk_line:
                    ens_prediction = 'OVER' if ens_total > dk_line else 'UNDER'
                    if over_prob is not None:
                        ens_edge = abs(over_prob - 0.5) * 100
                    else:
                        ens_edge = abs(ens_total - dk_line) / dk_line * 100
                else:
                    ens_prediction = None
                    ens_edge = None
                cur.execute('''
                    INSERT OR IGNORE INTO totals_predictions 
                    (game_id, over_under_line, projected_total, prediction, edge_pct, model_name, over_prob, under_prob)
                    VALUES (?, ?, ?, ?, ?, 'runs_ensemble', ?, ?)
                ''', (game_id, dk_line, ens_total, ens_prediction, ens_edge, over_prob, under_prob))
                line_str = f" (line {dk_line})" if dk_line else ""
                pred_str = f" → {ens_prediction}" if ens_prediction else ""
                if runner:
                    runner.info(f"  {'runs_ens':12}: projected total {ens_total:.1f}{line_str}{pred_str}")
                else:
                    print(f"  {'runs_ens':12}: projected total {ens_total:.1f}{line_str}{pred_str}")
                
                # Store per-component model totals (for weight adjustment tracking)
                for comp_name, comp_data in runs_result.get('model_breakdown', {}).items():
                    comp_total = comp_data.get('total', 0)
                    if comp_total > 0 and f'runs_{comp_name}' not in existing_totals:
                        if dk_line:
                            comp_pred = 'OVER' if comp_total > dk_line else 'UNDER'
                            comp_edge = abs(comp_total - dk_line) / dk_line * 100
                        else:
                            comp_pred = None
                            comp_edge = None
                        cur.execute('''
                            INSERT OR IGNORE INTO totals_predictions 
                            (game_id, over_under_line, projected_total, prediction, edge_pct, model_name)
                            VALUES (?, ?, ?, ?, ?, ?)
                        ''', (game_id, dk_line, comp_total, comp_pred, comp_edge, f'runs_{comp_name}'))
                        if runner:
                            runner.info(f"  {'runs_'+comp_name:12}: projected total {comp_total:.1f}{line_str}")
                        else:
                            print(f"  {'runs_'+comp_name:12}: projected total {comp_total:.1f}{line_str}")
            except Exception as e:
                if runner:
                    runner.warn(f"  {'runs_ens':12}: ERROR - {e}")
                else:
                    print(f"  {'runs_ens':12}: ERROR - {e}")
        
        # NN Slim Totals prediction
        if nn_slim_totals.is_trained() and 'nn_slim_totals' not in existing_totals:
            try:
                t_pred = nn_slim_totals.predict_game(home_id, away_id)
                proj_total = t_pred.get('projected_total', 0)
                dk = dk_line or 0
                prediction = 'OVER' if dk and proj_total > dk else 'UNDER' if dk else 'N/A'
                edge = abs(proj_total - dk) / dk * 100 if dk else 0
                cur.execute('''
                    INSERT OR IGNORE INTO totals_predictions 
                    (game_id, over_under_line, projected_total, prediction, edge_pct, model_name)
                    VALUES (?, ?, ?, ?, ?, 'nn_slim_totals')
                ''', (game_id, dk, proj_total, prediction, edge))
                if runner:
                    runner.info(f"  {'nn_slim_tot':12}: projected total {proj_total:.1f}" +
                               (f" (line {dk}) → {prediction}" if dk else ""))
                else:
                    print(f"  {'nn_slim_tot':12}: projected total {proj_total:.1f}" +
                          (f" (line {dk}) → {prediction}" if dk else ""))
            except Exception as e:
                if runner:
                    runner.warn(f"  {'nn_slim_tot':12}: ERROR - {e}")
                else:
                    print(f"  {'nn_slim_tot':12}: ERROR - {e}")

        # Old nn_spread removed — spreads disabled
    
    conn.commit()
    
    # Second pass: ensure ALL scheduled games have totals predictions (not just ones needing ML)
    cur.execute('''
        SELECT g.id, g.home_team_id, g.away_team_id, h.name, a.name
        FROM games g
        JOIN teams h ON g.home_team_id = h.id
        JOIN teams a ON g.away_team_id = a.id
        WHERE g.date BETWEEN ? AND ?
        AND g.status IN ('scheduled', 'in-progress')
        AND g.id NOT IN (SELECT DISTINCT game_id FROM totals_predictions WHERE model_name = 'runs_ensemble')
    ''', (date_start, date_end))
    totals_missing_games = cur.fetchall()
    
    if totals_missing_games:
        if runner:
            runner.info(f"Backfilling totals for {len(totals_missing_games)} games missing runs predictions")
        else:
            print(f"\nBackfilling totals for {len(totals_missing_games)} games missing runs predictions")
        
        for game_id, home_id, away_id, home_name, away_name in totals_missing_games:
            # Get DK line if available
            cur.execute('''
                SELECT over_under FROM betting_lines 
                WHERE home_team_id = ? AND away_team_id = ? AND over_under IS NOT NULL AND over_under > 0
                ORDER BY captured_at DESC LIMIT 1
            ''', (home_id, away_id))
            dk_row = cur.fetchone()
            dk_line = dk_row[0] if dk_row else None
            
            try:
                from models.runs_ensemble import predict as runs_predict
                runs_result = runs_predict(home_id, away_id, total_line=dk_line)
                
                ens_total = runs_result.get('projected_total', 0)
                ou_data = runs_result.get('over_under', {})
                over_prob = ou_data.get('over_prob')
                under_prob = ou_data.get('under_prob')
                if dk_line:
                    ens_prediction = 'OVER' if ens_total > dk_line else 'UNDER'
                    if over_prob is not None:
                        ens_edge = abs(over_prob - 0.5) * 100
                    else:
                        ens_edge = abs(ens_total - dk_line) / dk_line * 100
                else:
                    ens_prediction = None
                    ens_edge = None
                cur.execute('''
                    INSERT OR IGNORE INTO totals_predictions 
                    (game_id, over_under_line, projected_total, prediction, edge_pct, model_name, over_prob, under_prob)
                    VALUES (?, ?, ?, ?, ?, 'runs_ensemble', ?, ?)
                ''', (game_id, dk_line, ens_total, ens_prediction, ens_edge, over_prob, under_prob))
                
                for comp_name, comp_data in runs_result.get('model_breakdown', {}).items():
                    comp_total = comp_data.get('total', 0)
                    if comp_total > 0:
                        if dk_line:
                            comp_pred = 'OVER' if comp_total > dk_line else 'UNDER'
                            comp_edge = abs(comp_total - dk_line) / dk_line * 100
                        else:
                            comp_pred = None
                            comp_edge = None
                        cur.execute('''
                            INSERT OR IGNORE INTO totals_predictions 
                            (game_id, over_under_line, projected_total, prediction, edge_pct, model_name)
                            VALUES (?, ?, ?, ?, ?, ?)
                        ''', (game_id, dk_line, comp_total, comp_pred, comp_edge, f'runs_{comp_name}'))
                
                line_str = f" (line {dk_line})" if dk_line else ""
                if runner:
                    runner.info(f"  {away_name} @ {home_name}: {ens_total:.1f}{line_str}")
                else:
                    print(f"  {away_name} @ {home_name}: {ens_total:.1f}{line_str}")
            except Exception as e:
                if runner:
                    runner.warn(f"  {away_name} @ {home_name}: ERROR - {e}")
                else:
                    print(f"  {away_name} @ {home_name}: ERROR - {e}")
        
        conn.commit()
    
    # Validation: Check that all models have predictions for each game
    if runner:
        runner.info("--- Validation Check ---")
    else:
        print(f"\n--- Validation Check ---")
    missing = []
    for game_id, home_id, away_id, home_name, away_name in games:
        cur.execute('SELECT model_name FROM model_predictions WHERE game_id = ?', (game_id,))
        present_models = {row[0] for row in cur.fetchall()}
        missing_models = set(MODEL_NAMES) - present_models
        if missing_models:
            missing.append((game_id, away_name, home_name, missing_models))
    
    if missing:
        msg = f"{len(missing)} games missing predictions"
        if runner:
            runner.warn(msg)
            for game_id, away, home, models in missing:
                runner.info(f"  {away} @ {home}: missing {', '.join(sorted(models))}")
        else:
            print(f"⚠️  {msg}:")
            for game_id, away, home, models in missing:
                print(f"  {away} @ {home}: missing {', '.join(sorted(models))}")
    else:
        msg = f"All {len(MODEL_NAMES)} models have predictions for all {len(games)} games"
        if runner:
            runner.info(msg)
        else:
            print(f"✅ {msg}")
    
    conn.close()
    if runner:
        runner.info(f"Stored {predictions_made} predictions for {len(games)} games")
        runner.add_stat("games_predicted", len(games))
        runner.add_stat("models_run", models_run)
    else:
        print(f"\n✅ Stored {predictions_made} predictions for {len(games)} games")

def evaluate_predictions(date=None, runner=None):
    """Compare predictions to actual results.
    
    If date is None, evaluates ALL pending predictions (was_correct IS NULL)
    where the game has a final score. This ensures we catch up on any missed
    evaluations from past days.
    
    If date is specified, only evaluates predictions for that date.
    """
    conn = get_connection()
    cur = conn.cursor()
    
    # Get games with results and unevaluated predictions
    # Use LEFT JOIN for teams in case team IDs don't match teams table
    if date:
        cur.execute('''
            SELECT 
                g.id, COALESCE(h.name, g.home_team_id), COALESCE(a.name, g.away_team_id), 
                g.home_score, g.away_score,
                mp.model_name, mp.predicted_home_prob, g.date
            FROM games g
            LEFT JOIN teams h ON g.home_team_id = h.id
            LEFT JOIN teams a ON g.away_team_id = a.id
            JOIN model_predictions mp ON mp.game_id = g.id
            WHERE g.date = ?
            AND g.status = 'final'
            AND g.home_score IS NOT NULL
            AND mp.was_correct IS NULL
        ''', (date,))
    else:
        # Evaluate ALL pending predictions with completed games
        cur.execute('''
            SELECT 
                g.id, COALESCE(h.name, g.home_team_id), COALESCE(a.name, g.away_team_id), 
                g.home_score, g.away_score,
                mp.model_name, mp.predicted_home_prob, g.date
            FROM games g
            LEFT JOIN teams h ON g.home_team_id = h.id
            LEFT JOIN teams a ON g.away_team_id = a.id
            JOIN model_predictions mp ON mp.game_id = g.id
            WHERE g.status = 'final'
            AND g.home_score IS NOT NULL
            AND mp.was_correct IS NULL
            ORDER BY g.date
        ''')
    
    rows = cur.fetchall()
    date_str = date if date else "all pending"
    if runner:
        runner.info(f"Evaluating {len(rows)} predictions for {date_str}")
    else:
        print(f"Evaluating {len(rows)} predictions for {date_str}")
    
    updated = 0
    dates_evaluated = set()
    for game_id, home, away, home_score, away_score, model, home_prob, game_date in rows:
        home_won = home_score > away_score
        predicted_home = home_prob > 0.5
        correct = 1 if (home_won == predicted_home) else 0
        
        cur.execute('''
            UPDATE model_predictions 
            SET was_correct = ?
            WHERE game_id = ? AND model_name = ?
        ''', (correct, game_id, model))
        updated += 1
        dates_evaluated.add(game_date)
    
    conn.commit()
    conn.close()
    
    if dates_evaluated:
        msg = f"Updated {updated} predictions across {len(dates_evaluated)} date(s): {', '.join(sorted(dates_evaluated))}"
        if runner:
            runner.info(msg)
            runner.add_stat("games_evaluated", len(dates_evaluated))
        else:
            print(f"✅ {msg}")
    else:
        if runner:
            runner.info("No predictions needed evaluation")
        else:
            print(f"✅ No predictions needed evaluation")
    
    # Evaluate totals predictions
    conn = get_connection()
    cur = conn.cursor()
    cur.execute('''
        SELECT tp.rowid, tp.game_id, tp.prediction, tp.over_under_line, tp.model_name,
               g.home_score, g.away_score
        FROM totals_predictions tp
        JOIN games g ON tp.game_id = g.id
        WHERE tp.was_correct IS NULL
        AND g.home_score IS NOT NULL AND g.status = 'final'
    ''')
    totals_rows = cur.fetchall()
    totals_updated = 0
    for rowid, game_id, prediction, line, model_name, home_score, away_score in totals_rows:
        actual_total = home_score + away_score
        if prediction is None or line is None or prediction == 'N/A' or line == 0:
            # No DK line — just record actual total, no correct/incorrect
            correct = None
        elif actual_total == line:
            correct = None  # Push
        elif prediction == 'OVER':
            correct = 1 if actual_total > line else 0
        else:
            correct = 1 if actual_total < line else 0
        cur.execute('''
            UPDATE totals_predictions 
            SET was_correct = ?, actual_total = ?
            WHERE rowid = ?
        ''', (correct, actual_total, rowid))
        totals_updated += 1
    conn.commit()
    conn.close()
    if totals_updated:
        if runner:
            runner.info(f"Evaluated {totals_updated} totals predictions")
        else:
            print(f"✅ Evaluated {totals_updated} totals predictions")

def show_totals_accuracy():
    """Show totals model accuracy: MAE and O/U record"""
    conn = get_connection()
    cur = conn.cursor()

    # MAE per model
    cur.execute('''
        SELECT 
            tp.model_name,
            COUNT(*) as n,
            ROUND(AVG(ABS(tp.projected_total - (g.home_score + g.away_score))), 2) as mae,
            ROUND(AVG(tp.projected_total), 1) as avg_predicted,
            ROUND(AVG(g.home_score + g.away_score), 1) as avg_actual
        FROM totals_predictions tp
        JOIN games g ON tp.game_id = g.id
        WHERE g.status = 'final'
          AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL
        GROUP BY tp.model_name
        ORDER BY mae ASC
    ''')

    print("\n📊 TOTALS MODEL ACCURACY — MAE (Mean Absolute Error)")
    print("=" * 65)
    print(f"{'Model':<20} {'Games':>6} {'MAE':>7} {'Avg Pred':>9} {'Avg Actual':>11}")
    print("-" * 65)
    for row in cur.fetchall():
        print(f"{row[0]:<20} {row[1]:>6} {row[2]:>7.2f} {row[3]:>9.1f} {row[4]:>11.1f}")

    # O/U record per model
    cur.execute('''
        SELECT 
            model_name,
            COUNT(*) as total,
            SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN was_correct = 0 THEN 1 ELSE 0 END) as losses,
            ROUND(100.0 * SUM(was_correct) / COUNT(*), 1) as hit_rate
        FROM totals_predictions
        WHERE was_correct IS NOT NULL
        GROUP BY model_name
        ORDER BY hit_rate DESC
    ''')

    print("\n📊 TOTALS MODEL ACCURACY — O/U Record")
    print("=" * 55)
    print(f"{'Model':<20} {'W':>5} {'L':>5} {'Total':>6} {'Hit%':>7}")
    print("-" * 55)
    for row in cur.fetchall():
        print(f"{row[0]:<20} {row[2]:>5} {row[3]:>5} {row[1]:>6} {row[4]:>6.1f}%")

    # Over/Under split per model
    cur.execute('''
        SELECT 
            model_name,
            prediction,
            COUNT(*) as total,
            SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct,
            ROUND(100.0 * SUM(was_correct) / COUNT(*), 1) as hit_rate
        FROM totals_predictions
        WHERE was_correct IS NOT NULL
          AND prediction IN ('OVER', 'UNDER')
        GROUP BY model_name, prediction
        ORDER BY model_name, prediction
    ''')

    print("\n📊 TOTALS MODEL ACCURACY — Over/Under Split")
    print("=" * 60)
    print(f"{'Model':<20} {'Dir':<7} {'W':>5} {'Total':>6} {'Hit%':>7}")
    print("-" * 60)
    for row in cur.fetchall():
        correct = row[3]
        total = row[2]
        losses = total - correct
        print(f"{row[0]:<20} {row[1]:<7} {correct:>5} {total:>6} {row[4]:>6.1f}%")

    conn.close()


def show_accuracy():
    """Show model accuracy statistics"""
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute('''
        SELECT 
            model_name,
            COUNT(*) as total,
            SUM(was_correct) as correct,
            ROUND(100.0 * SUM(was_correct) / COUNT(*), 1) as accuracy
        FROM model_predictions
        WHERE was_correct IS NOT NULL
        GROUP BY model_name
        ORDER BY accuracy DESC
    ''')
    
    print("\n📊 MODEL ACCURACY")
    print("=" * 45)
    print(f"{'Model':<15} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("-" * 45)
    
    for model, total, correct, accuracy in cur.fetchall():
        print(f"{model:<15} {correct:>8} {total:>8} {accuracy:>9.1f}%")
    
    # Overall stats
    cur.execute('''
        SELECT COUNT(*), SUM(was_correct)
        FROM model_predictions
        WHERE was_correct IS NOT NULL AND model_name = 'ensemble'
    ''')
    total, correct = cur.fetchone()
    if total and total > 0:
        print("-" * 45)
        print(f"{'ENSEMBLE':<15} {correct:>8} {total:>8} {100*correct/total:>9.1f}%")
    
    conn.close()

def validate_predictions(date=None, fix=False):
    """Check for missing predictions and optionally backfill them.
    
    Args:
        date: Specific date to check, or None for all dates with predictions
        fix: If True, attempt to generate missing predictions
    """
    conn = get_connection()
    cur = conn.cursor()
    
    # Find games with incomplete predictions
    if date:
        cur.execute('''
            SELECT g.id, g.home_team_id, g.away_team_id, h.name, a.name, g.date,
                   COUNT(mp.model_name) as model_count
            FROM games g
            JOIN teams h ON g.home_team_id = h.id
            JOIN teams a ON g.away_team_id = a.id
            LEFT JOIN model_predictions mp ON g.id = mp.game_id
            WHERE g.date = ?
            GROUP BY g.id
            HAVING model_count < ?
        ''', (date, len(MODEL_NAMES)))
    else:
        cur.execute('''
            SELECT g.id, g.home_team_id, g.away_team_id, h.name, a.name, g.date,
                   COUNT(mp.model_name) as model_count
            FROM games g
            JOIN teams h ON g.home_team_id = h.id
            JOIN teams a ON g.away_team_id = a.id
            LEFT JOIN model_predictions mp ON g.id = mp.game_id
            WHERE g.id IN (SELECT DISTINCT game_id FROM model_predictions)
            GROUP BY g.id
            HAVING model_count < ?
        ''', (len(MODEL_NAMES),))
    
    incomplete = cur.fetchall()
    
    if not incomplete:
        print(f"✅ All games have complete predictions ({len(MODEL_NAMES)} models each)")
        conn.close()
        return
    
    print(f"⚠️  Found {len(incomplete)} games with missing predictions:")
    
    predictors = {name: Predictor(model=name) for name in MODEL_NAMES} if fix else {}
    fixed_count = 0
    
    for game_id, home_id, away_id, home_name, away_name, game_date, count in incomplete:
        # Find which models are missing
        cur.execute('SELECT model_name FROM model_predictions WHERE game_id = ?', (game_id,))
        present = {row[0] for row in cur.fetchall()}
        missing = set(MODEL_NAMES) - present
        
        print(f"\n  {away_name} @ {home_name} ({game_date}): {count}/{len(MODEL_NAMES)} models")
        print(f"    Missing: {', '.join(sorted(missing))}")
        
        if fix:
            # Get game status to determine if we should also evaluate
            cur.execute('SELECT status, home_score, away_score FROM games WHERE id = ?', (game_id,))
            game_info = cur.fetchone()
            is_final = game_info['status'] == 'final' and game_info['home_score'] is not None
            home_won = game_info['home_score'] > game_info['away_score'] if is_final else None
            
            for model_name in missing:
                try:
                    result = predictors[model_name].predict_game(home_name, away_name)
                    home_prob = result.get('home_win_probability', 0.5)
                    
                    # If game is final, evaluate immediately
                    was_correct = None
                    if is_final:
                        predicted_home_win = home_prob > 0.5
                        was_correct = 1 if predicted_home_win == home_won else 0
                    
                    cur.execute('''
                        INSERT INTO model_predictions 
                        (game_id, model_name, predicted_home_prob, was_correct)
                        VALUES (?, ?, ?, ?)
                    ''', (game_id, model_name, home_prob, was_correct))
                    
                    status = f"({'✓' if was_correct else '✗'})" if was_correct is not None else ""
                    print(f"    Fixed {model_name}: {home_prob*100:.1f}% {status}")
                    fixed_count += 1
                except Exception as e:
                    print(f"    Failed {model_name}: {e}")
            
            conn.commit()
    
    if fix:
        print(f"\n✅ Fixed {fixed_count} missing predictions")
    else:
        print(f"\nRun with --fix to backfill missing predictions")
    
    conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    cmd = sys.argv[1]
    date = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None
    fix = '--fix' in sys.argv
    refresh_existing = '--refresh-existing' in sys.argv
    refresh_all = '--refresh-all' in sys.argv
    
    if cmd == "predict":
        runner = ScriptRunner("predict_and_track_predict")
        predict_games(date=date, runner=runner, refresh_existing=refresh_existing, refresh_all=refresh_all)
        runner.finish()
    elif cmd == "evaluate":
        runner = ScriptRunner("predict_and_track_evaluate")
        evaluate_predictions(date, runner=runner)
        runner.finish()
    elif cmd == "accuracy":
        runner = ScriptRunner("predict_and_track_accuracy")
        show_accuracy()
        runner.finish()
    elif cmd == "totals_accuracy":
        runner = ScriptRunner("predict_and_track_totals_accuracy")
        show_totals_accuracy()
        runner.finish()
    elif cmd == "validate":
        runner = ScriptRunner("predict_and_track_validate")
        validate_predictions(date=date, fix=fix)
        runner.finish()
    elif cmd == "backfill_totals_probs":
        runner = ScriptRunner("backfill_totals_probs")
        runner.info("Backfilling over_prob/under_prob for existing totals_predictions...")
        import sqlite3
        conn = sqlite3.connect('data/baseball.db')
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        rows = cur.execute('''
            SELECT tp.id, tp.game_id, tp.over_under_line, tp.projected_total,
                   g.home_team_id, g.away_team_id
            FROM totals_predictions tp
            JOIN games g ON tp.game_id = g.id
            WHERE tp.model_name = 'runs_ensemble'
              AND tp.over_prob IS NULL
              AND tp.over_under_line IS NOT NULL
              AND tp.over_under_line > 0
        ''').fetchall()
        runner.info(f"Found {len(rows)} rows to backfill")
        updated = 0
        errors = 0
        from models.runs_ensemble import predict as runs_predict
        for row in rows:
            try:
                result = runs_predict(row['home_team_id'], row['away_team_id'], total_line=row['over_under_line'])
                ou_data = result.get('over_under', {})
                over_prob = ou_data.get('over_prob')
                under_prob = ou_data.get('under_prob')
                if over_prob is not None:
                    edge = abs(over_prob - 0.5) * 100
                    cur.execute('''
                        UPDATE totals_predictions SET over_prob = ?, under_prob = ?, edge_pct = ?
                        WHERE id = ?
                    ''', (over_prob, under_prob, edge, row['id']))
                    updated += 1
            except Exception as e:
                errors += 1
                if errors <= 5:
                    runner.warn(f"  Error for game {row['game_id']}: {e}")
        conn.commit()
        conn.close()
        runner.info(f"Backfilled {updated} rows ({errors} errors)")
        runner.finish()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
