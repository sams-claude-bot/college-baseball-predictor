"""
Models Blueprint - Model performance page
"""

import sys
from pathlib import Path
from collections import defaultdict
from flask import Blueprint, render_template

# Add paths for imports
base_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / "scripts"))

from database import get_connection
from models.compare_models import MODELS

from web.helpers import get_ensemble_weights, get_model_accuracy

models_bp = Blueprint('models', __name__)


@models_bp.route('/models/trends')
def model_trends():
    """Detailed model accuracy trends page"""
    conn = get_connection()
    c = conn.cursor()
    
    # Get daily accuracy per model
    c.execute('''
        SELECT mp.model_name, g.date, 
               COUNT(*) as total,
               SUM(CASE WHEN mp.was_correct = 1 THEN 1 ELSE 0 END) as correct
        FROM model_predictions mp
        JOIN games g ON mp.game_id = g.id
        WHERE mp.was_correct IS NOT NULL
        GROUP BY mp.model_name, g.date
        ORDER BY g.date
    ''')
    
    daily_data = defaultdict(list)
    for row in c.fetchall():
        daily_data[row['model_name']].append({
            'date': row['date'],
            'total': row['total'],
            'correct': row['correct'],
            'accuracy': round(row['correct'] / row['total'] * 100, 1) if row['total'] > 0 else 0
        })
    
    # Get cumulative accuracy per model
    c.execute('''
        SELECT mp.model_name, 
               COUNT(*) as total,
               SUM(CASE WHEN mp.was_correct = 1 THEN 1 ELSE 0 END) as correct
        FROM model_predictions mp
        WHERE mp.was_correct IS NOT NULL
        GROUP BY mp.model_name
        ORDER BY correct * 1.0 / total DESC
    ''')
    
    cumulative = []
    for row in c.fetchall():
        cumulative.append({
            'model': row['model_name'],
            'total': row['total'],
            'correct': row['correct'],
            'accuracy': round(row['correct'] / row['total'] * 100, 1) if row['total'] > 0 else 0
        })
    
    # Build rolling accuracy (30-game window) for chart
    c.execute('''
        SELECT mp.model_name, g.date, mp.was_correct
        FROM model_predictions mp
        JOIN games g ON mp.game_id = g.id
        WHERE mp.was_correct IS NOT NULL
        ORDER BY g.date, mp.id
    ''')
    
    model_history = defaultdict(list)
    for row in c.fetchall():
        model_history[row['model_name']].append({
            'date': row['date'],
            'correct': row['was_correct']
        })
    
    # Build daily accuracy (one point per day, not per prediction)
    rolling_data = {}
    for model_name, entries in model_history.items():
        # Group entries by date
        by_date = defaultdict(list)
        for entry in entries:
            by_date[entry['date']].append(entry['correct'])
        
        # Build daily points with cumulative and rolling (last 7 days)
        points = []
        cumulative_correct = 0
        cumulative_total = 0
        sorted_dates = sorted(by_date.keys())
        
        for i, date in enumerate(sorted_dates):
            day_results = by_date[date]
            cumulative_correct += sum(day_results)
            cumulative_total += len(day_results)
            
            # Rolling: last 7 days
            rolling_correct = 0
            rolling_total = 0
            for j in range(max(0, i - 6), i + 1):  # Last 7 days including today
                d = sorted_dates[j]
                rolling_correct += sum(by_date[d])
                rolling_total += len(by_date[d])
            
            points.append({
                'date': date,
                'cumulative': round(cumulative_correct / cumulative_total * 100, 1),
                'rolling': round(rolling_correct / rolling_total * 100, 1) if rolling_total > 0 else 0,
                'games': cumulative_total,
                'day_games': len(day_results)
            })
        rolling_data[model_name] = points
    
    # Get dates with games
    c.execute('''
        SELECT DISTINCT g.date 
        FROM model_predictions mp
        JOIN games g ON mp.game_id = g.id
        WHERE mp.was_correct IS NOT NULL
        ORDER BY g.date
    ''')
    dates = [row['date'] for row in c.fetchall()]

    # Defensive normalization: if neural points are one day ahead of canonical dates,
    # shift them back by one day for display consistency.
    if dates and 'neural' in rolling_data and rolling_data['neural']:
        max_date = max(dates)
        neural_max = max(p['date'] for p in rolling_data['neural'])
        if neural_max > max_date:
            from datetime import datetime, timedelta
            for p in rolling_data['neural']:
                p['date'] = (datetime.strptime(p['date'], '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
            # Re-sort after adjustment
            rolling_data['neural'].sort(key=lambda x: x['date'])
    
    conn.close()
    
    return render_template('model_trends.html',
                          daily_data=daily_data,
                          cumulative=cumulative,
                          rolling_data=rolling_data,
                          dates=dates)


@models_bp.route('/models')
def models():
    """Model performance page"""
    weights = get_ensemble_weights()
    accuracy = get_model_accuracy()

    # Combine weights and accuracy (include models with accuracy but not in ensemble)
    model_data = []
    all_model_names = set(weights.keys()) | set(accuracy.keys())
    for name in all_model_names:
        data = {
            'name': name,
            'weight': weights.get(name, 0),
            'weight_pct': weights.get(name, 0) * 100,
            'independent': name not in weights
        }
        if name in accuracy:
            data['accuracy'] = accuracy[name]
        model_data.append(data)

    # Sort: ensemble models by weight first, then independent models by accuracy
    model_data.sort(key=lambda x: (not x['independent'], x['weight']), reverse=True)

    # Get weight history from DB (if exists)
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute('''
            SELECT * FROM ensemble_weights_history
            ORDER BY recorded_at DESC
            LIMIT 100
        ''')
        weight_history = [dict(row) for row in c.fetchall()]
    except:
        weight_history = []
    conn.close()

    # Get prediction log (if exists)
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute('''
            SELECT p.*, g.home_score, g.away_score, g.status,
                   ht.name as home_team_name, at.name as away_team_name
            FROM predictions p
            LEFT JOIN games g ON p.game_id = g.id
            LEFT JOIN teams ht ON g.home_team_id = ht.id
            LEFT JOIN teams at ON g.away_team_id = at.id
            ORDER BY p.predicted_at DESC
            LIMIT 50
        ''')
        prediction_log = [dict(row) for row in c.fetchall()]
    except:
        prediction_log = []

    # Get runs ensemble weights
    try:
        import models.runs_ensemble as runs_ens
        runs_weights = runs_ens.get_weights()
    except:
        runs_weights = {}

    # Get totals prediction accuracy
    try:
        c.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct,
                SUM(CASE WHEN was_correct = 0 THEN 1 ELSE 0 END) as incorrect
            FROM totals_predictions
            WHERE was_correct IS NOT NULL
            AND prediction IS NOT NULL AND prediction != ''
        ''')
        totals_overall = dict(c.fetchone())

        # By type
        c.execute('''
            SELECT 
                prediction,
                COUNT(*) as total,
                SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct
            FROM totals_predictions
            WHERE was_correct IS NOT NULL
            AND prediction IS NOT NULL AND prediction != '' AND prediction != 'N/A'
            GROUP BY prediction
        ''')
        totals_by_type = [dict(row) for row in c.fetchall()]

        # By edge bucket
        c.execute('''
            SELECT 
                CASE 
                    WHEN edge_pct >= 30 THEN '30%+'
                    WHEN edge_pct >= 20 THEN '20-30%'
                    WHEN edge_pct >= 10 THEN '10-20%'
                    ELSE '<10%'
                END as edge_bucket,
                COUNT(*) as total,
                SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct
            FROM totals_predictions
            WHERE was_correct IS NOT NULL
            AND prediction IS NOT NULL AND prediction != '' AND prediction != 'N/A'
            GROUP BY edge_bucket
            ORDER BY MIN(edge_pct) DESC
        ''')
        totals_by_edge = [dict(row) for row in c.fetchall()]

        # No-line games: MAE stats for games without a betting line
        c.execute('''
            SELECT 
                COUNT(*) as total,
                ROUND(AVG(ABS(projected_total - actual_total)), 2) as mae,
                ROUND(AVG(projected_total), 1) as avg_projected,
                ROUND(AVG(actual_total), 1) as avg_actual
            FROM totals_predictions
            WHERE actual_total IS NOT NULL
            AND (prediction = 'N/A' OR over_under_line IS NULL)
            AND model_name = 'runs_ensemble'
        ''')
        totals_no_line = dict(c.fetchone())

        # Recent totals predictions
        c.execute('''
            SELECT tp.prediction, tp.over_under_line, tp.projected_total, tp.edge_pct,
                   tp.actual_total, tp.was_correct,
                   ht.name as home_name, at.name as away_name
            FROM totals_predictions tp
            JOIN games g ON tp.game_id = g.id
            JOIN teams ht ON g.home_team_id = ht.id
            JOIN teams at ON g.away_team_id = at.id
            WHERE tp.actual_total IS NOT NULL AND tp.model_name = 'runs_ensemble'
            AND tp.prediction IS NOT NULL AND tp.prediction != ''
            AND tp.over_under_line IS NOT NULL
            ORDER BY tp.predicted_at DESC
            LIMIT 10
        ''')
        recent_totals = [dict(row) for row in c.fetchall()]

        # MAE per totals model
        c.execute('''
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
        totals_mae_by_model = [dict(row) for row in c.fetchall()]

        # O/U split per model
        c.execute('''
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
        totals_ou_split = [dict(row) for row in c.fetchall()]

        # Weekly (7-day) MAE per totals model
        c.execute('''
            SELECT
                tp.model_name,
                COUNT(*) as n,
                ROUND(AVG(ABS(tp.projected_total - (g.home_score + g.away_score))), 2) as mae
            FROM totals_predictions tp
            JOIN games g ON tp.game_id = g.id
            WHERE g.status = 'final'
              AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL
              AND g.date >= date('now', '-7 days')
            GROUP BY tp.model_name
            ORDER BY mae ASC
        ''')
        totals_mae_weekly = {row['model_name']: {'n': row['n'], 'mae': row['mae']} for row in c.fetchall()}

        # Weekly O/U split per model
        c.execute('''
            SELECT
                model_name,
                COUNT(*) as total,
                SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct,
                ROUND(100.0 * SUM(was_correct) / COUNT(*), 1) as hit_rate
            FROM totals_predictions tp
            JOIN games g ON tp.game_id = g.id
            WHERE tp.was_correct IS NOT NULL
              AND tp.prediction IN ('OVER', 'UNDER')
              AND g.date >= date('now', '-7 days')
            GROUP BY model_name
            ORDER BY model_name
        ''')
        totals_weekly_ou = {row['model_name']: {'total': row['total'], 'correct': row['correct'], 'hit_rate': row['hit_rate']} for row in c.fetchall()}
    except Exception:
        totals_overall = {'total': 0, 'correct': 0, 'incorrect': 0}
        totals_by_type = []
        totals_by_edge = []
        totals_no_line = {'total': 0, 'mae': 0, 'avg_projected': 0, 'avg_actual': 0}
        recent_totals = []
        totals_mae_by_model = []
        totals_ou_split = []
        totals_mae_weekly = {}
        totals_weekly_ou = {}

    # Calibration report (if available)
    try:
        c.execute('''
            SELECT model_name, a, b, n_samples, updated_at
            FROM model_calibration
            ORDER BY model_name
        ''')
        calibration_report = [dict(row) for row in c.fetchall()]
    except Exception:
        calibration_report = []

    conn.close()

    # Meta-ensemble feature importance + accuracy
    meta_feature_importance = {}
    meta_accuracy = {}
    meta_training_size = 0
    try:
        from models.meta_ensemble import MetaEnsemble
        _meta = MetaEnsemble()
        meta_feature_importance = _meta.get_feature_importance()
        if meta_feature_importance:
            meta_feature_importance = dict(sorted(meta_feature_importance.items(), key=lambda x: -x[1]))
        # Get actual training data size
        try:
            rows, _ = _meta._extract_training_data()
            meta_training_size = len(rows)
        except Exception:
            pass
    except Exception:
        pass

    # Meta-ensemble accuracy from model_predictions
    try:
        conn_meta = get_connection()
        cm = conn_meta.cursor()
        cm.execute('''
            SELECT model_name,
                COUNT(*) as total,
                SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct,
                ROUND(100.0 * SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) / COUNT(*), 1) as accuracy
            FROM model_predictions
            WHERE was_correct IS NOT NULL
            AND model_name IN ('meta_ensemble', 'ensemble', 'prior', 'neural', 'elo', 'pear', 'quality')
            GROUP BY model_name
            ORDER BY accuracy DESC
        ''')
        meta_accuracy = {row['model_name']: {
            'total': row['total'], 'correct': row['correct'], 'accuracy': row['accuracy']
        } for row in cm.fetchall()}
        conn_meta.close()
    except Exception:
        pass

    # Rolling accuracy data for chart
    conn3 = get_connection()
    c3 = conn3.cursor()
    c3.execute('''
        SELECT mp.model_name, g.date, mp.was_correct
        FROM model_predictions mp
        JOIN games g ON mp.game_id = g.id
        WHERE mp.was_correct IS NOT NULL
        ORDER BY g.date, mp.id
    ''')

    # Build rolling accuracy (window=30) per model
    model_history = defaultdict(list)
    for row in c3.fetchall():
        model_history[row['model_name']].append({
            'date': row['date'],
            'correct': row['was_correct']
        })

    rolling_data = {}
    for model_name, entries in model_history.items():
        points = []
        window = []
        for entry in entries:
            window.append(entry['correct'])
            if len(window) > 30:
                window.pop(0)
            if len(window) >= 3:  # Show after just 3 games (early season friendly)
                points.append({
                    'date': entry['date'],
                    'accuracy': round(sum(window) / len(window) * 100, 1)
                })
        rolling_data[model_name] = points
    conn3.close()

    # Filter calibration: only show if any model has non-default values
    has_active_calibration = any(
        abs(r.get('a', 1.0) - 1.0) > 0.001 or abs(r.get('b', 0.0)) > 0.001
        for r in calibration_report
    )

    return render_template('models.html',
                          model_data=model_data,
                          weights=weights,
                          weight_history=weight_history,
                          prediction_log=prediction_log,
                          runs_weights=runs_weights,
                          totals_overall=totals_overall,
                          totals_by_type=totals_by_type,
                          totals_by_edge=totals_by_edge,
                          totals_no_line=totals_no_line,
                          recent_totals=recent_totals,
                          totals_mae_by_model=totals_mae_by_model,
                          totals_ou_split=totals_ou_split,
                          totals_mae_weekly=totals_mae_weekly,
                          totals_weekly_ou=totals_weekly_ou,
                          rolling_data=rolling_data,
                          calibration_report=calibration_report,
                          has_active_calibration=has_active_calibration,
                          meta_feature_importance=meta_feature_importance,
                          meta_accuracy=meta_accuracy,
                          meta_training_size=meta_training_size)
