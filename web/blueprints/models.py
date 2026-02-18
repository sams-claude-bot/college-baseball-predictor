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
    
    rolling_data = {}
    for model_name, entries in model_history.items():
        points = []
        window = []
        cumulative_correct = 0
        cumulative_total = 0
        for entry in entries:
            window.append(entry['correct'])
            cumulative_correct += entry['correct']
            cumulative_total += 1
            if len(window) > 30:
                window.pop(0)
            # Always show a point (even with 1 game)
            points.append({
                'date': entry['date'],
                'rolling': round(sum(window) / len(window) * 100, 1),
                'cumulative': round(cumulative_correct / cumulative_total * 100, 1)
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
            GROUP BY edge_bucket
            ORDER BY MIN(edge_pct) DESC
        ''')
        totals_by_edge = [dict(row) for row in c.fetchall()]

        # Recent totals predictions
        c.execute('''
            SELECT tp.prediction, tp.over_under_line, tp.projected_total, tp.edge_pct,
                   tp.actual_total, tp.was_correct,
                   ht.name as home_name, at.name as away_name
            FROM totals_predictions tp
            JOIN games g ON tp.game_id = g.id
            JOIN teams ht ON g.home_team_id = ht.id
            JOIN teams at ON g.away_team_id = at.id
            WHERE tp.actual_total IS NOT NULL
            ORDER BY tp.predicted_at DESC
            LIMIT 10
        ''')
        recent_totals = [dict(row) for row in c.fetchall()]
    except:
        totals_overall = {'total': 0, 'correct': 0, 'incorrect': 0}
        totals_by_type = []
        totals_by_edge = []
        recent_totals = []

    conn.close()

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

    return render_template('models.html',
                          model_data=model_data,
                          weights=weights,
                          weight_history=weight_history,
                          prediction_log=prediction_log,
                          runs_weights=runs_weights,
                          totals_overall=totals_overall,
                          totals_by_type=totals_by_type,
                          totals_by_edge=totals_by_edge,
                          recent_totals=recent_totals,
                          rolling_data=rolling_data)
