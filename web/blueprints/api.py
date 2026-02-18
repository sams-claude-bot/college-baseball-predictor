"""
API Blueprint - All API endpoints
"""

import sys
from pathlib import Path
from datetime import datetime
from flask import Blueprint, request, jsonify

# Add paths for imports
base_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / "scripts"))

from database import get_connection, get_team_record, get_team_runs
from models.compare_models import MODELS, normalize_team_id

from web.helpers import (
    get_all_teams, get_betting_games,
    american_to_implied_prob, compute_model_agreement
)

api_bp = Blueprint('api', __name__)


@api_bp.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    data = request.get_json()
    home_team = data.get('home_team')
    away_team = data.get('away_team')
    neutral_site = data.get('neutral_site', False)

    if not home_team or not away_team:
        return jsonify({'error': 'Both teams required'}), 400

    home_id = normalize_team_id(home_team)
    away_id = normalize_team_id(away_team)

    results = {}

    # Get predictions from all models
    for name, model in MODELS.items():
        try:
            pred = model.predict_game(home_id, away_id, neutral_site)
            results[name] = {
                'home_win_prob': pred['home_win_probability'],
                'away_win_prob': pred['away_win_probability'],
                'projected_home_runs': pred['projected_home_runs'],
                'projected_away_runs': pred['projected_away_runs'],
                'projected_total': pred.get('projected_total',
                                           pred['projected_home_runs'] + pred['projected_away_runs'])
            }
            if 'run_line' in pred:
                results[name]['run_line'] = pred['run_line']
            # Include momentum for ensemble
            if 'momentum' in pred:
                results[name]['momentum'] = pred['momentum']
            # Include component predictions and weights for ensemble
            if 'component_predictions' in pred:
                results[name]['component_predictions'] = pred['component_predictions']
            if 'weights' in pred:
                results[name]['weights'] = pred['weights']
            # Include model inputs for explainability
            if 'inputs' in pred:
                results[name]['inputs'] = pred['inputs']
        except Exception as e:
            results[name] = {'error': str(e)}

    # Series prediction
    try:
        ensemble = MODELS['ensemble']
        series = ensemble.predict_series(home_id, away_id, games=3, neutral_site=neutral_site)
        results['series'] = series
    except:
        results['series'] = None

    # Check for DK lines
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        SELECT * FROM betting_lines 
        WHERE home_team_id = ? AND away_team_id = ?
        ORDER BY captured_at DESC LIMIT 1
    ''', (home_id, away_id))
    line = c.fetchone()
    conn.close()

    if line:
        line_dict = dict(line)
        results['betting_line'] = {
            'home_ml': line_dict['home_ml'],
            'away_ml': line_dict['away_ml'],
            'over_under': line_dict['over_under']
        }

        # EV analysis
        if line_dict['home_ml'] and line_dict['away_ml']:
            dk_home = american_to_implied_prob(line_dict['home_ml'])
            dk_away = american_to_implied_prob(line_dict['away_ml'])
            total = dk_home + dk_away

            results['betting_line']['dk_home_prob'] = dk_home / total
            results['betting_line']['dk_away_prob'] = dk_away / total

            model_home = results['ensemble']['home_win_prob']
            results['betting_line']['home_edge'] = (model_home - dk_home/total) * 100
            results['betting_line']['away_edge'] = ((1-model_home) - dk_away/total) * 100

    # Add team context for explainability
    try:
        home_record = get_team_record(home_id)
        away_record = get_team_record(away_id)
        home_runs_data = get_team_runs(home_id)
        away_runs_data = get_team_runs(away_id)

        conn = get_connection()
        c = conn.cursor()
        c.execute('SELECT rating FROM elo_ratings WHERE team_id = ?', (home_id,))
        home_elo_row = c.fetchone()
        c.execute('SELECT rating FROM elo_ratings WHERE team_id = ?', (away_id,))
        away_elo_row = c.fetchone()
        c.execute('SELECT current_rank FROM teams WHERE id = ?', (home_id,))
        home_rank_row = c.fetchone()
        c.execute('SELECT current_rank FROM teams WHERE id = ?', (away_id,))
        away_rank_row = c.fetchone()
        conn.close()

        results['team_context'] = {
            'home': {
                'record': f"{home_record['wins']}-{home_record['losses']}",
                'wins': home_record['wins'],
                'losses': home_record['losses'],
                'runs_scored_avg': round(home_runs_data['runs_scored'] / home_runs_data['games'], 2) if home_runs_data.get('games') else None,
                'runs_allowed_avg': round(home_runs_data['runs_allowed'] / home_runs_data['games'], 2) if home_runs_data.get('games') else None,
                'elo': home_elo_row[0] if home_elo_row else None,
                'rank': home_rank_row[0] if home_rank_row and home_rank_row[0] else None
            },
            'away': {
                'record': f"{away_record['wins']}-{away_record['losses']}",
                'wins': away_record['wins'],
                'losses': away_record['losses'],
                'runs_scored_avg': round(away_runs_data['runs_scored'] / away_runs_data['games'], 2) if away_runs_data.get('games') else None,
                'runs_allowed_avg': round(away_runs_data['runs_allowed'] / away_runs_data['games'], 2) if away_runs_data.get('games') else None,
                'elo': away_elo_row[0] if away_elo_row else None,
                'rank': away_rank_row[0] if away_rank_row and away_rank_row[0] else None
            }
        }
    except Exception as e:
        results['team_context'] = {'error': str(e)}

    # Head-to-head history
    try:
        conn2 = get_connection()
        c2 = conn2.cursor()
        c2.execute('''
            SELECT date, home_team_id, home_score, away_score, winner_id
            FROM games 
            WHERE ((home_team_id = ? AND away_team_id = ?) OR (home_team_id = ? AND away_team_id = ?))
            AND status = 'final'
            ORDER BY date DESC LIMIT 10
        ''', (home_id, away_id, away_id, home_id))
        h2h = []
        for row in c2.fetchall():
            r = dict(row)
            if r['home_team_id'] == home_id:
                h2h.append({'date': r['date'], 'home_score': r['home_score'],
                           'away_score': r['away_score'], 'home_won': r['winner_id'] == home_id})
            else:
                h2h.append({'date': r['date'], 'home_score': r['away_score'],
                           'away_score': r['home_score'], 'home_won': r['winner_id'] == away_id})
        conn2.close()
        results['h2h'] = h2h
    except:
        results['h2h'] = []

    return jsonify(results)


@api_bp.route('/api/runs', methods=['POST'])
def api_runs():
    """API endpoint for detailed runs ensemble analysis"""
    data = request.get_json()
    home_team = data.get('home_team')
    away_team = data.get('away_team')
    total_line = data.get('total_line')

    if not home_team or not away_team:
        return jsonify({'error': 'Both teams required'}), 400

    home_id = normalize_team_id(home_team)
    away_id = normalize_team_id(away_team)

    try:
        import models.runs_ensemble as runs_ens
        result = runs_ens.predict(home_id, away_id, total_line=total_line)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/api/teams')
def api_teams():
    """API endpoint for team list"""
    teams = get_all_teams()
    return jsonify(teams)


@api_bp.route('/api/best-bets')
def api_best_bets():
    """Return best bets for a date with adjusted edge calculation.

    v2 Logic:
    - Underdog edges discounted 50% (market usually right)
    - Consensus bonus: +1% per model above 5 (max +5%)
    - Spreads DISABLED (0/5 historical)
    - Stricter thresholds: 8% favorites, 15% underdogs

    Query param: ?date=YYYY-MM-DD (defaults to today)
    """
    date_str = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    games = get_betting_games(date_str)

    # === v2 THRESHOLDS ===
    ML_EDGE_FAVORITE = 8.0
    ML_EDGE_UNDERDOG = 15.0
    UNDERDOG_DISCOUNT = 0.5
    CONSENSUS_BONUS_PER_MODEL = 1.0  # +1% per model above 5
    ML_MAX_FAVORITE = -200
    ML_MAX_FAVORITE_CONSENSUS = -300
    ML_MIN_UNDERDOG = 250
    SPREADS_ENABLED = False

    def calc_adjusted_edge(raw_edge, ml, models_agree=5):
        """Calculate adjusted edge with underdog discount and consensus bonus."""
        adj = raw_edge
        if ml and ml > 0:  # Underdog
            adj = raw_edge * UNDERDOG_DISCOUNT
        bonus = max(0, (models_agree - 5)) * CONSENSUS_BONUS_PER_MODEL
        return adj + bonus

    # Filter to only games with model predictions
    games = [g for g in games if g.get('best_pick')]

    # Build consensus lookup
    consensus_lookup = {}
    for g in games:
        if g.get('model_agreement') and g['model_agreement'].get('count', 0) >= 7:
            consensus_lookup[g['game_id']] = g['model_agreement']['count']

    # Best moneyline bets with adjusted edge
    ml_candidates = []
    for g in games:
        raw_edge = g.get('best_edge', 0)
        if g['best_pick'] == 'home':
            ml = g.get('home_ml')
        else:
            ml = g.get('away_ml')

        if ml is None:
            continue

        is_underdog = ml > 0
        threshold = ML_EDGE_UNDERDOG if is_underdog else ML_EDGE_FAVORITE

        if raw_edge < threshold:
            continue
        if ml < ML_MAX_FAVORITE:
            continue
        if ml > ML_MIN_UNDERDOG:
            continue

        models = consensus_lookup.get(g['game_id'], 5)
        adj_edge = calc_adjusted_edge(raw_edge, ml, models)

        ml_candidates.append({**g, 'adjusted_edge': adj_edge, 'models_agree': models})

    ml_candidates.sort(key=lambda x: x.get('adjusted_edge', 0), reverse=True)
    best_ml = ml_candidates[:6]

    ml_bets = []
    for g in best_ml:
        if g['best_pick'] == 'home':
            pick_id = g['home_team_id']
            pick_name = g['home_team_name']
            opp_name = g['away_team_name']
            ml = g['home_ml']
            prob = g['model_home_prob']
            dk_imp = g['dk_home_fair']
            is_home = 1
        else:
            pick_id = g['away_team_id']
            pick_name = g['away_team_name']
            opp_name = g['home_team_name']
            ml = g['away_ml']
            prob = g['model_away_prob']
            dk_imp = g['dk_away_fair']
            is_home = 0
        ml_bets.append({
            'game_id': g['game_id'], 'date': g['date'],
            'pick_team_id': pick_id, 'pick_team_name': pick_name,
            'opponent_name': opp_name, 'is_home': is_home,
            'moneyline': ml, 'model_prob': round(prob, 4),
            'dk_implied': round(dk_imp, 4),
            'edge': round(g['best_edge'], 2),
            'adjusted_edge': round(g.get('adjusted_edge', g['best_edge']), 2),
            'models_agree': g.get('models_agree', 5),
            'is_underdog': ml > 0 if ml else False
        })

    # Best totals (3+ runs edge, top 6)
    TOTALS_EDGE_RUNS = 3.0
    totals_candidates = [g for g in games
                        if abs(g.get('total_diff', 0)) >= TOTALS_EDGE_RUNS
                        and g.get('over_under')]
    totals_candidates.sort(key=lambda x: abs(x.get('total_diff', 0)), reverse=True)
    best_totals = totals_candidates[:6]

    totals_bets = []
    for g in best_totals:
        pick = g.get('total_lean', 'UNDER')
        odds = g.get('under_odds', -110) if pick == 'UNDER' else g.get('over_odds', -110)
        totals_bets.append({
            'game_id': g['game_id'], 'date': g['date'],
            'pick': pick, 'line': g['over_under'],
            'odds': odds or -110,
            'model_projection': round(g.get('projected_total', 0), 2),
            'edge': round(abs(g.get('total_diff', 0)), 2)
        })

    # Best spreads (top 6 by NN margin diff from line)
    spread_candidates = []
    for g in games:
        if g.get('home_spread') and g.get('nn_margin') is not None:
            margin = g['nn_margin']
            spread = g['home_spread']
            diff = abs(margin - spread)
            if diff >= 1.0:  # At least 1 run edge
                if margin > spread:
                    pick = g['home_team_name']
                    line = spread
                    odds = g.get('home_spread_odds', -110)
                else:
                    pick = g['away_team_name']
                    line = g.get('away_spread', -spread)
                    odds = g.get('away_spread_odds', -110)
                spread_candidates.append({
                    'game_id': g['game_id'], 'date': g['date'],
                    'pick': pick, 'line': line,
                    'odds': odds or -110,
                    'model_projection': round(margin, 2),
                    'edge': round(diff, 2)
                })
    spread_candidates.sort(key=lambda x: x['edge'], reverse=True)
    best_spreads = spread_candidates[:6] if SPREADS_ENABLED else []

    # Confident bets (7/10+ models agree, sorted by confidence score)
    confident_candidates = [g for g in games
                           if g.get('model_agreement')
                           and g['model_agreement']['count'] >= 7]
    confident_candidates.sort(key=lambda x: x['model_agreement']['confidence'], reverse=True)
    best_confident = confident_candidates[:6]

    confident_bets = []
    for g in best_confident:
        agreement = g['model_agreement']
        if agreement['pick'] == 'home':
            pick_id = g['home_team_id']
            pick_name = g['home_team_name']
            opp_name = g['away_team_name']
            ml = g.get('home_ml')
            is_home = 1
        else:
            pick_id = g['away_team_id']
            pick_name = g['away_team_name']
            opp_name = g['home_team_name']
            ml = g.get('away_ml')
            is_home = 0
        confident_bets.append({
            'game_id': g['game_id'], 'date': g['date'],
            'pick_team_id': pick_id, 'pick_team_name': pick_name,
            'opponent_name': opp_name, 'is_home': is_home,
            'moneyline': ml,
            'models_agree': agreement['count'],
            'models_total': agreement['total'],
            'avg_prob': round(agreement['avg_prob'], 4),
            'confidence': round(agreement['confidence'], 4),
            'models_for': agreement['models_for'],
            'models_against': agreement['models_against']
        })

    return jsonify({
        'date': date_str,
        'version': 2,
        'thresholds': {
            'ml_favorite': ML_EDGE_FAVORITE,
            'ml_underdog': ML_EDGE_UNDERDOG,
            'totals_runs': TOTALS_EDGE_RUNS,
            'underdog_discount': UNDERDOG_DISCOUNT,
            'spreads_enabled': SPREADS_ENABLED
        },
        'confident_bets': confident_bets,
        'moneylines': ml_bets,
        'totals': totals_bets,
        'spreads': best_spreads,
        'spreads_disabled_reason': 'Model not calibrated (0/5 historical)' if not SPREADS_ENABLED else None
    })
