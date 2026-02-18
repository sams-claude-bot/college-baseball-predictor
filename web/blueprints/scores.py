"""
Scores Blueprint - Scores, schedule, calendar, and game detail pages
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from flask import Blueprint, render_template, request, redirect

# Add paths for imports
base_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / "scripts"))

from database import get_connection, get_team_record
from models.compare_models import MODELS

from web.helpers import (
    get_all_conferences, get_available_dates,
    get_games_for_date_with_predictions, american_to_implied_prob
)

scores_bp = Blueprint('scores', __name__)


@scores_bp.route('/calendar')
def calendar():
    """Redirect calendar to scores page (merged functionality)"""
    date_str = request.args.get('date', '')
    conference = request.args.get('conference', '')

    # Build redirect URL with same params
    params = []
    if date_str:
        params.append(f'date={date_str}')
    if conference:
        params.append(f'conference={conference}')

    redirect_url = '/scores'
    if params:
        redirect_url += '?' + '&'.join(params)

    return redirect(redirect_url)


@scores_bp.route('/scores')
def scores():
    """Scores & Schedule page - merged scores + calendar with full model predictions"""
    # Default to most recent date with scored games, fallback to today
    conn_def = get_connection()
    latest = conn_def.execute(
        "SELECT date FROM games WHERE home_score IS NOT NULL ORDER BY date DESC LIMIT 1"
    ).fetchone()
    conn_def.close()
    default_date = latest['date'] if latest else datetime.now().strftime('%Y-%m-%d')
    date_str = request.args.get('date', default_date)
    conference = request.args.get('conference', '')

    # Parse date
    try:
        display_date = datetime.strptime(date_str, '%Y-%m-%d')
    except:
        display_date = datetime.now() - timedelta(days=1)
        date_str = display_date.strftime('%Y-%m-%d')

    # Get games for the date (base predictions from ensemble)
    games, correct_count, total_preds = get_games_for_date_with_predictions(date_str)

    # Get betting lines for the date
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        SELECT home_team_id, away_team_id, home_ml, away_ml, over_under
        FROM betting_lines
        WHERE date = ?
    ''', (date_str,))
    betting_lines_map = {}
    for row in c.fetchall():
        key = (row['home_team_id'], row['away_team_id'])
        betting_lines_map[key] = {
            'home_ml': row['home_ml'],
            'away_ml': row['away_ml'],
            'over_under': row['over_under']
        }
    conn.close()

    # Load stored pre-game predictions for neural model
    conn2 = get_connection()
    c2 = conn2.cursor()
    c2.execute('''
        SELECT game_id, predicted_home_prob
        FROM model_predictions
        WHERE model_name = 'neural'
          AND game_id IN (SELECT id FROM games WHERE date = ?)
    ''', (date_str,))
    stored_nn = {row['game_id']: row['predicted_home_prob'] for row in c2.fetchall()}
    conn2.close()

    # Add neural predictions and betting lines — stored predictions only, no live models
    nn_correct = 0
    nn_total = 0

    for game in games:
        # Add betting lines
        key = (game.get('home_team_id'), game.get('away_team_id'))
        if key in betting_lines_map:
            game['home_ml'] = betting_lines_map[key]['home_ml']
            game['away_ml'] = betting_lines_map[key]['away_ml']
            game['over_under'] = betting_lines_map[key]['over_under']

        # Neural model prediction — stored only
        game_id = game.get('id')
        stored_nn_prob = stored_nn.get(game_id)

        if stored_nn_prob is not None:
            game['nn_home_prob'] = stored_nn_prob
            game['nn_winner'] = game['home_team_id'] if stored_nn_prob > 0.5 else game['away_team_id']
            game['nn_confidence'] = max(stored_nn_prob, 1 - stored_nn_prob)
            game['nn_source'] = 'pre-game'

        if game.get('nn_winner') and game['status'] == 'final' and game.get('winner_id'):
            game['nn_correct'] = game['nn_winner'] == game['winner_id']
            nn_total += 1
            if game['nn_correct']:
                nn_correct += 1

    # Filter by conference if specified
    if conference:
        games = [g for g in games if g.get('home_conf') == conference or g.get('away_conf') == conference]
        # Recalculate accuracy for filtered games
        correct_count = sum(1 for g in games if g.get('pred_correct'))
        total_preds = sum(1 for g in games if g.get('pred_correct') is not None)
        nn_correct = sum(1 for g in games if g.get('nn_correct'))
        nn_total = sum(1 for g in games if g.get('nn_correct') is not None)

    # Split into completed, in-progress, and scheduled
    completed_games = [g for g in games if g['status'] == 'final']
    in_progress_games = [g for g in games if g['status'] == 'in-progress']
    scheduled_games = [g for g in games if g['status'] not in ('final', 'in-progress')]

    # Calculate prev/next dates
    prev_date = (display_date - timedelta(days=1)).strftime('%Y-%m-%d')
    next_date = (display_date + timedelta(days=1)).strftime('%Y-%m-%d')

    # Get conferences for filter
    conferences = get_all_conferences()

    # Stats
    total_runs = sum((g.get('home_score') or 0) + (g.get('away_score') or 0) for g in completed_games)
    avg_runs = round(total_runs / len(completed_games), 1) if completed_games else 0
    upsets = sum(1 for g in completed_games if g.get('is_upset'))
    home_wins = sum(1 for g in completed_games if g.get('winner_id') == g.get('home_team_id'))
    home_win_pct = round(home_wins / len(completed_games) * 100) if completed_games else 0

    accuracy_pct = round(correct_count / total_preds * 100, 1) if total_preds > 0 else None
    nn_accuracy_pct = round(nn_correct / nn_total * 100, 1) if nn_total > 0 else None

    # Quick links - get dates with completed games
    available_dates = get_available_dates()

    return render_template('scores.html',
                          completed_games=completed_games,
                          in_progress_games=in_progress_games,
                          scheduled_games=scheduled_games,
                          selected_date=date_str,
                          display_date=display_date,
                          prev_date=prev_date,
                          next_date=next_date,
                          conferences=conferences,
                          selected_conference=conference,
                          available_dates=available_dates,
                          total_completed=len(completed_games),
                          total_scheduled=len(scheduled_games),
                          avg_runs=avg_runs,
                          upsets=upsets,
                          home_win_pct=home_win_pct,
                          accuracy_pct=accuracy_pct,
                          correct_count=correct_count,
                          total_preds=total_preds,
                          nn_accuracy_pct=nn_accuracy_pct,
                          nn_correct=nn_correct,
                          nn_total=nn_total)


@scores_bp.route('/game/<game_id>')
def game_detail(game_id):
    """Individual game detail page with predictions and H2H history"""
    conn = get_connection()
    c = conn.cursor()

    c.execute('''
        SELECT g.*, ht.name as home_name, at.name as away_name,
               ht.current_rank as home_rank, at.current_rank as away_rank,
               ht.conference as home_conf, at.conference as away_conf,
               he.rating as home_elo, ae.rating as away_elo
        FROM games g
        LEFT JOIN teams ht ON g.home_team_id = ht.id
        LEFT JOIN teams at ON g.away_team_id = at.id
        LEFT JOIN elo_ratings he ON g.home_team_id = he.team_id
        LEFT JOIN elo_ratings ae ON g.away_team_id = ae.team_id
        WHERE g.id = ?
    ''', (game_id,))
    game = c.fetchone()
    if not game:
        conn.close()
        return render_template('404.html', message="Game not found"), 404
    game = dict(game)

    home_id = game['home_team_id']
    away_id = game['away_team_id']
    home_record = get_team_record(home_id)
    away_record = get_team_record(away_id)

    home = {'id': home_id, 'name': game['home_name'], 'rank': game['home_rank'],
            'elo': game['home_elo'] or 1500, 'record': f"{home_record['wins']}-{home_record['losses']}"}
    away = {'id': away_id, 'name': game['away_name'], 'rank': game['away_rank'],
            'elo': game['away_elo'] or 1500, 'record': f"{away_record['wins']}-{away_record['losses']}"}

    # Predictions from all models
    prediction = None
    models_list = []
    try:
        for name, model in MODELS.items():
            if name in ('nn_totals', 'nn_spread', 'nn_dow_totals'):
                continue
            try:
                pred = model.predict_game(home_id, away_id)
                entry = {
                    'name': name,
                    'home_prob': pred['home_win_probability'],
                    'away_prob': pred['away_win_probability'],
                    'home_runs': pred['projected_home_runs'],
                    'away_runs': pred['projected_away_runs']
                }
                models_list.append(entry)
                if name == 'ensemble':
                    prediction = {
                        'home_win_prob': pred['home_win_probability'],
                        'away_win_prob': pred['away_win_probability'],
                        'projected_home_runs': pred['projected_home_runs'],
                        'projected_away_runs': pred['projected_away_runs'],
                        'projected_total': pred.get('projected_total',
                            pred['projected_home_runs'] + pred['projected_away_runs'])
                    }
            except:
                pass
    except:
        pass

    # Sort: ensemble first, then by home_prob desc
    models_list.sort(key=lambda x: (0 if x['name'] == 'ensemble' else 1, -x['home_prob']))

    # DK lines
    c.execute('''
        SELECT * FROM betting_lines 
        WHERE home_team_id = ? AND away_team_id = ?
        ORDER BY captured_at DESC LIMIT 1
    ''', (home_id, away_id))
    line_row = c.fetchone()
    betting_line = None
    if line_row:
        bl = dict(line_row)
        betting_line = bl
        if prediction and bl.get('home_ml') and bl.get('away_ml'):
            dk_home = american_to_implied_prob(bl['home_ml'])
            dk_away = american_to_implied_prob(bl['away_ml'])
            total = dk_home + dk_away
            betting_line['home_edge'] = (prediction['home_win_prob'] - dk_home/total) * 100
            betting_line['away_edge'] = (prediction['away_win_prob'] - dk_away/total) * 100

    # Head-to-head history
    c.execute('''
        SELECT date, home_score, away_score, winner_id, home_team_id
        FROM games 
        WHERE ((home_team_id = ? AND away_team_id = ?) OR (home_team_id = ? AND away_team_id = ?))
        AND status = 'final'
        ORDER BY date DESC
        LIMIT 20
    ''', (home_id, away_id, away_id, home_id))

    h2h_games = []
    h2h_record = {'home_wins': 0, 'away_wins': 0}
    for row in c.fetchall():
        r = dict(row)
        # Normalize so away/home match the current game
        if r['home_team_id'] == home_id:
            winner = 'home' if r['winner_id'] == home_id else 'away'
            h2h_games.append({'date': r['date'], 'home_score': r['home_score'],
                            'away_score': r['away_score'], 'winner': winner})
        else:
            winner = 'home' if r['winner_id'] == away_id else 'away'
            # Flip scores since home/away are reversed
            h2h_games.append({'date': r['date'], 'home_score': r['away_score'],
                            'away_score': r['home_score'], 'winner': 'away' if winner == 'home' else 'home'})
        if winner == 'home':
            h2h_record['home_wins'] += 1
        else:
            h2h_record['away_wins'] += 1

    conn.close()

    return render_template('game.html',
        game=game, home=home, away=away,
        prediction=prediction, models=models_list,
        betting_line=betting_line,
        h2h_games=h2h_games, h2h_record=h2h_record)
