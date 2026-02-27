"""
Scores Blueprint - Scores, schedule, calendar, and game detail pages
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from flask import Blueprint, render_template, request, redirect, current_app

# Add paths for imports
base_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / "scripts"))

from database import get_connection, get_team_record
from web.helpers import (
    get_all_conferences, get_available_dates,
    get_games_for_date_with_predictions, american_to_implied_prob
)
from web.services.game_quality import compute_gqi, gqi_label, gqi_color
from web.services.win_quality import get_game_resume_impact

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
    cache = current_app.cache

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

    cache_key = f'scores:{date_str}:{conference}'
    cached = cache.get(cache_key)
    if cached:
        return cached

    # Get games for the date (base predictions from ensemble)
    games, correct_count, total_preds = get_games_for_date_with_predictions(date_str)

    # Get betting lines for the date
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        SELECT home_team_id, away_team_id, home_ml, away_ml, over_under
        FROM betting_lines
        WHERE date = ? AND book = 'draftkings'
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

    # Load stored runs ensemble totals
    c2 = conn2.cursor()
    c2.execute('''
        SELECT game_id, projected_total
        FROM totals_predictions
        WHERE model_name = 'runs_ensemble'
          AND game_id IN (SELECT id FROM games WHERE date = ?)
    ''', (date_str,))
    stored_totals = {row['game_id']: row['projected_total'] for row in c2.fetchall()}
    conn2.close()

    # Add neural predictions, totals, and betting lines — stored predictions only, no live models
    nn_correct = 0
    nn_total = 0

    for game in games:
        # Add betting lines
        key = (game.get('home_team_id'), game.get('away_team_id'))
        if key in betting_lines_map:
            game['home_ml'] = betting_lines_map[key]['home_ml']
            game['away_ml'] = betting_lines_map[key]['away_ml']
            game['over_under'] = betting_lines_map[key]['over_under']

        # Projected total from runs ensemble (stored first, live fallback)
        game_id = game.get('id')
        if game_id in stored_totals:
            game['nn_projected_total'] = stored_totals[game_id]
        else:
            try:
                from models.runs_ensemble import predict as runs_predict
                rp = runs_predict(game['home_team_id'], game['away_team_id'], game_id=game_id)
                game['nn_projected_total'] = rp.get('projected_total')
            except Exception:
                pass

        # Neural model prediction — stored only
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

    # Compute GQI for each game
    for game in games:
        home_elo = game.get('home_elo') or 1500
        away_elo = game.get('away_elo') or 1500
        game['gqi'] = compute_gqi(home_elo, away_elo, game.get('home_rank'), game.get('away_rank'))
        game['gqi_label'] = gqi_label(game['gqi'])
        game['gqi_color'] = gqi_color(game['gqi'])

    # Filter by conference if specified
    if conference:
        games = [g for g in games if g.get('home_conf') == conference or g.get('away_conf') == conference]
        # Recalculate accuracy for filtered games
        correct_count = sum(1 for g in games if g.get('pred_correct'))
        total_preds = sum(1 for g in games if g.get('pred_correct') is not None)
        nn_correct = sum(1 for g in games if g.get('nn_correct'))
        nn_total = sum(1 for g in games if g.get('nn_correct') is not None)

    # Split into completed, in-progress, scheduled, and postponed
    completed_games = [g for g in games if g['status'] == 'final']
    all_in_progress = [g for g in games if g['status'] == 'in-progress']
    postponed_games = [g for g in games if g['status'] in ('postponed', 'canceled')]
    scheduled_games = [g for g in games if g['status'] not in ('final', 'in-progress', 'postponed', 'canceled')]

    # Split live games: those with StatBroadcast coverage vs ESPN-only
    live_with_stats = []
    live_espn_only = []
    for g in all_in_progress:
        sit = g.get('situation')
        if sit and sit.get('sb_outs') is not None or sit and sit.get('sb_pitcher'):
            live_with_stats.append(g)
        else:
            live_espn_only.append(g)
    in_progress_games = all_in_progress  # keep for backward compat

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

    # Totals (O/U) accuracy for the date
    totals_correct = 0
    totals_total = 0
    for game in completed_games:
        game_id = game.get('id')
        proj = stored_totals.get(game_id)
        ou_line = game.get('over_under')
        if proj and ou_line and game.get('home_score') is not None:
            actual = game['home_score'] + game['away_score']
            if actual != ou_line:  # skip pushes
                was_over = actual > ou_line
                predicted_over = proj > ou_line
                if was_over == predicted_over:
                    totals_correct += 1
                totals_total += 1
    totals_accuracy_pct = round(totals_correct / totals_total * 100, 1) if totals_total > 0 else None

    # Quick links - get dates with completed games
    available_dates = get_available_dates()

    result = render_template('scores.html',
                          completed_games=completed_games,
                          in_progress_games=in_progress_games,
                          live_with_stats=live_with_stats,
                          live_espn_only=live_espn_only,
                          scheduled_games=scheduled_games,
                          postponed_games=postponed_games,
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
                          nn_total=nn_total,
                          totals_accuracy_pct=totals_accuracy_pct,
                          totals_correct=totals_correct,
                          totals_total=totals_total)
    cache.set(cache_key, result, timeout=600)
    return result


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

    # Parse JSON fields for live game state
    import json as _json
    if game.get('situation_json'):
        try:
            game['situation'] = _json.loads(game['situation_json'])
        except:
            game['situation'] = None
    else:
        game['situation'] = None
    if game.get('linescore_json'):
        try:
            game['linescore'] = _json.loads(game['linescore_json'])
        except:
            game['linescore'] = None
    else:
        game['linescore'] = None

    home_id = game['home_team_id']
    away_id = game['away_team_id']
    home_record = get_team_record(home_id)
    away_record = get_team_record(away_id)

    home = {'id': home_id, 'name': game['home_name'], 'rank': game['home_rank'],
            'elo': game['home_elo'] or 1500, 'record': f"{home_record['wins']}-{home_record['losses']}"}
    away = {'id': away_id, 'name': game['away_name'], 'rank': game['away_rank'],
            'elo': game['away_elo'] or 1500, 'record': f"{away_record['wins']}-{away_record['losses']}"}

    # Predictions from stored model_predictions (no live model execution)
    prediction = None
    models_list = []
    c.execute('''
        SELECT model_name, predicted_home_prob, predicted_home_runs, predicted_away_runs
        FROM model_predictions
        WHERE game_id = ?
    ''', (game_id,))
    for row in c.fetchall():
        r = dict(row)
        home_prob = r['predicted_home_prob'] or 0.5
        entry = {
            'name': r['model_name'],
            'home_prob': home_prob,
            'away_prob': 1 - home_prob,
            'home_runs': r['predicted_home_runs'],
            'away_runs': r['predicted_away_runs']
        }
        models_list.append(entry)
        # Prefer meta_ensemble as primary; fall back to ensemble
        if r['model_name'] == 'meta_ensemble':
            prediction = {
                'home_win_prob': home_prob,
                'away_win_prob': 1 - home_prob,
                'projected_home_runs': r['predicted_home_runs'],
                'projected_away_runs': r['predicted_away_runs'],
                'projected_total': (r['predicted_home_runs'] + r['predicted_away_runs']) if (r['predicted_home_runs'] is not None and r['predicted_away_runs'] is not None) else None
            }
        elif r['model_name'] == 'ensemble' and prediction is None:
            home_runs = r['predicted_home_runs']
            away_runs = r['predicted_away_runs']
            prediction = {
                'home_win_prob': home_prob,
                'away_win_prob': 1 - home_prob,
                'projected_home_runs': home_runs,
                'projected_away_runs': away_runs,
                'projected_total': (home_runs + away_runs) if (home_runs is not None and away_runs is not None) else None
            }

    # Sort: meta_ensemble first, then ensemble, then by home_prob desc
    models_list.sort(key=lambda x: (0 if x['name'] == 'meta_ensemble' else 1 if x['name'] == 'ensemble' else 2, -x['home_prob']))

    # DK lines
    c.execute('''
        SELECT * FROM betting_lines 
        WHERE home_team_id = ? AND away_team_id = ? AND book = 'draftkings'
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

    # Totals model breakdown (runs predictions from totals_predictions)
    totals_models = []
    try:
        c.execute('''
            SELECT model_name, prediction, over_under_line, projected_total, actual_total, was_correct
            FROM totals_predictions
            WHERE game_id = ?
            ORDER BY CASE model_name WHEN 'runs_ensemble' THEN 0 ELSE 1 END, model_name
        ''', (game_id,))
        totals_models = [dict(row) for row in c.fetchall()]
    except:
        pass

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

    # Series probabilities for upcoming games
    series_probs = None
    if prediction and game.get('status') != 'final':
        from web.services.series_probability import compute_series_probs
        series_probs = compute_series_probs(prediction['home_win_prob'])

    # Cross-matchup: offense vs pitching percentiles
    cross_matchup = None
    try:
        from web.services.cross_matchup import build_cross_matchup
        cross_matchup = build_cross_matchup(home_id, away_id)
    except Exception:
        pass

    # Game Quality Index
    gqi_val = compute_gqi(home['elo'], away['elo'], home.get('rank'), away.get('rank'))

    # Resume Impact (win quality / loss damage)
    resume_impact = None
    try:
        resume_impact = get_game_resume_impact(home_id, away_id)
    except Exception:
        pass

    return render_template('game.html',
        game=game, home=home, away=away,
        prediction=prediction, models=models_list,
        betting_line=betting_line,
        h2h_games=h2h_games, h2h_record=h2h_record,
        totals_models=totals_models,
        series_probs=series_probs,
        cross_matchup=cross_matchup,
        resume_impact=resume_impact,
        gqi=gqi_val, gqi_label=gqi_label(gqi_val), gqi_color=gqi_color(gqi_val))
