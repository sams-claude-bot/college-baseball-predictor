#!/usr/bin/env python3
"""
College Baseball Predictor - Web Dashboard

Flask web interface for exploring predictions, teams, rankings, and betting data.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

# Add paths for imports
base_dir = Path(__file__).parent.parent
sys.path.insert(0, str(base_dir / "scripts"))
sys.path.insert(0, str(base_dir / "models"))

from flask import Flask, render_template, request, jsonify

from database import (
    get_connection, get_team_record, get_team_runs, 
    get_recent_games, get_upcoming_games, get_current_top_25
)
from compare_models import MODELS, normalize_team_id
from betting_lines import american_to_implied_prob

app = Flask(__name__)

# ============================================
# Helper Functions
# ============================================

def get_all_teams():
    """Get all teams with their records and ratings"""
    conn = get_connection()
    c = conn.cursor()
    
    c.execute('''
        SELECT t.id, t.name, t.nickname, t.conference, t.current_rank,
               e.rating as elo_rating
        FROM teams t
        LEFT JOIN elo_ratings e ON t.id = e.team_id
        ORDER BY t.name
    ''')
    
    teams = []
    for row in c.fetchall():
        team = dict(row)
        record = get_team_record(team['id'])
        team['wins'] = record['wins']
        team['losses'] = record['losses']
        team['win_pct'] = record['pct']
        teams.append(team)
    
    conn.close()
    return teams

def get_team_detail(team_id):
    """Get detailed info for a single team"""
    conn = get_connection()
    c = conn.cursor()
    
    c.execute('''
        SELECT t.*, e.rating as elo_rating
        FROM teams t
        LEFT JOIN elo_ratings e ON t.id = e.team_id
        WHERE t.id = ?
    ''', (team_id,))
    
    row = c.fetchone()
    if not row:
        conn.close()
        return None
    
    team = dict(row)
    
    # Get records
    overall = get_team_record(team_id)
    conf = get_team_record(team_id, conference_only=True)
    runs = get_team_runs(team_id)
    
    team['overall_record'] = overall
    team['conference_record'] = conf
    team['runs'] = runs
    
    # Get schedule/results
    c.execute('''
        SELECT g.*, 
               ht.name as home_team_name,
               at.name as away_team_name
        FROM games g
        LEFT JOIN teams ht ON g.home_team_id = ht.id
        LEFT JOIN teams at ON g.away_team_id = at.id
        WHERE g.home_team_id = ? OR g.away_team_id = ?
        ORDER BY g.date, g.time
    ''', (team_id, team_id))
    team['schedule'] = [dict(r) for r in c.fetchall()]
    
    # Get roster
    c.execute('''
        SELECT * FROM players
        WHERE team_id = ?
        ORDER BY position, name
    ''', (team_id,))
    team['roster'] = [dict(r) for r in c.fetchall()]
    
    # Get Elo history (if exists)
    c.execute('''
        SELECT rating, games_played, updated_at
        FROM elo_ratings
        WHERE team_id = ?
    ''', (team_id,))
    elo_row = c.fetchone()
    if elo_row:
        team['elo_history'] = [dict(elo_row)]
    else:
        team['elo_history'] = []
    
    conn.close()
    return team

def get_todays_games():
    """Get games scheduled for today"""
    today = datetime.now().strftime('%Y-%m-%d')
    
    conn = get_connection()
    c = conn.cursor()
    
    c.execute('''
        SELECT g.*, 
               ht.name as home_team_name, ht.current_rank as home_rank,
               at.name as away_team_name, at.current_rank as away_rank,
               he.rating as home_elo, ae.rating as away_elo,
               b.home_ml, b.away_ml, b.over_under
        FROM games g
        LEFT JOIN teams ht ON g.home_team_id = ht.id
        LEFT JOIN teams at ON g.away_team_id = at.id
        LEFT JOIN elo_ratings he ON g.home_team_id = he.team_id
        LEFT JOIN elo_ratings ae ON g.away_team_id = ae.team_id
        LEFT JOIN betting_lines b ON g.id = b.game_id
        WHERE g.date = ?
        ORDER BY g.time
    ''', (today,))
    
    games = [dict(row) for row in c.fetchall()]
    conn.close()
    
    # Add predictions to each game
    for game in games:
        try:
            ensemble = MODELS['ensemble']
            pred = ensemble.predict_game(game['home_team_id'], game['away_team_id'])
            game['prediction'] = pred
        except Exception as e:
            game['prediction'] = None
            game['prediction_error'] = str(e)
    
    return games

def get_value_picks(limit=5):
    """Get top value picks based on model vs DraftKings edge"""
    conn = get_connection()
    c = conn.cursor()
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    c.execute('''
        SELECT b.*, 
               ht.name as home_team_name,
               at.name as away_team_name
        FROM betting_lines b
        LEFT JOIN teams ht ON b.home_team_id = ht.id
        LEFT JOIN teams at ON b.away_team_id = at.id
        WHERE b.date >= ?
        ORDER BY b.date
    ''', (today,))
    
    lines = [dict(row) for row in c.fetchall()]
    conn.close()
    
    picks = []
    for line in lines:
        if not line['home_ml'] or not line['away_ml']:
            continue
            
        try:
            # Get DK implied probability
            dk_home_prob = american_to_implied_prob(line['home_ml'])
            dk_away_prob = american_to_implied_prob(line['away_ml'])
            total_prob = dk_home_prob + dk_away_prob
            dk_home_fair = dk_home_prob / total_prob
            
            # Get model prediction
            ensemble = MODELS['ensemble']
            pred = ensemble.predict_game(line['home_team_id'], line['away_team_id'])
            model_home_prob = pred['home_win_probability']
            
            # Calculate edge
            home_edge = (model_home_prob - dk_home_fair) * 100
            away_edge = ((1 - model_home_prob) - (1 - dk_home_fair)) * 100
            
            best_edge = home_edge if home_edge > abs(away_edge) else away_edge
            best_pick = line['home_team_name'] if home_edge > 0 else line['away_team_name']
            best_ml = line['home_ml'] if home_edge > 0 else line['away_ml']
            
            picks.append({
                'date': line['date'],
                'game': f"{line['away_team_name']} @ {line['home_team_name']}",
                'pick': best_pick,
                'edge': abs(best_edge),
                'moneyline': best_ml,
                'model_prob': model_home_prob if home_edge > 0 else 1 - model_home_prob,
                'dk_implied': dk_home_fair if home_edge > 0 else 1 - dk_home_fair,
                'home_team_id': line['home_team_id'],
                'away_team_id': line['away_team_id']
            })
        except Exception:
            continue
    
    # Sort by edge and return top picks
    picks.sort(key=lambda x: x['edge'], reverse=True)
    return picks[:limit]

def get_quick_stats():
    """Get dashboard quick statistics"""
    conn = get_connection()
    c = conn.cursor()
    
    c.execute("SELECT COUNT(*) FROM teams")
    total_teams = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM games WHERE status='final'")
    games_played = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM games WHERE status='scheduled'")
    games_scheduled = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM players")
    players_tracked = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM betting_lines")
    betting_lines = c.fetchone()[0]
    
    conn.close()
    
    return {
        'total_teams': total_teams,
        'games_played': games_played,
        'games_scheduled': games_scheduled,
        'players_tracked': players_tracked,
        'betting_lines': betting_lines
    }

def get_betting_games():
    """Get all games with betting lines"""
    conn = get_connection()
    c = conn.cursor()
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    c.execute('''
        SELECT b.*, 
               ht.name as home_team_name, ht.current_rank as home_rank,
               at.name as away_team_name, at.current_rank as away_rank,
               g.status, g.home_score, g.away_score
        FROM betting_lines b
        LEFT JOIN teams ht ON b.home_team_id = ht.id
        LEFT JOIN teams at ON b.away_team_id = at.id
        LEFT JOIN games g ON b.game_id = g.id
        WHERE b.date >= ?
        ORDER BY b.date, b.captured_at DESC
    ''', (today,))
    
    lines = [dict(row) for row in c.fetchall()]
    conn.close()
    
    # Add model analysis to each
    for line in lines:
        if not line['home_ml'] or not line['away_ml']:
            continue
            
        try:
            # DK implied probability (remove vig)
            dk_home_prob = american_to_implied_prob(line['home_ml'])
            dk_away_prob = american_to_implied_prob(line['away_ml'])
            total = dk_home_prob + dk_away_prob
            line['dk_home_fair'] = dk_home_prob / total
            line['dk_away_fair'] = dk_away_prob / total
            
            # Model prediction
            ensemble = MODELS['ensemble']
            pred = ensemble.predict_game(line['home_team_id'], line['away_team_id'])
            line['model_home_prob'] = pred['home_win_probability']
            line['model_away_prob'] = pred['away_win_probability']
            line['projected_total'] = pred['projected_total']
            
            # Edges
            line['home_edge'] = (pred['home_win_probability'] - line['dk_home_fair']) * 100
            line['away_edge'] = (pred['away_win_probability'] - line['dk_away_fair']) * 100
            
            # Best pick
            if line['home_edge'] > abs(line['away_edge']):
                line['best_pick'] = 'home'
                line['best_edge'] = line['home_edge']
            else:
                line['best_pick'] = 'away'
                line['best_edge'] = abs(line['away_edge'])
            
            # Totals analysis
            if line['over_under']:
                line['total_diff'] = pred['projected_total'] - line['over_under']
                line['total_lean'] = 'OVER' if line['total_diff'] > 0 else 'UNDER'
            
            # EV calculation (per $100)
            if line['best_pick'] == 'home':
                ml = line['home_ml']
                prob = pred['home_win_probability']
            else:
                ml = line['away_ml']
                prob = pred['away_win_probability']
            
            if ml > 0:
                win_amount = ml
            else:
                win_amount = 100 / abs(ml) * 100
            
            line['ev_per_100'] = (prob * win_amount) - ((1 - prob) * 100)
            
        except Exception as e:
            line['error'] = str(e)
    
    return lines

def get_rankings_history():
    """Get historical rankings data"""
    conn = get_connection()
    c = conn.cursor()
    
    c.execute('''
        SELECT DISTINCT date FROM rankings_history
        ORDER BY date DESC
        LIMIT 10
    ''')
    dates = [row[0] for row in c.fetchall()]
    
    conn.close()
    return dates

def get_rankings_for_date(date):
    """Get rankings for a specific date"""
    conn = get_connection()
    c = conn.cursor()
    
    c.execute('''
        SELECT rh.team_id, rh.rank, rh.poll, t.name, t.conference,
               e.rating as elo_rating
        FROM rankings_history rh
        LEFT JOIN teams t ON rh.team_id = t.id
        LEFT JOIN elo_ratings e ON rh.team_id = e.team_id
        WHERE rh.date = ?
        ORDER BY rh.rank
    ''', (date,))
    
    rankings = [dict(row) for row in c.fetchall()]
    conn.close()
    return rankings

def get_ensemble_weights():
    """Get current ensemble model weights"""
    ensemble = MODELS.get('ensemble')
    if ensemble and hasattr(ensemble, 'weights'):
        return ensemble.weights
    return {}

def get_model_accuracy():
    """Get model accuracy statistics"""
    ensemble = MODELS.get('ensemble')
    if ensemble and hasattr(ensemble, 'get_model_accuracy'):
        return ensemble.get_model_accuracy()
    return {}

# ============================================
# Routes
# ============================================

@app.route('/')
def dashboard():
    """Main dashboard page"""
    todays_games = get_todays_games()
    value_picks = get_value_picks(5)
    top_25 = get_current_top_25()
    stats = get_quick_stats()
    
    # Add Elo ratings to top 25
    conn = get_connection()
    c = conn.cursor()
    for team in top_25:
        c.execute('SELECT rating FROM elo_ratings WHERE team_id = ?', (team['id'],))
        row = c.fetchone()
        team['elo_rating'] = row[0] if row else None
    conn.close()
    
    return render_template('dashboard.html',
                          todays_games=todays_games,
                          value_picks=value_picks,
                          top_25=top_25,
                          stats=stats,
                          today=datetime.now().strftime('%B %d, %Y'))

@app.route('/teams')
def teams():
    """Teams listing page"""
    all_teams = get_all_teams()
    
    # Get unique conferences
    conferences = sorted(set(t['conference'] for t in all_teams if t['conference']))
    
    # Sort parameter
    sort_by = request.args.get('sort', 'name')
    
    if sort_by == 'elo':
        all_teams.sort(key=lambda x: x.get('elo_rating') or 0, reverse=True)
    elif sort_by == 'rank':
        all_teams.sort(key=lambda x: (x.get('current_rank') or 999, x['name']))
    elif sort_by == 'win_pct':
        all_teams.sort(key=lambda x: x.get('win_pct', 0), reverse=True)
    elif sort_by == 'conference':
        all_teams.sort(key=lambda x: (x.get('conference') or 'ZZZ', x['name']))
    else:
        all_teams.sort(key=lambda x: x['name'])
    
    return render_template('teams.html',
                          teams=all_teams,
                          conferences=conferences,
                          sort_by=sort_by)

@app.route('/team/<team_id>')
def team_detail(team_id):
    """Individual team detail page"""
    team = get_team_detail(team_id)
    
    if not team:
        return render_template('404.html', message="Team not found"), 404
    
    # Split roster into batters and pitchers
    batters = [p for p in team['roster'] if p.get('position') not in ('P', 'RHP', 'LHP')]
    pitchers = [p for p in team['roster'] if p.get('position') in ('P', 'RHP', 'LHP')]
    
    # Get recent form (last 10 games)
    completed_games = [g for g in team['schedule'] if g['status'] == 'final']
    recent_form = completed_games[-10:] if completed_games else []
    
    return render_template('team_detail.html',
                          team=team,
                          batters=batters,
                          pitchers=pitchers,
                          recent_form=recent_form)

@app.route('/predict')
def predict():
    """Prediction tool page"""
    all_teams = get_all_teams()
    all_teams.sort(key=lambda x: x['name'])
    
    return render_template('predict.html', teams=all_teams)

@app.route('/api/predict', methods=['POST'])
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
    
    return jsonify(results)

@app.route('/rankings')
def rankings():
    """Rankings page"""
    top_25 = get_current_top_25()
    history_dates = get_rankings_history()
    
    # Add Elo and records to top 25
    conn = get_connection()
    c = conn.cursor()
    for team in top_25:
        c.execute('SELECT rating FROM elo_ratings WHERE team_id = ?', (team['id'],))
        row = c.fetchone()
        team['elo_rating'] = row[0] if row else None
        
        record = get_team_record(team['id'])
        team['wins'] = record['wins']
        team['losses'] = record['losses']
    conn.close()
    
    # Get selected week rankings
    selected_date = request.args.get('week')
    historical = None
    if selected_date:
        historical = get_rankings_for_date(selected_date)
    
    return render_template('rankings.html',
                          top_25=top_25,
                          history_dates=history_dates,
                          selected_date=selected_date,
                          historical=historical)

@app.route('/betting')
def betting():
    """Betting analysis page"""
    games = get_betting_games()
    
    # Sort by edge
    games_with_edge = [g for g in games if g.get('best_edge')]
    games_with_edge.sort(key=lambda x: x.get('best_edge', 0), reverse=True)
    
    # Best bets (edge > 5%)
    best_bets = [g for g in games_with_edge if g.get('best_edge', 0) >= 5]
    
    return render_template('betting.html',
                          games=games_with_edge,
                          best_bets=best_bets)

@app.route('/models')
def models():
    """Model performance page"""
    weights = get_ensemble_weights()
    accuracy = get_model_accuracy()
    
    # Combine weights and accuracy
    model_data = []
    for name in weights:
        data = {
            'name': name,
            'weight': weights.get(name, 0),
            'weight_pct': weights.get(name, 0) * 100
        }
        if name in accuracy:
            data['accuracy'] = accuracy[name]
        model_data.append(data)
    
    # Sort by weight
    model_data.sort(key=lambda x: x['weight'], reverse=True)
    
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
    conn.close()
    
    return render_template('models.html',
                          model_data=model_data,
                          weights=weights,
                          weight_history=weight_history,
                          prediction_log=prediction_log)

@app.route('/api/teams')
def api_teams():
    """API endpoint for team list"""
    teams = get_all_teams()
    return jsonify(teams)

# ============================================
# Template Filters
# ============================================

@app.template_filter('format_odds')
def format_odds(value):
    """Format American odds with + prefix for positive"""
    if value is None:
        return 'N/A'
    return f"+{value}" if value > 0 else str(value)

@app.template_filter('format_pct')
def format_pct(value):
    """Format decimal as percentage"""
    if value is None:
        return 'N/A'
    return f"{value * 100:.1f}%"

@app.template_filter('format_edge')
def format_edge(value):
    """Format edge with sign"""
    if value is None:
        return 'N/A'
    return f"+{value:.1f}%" if value > 0 else f"{value:.1f}%"

# ============================================
# Error Handlers
# ============================================

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html', error=str(e)), 500

# ============================================
# Main
# ============================================

if __name__ == '__main__':
    print("🏟️  College Baseball Predictor Dashboard")
    print("=" * 40)
    print(f"Running on http://0.0.0.0:5000")
    print("Press Ctrl+C to stop")
    print()
    app.run(host='0.0.0.0', port=5000, debug=True)
