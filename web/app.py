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
sys.path.insert(0, str(base_dir))  # Project root for models.* imports
sys.path.insert(0, str(base_dir / "scripts"))  # For database.py etc

from flask import Flask, render_template, request, jsonify

from database import (
    get_connection, get_team_record, get_team_runs, 
    get_recent_games, get_upcoming_games, get_current_top_25
)
from models.compare_models import MODELS, normalize_team_id
from betting_lines import american_to_implied_prob

app = Flask(__name__)

# ============================================
# Helper Functions
# ============================================

def get_all_conferences():
    """Get all unique conferences"""
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        SELECT DISTINCT conference FROM teams 
        WHERE conference IS NOT NULL AND conference != ''
        ORDER BY conference
    ''')
    conferences = [row[0] for row in c.fetchall()]
    conn.close()
    return conferences

def get_games_by_date(date_str, conference=None):
    """Get all games for a specific date, optionally filtered by conference"""
    conn = get_connection()
    c = conn.cursor()
    
    query = '''
        SELECT g.id, g.date, g.time, g.status,
               g.home_team_id, g.away_team_id,
               g.home_score, g.away_score, g.winner_id, g.innings,
               g.venue, g.is_neutral_site, g.is_conference_game,
               ht.name as home_team_name, ht.current_rank as home_rank, ht.conference as home_conf,
               at.name as away_team_name, at.current_rank as away_rank, at.conference as away_conf,
               he.rating as home_elo, ae.rating as away_elo,
               b.home_ml, b.away_ml, b.over_under
        FROM games g
        LEFT JOIN teams ht ON g.home_team_id = ht.id
        LEFT JOIN teams at ON g.away_team_id = at.id
        LEFT JOIN elo_ratings he ON g.home_team_id = he.team_id
        LEFT JOIN elo_ratings ae ON g.away_team_id = ae.team_id
        LEFT JOIN betting_lines b ON g.home_team_id = b.home_team_id 
            AND g.away_team_id = b.away_team_id AND g.date = b.date
        WHERE g.date = ?
    '''
    
    params = [date_str]
    
    if conference:
        query += " AND (ht.conference = ? OR at.conference = ?)"
        params.extend([conference, conference])
    
    query += " ORDER BY g.time, g.id"
    
    c.execute(query, params)
    games = [dict(row) for row in c.fetchall()]
    conn.close()
    
    return games

def get_available_dates():
    """Get all dates that have games (for calendar navigation)"""
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        SELECT DISTINCT date, 
               COUNT(*) as game_count,
               SUM(CASE WHEN status = 'final' THEN 1 ELSE 0 END) as completed
        FROM games 
        GROUP BY date 
        ORDER BY date DESC
        LIMIT 60
    ''')
    dates = [dict(row) for row in c.fetchall()]
    conn.close()
    return dates

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
        SELECT * FROM player_stats
        WHERE team_id = ?
        ORDER BY position, number
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
    """Get all games scheduled for today, with DK lines if available"""
    today = datetime.now().strftime('%Y-%m-%d')
    
    conn = get_connection()
    c = conn.cursor()
    
    # Combine games from both games table and betting_lines (DK may have games not in schedule)
    c.execute('''
        SELECT g.id, g.date, g.time, g.status,
               g.home_team_id, g.away_team_id,
               g.home_score, g.away_score,
               b.home_ml, b.away_ml, b.over_under, 
               b.home_spread as run_line, b.home_spread_odds as run_line_odds,
               ht.name as home_team_name, ht.current_rank as home_rank, ht.conference as home_conf,
               at.name as away_team_name, at.current_rank as away_rank, at.conference as away_conf,
               he.rating as home_elo, ae.rating as away_elo
        FROM games g
        LEFT JOIN betting_lines b ON g.home_team_id = b.home_team_id 
            AND g.away_team_id = b.away_team_id AND g.date = b.date
        LEFT JOIN teams ht ON g.home_team_id = ht.id
        LEFT JOIN teams at ON g.away_team_id = at.id
        LEFT JOIN elo_ratings he ON g.home_team_id = he.team_id
        LEFT JOIN elo_ratings ae ON g.away_team_id = ae.team_id
        WHERE g.date = ?
        
        UNION
        
        SELECT b.game_id as id, b.date, NULL as time, NULL as status,
               b.home_team_id, b.away_team_id,
               NULL as home_score, NULL as away_score,
               b.home_ml, b.away_ml, b.over_under,
               b.home_spread as run_line, b.home_spread_odds as run_line_odds,
               ht.name as home_team_name, ht.current_rank as home_rank, ht.conference as home_conf,
               at.name as away_team_name, at.current_rank as away_rank, at.conference as away_conf,
               he.rating as home_elo, ae.rating as away_elo
        FROM betting_lines b
        LEFT JOIN teams ht ON b.home_team_id = ht.id
        LEFT JOIN teams at ON b.away_team_id = at.id
        LEFT JOIN elo_ratings he ON b.home_team_id = he.team_id
        LEFT JOIN elo_ratings ae ON b.away_team_id = ae.team_id
        WHERE b.date = ?
        AND NOT EXISTS (
            SELECT 1 FROM games g 
            WHERE g.home_team_id = b.home_team_id 
            AND g.away_team_id = b.away_team_id 
            AND g.date = b.date
        )
        
        ORDER BY time, id
    ''', (today, today))
    
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
    # Only look at next 3 days to avoid running predictions on 2000+ games
    three_days = (datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d')
    
    c.execute('''
        SELECT b.*, 
               ht.name as home_team_name,
               at.name as away_team_name
        FROM betting_lines b
        LEFT JOIN teams ht ON b.home_team_id = ht.id
        LEFT JOIN teams at ON b.away_team_id = at.id
        WHERE b.date >= ? AND b.date <= ?
        ORDER BY b.date
        LIMIT 100
    ''', (today, three_days))
    
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
            
            # Get blended model prediction (neural + ensemble)
            pred = get_blended_prediction(line['home_team_id'], line['away_team_id'])
            if not pred:
                continue
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
    
    c.execute("SELECT COUNT(*) FROM player_stats")
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

def get_blended_prediction(home_team_id, away_team_id):
    """Blend ensemble + neural predictions for betting analysis.
    
    Neural gets 60% weight while ensemble warms up (few evaluated predictions).
    As ensemble accumulates more data, weights will shift toward equal.
    """
    ensemble = MODELS.get('ensemble')
    neural = MODELS.get('neural')
    
    ensemble_pred = None
    neural_pred = None
    
    if ensemble:
        try:
            ensemble_pred = ensemble.predict_game(home_team_id, away_team_id)
        except Exception:
            pass
    
    if neural:
        try:
            neural_pred = neural.predict_game(home_team_id, away_team_id)
        except Exception:
            pass
    
    # If only one model available, use it
    if ensemble_pred and not neural_pred:
        return ensemble_pred
    if neural_pred and not ensemble_pred:
        return neural_pred
    if not ensemble_pred and not neural_pred:
        return None
    
    # Blend: neural 60%, ensemble 40% (early season weighting)
    nn_weight = 0.60
    ens_weight = 0.40
    
    home_prob = (neural_pred['home_win_probability'] * nn_weight + 
                 ensemble_pred['home_win_probability'] * ens_weight)
    away_prob = 1 - home_prob
    
    # Blend run projections
    home_runs = (neural_pred.get('projected_home_runs', 0) * nn_weight +
                 ensemble_pred.get('projected_home_runs', 0) * ens_weight)
    away_runs = (neural_pred.get('projected_away_runs', 0) * nn_weight +
                 ensemble_pred.get('projected_away_runs', 0) * ens_weight)
    
    return {
        'home_win_probability': home_prob,
        'away_win_probability': away_prob,
        'projected_home_runs': home_runs,
        'projected_away_runs': away_runs,
        'projected_total': home_runs + away_runs,
        'neural_prob': neural_pred['home_win_probability'],
        'ensemble_prob': ensemble_pred['home_win_probability'],
        'blend': f"NN {int(nn_weight*100)}% / Ens {int(ens_weight*100)}%"
    }


def get_betting_games():
    """Get all games with betting lines"""
    conn = get_connection()
    c = conn.cursor()
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    c.execute('''
        SELECT b.*, 
               ht.name as home_team_name, ht.current_rank as home_rank, ht.conference as home_conf,
               at.name as away_team_name, at.current_rank as away_rank, at.conference as away_conf,
               g.status, g.home_score, g.away_score
        FROM betting_lines b
        LEFT JOIN teams ht ON b.home_team_id = ht.id
        LEFT JOIN teams at ON b.away_team_id = at.id
        LEFT JOIN games g ON b.game_id = g.id
        WHERE b.date = ?
        ORDER BY b.captured_at DESC
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
            
            # Model prediction (blended neural + ensemble)
            pred = get_blended_prediction(line['home_team_id'], line['away_team_id'])
            if not pred:
                continue
            line['model_home_prob'] = pred['home_win_probability']
            line['model_away_prob'] = pred['away_win_probability']
            line['projected_total'] = pred.get('projected_total', 0)
            line['blend_info'] = pred.get('blend', '')
            line['nn_prob'] = pred.get('neural_prob')
            line['ens_prob'] = pred.get('ensemble_prob')
            
            # Edges
            line['home_edge'] = (pred['home_win_probability'] - line['dk_home_fair']) * 100
            line['away_edge'] = (pred['away_win_probability'] - line['dk_away_fair']) * 100
            
            # Best pick (pick whichever side has positive edge)
            if line['home_edge'] >= 0:
                line['best_pick'] = 'home'
                line['best_edge'] = line['home_edge']
            else:
                line['best_pick'] = 'away'
                line['best_edge'] = abs(line['away_edge'])
            
            # Totals analysis (simple version for speed)
            if line['over_under']:
                line['total_diff'] = pred['projected_total'] - line['over_under']
                line['total_lean'] = 'OVER' if line['total_diff'] > 0 else 'UNDER'
                # Estimate edge from diff (rough approximation)
                line['total_edge'] = min(abs(line['total_diff']) * 8, 50)  # ~8% edge per run diff
            
            # NN Totals model
            nn_totals = MODELS.get('nn_totals')
            if nn_totals and nn_totals.is_trained():
                try:
                    totals_pred = nn_totals.predict_game(
                        line['home_team_id'], line['away_team_id'],
                        over_under_line=line.get('over_under'))
                    line['nn_total'] = totals_pred.get('projected_total')
                    line['nn_over_prob'] = totals_pred.get('over_prob')
                    line['nn_under_prob'] = totals_pred.get('under_prob')
                except Exception:
                    pass
            
            # NN Spread model
            nn_spread = MODELS.get('nn_spread')
            if nn_spread and nn_spread.is_trained():
                try:
                    spread_pred = nn_spread.predict_game(
                        line['home_team_id'], line['away_team_id'])
                    line['nn_margin'] = spread_pred.get('projected_margin')
                    line['nn_cover_prob'] = spread_pred.get('cover_prob')
                except Exception:
                    pass
            
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
    """Get model accuracy statistics from database"""
    conn = get_connection()
    c = conn.cursor()
    
    c.execute('''
        SELECT model_name,
               SUM(was_correct) as correct,
               COUNT(*) as total
        FROM model_predictions 
        WHERE was_correct IS NOT NULL
        GROUP BY model_name
    ''')
    
    result = {}
    for row in c.fetchall():
        model_name, correct, total = row
        result[model_name] = {
            'all_time_accuracy': correct / total if total > 0 else None,
            'all_time_predictions': total,
            'recent_accuracy': correct / total if total > 0 else None,  # TODO: add rolling window
            'recent_predictions': total,
            'current_weight': MODELS.get('ensemble').weights.get(model_name, 0) if MODELS.get('ensemble') else 0
        }
    
    conn.close()
    return result

# ============================================
# Routes
# ============================================

def get_featured_team_info(team_id='mississippi-state'):
    """Get featured team focus data for dashboard"""
    conn = get_connection()
    c = conn.cursor()
    
    # Record
    record = get_team_record(team_id)
    
    # Elo
    c.execute('SELECT rating FROM elo_ratings WHERE team_id = ?', (team_id,))
    row = c.fetchone()
    elo = row[0] if row else 1500
    
    # Current rank
    c.execute('SELECT current_rank FROM teams WHERE id = ?', (team_id,))
    row = c.fetchone()
    rank = row[0] if row else None
    
    # Last 5 results
    c.execute('''
        SELECT g.date, g.home_team_id, g.away_team_id, g.home_score, g.away_score, g.winner_id,
               ht.name as home_name, at.name as away_name
        FROM games g
        LEFT JOIN teams ht ON g.home_team_id = ht.id
        LEFT JOIN teams at ON g.away_team_id = at.id
        WHERE (g.home_team_id = ? OR g.away_team_id = ?) AND g.status = 'final'
        ORDER BY g.date DESC
        LIMIT 5
    ''', (team_id, team_id))
    last_5 = []
    for r in c.fetchall():
        r = dict(r)
        won = r['winner_id'] == team_id
        if r['home_team_id'] == team_id:
            opponent = r['away_name']
            score = f"{r['home_score']}-{r['away_score']}"
        else:
            opponent = r['home_name']
            score = f"{r['away_score']}-{r['home_score']}"
        last_5.append({'won': won, 'opponent': opponent, 'score': score, 'date': r['date']})
    
    # Next game
    today = datetime.now().strftime('%Y-%m-%d')
    c.execute('''
        SELECT g.date, g.time, g.home_team_id, g.away_team_id,
               ht.name as home_name, at.name as away_name
        FROM games g
        LEFT JOIN teams ht ON g.home_team_id = ht.id
        LEFT JOIN teams at ON g.away_team_id = at.id
        WHERE (g.home_team_id = ? OR g.away_team_id = ?) AND g.status = 'scheduled' AND g.date >= ?
        ORDER BY g.date, g.time
        LIMIT 1
    ''', (team_id, team_id, today))
    next_game = None
    row = c.fetchone()
    if row:
        r = dict(row)
        if r['home_team_id'] == team_id:
            next_game = {'opponent': r['away_name'], 'location': 'vs', 'date': r['date'], 'time': r['time']}
        else:
            next_game = {'opponent': r['home_name'], 'location': '@', 'date': r['date'], 'time': r['time']}
    
    # Team name
    c2 = conn.cursor()
    c2.execute('SELECT name FROM teams WHERE id = ?', (team_id,))
    name_row = c2.fetchone()
    team_name = name_row[0] if name_row else team_id.replace('-', ' ').title()
    
    conn.close()
    return {
        'team_id': team_id,
        'team_name': team_name,
        'record': record,
        'elo': elo,
        'rank': rank,
        'last_5': last_5,
        'next_game': next_game
    }

def get_model_accuracy_summary():
    """Get quick model accuracy numbers for dashboard"""
    conn = get_connection()
    c = conn.cursor()
    
    result = {}
    for model_name in ('ensemble', 'neural'):
        c.execute('''
            SELECT SUM(was_correct) as correct, COUNT(*) as total
            FROM model_predictions 
            WHERE was_correct IS NOT NULL AND model_name = ?
        ''', (model_name,))
        row = c.fetchone()
        if row and row['total'] > 0:
            result[model_name] = {'correct': row['correct'], 'total': row['total'],
                                  'pct': round(row['correct'] / row['total'] * 100, 1)}
        else:
            result[model_name] = {'correct': 0, 'total': 0, 'pct': None}
    
    conn.close()
    return result

@app.route('/')
def dashboard():
    """Main dashboard page"""
    featured_team = request.args.get('team', 'mississippi-state')
    todays_games = get_todays_games()
    value_picks = get_value_picks(5)
    stats = get_quick_stats()
    featured = get_featured_team_info(featured_team)
    accuracy = get_model_accuracy_summary()
    
    # Model snapshot - all models
    all_accuracy = get_model_accuracy()
    
    return render_template('dashboard.html',
                          todays_games=todays_games,
                          value_picks=value_picks,
                          stats=stats,
                          featured=featured,
                          accuracy=accuracy,
                          all_accuracy=all_accuracy,
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
    pitcher_positions = ('P', 'RHP', 'LHP')
    pitchers = [p for p in team['roster'] if (p.get('position') or '') in pitcher_positions 
                or (p.get('position') or '').startswith(('RHP', 'LHP'))]
    batters = [p for p in team['roster'] if p not in pitchers]
    
    # Get recent form (last 10 games)
    completed_games = [g for g in team['schedule'] if g['status'] == 'final']
    recent_form = completed_games[-10:] if completed_games else []
    
    return render_template('team_detail.html',
                          team=team,
                          batters=batters,
                          pitchers=pitchers,
                          recent_form=recent_form)

@app.route('/game/<game_id>')
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


@app.route('/standings')
def standings():
    """Conference standings page"""
    selected_conf = request.args.get('conference', '')
    
    conferences_to_show = ['SEC', 'Big Ten', 'ACC', 'Big 12']
    if selected_conf:
        conferences_to_show = [selected_conf]
    
    conn = get_connection()
    c = conn.cursor()
    
    standings_data = {}
    for conf in conferences_to_show:
        c.execute('''
            SELECT t.id, t.name, t.current_rank, e.rating as elo
            FROM teams t
            LEFT JOIN elo_ratings e ON t.id = e.team_id
            WHERE t.conference = ?
            ORDER BY t.name
        ''', (conf,))
        
        teams = []
        for row in c.fetchall():
            t = dict(row)
            record = get_team_record(t['id'])
            runs = get_team_runs(t['id'])
            games = runs['games'] or 1
            
            # Conference record
            c.execute('''
                SELECT 
                    SUM(CASE WHEN winner_id = ? THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN winner_id != ? AND winner_id IS NOT NULL THEN 1 ELSE 0 END) as losses
                FROM games
                WHERE ((home_team_id = ? AND away_team_id IN (SELECT id FROM teams WHERE conference = ?))
                    OR (away_team_id = ? AND home_team_id IN (SELECT id FROM teams WHERE conference = ?)))
                AND status = 'final' AND is_conference_game = 1
            ''', (t['id'], t['id'], t['id'], conf, t['id'], conf))
            conf_rec = c.fetchone()
            
            # Streak
            c.execute('''
                SELECT winner_id FROM games
                WHERE (home_team_id = ? OR away_team_id = ?) AND status = 'final'
                ORDER BY date DESC, id DESC LIMIT 10
            ''', (t['id'], t['id']))
            streak_rows = c.fetchall()
            streak = ''
            if streak_rows:
                streak_type = 'W' if streak_rows[0]['winner_id'] == t['id'] else 'L'
                streak_count = 0
                for sr in streak_rows:
                    if (sr['winner_id'] == t['id']) == (streak_type == 'W'):
                        streak_count += 1
                    else:
                        break
                streak = f"{streak_type}{streak_count}"
            
            rs_avg = round(runs['runs_scored'] / games, 1)
            ra_avg = round(runs['runs_allowed'] / games, 1)
            
            teams.append({
                'id': t['id'],
                'name': t['name'],
                'rank': t['current_rank'],
                'wins': record['wins'],
                'losses': record['losses'],
                'win_pct': f".{int(record['pct']*1000):03d}" if record['pct'] < 1 else '1.000',
                'conf_wins': conf_rec['wins'] or 0 if conf_rec else 0,
                'conf_losses': conf_rec['losses'] or 0 if conf_rec else 0,
                'rs_avg': rs_avg,
                'ra_avg': ra_avg,
                'run_diff': round(rs_avg - ra_avg, 1),
                'elo': t['elo'] or 1500,
                'streak': streak
            })
        
        teams.sort(key=lambda x: (-x['wins'], x['losses'], -x['elo']))
        standings_data[conf] = teams
    
    conn.close()
    
    all_conferences = get_all_conferences()
    
    return render_template('standings.html',
        standings=standings_data,
        conferences=all_conferences,
        selected_conf=selected_conf)


@app.route('/predict')
def predict():
    """Prediction tool page"""
    all_teams = get_all_teams()
    all_teams.sort(key=lambda x: x['name'])
    conferences = get_all_conferences()
    
    return render_template('predict.html', teams=all_teams, conferences=conferences)

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

@app.route('/api/runs', methods=['POST'])
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

@app.route('/rankings')
def rankings():
    """Rankings page"""
    conference = request.args.get('conference', '')
    
    top_25 = get_current_top_25()
    history_dates = get_rankings_history()
    conferences = get_all_conferences()
    
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
    
    # Filter by conference if specified
    if conference:
        top_25 = [t for t in top_25 if t.get('conference') == conference]
    
    # Get selected week rankings
    selected_date = request.args.get('week')
    historical = None
    if selected_date:
        historical = get_rankings_for_date(selected_date)
        if conference:
            historical = [t for t in historical if t.get('conference') == conference]
    
    # Get Elo Top 25
    conn2 = get_connection()
    c2 = conn2.cursor()
    elo_limit = 200 if conference else 25
    elo_query = '''
        SELECT e.team_id, e.rating, t.name, t.conference, t.current_rank
        FROM elo_ratings e
        JOIN teams t ON e.team_id = t.id
    '''
    elo_params = []
    if conference:
        elo_query += ' WHERE t.conference = ?'
        elo_params.append(conference)
    elo_query += ' ORDER BY e.rating DESC LIMIT ?'
    elo_params.append(elo_limit)
    
    c2.execute(elo_query, elo_params)
    elo_top_25 = [dict(row) for row in c2.fetchall()]
    for i, team in enumerate(elo_top_25):
        team['elo_rank'] = i + 1
        record = get_team_record(team['team_id'])
        team['wins'] = record['wins']
        team['losses'] = record['losses']
        # Add SOS if available
        sos_row = c2.execute('SELECT past_sos, overall_sos FROM team_sos WHERE team_id = ?', 
                            (team['team_id'],)).fetchone()
        team['sos'] = round(sos_row['past_sos']) if sos_row else None
    conn2.close()
    
    # Find disagreements - teams in AP but not Elo top 25, and vice versa
    ap_ids = set(t['id'] for t in top_25)
    elo_ids = set(t['team_id'] for t in elo_top_25)
    
    return render_template('rankings.html',
                          top_25=top_25,
                          elo_top_25=elo_top_25,
                          ap_ids=ap_ids,
                          elo_ids=elo_ids,
                          history_dates=history_dates,
                          selected_date=selected_date,
                          historical=historical,
                          conferences=conferences,
                          selected_conference=conference)

@app.route('/betting')
def betting():
    """Betting analysis page"""
    conference = request.args.get('conference', '')
    conferences = get_all_conferences()
    
    games = get_betting_games()
    
    # Filter by conference if specified
    if conference:
        games = [g for g in games 
                if g.get('home_conf') == conference or g.get('away_conf') == conference]
    
    # Sort by edge
    games_with_edge = [g for g in games if g.get('best_edge')]
    games_with_edge.sort(key=lambda x: x.get('best_edge', 0), reverse=True)
    
    # Best bets (edge > 5%)
    best_bets = [g for g in games_with_edge if g.get('best_edge', 0) >= 5]
    
    # Best totals (edge > 15% on over/under)
    games_with_totals = [g for g in games if g.get('total_edge') and g.get('over_under')]
    games_with_totals.sort(key=lambda x: x.get('total_edge', 0), reverse=True)
    best_totals = [g for g in games_with_totals if g.get('total_edge', 0) >= 15]
    
    return render_template('betting.html',
                          games=games_with_edge,
                          best_bets=best_bets,
                          best_totals=best_totals,
                          conferences=conferences,
                          selected_conference=conference)

@app.route('/models')
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
    from collections import defaultdict
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
            if len(window) >= 10:  # need at least 10 to show
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

@app.route('/calendar')
def calendar():
    """Calendar view for historical game data"""
    # Get date from query param, default to today
    date_str = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    conference = request.args.get('conference', '')
    
    # Get games for selected date
    games = get_games_by_date(date_str, conference if conference else None)
    
    # Get available dates for navigation
    available_dates = get_available_dates()
    
    # Get conferences for filter
    conferences = get_all_conferences()
    
    # Parse date for display
    try:
        display_date = datetime.strptime(date_str, '%Y-%m-%d')
    except:
        display_date = datetime.now()
    
    # Calculate prev/next dates
    prev_date = (display_date - timedelta(days=1)).strftime('%Y-%m-%d')
    next_date = (display_date + timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Add predictions to each game
    ensemble = MODELS.get('ensemble')
    neural = MODELS.get('neural')
    correct_predictions = 0
    total_predictions = 0
    nn_correct = 0
    nn_total = 0
    
    for game in games:
        try:
            if ensemble and game.get('home_team_id') and game.get('away_team_id'):
                pred = ensemble.predict_game(game['home_team_id'], game['away_team_id'])
                game['pred_home_prob'] = pred.get('home_win_probability', 0.5)
                game['pred_away_prob'] = pred.get('away_win_probability', 0.5)
                game['pred_winner'] = game['home_team_id'] if game['pred_home_prob'] > 0.5 else game['away_team_id']
                game['pred_confidence'] = max(game['pred_home_prob'], game['pred_away_prob'])
                
                # Check if prediction was correct for final games
                if game['status'] == 'final' and game.get('winner_id'):
                    game['pred_correct'] = game['pred_winner'] == game['winner_id']
                    total_predictions += 1
                    if game['pred_correct']:
                        correct_predictions += 1
        except Exception:
            game['pred_winner'] = None
            game['pred_confidence'] = None
        
        # NN Totals and Spread predictions
        try:
            nn_totals_model = MODELS.get('nn_totals')
            nn_spread_model = MODELS.get('nn_spread')
            if game.get('home_team_id') and game.get('away_team_id'):
                if nn_totals_model and nn_totals_model.is_trained():
                    t_pred = nn_totals_model.predict_game(game['home_team_id'], game['away_team_id'])
                    game['nn_projected_total'] = t_pred.get('projected_total')
                if nn_spread_model and nn_spread_model.is_trained():
                    s_pred = nn_spread_model.predict_game(game['home_team_id'], game['away_team_id'])
                    game['nn_projected_margin'] = s_pred.get('projected_margin')
                    game['nn_cover_prob'] = s_pred.get('cover_prob')
        except Exception:
            pass
        
        # Neural model prediction (independent)
        try:
            if neural and game.get('home_team_id') and game.get('away_team_id'):
                nn_pred = neural.predict_game(game['home_team_id'], game['away_team_id'])
                game['nn_home_prob'] = nn_pred.get('home_win_probability', 0.5)
                game['nn_winner'] = game['home_team_id'] if game['nn_home_prob'] > 0.5 else game['away_team_id']
                game['nn_confidence'] = max(game['nn_home_prob'], 1 - game['nn_home_prob'])
                
                if game['status'] == 'final' and game.get('winner_id'):
                    game['nn_correct'] = game['nn_winner'] == game['winner_id']
                    nn_total += 1
                    if game['nn_correct']:
                        nn_correct += 1
        except Exception:
            game['nn_winner'] = None
    
    # Model accuracy for the day
    model_accuracy = round((correct_predictions / total_predictions) * 100) if total_predictions > 0 else None
    
    # Stats for the day
    total_games = len(games)
    completed = sum(1 for g in games if g['status'] == 'final')
    scheduled = sum(1 for g in games if g['status'] == 'scheduled')
    
    # Calculate aggregate stats
    final_games = [g for g in games if g['status'] == 'final']
    total_runs = sum((g.get('home_score') or 0) + (g.get('away_score') or 0) for g in final_games)
    avg_runs = round(total_runs / completed, 1) if completed > 0 else 0
    
    home_wins = sum(1 for g in final_games if g.get('winner_id') == g.get('home_team_id'))
    home_win_pct = round((home_wins / completed) * 100) if completed > 0 else 0
    
    return render_template('calendar.html',
                          games=games,
                          selected_date=date_str,
                          display_date=display_date,
                          prev_date=prev_date,
                          next_date=next_date,
                          available_dates=available_dates,
                          conferences=conferences,
                          selected_conference=conference,
                          total_games=total_games,
                          completed=completed,
                          scheduled=scheduled,
                          avg_runs=avg_runs,
                          home_win_pct=home_win_pct,
                          model_accuracy=model_accuracy,
                          correct_predictions=correct_predictions,
                          total_predictions=total_predictions,
                          nn_accuracy=round((nn_correct / nn_total) * 100) if nn_total > 0 else None,
                          nn_correct=nn_correct,
                          nn_total=nn_total)

@app.route('/tracker')
def tracker():
    """P&L Tracker - reads from tracked_bets table (recorded by record_bets.py).
    
    This ensures P&L exactly matches what was shown as 'best bets' on the betting page,
    since both use the same blended prediction logic at the same point in time.
    """
    conn = get_connection()
    c = conn.cursor()
    
    # --- MONEYLINE BETS ---
    c.execute('''
        SELECT tb.*, g.status, g.home_score, g.away_score
        FROM tracked_bets tb
        LEFT JOIN games g ON tb.game_id = g.id
        ORDER BY tb.date, tb.game_id
    ''')
    all_bets = [dict(r) for r in c.fetchall()]
    
    bets = []
    pending_bets = []
    for b in all_bets:
        entry = {
            'date': b['date'],
            'game': f"{'vs ' if b['is_home'] else '@ '}{b['opponent_name']}",
            'pick': b['pick_team_name'],
            'moneyline': b['moneyline'],
            'edge': round(b['edge'], 1),
            'model_prob': b['model_prob'],
            'dk_implied': b['dk_implied'],
            'won': b['won'],
            'profit': b.get('profit', 0) or 0,
        }
        if b['won'] is not None:
            bets.append(entry)
        else:
            pending_bets.append(entry)
    
    # --- SPREAD & TOTAL BETS ---
    spread_bets = []
    total_bets_list = []
    pending_spread = []
    pending_totals = []
    
    c.execute('''
        SELECT tb.*, g.home_score, g.away_score,
               COALESCE(ht.name, bht.name) as home_name,
               COALESCE(at.name, bat.name) as away_name
        FROM tracked_bets_spreads tb
        LEFT JOIN games g ON tb.game_id = g.id
        LEFT JOIN teams ht ON g.home_team_id = ht.id
        LEFT JOIN teams at ON g.away_team_id = at.id
        LEFT JOIN betting_lines bl ON tb.game_id = bl.game_id AND tb.date = bl.date
        LEFT JOIN teams bht ON bl.home_team_id = bht.id
        LEFT JOIN teams bat ON bl.away_team_id = bat.id
        ORDER BY tb.date, tb.game_id
    ''')
    all_spread_bets = [dict(r) for r in c.fetchall()]
    conn.close()
    
    for b in all_spread_bets:
        game_label = f"{b.get('away_name', '?')} @ {b.get('home_name', '?')}"
        entry = {
            'date': b['date'],
            'game': game_label,
            'pick': b['pick'],
            'line': b['line'],
            'odds': b['odds'],
            'model_projection': b['model_projection'],
            'edge': round(b['edge'], 1),
            'won': b['won'],
            'profit': b.get('profit', 0) or 0,
            'bet_type': b['bet_type'],
        }
        if b['bet_type'] == 'spread':
            if b['won'] is not None:
                spread_bets.append(entry)
            else:
                pending_spread.append(entry)
        else:
            if b['won'] is not None:
                total_bets_list.append(entry)
            else:
                pending_totals.append(entry)
    
    def calc_stats(bet_list):
        total = len(bet_list)
        wins = sum(1 for b in bet_list if b['won'])
        pl = sum(b['profit'] for b in bet_list)
        return {
            'total': total,
            'wins': wins,
            'win_rate': round(wins / total * 100, 1) if total > 0 else 0,
            'pl': round(pl, 2),
            'roi': round(pl / (total * 100) * 100, 1) if total > 0 else 0,
        }
    
    ml_stats = calc_stats(bets)
    spread_stats = calc_stats(spread_bets)
    totals_stats = calc_stats(total_bets_list)
    
    # Combined P&L
    all_completed = []
    for b in bets:
        all_completed.append({'date': b['date'], 'profit': b['profit'], 'won': b['won'], 'type': 'ML', 'pick': b['pick']})
    for b in spread_bets:
        all_completed.append({'date': b['date'], 'profit': b['profit'], 'won': b['won'], 'type': 'SPR', 'pick': b['pick']})
    for b in total_bets_list:
        all_completed.append({'date': b['date'], 'profit': b['profit'], 'won': b['won'], 'type': 'TOT', 'pick': b['pick']})
    all_completed.sort(key=lambda x: x['date'])
    
    combined_stats = calc_stats(all_completed)
    
    # Running P&L for chart (combined)
    running_pl = []
    cumulative = 0
    for b in all_completed:
        cumulative += b['profit']
        running_pl.append({'date': b['date'], 'pl': round(cumulative, 2), 
                          'type': b['type'], 'pick': b['pick'], 'profit': b['profit']})
    
    # Per-type running P&L for individual charts
    def build_running_pl(bet_list, bet_type):
        result = []
        cum = 0
        sorted_bets = sorted(bet_list, key=lambda x: x['date'])
        for b in sorted_bets:
            cum += b['profit']
            result.append({'date': b['date'], 'pl': round(cum, 2), 
                          'pick': b['pick'], 'profit': b['profit']})
        return result
    
    ml_running_pl = build_running_pl(bets, 'ML')
    spread_running_pl = build_running_pl(spread_bets, 'SPR')
    totals_running_pl = build_running_pl(total_bets_list, 'TOT')
    
    # Edge buckets (moneyline only for backward compat)
    buckets = {
        '5-10%': {'bets': [], 'label': '5-10%'},
        '10-20%': {'bets': [], 'label': '10-20%'},
        '20%+': {'bets': [], 'label': '20%+'}
    }
    for b in bets:
        if b['edge'] >= 20:
            buckets['20%+']['bets'].append(b)
        elif b['edge'] >= 10:
            buckets['10-20%']['bets'].append(b)
        else:
            buckets['5-10%']['bets'].append(b)
    
    bucket_stats = []
    for key in ('5-10%', '10-20%', '20%+'):
        bb = buckets[key]['bets']
        if bb:
            bw = sum(1 for x in bb if x['won'])
            bp = sum(x['profit'] for x in bb)
            bucket_stats.append({
                'label': key,
                'count': len(bb),
                'wins': bw,
                'win_rate': round(bw / len(bb) * 100, 1),
                'pl': round(bp, 2),
                'roi': round(bp / (len(bb) * 100) * 100, 1)
            })
        else:
            bucket_stats.append({'label': key, 'count': 0, 'wins': 0, 'win_rate': 0, 'pl': 0, 'roi': 0})
    
    return render_template('tracker.html',
                          bets=bets,
                          pending_bets=pending_bets,
                          spread_bets=spread_bets,
                          pending_spread=pending_spread,
                          total_bets_list=total_bets_list,
                          pending_totals=pending_totals,
                          ml_stats=ml_stats,
                          spread_stats=spread_stats,
                          totals_stats=totals_stats,
                          combined_stats=combined_stats,
                          total_bets=ml_stats['total'],
                          wins=ml_stats['wins'],
                          win_rate=ml_stats['win_rate'],
                          total_pl=ml_stats['pl'],
                          roi=ml_stats['roi'],
                          running_pl=running_pl,
                          ml_running_pl=ml_running_pl,
                          spread_running_pl=spread_running_pl,
                          totals_running_pl=totals_running_pl,
                          bucket_stats=bucket_stats)

@app.route('/api/teams')
def api_teams():
    """API endpoint for team list"""
    teams = get_all_teams()
    return jsonify(teams)

@app.route('/api/best-bets')
def api_best_bets():
    """Return today's best bets — same logic as the Betting page.
    Used by record_daily_bets.py to populate the P&L tracker.
    """
    games = get_betting_games()
    
    # Best moneyline bets (5%+ edge, top 6)
    ml_candidates = [g for g in games if g.get('best_edge', 0) >= 5]
    ml_candidates.sort(key=lambda x: x.get('best_edge', 0), reverse=True)
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
            'dk_implied': round(dk_imp, 4), 'edge': round(g['best_edge'], 2)
        })
    
    # Best totals (15%+ edge, top 6)
    totals_candidates = [g for g in games if g.get('total_edge', 0) >= 15 and g.get('over_under')]
    totals_candidates.sort(key=lambda x: x.get('total_edge', 0), reverse=True)
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
    best_spreads = spread_candidates[:6]
    
    return jsonify({
        'date': datetime.now().strftime('%Y-%m-%d'),
        'moneylines': ml_bets,
        'totals': totals_bets,
        'spreads': best_spreads
    })

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
# Debug Page
# ============================================

@app.route('/debug')
def debug():
    conn = get_connection()
    c = conn.cursor()
    
    # Teams with more than 5 games in any Mon-Sun week
    # Get all weeks that have final games
    c.execute("SELECT DISTINCT date FROM games WHERE status='final' ORDER BY date")
    all_dates = [row[0] for row in c.fetchall()]
    
    # Build Monday-Sunday week boundaries
    from datetime import date as date_type
    weeks = set()
    for d_str in all_dates:
        d = datetime.strptime(d_str, '%Y-%m-%d').date()
        monday = d - timedelta(days=d.weekday())
        weeks.add(monday.strftime('%Y-%m-%d'))
    
    suspicious_teams = []
    for monday_str in sorted(weeks):
        monday = datetime.strptime(monday_str, '%Y-%m-%d').date()
        sunday = monday + timedelta(days=6)
        
        c.execute('''
            SELECT t.id, t.name, t.conference,
                COUNT(g.id) as games_this_week,
                SUM(CASE WHEN g.winner_id = t.id THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN g.winner_id != t.id AND g.winner_id IS NOT NULL THEN 1 ELSE 0 END) as losses
            FROM teams t
            JOIN games g ON (g.home_team_id = t.id OR g.away_team_id = t.id) AND g.status = 'final'
            WHERE g.date >= ? AND g.date <= ?
            GROUP BY t.id
            HAVING games_this_week > 5
            ORDER BY games_this_week DESC
        ''', (monday_str, sunday.strftime('%Y-%m-%d')))
        
        for row in c.fetchall():
            entry = dict(row)
            entry['week'] = f"{monday_str} to {sunday.strftime('%Y-%m-%d')}"
            suspicious_teams.append(entry)
    teams_over_3 = suspicious_teams
    
    # Load flags from JSON file
    flags_path = base_dir / 'data' / 'debug_flags.json'
    flags = {}
    if flags_path.exists():
        with open(flags_path) as f:
            flags = json.load(f)
    
    # Data quality audit stats
    c.execute("SELECT COUNT(*) FROM games WHERE status='final'")
    total_final = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM games WHERE status='scheduled'")
    total_scheduled = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM games WHERE status IN ('phantom','postponed','cancelled')")
    total_other = c.fetchone()[0]
    
    # Duplicate check
    c.execute('''
        SELECT COUNT(*) FROM (
            SELECT home_team_id, away_team_id, home_score, away_score, date, COUNT(*) c
            FROM games WHERE status='final'
            GROUP BY home_team_id, away_team_id, home_score, away_score, date
            HAVING c > 1
        )
    ''')
    dupe_count = c.fetchone()[0]
    
    # Orphan team IDs
    c.execute('''
        SELECT COUNT(*) FROM (
            SELECT DISTINCT t.id FROM (
                SELECT home_team_id as id FROM games
                UNION SELECT away_team_id as id FROM games
            ) t LEFT JOIN teams ON teams.id = t.id
            WHERE teams.id IS NULL
        )
    ''')
    orphan_count = c.fetchone()[0]
    
    # Recent dates summary
    c.execute('''
        SELECT date,
            SUM(CASE WHEN status='final' THEN 1 ELSE 0 END) as final,
            SUM(CASE WHEN status='scheduled' THEN 1 ELSE 0 END) as scheduled,
            SUM(CASE WHEN status NOT IN ('final','scheduled') THEN 1 ELSE 0 END) as other
        FROM games
        WHERE date >= date('now', '-7 days') AND date <= date('now', '+1 day')
        GROUP BY date ORDER BY date
    ''')
    recent_dates = [dict(row) for row in c.fetchall()]
    
    conn.close()
    
    # Bug reports
    reports_path = base_dir / 'data' / 'bug_reports.json'
    bug_reports = []
    if reports_path.exists():
        with open(reports_path) as f:
            bug_reports = json.load(f)
    
    return render_template('debug.html',
        teams=teams_over_3,
        flags=flags,
        total_final=total_final,
        total_scheduled=total_scheduled,
        total_other=total_other,
        dupe_count=dupe_count,
        orphan_count=orphan_count,
        recent_dates=recent_dates,
        bug_reports=bug_reports
    )


@app.route('/api/debug/flag', methods=['POST'])
def api_debug_flag():
    """Flag a team as correct/incorrect for review."""
    data = request.get_json()
    team_id = data.get('team_id')
    flag = data.get('flag')  # 'correct', 'incorrect', or 'clear'
    note = data.get('note', '')
    
    if not team_id or not flag:
        return jsonify({'error': 'team_id and flag required'}), 400
    
    flags_path = base_dir / 'data' / 'debug_flags.json'
    flags = {}
    if flags_path.exists():
        with open(flags_path) as f:
            flags = json.load(f)
    
    if flag == 'clear':
        flags.pop(team_id, None)
    else:
        flags[team_id] = {
            'flag': flag,
            'note': note,
            'flagged_at': datetime.now().isoformat()
        }
    
    with open(flags_path, 'w') as f:
        json.dump(flags, f, indent=2)
    
    return jsonify({'ok': True, 'team_id': team_id, 'flag': flag})


@app.route('/api/bug-report', methods=['POST'])
def api_bug_report():
    """Submit a bug report."""
    data = request.get_json()
    description = data.get('description', '').strip()
    page = data.get('page', '')
    
    if not description:
        return jsonify({'error': 'Description required'}), 400
    
    reports_path = base_dir / 'data' / 'bug_reports.json'
    reports = []
    if reports_path.exists():
        with open(reports_path) as f:
            reports = json.load(f)
    
    reports.append({
        'id': len(reports) + 1,
        'description': description,
        'page': page,
        'status': 'open',
        'submitted_at': datetime.now().isoformat()
    })
    
    with open(reports_path, 'w') as f:
        json.dump(reports, f, indent=2)
    
    return jsonify({'ok': True, 'id': len(reports)})


@app.route('/api/bug-report/<int:bug_id>', methods=['PATCH'])
def api_bug_update(bug_id):
    """Update bug report status."""
    data = request.get_json()
    status = data.get('status', 'closed')
    
    reports_path = base_dir / 'data' / 'bug_reports.json'
    if not reports_path.exists():
        return jsonify({'error': 'No reports'}), 404
    
    with open(reports_path) as f:
        reports = json.load(f)
    
    for r in reports:
        if r['id'] == bug_id:
            r['status'] = status
            break
    
    with open(reports_path, 'w') as f:
        json.dump(reports, f, indent=2)
    
    return jsonify({'ok': True})


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
    app.run(host='0.0.0.0', port=5000, debug=False)
