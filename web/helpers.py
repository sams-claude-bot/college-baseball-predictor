"""
Shared helper functions and utilities for the College Baseball Dashboard.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

# Add paths for imports
base_dir = Path(__file__).parent.parent
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / "scripts"))

from database import (
    get_connection, get_team_record, get_team_runs,
    get_recent_games, get_upcoming_games, get_current_top_25
)
from models.compare_models import MODELS, normalize_team_id

# ============================================
# Constants
# ============================================

# Models to use for consensus voting (12 total - includes XGBoost and LightGBM)
CONSENSUS_MODELS = ['pythagorean', 'elo', 'log5', 'advanced', 'pitching',
                    'conference', 'prior', 'poisson', 'neural', 'xgboost', 'lightgbm', 'ensemble']


# ============================================
# Odds/Probability Helpers
# ============================================

def american_to_implied_prob(american_odds):
    """Convert American odds to implied probability."""
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)


# ============================================
# Database Query Helpers
# ============================================

def get_all_conferences():
    """Get all unique conferences"""
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        SELECT DISTINCT conference FROM teams 
        WHERE conference IS NOT NULL AND conference != '' AND conference != 'Non-D1'
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
        WHERE t.conference != 'Non-D1'
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

    # Get Elo history (game-by-game)
    c.execute('''
        SELECT eh.rating, eh.game_date, eh.opponent_id, eh.rating_change,
               t.name as opponent_name
        FROM elo_history eh
        LEFT JOIN teams t ON eh.opponent_id = t.id
        WHERE eh.team_id = ?
        ORDER BY eh.game_date, eh.id
    ''', (team_id,))
    team['elo_history'] = [dict(r) for r in c.fetchall()]

    # Add starting point at 1500
    if team['elo_history']:
        first_date = team['elo_history'][0]['game_date']
        team['elo_history'].insert(0, {
            'rating': 1500, 'game_date': first_date,
            'opponent_name': 'Start', 'rating_change': 0
        })

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
               he.rating as home_elo, ae.rating as away_elo,
               gw.temp_f as temperature, gw.wind_speed_mph as wind_speed, gw.wind_direction_deg as wind_direction, 
               gw.humidity_pct as humidity, gw.precip_prob_pct as precipitation_prob
        FROM games g
        LEFT JOIN betting_lines b ON g.home_team_id = b.home_team_id 
            AND g.away_team_id = b.away_team_id AND g.date = b.date
        LEFT JOIN teams ht ON g.home_team_id = ht.id
        LEFT JOIN teams at ON g.away_team_id = at.id
        LEFT JOIN elo_ratings he ON g.home_team_id = he.team_id
        LEFT JOIN elo_ratings ae ON g.away_team_id = ae.team_id
        LEFT JOIN game_weather gw ON g.id = gw.game_id
        WHERE g.date = ?

        UNION

        SELECT b.game_id as id, b.date, NULL as time, NULL as status,
               b.home_team_id, b.away_team_id,
               NULL as home_score, NULL as away_score,
               b.home_ml, b.away_ml, b.over_under,
               b.home_spread as run_line, b.home_spread_odds as run_line_odds,
               ht.name as home_team_name, ht.current_rank as home_rank, ht.conference as home_conf,
               at.name as away_team_name, at.current_rank as away_rank, at.conference as away_conf,
               he.rating as home_elo, ae.rating as away_elo,
               NULL as temperature, NULL as wind_speed, NULL as wind_direction, NULL as humidity, NULL as precipitation_prob
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
    """Get top value picks using v2 logic with adjusted edges.

    v2 changes:
    - Underdog edges discounted 50%
    - Consensus bonus: +1% per model above 5
    - Higher thresholds: 8% favorites, 15% underdogs
    """
    # Edge thresholds
    ML_EDGE_FAVORITE = 8.0
    CONSENSUS_BONUS_PER_MODEL = 1.0

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

            # Get model agreement for consensus bonus
            agreement = compute_model_agreement(line['home_team_id'], line['away_team_id'])
            models_agree = agreement.get('count', 5) if agreement else 5

            # Calculate edge
            home_edge = (model_home_prob - dk_home_fair) * 100
            away_edge = ((1 - model_home_prob) - (1 - dk_home_fair)) * 100

            # Determine best pick - pick the side with POSITIVE edge
            # (home_edge and away_edge always sum to ~0, so one is positive, one negative)
            if home_edge > 0:
                raw_edge = home_edge
                best_pick = line['home_team_name']
                best_ml = line['home_ml']
                model_prob = model_home_prob
                dk_implied = dk_home_fair
            else:
                raw_edge = away_edge  # This will be positive when home_edge is negative
                best_pick = line['away_team_name']
                best_ml = line['away_ml']
                model_prob = 1 - model_home_prob
                dk_implied = 1 - dk_home_fair

            is_underdog = best_ml > 0

            # Raw edge — no discount, no bonus
            adjusted_edge = raw_edge
            consensus_bonus = max(0, (models_agree - 5)) * CONSENSUS_BONUS_PER_MODEL

            # Single threshold for all picks
            if raw_edge < ML_EDGE_FAVORITE:
                continue

            picks.append({
                'date': line['date'],
                'game': f"{line['away_team_name']} @ {line['home_team_name']}",
                'pick': best_pick,
                'edge': raw_edge,
                'adjusted_edge': adjusted_edge,
                'moneyline': best_ml,
                'model_prob': model_prob,
                'dk_implied': dk_implied,
                'home_team_id': line['home_team_id'],
                'away_team_id': line['away_team_id'],
                'models_agree': models_agree,
                'is_underdog': is_underdog,
                'consensus_bonus': consensus_bonus
            })
        except Exception:
            continue

    # Sort by raw edge
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


def compute_model_agreement(home_team_id, away_team_id):
    """
    Run all consensus models and compute agreement/confidence scores.

    Returns dict with:
        pick: 'home' or 'away' (consensus pick)
        count: number of models agreeing on pick
        total: total models that returned predictions
        avg_prob: average probability for the consensus pick
        confidence: agreement_pct * avg_prob (0-1 scale)
        models_for: list of model names picking the consensus side
        models_against: list of model names picking the opposite side
    """
    home_votes = []
    away_votes = []

    for model_name in CONSENSUS_MODELS:
        model = MODELS.get(model_name)
        if not model:
            continue
        try:
            pred = model.predict_game(home_team_id, away_team_id)
            if pred and 'home_win_probability' in pred:
                prob = pred['home_win_probability']
                if prob > 0.5:
                    home_votes.append((model_name, prob))
                else:
                    away_votes.append((model_name, 1 - prob))
        except Exception:
            pass  # Skip models that fail for this matchup

    total = len(home_votes) + len(away_votes)
    if total == 0:
        return None

    # Determine consensus
    if len(home_votes) >= len(away_votes):
        pick = 'home'
        count = len(home_votes)
        models_for = [m[0] for m in home_votes]
        models_against = [m[0] for m in away_votes]
        avg_prob = sum(m[1] for m in home_votes) / len(home_votes) if home_votes else 0.5
    else:
        pick = 'away'
        count = len(away_votes)
        models_for = [m[0] for m in away_votes]
        models_against = [m[0] for m in home_votes]
        avg_prob = sum(m[1] for m in away_votes) / len(away_votes) if away_votes else 0.5

    agreement_pct = count / total
    confidence = agreement_pct * avg_prob

    return {
        'pick': pick,
        'count': count,
        'total': total,
        'avg_prob': avg_prob,
        'confidence': confidence,
        'models_for': models_for,
        'models_against': models_against
    }


def get_betting_games(date_str=None):
    """Get all games with betting lines for a given date (defaults to today)"""
    conn = get_connection()
    c = conn.cursor()

    today = date_str or datetime.now().strftime('%Y-%m-%d')

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

            # Model consensus (run all 10 models)
            agreement = compute_model_agreement(line['home_team_id'], line['away_team_id'])
            if agreement:
                line['model_agreement'] = agreement

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

            # Totals analysis — use dedicated runs ensemble
            if line['over_under']:
                try:
                    import models.runs_ensemble as runs_ens
                    runs_result = runs_ens.predict(line['home_team_id'], line['away_team_id'], total_line=line['over_under'])
                    line['projected_total'] = runs_result['projected_total']
                    line['total_diff'] = runs_result['projected_total'] - line['over_under']
                    line['total_lean'] = 'OVER' if line['total_diff'] > 0 else 'UNDER'
                    ou = runs_result.get('over_under', {})
                    line['total_edge'] = ou.get('edge', min(abs(line['total_diff']) * 8, 50))
                    line['runs_breakdown'] = runs_result.get('model_breakdown', {})
                except Exception:
                    line['total_diff'] = pred['projected_total'] - line['over_under']
                    line['total_lean'] = 'OVER' if line['total_diff'] > 0 else 'UNDER'
                    line['total_edge'] = min(abs(line['total_diff']) * 8, 50)

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

    # Filter out games where either team name is missing (non-tracked teams)
    lines = [l for l in lines if l.get('home_team_name') and l.get('away_team_name')]

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

    # All-time accuracy
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
            'recent_accuracy': None,
            'recent_predictions': 0,
            'current_weight': MODELS.get('ensemble').weights.get(model_name, 0) if MODELS.get('ensemble') else 0
        }

    # Rolling 7-day accuracy
    c.execute('''
        SELECT mp.model_name,
               SUM(mp.was_correct) as correct,
               COUNT(*) as total
        FROM model_predictions mp
        JOIN games g ON mp.game_id = g.id
        WHERE mp.was_correct IS NOT NULL
        AND g.date >= date('now', '-7 days')
        GROUP BY mp.model_name
    ''')

    for row in c.fetchall():
        model_name, correct, total = row
        if model_name in result:
            result[model_name]['recent_accuracy'] = correct / total if total > 0 else None
            result[model_name]['recent_predictions'] = total

    conn.close()
    return result


def get_recent_results(days_back=3):
    """Get completed games from the last N days with scores and prediction accuracy"""
    conn = get_connection()
    c = conn.cursor()

    today = datetime.now()
    start_date = (today - timedelta(days=days_back)).strftime('%Y-%m-%d')
    today_str = today.strftime('%Y-%m-%d')

    c.execute('''
        SELECT g.id, g.date, g.time, g.status,
               g.home_team_id, g.away_team_id,
               g.home_score, g.away_score, g.winner_id, g.innings,
               g.is_conference_game,
               ht.name as home_team_name, ht.current_rank as home_rank, ht.conference as home_conf,
               at.name as away_team_name, at.current_rank as away_rank, at.conference as away_conf,
               he.rating as home_elo, ae.rating as away_elo
        FROM games g
        LEFT JOIN teams ht ON g.home_team_id = ht.id
        LEFT JOIN teams at ON g.away_team_id = at.id
        LEFT JOIN elo_ratings he ON g.home_team_id = he.team_id
        LEFT JOIN elo_ratings ae ON g.away_team_id = ae.team_id
        WHERE g.date >= ? AND g.date < ? AND g.status = 'final'
        ORDER BY g.date DESC, g.time DESC, g.id DESC
    ''', (start_date, today_str))

    games = [dict(row) for row in c.fetchall()]
    conn.close()

    # Add predictions and check correctness
    ensemble = MODELS.get('ensemble')

    for game in games:
        game['is_upset'] = False
        try:
            if ensemble and game.get('home_team_id') and game.get('away_team_id'):
                pred = ensemble.predict_game(game['home_team_id'], game['away_team_id'])
                game['pred_home_prob'] = pred.get('home_win_probability', 0.5)
                game['pred_winner'] = game['home_team_id'] if game['pred_home_prob'] > 0.5 else game['away_team_id']
                game['pred_confidence'] = max(game['pred_home_prob'], 1 - game['pred_home_prob'])

                # Check if prediction was correct
                if game.get('winner_id'):
                    game['pred_correct'] = game['pred_winner'] == game['winner_id']

                    # Check for upset (lower rank beat higher rank)
                    winner_rank = game['home_rank'] if game['winner_id'] == game['home_team_id'] else game['away_rank']
                    loser_rank = game['away_rank'] if game['winner_id'] == game['home_team_id'] else game['home_rank']
                    if winner_rank and loser_rank and winner_rank > loser_rank:
                        game['is_upset'] = True
                    # Unranked beat ranked
                    elif loser_rank and not winner_rank:
                        game['is_upset'] = True
        except Exception:
            game['pred_winner'] = None
            game['pred_correct'] = None

    # Group by date
    by_date = defaultdict(list)
    for game in games:
        by_date[game['date']].append(game)

    # Calculate accuracy per date
    results_by_date = []
    for date in sorted(by_date.keys(), reverse=True):
        date_games = by_date[date]
        correct = sum(1 for g in date_games if g.get('pred_correct'))
        total_with_preds = sum(1 for g in date_games if g.get('pred_correct') is not None)

        results_by_date.append({
            'date': date,
            'display_date': datetime.strptime(date, '%Y-%m-%d').strftime('%A, %b %d'),
            'games': date_games,
            'correct': correct,
            'total': total_with_preds,
            'accuracy': round(correct / total_with_preds * 100, 1) if total_with_preds > 0 else None
        })

    return results_by_date


def get_games_for_date_with_predictions(date_str):
    """Get all games for a date with predictions - for scores page"""
    conn = get_connection()
    c = conn.cursor()

    c.execute('''
        SELECT g.id, g.date, g.time, g.status,
               g.home_team_id, g.away_team_id,
               g.home_score, g.away_score, g.winner_id, g.innings,
               g.is_conference_game,
               ht.name as home_team_name, ht.current_rank as home_rank, ht.conference as home_conf,
               at.name as away_team_name, at.current_rank as away_rank, at.conference as away_conf,
               he.rating as home_elo, ae.rating as away_elo
        FROM games g
        LEFT JOIN teams ht ON g.home_team_id = ht.id
        LEFT JOIN teams at ON g.away_team_id = at.id
        LEFT JOIN elo_ratings he ON g.home_team_id = he.team_id
        LEFT JOIN elo_ratings ae ON g.away_team_id = ae.team_id
        WHERE g.date = ?
        ORDER BY g.time, g.id
    ''', (date_str,))

    games = [dict(row) for row in c.fetchall()]
    conn.close()

    # Load pre-game predictions from model_predictions table (recorded BEFORE games)
    conn2 = get_connection()
    c2 = conn2.cursor()
    c2.execute('''
        SELECT game_id, model_name, predicted_home_prob
        FROM model_predictions
        WHERE model_name IN ('ensemble', 'neural')
          AND game_id IN (SELECT id FROM games WHERE date = ?)
    ''', (date_str,))
    stored_preds = {}
    for row in c2.fetchall():
        key = (row['game_id'], row['model_name'])
        stored_preds[key] = row['predicted_home_prob']
    conn2.close()

    # Fall back to live predictions only for scheduled (future) games
    ensemble = MODELS.get('ensemble')
    correct_count = 0
    total_preds = 0

    for game in games:
        game['is_upset'] = False
        game_id = game['id']

        # Try stored pre-game prediction first
        stored_ens = stored_preds.get((game_id, 'ensemble'))

        if stored_ens is not None:
            # Use the pre-game prediction (honest accuracy)
            game['pred_home_prob'] = stored_ens
            game['pred_winner'] = game['home_team_id'] if stored_ens > 0.5 else game['away_team_id']
            game['pred_confidence'] = max(stored_ens, 1 - stored_ens)
            game['pred_source'] = 'pre-game'
        elif ensemble and game.get('home_team_id') and game.get('away_team_id'):
            # No stored prediction — use live model (for future/untracked games)
            try:
                pred = ensemble.predict_game(game['home_team_id'], game['away_team_id'])
                game['pred_home_prob'] = pred.get('home_win_probability', 0.5)
                game['pred_winner'] = game['home_team_id'] if game['pred_home_prob'] > 0.5 else game['away_team_id']
                game['pred_confidence'] = max(game['pred_home_prob'], 1 - game['pred_home_prob'])
                game['pred_source'] = 'live'
            except Exception:
                game['pred_winner'] = None
                game['pred_correct'] = None
                continue
        else:
            game['pred_winner'] = None
            game['pred_correct'] = None
            continue

        if game['status'] == 'final' and game.get('winner_id'):
            game['pred_correct'] = game['pred_winner'] == game['winner_id']
            total_preds += 1
            if game['pred_correct']:
                correct_count += 1

            # Check for upset
            winner_rank = game['home_rank'] if game['winner_id'] == game['home_team_id'] else game['away_rank']
            loser_rank = game['away_rank'] if game['winner_id'] == game['home_team_id'] else game['home_rank']
            if winner_rank and loser_rank and winner_rank > loser_rank:
                game['is_upset'] = True
            elif loser_rank and not winner_rank:
                game['is_upset'] = True

    return games, correct_count, total_preds


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
