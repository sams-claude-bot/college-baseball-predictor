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

# Models to use for consensus voting (11 component models, NOT ensemble)
# Ensemble is excluded because it's already a weighted blend of these models
CONSENSUS_MODELS = ['pythagorean', 'elo', 'log5', 'advanced', 'pitching',
                    'conference', 'prior', 'poisson', 'neural', 'xgboost', 'lightgbm']


# ============================================
# Odds/Probability Helpers
# ============================================

def american_to_implied_prob(american_odds):
    """Convert American odds to implied probability."""
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)


# Betting edge adjustment constants (from shared config)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.model_config import BET_UNDERDOG_DISCOUNT, BET_CONSENSUS_BONUS

UNDERDOG_EDGE_DISCOUNT = BET_UNDERDOG_DISCOUNT
CONSENSUS_BONUS_PER_MODEL = BET_CONSENSUS_BONUS


def calculate_adjusted_edge(raw_edge: float, moneyline: int = None, models_agree: int = 5) -> float:
    """
    Calculate adjusted betting edge with underdog discount and consensus bonus.
    
    Args:
        raw_edge: Raw edge percentage from model vs implied odds
        moneyline: American odds (positive = underdog)
        models_agree: Number of models agreeing on the pick (out of 11)
    
    Returns:
        Adjusted edge percentage
    """
    adj = raw_edge
    
    # Underdog discount: market is usually right about favorites
    if moneyline and moneyline > 0:
        adj = raw_edge * UNDERDOG_EDGE_DISCOUNT
    
    # Consensus bonus: +1% for each model above 5 agreeing (max +6%)
    bonus = max(0, (models_agree - 5)) * CONSENSUS_BONUS_PER_MODEL
    
    return adj + bonus


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
               g.home_score, g.away_score, g.winner_id, g.innings, g.inning_text,
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
        LEFT JOIN (SELECT *, MAX(id) as _latest FROM betting_lines WHERE book = 'draftkings' GROUP BY date, home_team_id, away_team_id) b ON g.home_team_id = b.home_team_id 
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
               e.rating as elo_rating,
               r.sams_rpi, r.sams_rank,
               ts.overall_sos
        FROM teams t
        LEFT JOIN elo_ratings e ON t.id = e.team_id
        LEFT JOIN team_rpi r ON t.id = r.team_id
        LEFT JOIN team_sos ts ON t.id = ts.team_id
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
        SELECT t.*, e.rating as elo_rating,
               r.sams_rpi, r.sams_rank, r.wp as rpi_wp, r.owp as rpi_owp, r.oowp as rpi_oowp,
               r.ncaa_rpi, r.ncaa_rank
        FROM teams t
        LEFT JOIN elo_ratings e ON t.id = e.team_id
        LEFT JOIN team_rpi r ON t.id = r.team_id
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

    # Assumed lineup: top 9 batters by at-bats
    c.execute('''
        SELECT name, position, at_bats, batting_avg, ops, home_runs, rbi,
               woba, wrc_plus
        FROM player_stats
        WHERE team_id = ? AND at_bats > 0
              AND COALESCE(position, '') NOT IN ('P', 'RHP', 'LHP')
              AND (innings_pitched IS NULL OR innings_pitched = 0 OR at_bats > 10)
        ORDER BY at_bats DESC
        LIMIT 9
    ''', (team_id,))
    team['assumed_lineup'] = [dict(r) for r in c.fetchall()]

    # Recent starting pitchers (rotation)
    c.execute('''
        SELECT starting_pitcher, game_date, opponent
        FROM lineup_history
        WHERE team_id = ?
        ORDER BY game_date DESC
        LIMIT 5
    ''', (team_id,))
    team['recent_starters'] = [dict(r) for r in c.fetchall()]

    # Projected starters from pitching_matchups
    today = datetime.now().strftime('%Y-%m-%d')
    c.execute('''
        SELECT pm.game_id, g.date,
               CASE WHEN g.home_team_id = ? THEN pm.home_starter_name ELSE pm.away_starter_name END as starter_name,
               CASE WHEN g.home_team_id = ? THEN pm.home_starter_id ELSE pm.away_starter_id END as starter_id,
               CASE WHEN g.home_team_id = ? THEN at.name ELSE ht.name END as opponent,
               CASE WHEN g.home_team_id = ? THEN 'vs' ELSE '@' END as location,
               pm.notes
        FROM pitching_matchups pm
        JOIN games g ON pm.game_id = g.id
        LEFT JOIN teams ht ON g.home_team_id = ht.id
        LEFT JOIN teams at ON g.away_team_id = at.id
        WHERE (g.home_team_id = ? OR g.away_team_id = ?)
        AND g.date >= ?
        AND g.status = 'scheduled'
        ORDER BY g.date
        LIMIT 5
    ''', (team_id, team_id, team_id, team_id, team_id, team_id, today))
    team['projected_starters'] = [dict(r) for r in c.fetchall()]

    conn.close()
    return team


def get_todays_games():
    """Get all games scheduled for today, with DK lines if available"""
    today = datetime.now().strftime('%Y-%m-%d')

    conn = get_connection()
    c = conn.cursor()

    # Games from schedule only — betting lines joined on when available
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
        LEFT JOIN (SELECT *, MAX(id) as _latest FROM betting_lines WHERE book = 'draftkings' GROUP BY date, home_team_id, away_team_id) b ON g.home_team_id = b.home_team_id 
            AND g.away_team_id = b.away_team_id AND g.date = b.date
        LEFT JOIN teams ht ON g.home_team_id = ht.id
        LEFT JOIN teams at ON g.away_team_id = at.id
        LEFT JOIN elo_ratings he ON g.home_team_id = he.team_id
        LEFT JOIN elo_ratings ae ON g.away_team_id = ae.team_id
        LEFT JOIN game_weather gw ON g.id = gw.game_id
        WHERE g.date = ?
        ORDER BY g.time, g.id
    ''', (today,))

    games = [dict(row) for row in c.fetchall()]
    conn.close()

    # Fill in missing team names from team IDs (for opponents not in teams table)
    for game in games:
        if not game.get('home_team_name') and game.get('home_team_id'):
            game['home_team_name'] = game['home_team_id'].replace('-', ' ').title()
        if not game.get('away_team_name') and game.get('away_team_id'):
            game['away_team_name'] = game['away_team_id'].replace('-', ' ').title()

    # Add stored predictions to each game (no live model calls)
    conn2 = get_connection()
    stored = {}
    for row in conn2.execute('''
        SELECT game_id, predicted_home_prob, predicted_home_runs, predicted_away_runs
        FROM model_predictions WHERE model_name = 'ensemble'
        AND game_id IN (SELECT id FROM games WHERE date = ?)
    ''', (today,)).fetchall():
        stored[row['game_id']] = row
    conn2.close()

    for game in games:
        sp = stored.get(game.get('id'))
        if sp:
            game['prediction'] = {
                'home_win_probability': sp['predicted_home_prob'],
                'away_win_probability': 1 - sp['predicted_home_prob'],
                'projected_home_runs': sp['predicted_home_runs'] or 0,
                'projected_away_runs': sp['predicted_away_runs'] or 0,
                'projected_total': (sp['predicted_home_runs'] or 0) + (sp['predicted_away_runs'] or 0)
            }
        else:
            game['prediction'] = None

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
        WHERE b.date >= ? AND b.date <= ? AND b.book = 'draftkings'
        ORDER BY b.date
        LIMIT 100
    ''', (today, three_days))

    lines = [dict(row) for row in c.fetchall()]
    conn.close()

    # Load stored predictions for value picks
    conn_vp = get_connection()
    stored_vp = {}
    stored_vp_agreement = {}
    for row in conn_vp.execute('''
        SELECT game_id, model_name, predicted_home_prob
        FROM model_predictions
        WHERE game_id IN (SELECT id FROM games WHERE date >= ? AND date <= ?)
    ''', (today, three_days)).fetchall():
        if row['model_name'] == 'ensemble':
            stored_vp[row['game_id']] = {'prob': row['predicted_home_prob']}
        if row['game_id'] not in stored_vp_agreement:
            stored_vp_agreement[row['game_id']] = {}
        stored_vp_agreement[row['game_id']][row['model_name']] = row['predicted_home_prob']
    conn_vp.close()

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

            # Use stored ensemble prediction
            game_id = line.get('game_id')
            stored_ens = stored_vp.get(game_id)
            if not stored_ens:
                continue
            model_home_prob = stored_ens['prob']

            # Model agreement from stored predictions
            game_models = stored_vp_agreement.get(game_id, {})
            ens_home = model_home_prob > 0.5
            models_agree = sum(1 for m, p in game_models.items() if m != 'ensemble' and (p > 0.5) == ens_home) if game_models else 5

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
                'models_total': sum(1 for m in game_models if m != 'ensemble') if game_models else 11,
                'ensemble_confidence': max(model_home_prob, 1 - model_home_prob),
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

    # Query DraftKings lines
    c.execute('''
        SELECT b.*, 
               ht.name as home_team_name, ht.current_rank as home_rank, ht.conference as home_conf,
               at.name as away_team_name, at.current_rank as away_rank, at.conference as away_conf,
               g.status, g.home_score, g.away_score
        FROM betting_lines b
        LEFT JOIN teams ht ON b.home_team_id = ht.id
        LEFT JOIN teams at ON b.away_team_id = at.id
        LEFT JOIN games g ON b.game_id = g.id
        WHERE b.date = ? AND b.book = 'draftkings'
        ORDER BY b.captured_at DESC
    ''', (today,))
    dk_lines = {row['game_id']: dict(row) for row in c.fetchall()}

    # Query FanDuel lines
    c.execute('''
        SELECT b.home_team_id, b.away_team_id, b.game_id,
               b.home_ml as fd_home_ml, b.away_ml as fd_away_ml,
               b.over_under as fd_over_under, b.over_odds as fd_over_odds,
               b.under_odds as fd_under_odds
        FROM betting_lines b
        WHERE b.date = ? AND b.book = 'fanduel'
    ''', (today,))
    fd_lines = {row['game_id']: dict(row) for row in c.fetchall()}

    # Merge: start with DK as base, attach FD fields
    # Skip final games - they shouldn't appear on betting page
    lines = []
    for game_id, line in dk_lines.items():
        if line.get('status') == 'final':
            continue  # Don't show completed games
        fd = fd_lines.get(game_id, {})
        line['fd_home_ml'] = fd.get('fd_home_ml')
        line['fd_away_ml'] = fd.get('fd_away_ml')
        line['fd_over_under'] = fd.get('fd_over_under')
        line['fd_over_odds'] = fd.get('fd_over_odds')
        line['fd_under_odds'] = fd.get('fd_under_odds')
        lines.append(line)

    # Also add FD-only games (no DK line)
    for game_id, fd in fd_lines.items():
        if game_id not in dk_lines:
            # Need full game info for FD-only lines
            c.execute('''
                SELECT b.*, 
                       ht.name as home_team_name, ht.current_rank as home_rank, ht.conference as home_conf,
                       at.name as away_team_name, at.current_rank as away_rank, at.conference as away_conf,
                       g.status, g.home_score, g.away_score
                FROM betting_lines b
                LEFT JOIN teams ht ON b.home_team_id = ht.id
                LEFT JOIN teams at ON b.away_team_id = at.id
                LEFT JOIN games g ON b.game_id = g.id
                WHERE b.game_id = ? AND b.book = 'fanduel'
            ''', (game_id,))
            row = c.fetchone()
            if row:
                line = dict(row)
                if line.get('status') == 'final':
                    continue  # Don't show completed games
                # Copy ML/OU from FD into the main fields (used for edge calc)
                line['fd_home_ml'] = line['home_ml']
                line['fd_away_ml'] = line['away_ml']
                line['fd_over_under'] = line['over_under']
                line['fd_over_odds'] = line.get('over_odds')
                line['fd_under_odds'] = line.get('under_odds')
                lines.append(line)

    conn.close()

    # Load stored pre-game predictions (calculated once daily by predict_and_track)
    conn_sp = get_connection()
    sp_rows = conn_sp.execute('''
        SELECT game_id, model_name, predicted_home_prob, predicted_home_runs, predicted_away_runs
        FROM model_predictions
        WHERE game_id IN (SELECT id FROM games WHERE date = ?)
    ''', (today,)).fetchall()
    stored_preds = {}
    stored_model_agreement = {}
    for row in sp_rows:
        stored_preds[(row['game_id'], row['model_name'])] = {
            'prob': row['predicted_home_prob'],
            'home_runs': row['predicted_home_runs'] or 0,
            'away_runs': row['predicted_away_runs'] or 0
        }
        # Track all models per game for agreement
        if row['game_id'] not in stored_model_agreement:
            stored_model_agreement[row['game_id']] = {}
        stored_model_agreement[row['game_id']][row['model_name']] = row['predicted_home_prob']
    
    tp_rows = conn_sp.execute('''
        SELECT game_id, projected_total FROM totals_predictions
        WHERE model_name = 'runs_ensemble'
    ''').fetchall()
    stored_totals = {row['game_id']: row['projected_total'] for row in tp_rows}
    conn_sp.close()

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

            # Model prediction — use stored pre-game predictions
            game_id = line.get('game_id')
            stored_ens = stored_preds.get((game_id, 'ensemble'))
            stored_nn = stored_preds.get((game_id, 'neural'))
            
            if stored_ens:
                line['model_home_prob'] = stored_ens['prob']
                line['model_away_prob'] = 1 - stored_ens['prob']
                line['projected_total'] = stored_ens['home_runs'] + stored_ens['away_runs']
                line['ens_prob'] = stored_ens['prob']
            if stored_nn:
                line['nn_prob'] = stored_nn['prob']
            
            if not stored_ens:
                continue
            
            line['blend_info'] = 'pre-game'

            # Model consensus from stored predictions
            game_models = stored_model_agreement.get(game_id, {})
            if game_models:
                ens_home = stored_ens['prob'] > 0.5
                models_for = [m for m, p in game_models.items() if (p > 0.5) == ens_home and m != 'ensemble']
                models_against = [m for m, p in game_models.items() if (p > 0.5) != ens_home and m != 'ensemble']
                all_probs = [p if ens_home else (1-p) for m, p in game_models.items() if m != 'ensemble']
                avg_prob = sum(all_probs) / len(all_probs) if all_probs else 0.5
                line['model_agreement'] = {
                    'count': len(models_for),
                    'total': len(models_for) + len(models_against),
                    'pick': 'home' if ens_home else 'away',
                    'avg_prob': avg_prob,
                    'confidence': avg_prob,
                    'models_for': models_for,
                    'models_against': models_against
                }

            # Edges
            line['home_edge'] = (line['model_home_prob'] - line['dk_home_fair']) * 100
            line['away_edge'] = (line['model_away_prob'] - line['dk_away_fair']) * 100

            # Best pick
            if line['home_edge'] >= 0:
                line['best_pick'] = 'home'
                line['best_edge'] = line['home_edge']
            else:
                line['best_pick'] = 'away'
                line['best_edge'] = abs(line['away_edge'])

            # Totals analysis — use stored runs ensemble prediction
            if line['over_under']:
                stored_total = stored_totals.get(game_id)
                if stored_total:
                    line['projected_total'] = stored_total
                    line['total_diff'] = stored_total - line['over_under']
                else:
                    line['total_diff'] = line['projected_total'] - line['over_under']
                line['total_lean'] = 'OVER' if line['total_diff'] > 0 else 'UNDER'
                line['total_edge'] = min(abs(line['total_diff']) * 8, 50)

            # EV calculation (per $100)
            if line['best_pick'] == 'home':
                ml = line['home_ml']
                prob = line['model_home_prob']
            else:
                ml = line['away_ml']
                prob = line['model_away_prob']

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
               g.home_score, g.away_score, g.winner_id, g.innings, g.inning_text,
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

    # Add stored predictions — no live model calls
    conn_rr = get_connection()
    stored_rr = {}
    for row in conn_rr.execute('''
        SELECT game_id, predicted_home_prob FROM model_predictions
        WHERE model_name = 'ensemble'
        AND game_id IN (SELECT id FROM games WHERE date >= ? AND date < ?)
    ''', (start_date, today_str)).fetchall():
        stored_rr[row['game_id']] = row['predicted_home_prob']
    conn_rr.close()

    for game in games:
        game['is_upset'] = False
        try:
            sp = stored_rr.get(game.get('id'))
            if sp is not None:
                game['pred_home_prob'] = sp
                game['pred_winner'] = game['home_team_id'] if sp > 0.5 else game['away_team_id']
                game['pred_confidence'] = max(sp, 1 - sp)

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
               g.home_score, g.away_score, g.winner_id, g.innings, g.inning_text,
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

    # Fill in missing team names from team IDs
    for game in games:
        if not game.get('home_team_name') and game.get('home_team_id'):
            game['home_team_name'] = game['home_team_id'].replace('-', ' ').title()
        if not game.get('away_team_name') and game.get('away_team_id'):
            game['away_team_name'] = game['away_team_id'].replace('-', ' ').title()

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

    # Use stored predictions only — no live model calls
    correct_count = 0
    total_preds = 0

    for game in games:
        game['is_upset'] = False
        game_id = game['id']

        # Use stored pre-game prediction
        stored_ens = stored_preds.get((game_id, 'ensemble'))

        if stored_ens is not None:
            game['pred_home_prob'] = stored_ens
            game['pred_winner'] = game['home_team_id'] if stored_ens > 0.5 else game['away_team_id']
            game['pred_confidence'] = max(stored_ens, 1 - stored_ens)
            game['pred_source'] = 'pre-game'
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
