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
from models.calibration import Calibrator

# Lazy-loaded calibrator for betting page
_api_calibrator = None

def _get_calibrator():
    global _api_calibrator
    if _api_calibrator is None:
        cal = Calibrator()
        if cal._load():
            _api_calibrator = cal
        else:
            _api_calibrator = False
    return _api_calibrator if _api_calibrator is not False else None

from web.helpers import (
    get_all_teams, get_betting_games,
    american_to_implied_prob, compute_model_agreement,
    calculate_adjusted_edge, UNDERDOG_EDGE_DISCOUNT
)
from web.bet_quality import (
    passes_quality_gate, bet_quality_score, has_vegas_disagreement,
    vegas_implied_prob, MAX_PER_TYPE, get_meta_ensemble_prob
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
        WHERE home_team_id = ? AND away_team_id = ? AND book = 'draftkings'
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
    """Return best bets for a date with quality-gated selection.

    v3 Logic (2026-02-24 overhaul based on -$742 P&L analysis):
    - NO underdog bets (2W-6L, 25% win rate, -$325)
    - Require model prob >= 65%
    - Skip when model disagrees with Vegas by >25pp (model error, not value)
    - Require 5pp+ margin over breakeven
    - Consensus: 8+ models must agree
    - EV: 10%+ edge required
    - Max 4 bets per category (quality > quantity)
    - Quality score ranking instead of raw edge

    Query param: ?date=YYYY-MM-DD (defaults to today)
    """
    date_str = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    games = get_betting_games(date_str)

    SPREADS_ENABLED = False

    # Filter to only games with model predictions
    games = [g for g in games if g.get('best_pick')]

    # Build consensus lookup
    consensus_lookup = {}
    for g in games:
        if g.get('model_agreement') and g['model_agreement'].get('count', 0) >= 7:
            consensus_lookup[g['game_id']] = g['model_agreement']['count']

    # --- CONFIDENT BETS (Model Consensus) with quality gates ---
    def _pick_ml(g):
        pick = g.get('model_agreement', {}).get('pick', 'home')
        return g.get('home_ml') if pick == 'home' else g.get('away_ml')

    confident_candidates = [g for g in games
                           if g.get('model_agreement')
                           and g['model_agreement']['count'] >= 7]

    confident_bets = []
    confident_rejections = []
    for g in confident_candidates:
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

        # Try to get meta_ensemble probability
        meta_prob = get_meta_ensemble_prob(g['game_id'], pick_id)
        if meta_prob is not None:
            # Adjust for pick direction
            if agreement['pick'] == 'away':
                meta_prob = 1.0 - meta_prob

        bet_info = {
            'game_id': g['game_id'], 'date': g['date'],
            'pick_team_id': pick_id, 'pick_team_name': pick_name,
            'opponent_name': opp_name, 'is_home': is_home,
            'moneyline': ml,
            'models_agree': agreement['count'],
            'models_total': agreement['total'],
            'avg_prob': round(agreement['avg_prob'], 4),
            'meta_prob': round(meta_prob, 4) if meta_prob else None,
            'confidence': round(agreement['confidence'], 4),
            'models_for': agreement['models_for'],
            'models_against': agreement['models_against']
        }

        # Apply quality gate
        passes, reason = passes_quality_gate(bet_info, category='consensus')

        # Add Vegas disagreement flag
        prob_for_check = meta_prob or agreement['avg_prob']
        disagrees, mp, vi, diff = has_vegas_disagreement(prob_for_check, ml)
        bet_info['vegas_disagreement'] = disagrees
        bet_info['vegas_implied'] = round(vi, 4) if vi else None
        bet_info['disagreement_pp'] = round(diff * 100, 1)

        if passes:
            bet_info['quality_score'] = bet_quality_score(bet_info, 'consensus')
            # Add calibrated probability + edge
            cal = _get_calibrator()
            if cal and ml:
                raw_p = meta_prob or agreement['avg_prob']
                cal_p = cal.calibrate(raw_p)
                implied = abs(ml)/(abs(ml)+100) if ml < 0 else 100/(100+ml)
                bet_info['calibrated_prob'] = round(cal_p, 4)
                bet_info['calibrated_edge'] = round((cal_p - implied) * 100, 1)
            confident_bets.append(bet_info)
        else:
            bet_info['rejection_reason'] = reason
            confident_rejections.append(bet_info)

    # Sort by quality score, take top MAX_PER_TYPE
    confident_bets.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
    confident_bets = confident_bets[:MAX_PER_TYPE]

    # --- EV MONEYLINE BETS with quality gates ---
    ml_bets = []
    ml_rejections = []
    confident_ids = {g['game_id'] for g in confident_bets}

    for g in games:
        raw_edge = g.get('best_edge', 0)
        if g['best_pick'] == 'home':
            pick_id = g['home_team_id']
            pick_name = g['home_team_name']
            opp_name = g['away_team_name']
            ml = g.get('home_ml')
            prob = g.get('model_home_prob', 0.5)
            dk_imp = g.get('dk_home_fair', 0.5)
        else:
            pick_id = g['away_team_id']
            pick_name = g['away_team_name']
            opp_name = g['home_team_name']
            ml = g.get('away_ml')
            prob = g.get('model_away_prob', 0.5)
            dk_imp = g.get('dk_away_fair', 0.5)

        if ml is None or g['game_id'] in confident_ids:
            continue

        # Get meta_ensemble probability
        meta_prob = get_meta_ensemble_prob(g['game_id'], pick_id)
        if meta_prob is not None and g['best_pick'] == 'away':
            meta_prob = 1.0 - meta_prob

        models = consensus_lookup.get(g['game_id'], 5)

        bet_info = {
            'game_id': g['game_id'], 'date': g['date'],
            'pick_team_id': pick_id, 'pick_team_name': pick_name,
            'opponent_name': opp_name, 'is_home': 1 if g['best_pick'] == 'home' else 0,
            'moneyline': ml, 'model_prob': round(prob, 4),
            'meta_prob': round(meta_prob, 4) if meta_prob else None,
            'dk_implied': round(dk_imp, 4),
            'edge': round(raw_edge, 2),
            'models_agree': models,
        }

        # Apply quality gate
        passes, reason = passes_quality_gate(bet_info, category='ev')

        # Add Vegas disagreement flag
        prob_for_check = meta_prob or prob
        disagrees, mp, vi, diff = has_vegas_disagreement(prob_for_check, ml)
        bet_info['vegas_disagreement'] = disagrees
        bet_info['vegas_implied'] = round(vi, 4) if vi else None
        bet_info['disagreement_pp'] = round(diff * 100, 1)
        bet_info['is_underdog'] = ml > 0 if ml else False

        if passes:
            bet_info['quality_score'] = bet_quality_score(bet_info, 'ev')
            # Add calibrated probability + edge
            cal = _get_calibrator()
            if cal and ml:
                raw_p = meta_prob or prob
                cal_p = cal.calibrate(raw_p)
                implied = abs(ml)/(abs(ml)+100) if ml < 0 else 100/(100+ml)
                bet_info['calibrated_prob'] = round(cal_p, 4)
                bet_info['calibrated_edge'] = round((cal_p - implied) * 100, 1)
            ml_bets.append(bet_info)
        else:
            bet_info['rejection_reason'] = reason
            ml_rejections.append(bet_info)

    ml_bets.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
    ml_bets = ml_bets[:MAX_PER_TYPE]

    # --- TOTALS with quality gates ---
    totals_bets = []
    totals_rejections = []
    for g in games:
        if not g.get('over_under') or not g.get('total_diff'):
            continue
        pick = g.get('total_lean', 'UNDER')
        odds = g.get('under_odds', -110) if pick == 'UNDER' else g.get('over_odds', -110)

        bet_info = {
            'game_id': g['game_id'], 'date': g['date'],
            'pick': pick, 'line': g['over_under'],
            'odds': odds or -110,
            'model_projection': round(g.get('projected_total', 0), 2),
            'edge': round(abs(g.get('total_diff', 0)), 2),
            'over_prob': g.get('over_prob'),
            'under_prob': g.get('under_prob'),
        }

        passes, reason = passes_quality_gate(bet_info, category='totals')
        if passes:
            totals_bets.append(bet_info)
        else:
            bet_info['rejection_reason'] = reason
            totals_rejections.append(bet_info)

    totals_bets.sort(key=lambda x: x.get('edge', 0), reverse=True)
    totals_bets = totals_bets[:MAX_PER_TYPE]

    # --- SPREADS (still disabled) ---
    best_spreads = []

    return jsonify({
        'date': date_str,
        'version': 3,
        'quality_gates': {
            'max_per_type': MAX_PER_TYPE,
            'no_underdogs': True,
            'min_model_prob': 0.65,
            'max_vegas_disagreement_pp': 25,
            'min_margin_pp': 5,
            'consensus_min_models': 8,
            'ev_min_edge': 10.0,
        },
        'confident_bets': confident_bets,
        'confident_rejections': confident_rejections[:6],  # Show top rejections for transparency
        'moneylines': ml_bets,
        'ml_rejections': ml_rejections[:6],
        'totals': totals_bets,
        'totals_rejections': totals_rejections[:6],
        'spreads': best_spreads,
        'spreads_disabled_reason': 'Model not calibrated (0/5 historical)'
    })


@api_bp.route('/api/live-scores')
def live_scores():
    """Lightweight endpoint for live score polling — returns only in-progress and recently-finished games."""
    from datetime import datetime
    date_str = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    
    conn = get_connection()
    rows = conn.execute('''
        SELECT g.id, g.status, g.home_score, g.away_score, g.inning_text, g.innings,
               g.winner_id, g.home_team_id, g.away_team_id,
               g.home_hits, g.away_hits, g.home_errors, g.away_errors,
               g.situation_json
        FROM games g
        WHERE g.date = ?
    ''', (date_str,)).fetchall()
    conn.close()
    
    import json as _json
    games = {}
    has_live = False
    for r in rows:
        game_data = {
            'status': r['status'],
            'home_score': r['home_score'],
            'away_score': r['away_score'],
            'inning_text': r['inning_text'],
            'innings': r['innings'],
            'winner_id': r['winner_id'],
            'home_hits': r['home_hits'],
            'away_hits': r['away_hits'],
            'home_errors': r['home_errors'],
            'away_errors': r['away_errors'],
        }
        # Parse situation for live games
        if r['situation_json']:
            try:
                game_data['situation'] = _json.loads(r['situation_json'])
            except:
                game_data['situation'] = None
        else:
            game_data['situation'] = None
        games[r['id']] = game_data
        if r['status'] == 'in-progress':
            has_live = True
    
    return jsonify({'games': games, 'has_live': has_live, 'date': date_str})


@api_bp.route('/api/elo-card')
def elo_card_image():
    """Generate a shareable Elo Top 25 PNG image."""
    from PIL import Image, ImageDraw, ImageFont
    from io import BytesIO
    from flask import send_file
    
    conn = get_connection()
    
    # Get Elo Top 25
    rows = conn.execute('''
        SELECT t.id as team_id, t.name, t.conference, e.rating,
               (SELECT COUNT(*) FROM games WHERE (home_team_id = t.id AND winner_id = t.id) OR (away_team_id = t.id AND winner_id = t.id)) as wins,
               (SELECT COUNT(*) FROM games WHERE ((home_team_id = t.id OR away_team_id = t.id) AND winner_id IS NOT NULL AND winner_id != t.id)) as losses
        FROM elo_ratings e
        JOIN teams t ON e.team_id = t.id
        ORDER BY e.rating DESC
        LIMIT 25
    ''').fetchall()
    conn.close()
    
    # Image dimensions
    W, H = 600, 900
    
    # Colors
    bg_dark = (15, 23, 42)
    bg_row_top5 = (20, 30, 60)
    bg_row_mid = (18, 27, 50)
    bg_row = (17, 25, 45)
    white = (232, 232, 232)
    blue = (96, 165, 250)
    purple = (167, 139, 250)
    gray = (100, 116, 139)
    light_gray = (148, 163, 184)
    
    img = Image.new('RGB', (W, H), bg_dark)
    draw = ImageDraw.Draw(img)
    
    # Fonts
    try:
        font_bold = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 14)
        font_reg = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
        font_small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 11)
        font_title = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 24)
        font_subtitle = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    except:
        font_bold = font_reg = font_small = font_title = font_subtitle = ImageFont.load_default()
    
    # Header
    draw.text((W//2, 24), 'COLLEGE BASEBALL PREDICTOR', fill=gray, font=font_small, anchor='mt')
    draw.text((W//2, 48), "Sam's Top 25", fill=white, font=font_title, anchor='mt')
    
    from datetime import datetime
    draw.text((W//2, 78), datetime.now().strftime('%B %d, %Y'), fill=gray, font=font_subtitle, anchor='mt')
    
    # Rankings
    y = 100
    row_h = 30
    logo_dir = Path(__file__).parent.parent / 'static' / 'logos'
    
    for i, team in enumerate(rows):
        rank = i + 1
        row_y = y + i * row_h
        
        # Row background
        if rank <= 5:
            row_bg = bg_row_top5
        elif rank <= 10:
            row_bg = bg_row_mid
        else:
            row_bg = bg_row
        
        # Alternating subtle stripe
        if rank % 2 == 0:
            row_bg = tuple(min(c + 3, 255) for c in row_bg)
        
        draw.rounded_rectangle([(16, row_y), (W - 16, row_y + row_h - 2)], radius=6, fill=row_bg)
        
        # Rank number
        rank_color = blue if rank <= 5 else (purple if rank <= 10 else gray)
        draw.text((36, row_y + row_h//2), str(rank), fill=rank_color, font=font_bold, anchor='mm')
        
        # Team logo
        logo_path = logo_dir / f"{team['team_id']}.png"
        if logo_path.exists():
            try:
                logo = Image.open(logo_path).convert('RGBA')
                logo = logo.resize((22, 22), Image.LANCZOS)
                # Add subtle circle behind logo for dark logos
                circle_bg = Image.new('RGBA', (26, 26), (0, 0, 0, 0))
                circle_draw = ImageDraw.Draw(circle_bg)
                circle_draw.ellipse([(0, 0), (25, 25)], fill=(30, 40, 65, 200))
                img.paste(circle_bg, (52, row_y + 2), circle_bg)
                img.paste(logo, (54, row_y + 4), logo)
            except:
                pass
        
        # Team name
        draw.text((80, row_y + 6), team['name'], fill=white, font=font_bold)
        
        # Conference
        conf = team['conference'] or ''
        if conf:
            name_width = draw.textlength(team['name'], font=font_bold)
            draw.text((84 + name_width, row_y + 8), conf, fill=gray, font=font_small)
        
        # Record
        record = f"{team['wins']}-{team['losses']}"
        draw.text((W - 120, row_y + 8), record, fill=light_gray, font=font_reg)
        
        # Elo rating
        rating = int(team['rating'])
        elo_color = blue if rating >= 1600 else (purple if rating >= 1550 else white)
        draw.text((W - 36, row_y + 6), str(rating), fill=elo_color, font=font_bold, anchor='rt')
    
    # Footer line
    footer_y = y + 25 * row_h + 10
    draw.line([(50, footer_y), (W - 50, footer_y)], fill=(30, 40, 60), width=1)
    draw.text((W//2, footer_y + 14), 'baseball.mcdevitt.page', fill=gray, font=font_small, anchor='mt')
    
    # Save to buffer
    buf = BytesIO()
    img.save(buf, format='PNG', optimize=True)
    buf.seek(0)
    
    return send_file(buf, mimetype='image/png', download_name='elo-top-25.png',
                     as_attachment=request.args.get('download', '0') == '1')
