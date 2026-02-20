"""
Betting Blueprint - Betting analysis and P&L tracker pages
"""

import sys
from pathlib import Path
from flask import Blueprint, render_template, request

# Add paths for imports
base_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / "scripts"))

from database import get_connection

from web.helpers import get_all_conferences, get_betting_games

betting_bp = Blueprint('betting', __name__)


@betting_bp.route('/betting')
def betting():
    """Betting analysis page - v2 logic with adjusted edges"""
    conference = request.args.get('conference', '')
    conferences = get_all_conferences()

    games = get_betting_games()

    # === v2 THRESHOLDS ===
    ML_EDGE_FAVORITE = 8.0
    ML_EDGE_UNDERDOG = 15.0
    UNDERDOG_DISCOUNT = 0.5
    CONSENSUS_BONUS_PER_MODEL = 1.0
    TOTALS_EDGE_RUNS = 3.0
    SPREADS_ENABLED = False

    def calc_adjusted_edge(raw_edge, ml, models_agree=5):
        adj = raw_edge
        if ml and ml > 0:  # Underdog discount
            adj = raw_edge * UNDERDOG_DISCOUNT
        bonus = max(0, (models_agree - 5)) * CONSENSUS_BONUS_PER_MODEL
        return adj + bonus

    # Filter by conference if specified
    if conference:
        games = [g for g in games
                if g.get('home_conf') == conference or g.get('away_conf') == conference]

    # Build consensus lookup
    consensus_lookup = {}
    for g in games:
        if g.get('model_agreement') and g['model_agreement']['count'] >= 7:
            consensus_lookup[g['game_id']] = g['model_agreement']['count']

    # Add adjusted_edge to all games
    for g in games:
        if g.get('best_edge'):
            ml = g.get('home_ml') if g.get('best_pick') == 'home' else g.get('away_ml')
            models = consensus_lookup.get(g['game_id'], 5)
            g['adjusted_edge'] = calc_adjusted_edge(g['best_edge'], ml, models)
            g['models_agree'] = models
            g['is_underdog'] = ml > 0 if ml else False

    # Sort by adjusted edge
    games_with_edge = [g for g in games if g.get('adjusted_edge')]
    games_with_edge.sort(key=lambda x: x.get('adjusted_edge', 0), reverse=True)

    # Confident bets (7/10+ models agree, sorted by adjusted edge)
    # Cap at -300: heavy favorites aren't worth betting even with consensus
    MAX_FAVORITE_ML = -300

    def passes_favorite_cap(g):
        ml = g.get('home_ml') if g.get('best_pick') == 'home' else g.get('away_ml')
        return ml is None or ml >= MAX_FAVORITE_ML

    confident_candidates = [g for g in games
                           if g.get('model_agreement')
                           and g['model_agreement']['count'] >= 7
                           and passes_favorite_cap(g)]
    confident_candidates.sort(key=lambda x: x.get('adjusted_edge', 0), reverse=True)
    confident_bets = confident_candidates[:6]

    # EV bets — pure raw edge over DK line (exclude consensus picks)
    confident_ids = {g['game_id'] for g in confident_bets}
    ev_candidates = [g for g in games_with_edge
                     if g['game_id'] not in confident_ids
                     and g.get('best_edge', 0) >= ML_EDGE_FAVORITE
                     and passes_favorite_cap(g)]
    ev_candidates.sort(key=lambda x: x.get('best_edge', 0), reverse=True)
    ev_bets = ev_candidates[:6]

    # Best totals (3+ runs edge)
    games_with_totals = [g for g in games if g.get('over_under')]
    games_with_totals.sort(key=lambda x: abs(x.get('total_diff', 0)), reverse=True)
    best_totals = [g for g in games_with_totals if abs(g.get('total_diff', 0)) >= TOTALS_EDGE_RUNS]

    return render_template('betting.html',
                          games=games_with_edge,
                          confident_bets=confident_bets,
                          ev_bets=ev_bets,
                          best_totals=best_totals,
                          conferences=conferences,
                          selected_conference=conference,
                          spreads_enabled=SPREADS_ENABLED,
                          v2_thresholds={
                              'ml_favorite': ML_EDGE_FAVORITE,
                              'ml_underdog': ML_EDGE_UNDERDOG,
                              'totals_runs': TOTALS_EDGE_RUNS,
                              'underdog_discount': UNDERDOG_DISCOUNT
                          })


@betting_bp.route('/tracker')
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
