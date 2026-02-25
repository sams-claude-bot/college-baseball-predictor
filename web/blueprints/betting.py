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

from web.services.betting_page import build_betting_page_context
from web.services.risk_engine_page import build_risk_engine_page_context

betting_bp = Blueprint('betting', __name__)


@betting_bp.route('/betting')
def betting():
    """Betting analysis page - v2 logic with adjusted edges"""
    conference = request.args.get('conference', '')
    _view = request.args.get('view', '')
    return render_template('betting.html', **build_betting_page_context(conference=conference))


@betting_bp.route('/experimental/risk-engine')
def risk_engine_experimental():
    """Experimental dashboard for bet risk engine outputs."""
    return render_template('risk_engine.html', **build_risk_engine_page_context())


@betting_bp.route('/tracker')
def tracker():
    """P&L Tracker - three categories: Consensus ML, EV ML, and Totals."""
    conn = get_connection()
    c = conn.cursor()

    # --- CONSENSUS ML BETS (from tracked_confident_bets) ---
    c.execute('''
        SELECT tc.*, g.status, g.home_score, g.away_score
        FROM tracked_confident_bets tc
        LEFT JOIN games g ON tc.game_id = g.id
        ORDER BY tc.date, tc.game_id
    ''')
    all_consensus = [dict(r) for r in c.fetchall()]

    consensus_bets = []
    pending_consensus = []
    for b in all_consensus:
        entry = {
            'date': b['date'],
            'game': f"{'vs ' if b['is_home'] else '@ '}{b['opponent_name']}",
            'pick': b['pick_team_name'],
            'moneyline': b['moneyline'],
            'models_agree': b.get('models_agree', 0),
            'edge': round((b.get('avg_prob', 0.5) - (abs(b['moneyline']) / (abs(b['moneyline']) + 100) if b['moneyline'] < 0 else 100 / (100 + b['moneyline']))) * 100, 1) if b['moneyline'] else 0,
            'model_prob': b.get('avg_prob', 0),
            'won': b['won'],
            'profit': b.get('profit', 0) or 0,
        }
        if b['won'] is not None:
            consensus_bets.append(entry)
        else:
            pending_consensus.append(entry)

    # --- EV ML BETS (from tracked_bets) ---
    c.execute('''
        SELECT tb.*, g.status, g.home_score, g.away_score
        FROM tracked_bets tb
        LEFT JOIN games g ON tb.game_id = g.id
        ORDER BY tb.date, tb.game_id
    ''')
    all_ev = [dict(r) for r in c.fetchall()]

    ev_bets = []
    pending_ev = []
    for b in all_ev:
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
            ev_bets.append(entry)
        else:
            pending_ev.append(entry)

    # --- TOTALS BETS ---
    total_bets_list = []
    pending_totals = []

    c.execute('''
        SELECT tb.*, g.home_score, g.away_score,
               ht.name as home_name,
               at.name as away_name
        FROM tracked_bets_spreads tb
        LEFT JOIN games g ON tb.game_id = g.id
        LEFT JOIN teams ht ON g.home_team_id = ht.id
        LEFT JOIN teams at ON g.away_team_id = at.id
        WHERE tb.bet_type = 'total'
        ORDER BY tb.date, tb.game_id
    ''')
    all_totals = [dict(r) for r in c.fetchall()]
    conn.close()

    for b in all_totals:
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
        }
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

    consensus_stats = calc_stats(consensus_bets)
    ev_stats = calc_stats(ev_bets)
    totals_stats = calc_stats(total_bets_list)

    # Combined P&L
    all_completed = []
    for b in consensus_bets:
        all_completed.append({'date': b['date'], 'profit': b['profit'], 'won': b['won'], 'type': 'ML', 'pick': b['pick']})
    for b in ev_bets:
        all_completed.append({'date': b['date'], 'profit': b['profit'], 'won': b['won'], 'type': 'EV', 'pick': b['pick']})
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
    def build_running_pl(bet_list):
        result = []
        cum = 0
        sorted_bets = sorted(bet_list, key=lambda x: x['date'])
        for b in sorted_bets:
            cum += b['profit']
            result.append({'date': b['date'], 'pl': round(cum, 2),
                          'pick': b['pick'], 'profit': b['profit']})
        return result

    consensus_running_pl = build_running_pl(consensus_bets)
    ev_running_pl = build_running_pl(ev_bets)
    totals_running_pl = build_running_pl(total_bets_list)

    # --- PARLAY BETS ---
    conn2 = get_connection()
    c2 = conn2.cursor()
    c2.execute('SELECT * FROM tracked_parlays ORDER BY date')
    all_parlays = [dict(r) for r in c2.fetchall()]
    conn2.close()

    import json as _json
    parlay_bets = []
    pending_parlays = []
    for p in all_parlays:
        legs = _json.loads(p['legs_json'])
        entry = {
            'date': p['date'],
            'legs': legs,
            'num_legs': p['num_legs'],
            'american_odds': p['american_odds'],
            'decimal_odds': p['decimal_odds'],
            'model_prob': p['model_prob'],
            'bet_amount': p['bet_amount'],
            'payout': p['payout'],
            'won': p['won'],
            'profit': p.get('profit', 0) or 0,
            'legs_won': p.get('legs_won', 0) or 0,
        }
        if p['won'] is not None:
            parlay_bets.append(entry)
        else:
            pending_parlays.append(entry)

    parlay_stats = calc_stats(parlay_bets)
    parlay_running_pl = build_running_pl([{'date': p['date'], 'pick': f"Parlay +{p['american_odds']}", 'profit': p['profit']} for p in parlay_bets])

    # Update combined stats to include parlays
    for p in parlay_bets:
        all_completed.append({'date': p['date'], 'profit': p['profit'], 'won': p['won'], 'type': 'PARLAY', 'pick': f"Parlay +{p['american_odds']}"})
    all_completed.sort(key=lambda x: x['date'])
    combined_stats = calc_stats(all_completed)

    # Rebuild combined running P&L with parlays included
    running_pl = []
    cumulative = 0
    for b in all_completed:
        cumulative += b['profit']
        running_pl.append({'date': b['date'], 'pl': round(cumulative, 2),
                          'type': b['type'], 'pick': b['pick'], 'profit': b['profit']})

    # Edge buckets (consensus ML)
    buckets = {
        '5-10%': {'bets': [], 'label': '5-10%'},
        '10-20%': {'bets': [], 'label': '10-20%'},
        '20%+': {'bets': [], 'label': '20%+'}
    }
    for b in consensus_bets:
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

    # --- ML BUCKET ANALYSIS (by moneyline range) ---
    # Combines consensus + EV bets for ML bucket breakdown
    all_ml_bets = []
    for b in consensus_bets:
        all_ml_bets.append({'ml': b.get('moneyline', 0), 'won': b['won'], 'profit': b['profit']})
    for b in ev_bets:
        all_ml_bets.append({'ml': b.get('moneyline', 0), 'won': b['won'], 'profit': b['profit']})

    ml_bucket_stats = []
    ml_bucket_defs = [
        ('Heavy Fav (<-200)', lambda ml: ml is not None and ml < -200),
        ('Light Fav (-200 to -100)', lambda ml: ml is not None and -200 <= ml <= -100),
        ('Underdog (+ML)', lambda ml: ml is not None and ml > 0),
    ]
    for label, test_fn in ml_bucket_defs:
        bb = [b for b in all_ml_bets if test_fn(b['ml'])]
        if bb:
            bw = sum(1 for x in bb if x['won'])
            bp = sum(x['profit'] for x in bb)
            ml_bucket_stats.append({
                'label': label,
                'count': len(bb),
                'wins': bw,
                'win_rate': round(bw / len(bb) * 100, 1),
                'pl': round(bp, 2),
                'roi': round(bp / (len(bb) * 100) * 100, 1)
            })
        else:
            ml_bucket_stats.append({'label': label, 'count': 0, 'wins': 0, 'win_rate': 0, 'pl': 0, 'roi': 0})

    # Recommendation based on data
    underdog_bucket = ml_bucket_stats[2] if len(ml_bucket_stats) > 2 else None
    bet_recommendation = None
    if underdog_bucket and underdog_bucket['count'] >= 3 and underdog_bucket['win_rate'] < 40:
        bet_recommendation = f"⚠️ Underdog bets: {underdog_bucket['wins']}W-{underdog_bucket['count'] - underdog_bucket['wins']}L ({underdog_bucket['win_rate']}%), ${underdog_bucket['pl']:+.0f}. New filters now skip all underdogs."

    return render_template('tracker.html',
                          consensus_bets=consensus_bets,
                          pending_consensus=pending_consensus,
                          ev_bets=ev_bets,
                          pending_ev=pending_ev,
                          total_bets_list=total_bets_list,
                          pending_totals=pending_totals,
                          consensus_stats=consensus_stats,
                          ev_stats=ev_stats,
                          totals_stats=totals_stats,
                          combined_stats=combined_stats,
                          running_pl=running_pl,
                          consensus_running_pl=consensus_running_pl,
                          ev_running_pl=ev_running_pl,
                          totals_running_pl=totals_running_pl,
                          bucket_stats=bucket_stats,
                          ml_bucket_stats=ml_bucket_stats,
                          bet_recommendation=bet_recommendation,
                          parlay_bets=parlay_bets,
                          pending_parlays=pending_parlays,
                          parlay_stats=parlay_stats,
                          parlay_running_pl=parlay_running_pl)
