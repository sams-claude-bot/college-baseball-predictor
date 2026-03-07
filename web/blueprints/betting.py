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
from strategy_pl_report import generate_report as generate_strategy_report
from line_tracker import get_line_movement

from web.helpers import get_clv_summary
from web.services.betting_page import build_betting_page_context
from web.services.risk_engine_page import build_risk_engine_page_context

betting_bp = Blueprint('betting', __name__)


@betting_bp.route('/betting')
def betting():
    """Betting analysis page - v2 logic with adjusted edges"""
    conference = request.args.get('conference', '')
    _view = request.args.get('view', '')
    ctx = build_betting_page_context(conference=conference)

    # Top movers: games with biggest ML movement today (>1pp)
    try:
        movements = get_line_movement()
        with_moves = [m for m in movements if m['home_prob_move'] is not None and abs(m['home_prob_move']) > 1.0]
        with_moves.sort(key=lambda m: abs(m['home_prob_move']), reverse=True)
        ctx['top_movers'] = with_moves[:5]
    except Exception:
        ctx['top_movers'] = []

    return render_template('betting.html', **ctx)


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

    # --- EV ML BETS (from tracked_bets) ---
    c.execute('''
        SELECT tb.*, g.status, g.home_score, g.away_score
        FROM tracked_bets tb
        LEFT JOIN games g ON tb.game_id = g.id
        ORDER BY tb.date, tb.game_id
    ''')
    all_ev = [dict(r) for r in c.fetchall()]

    # Prefetch opening/closing snapshot context for ML CLV display.
    game_ids_for_clv = sorted({
        *(b['game_id'] for b in all_consensus if b.get('game_id')),
        *(b['game_id'] for b in all_ev if b.get('game_id')),
    })
    opening_by_game = {}
    closing_by_game = {}

    if game_ids_for_clv:
        placeholders = ','.join('?' for _ in game_ids_for_clv)

        # Opening snapshot (when available from odds ingest)
        opening_rows = c.execute(f'''
            SELECT blh.game_id, blh.home_ml, blh.away_ml, blh.snapshot_type, blh.captured_at
            FROM betting_line_history blh
            JOIN (
                SELECT game_id, MAX(captured_at) AS mx
                FROM betting_line_history
                WHERE game_id IN ({placeholders})
                  AND book = 'draftkings'
                  AND snapshot_type = 'opening'
                GROUP BY game_id
            ) x
              ON x.game_id = blh.game_id
             AND x.mx = blh.captured_at
        ''', game_ids_for_clv).fetchall()
        opening_by_game = {r['game_id']: dict(r) for r in opening_rows}

        # Best pregame close context. Prefer explicit 'closing', fallback to 'pregame'.
        closing_rows = c.execute(f'''
            SELECT game_id, home_ml, away_ml, snapshot_type, captured_at
            FROM betting_line_history
            WHERE game_id IN ({placeholders})
              AND book = 'draftkings'
              AND snapshot_type IN ('closing', 'pregame')
            ORDER BY captured_at DESC
        ''', game_ids_for_clv).fetchall()

        priority = {'closing': 2, 'pregame': 1}
        for row in closing_rows:
            d = dict(row)
            gid = d['game_id']
            p = priority.get(d.get('snapshot_type'), 0)
            existing = closing_by_game.get(gid)
            if existing is None or p > existing.get('_priority', 0):
                d['_priority'] = p
                closing_by_game[gid] = d

    def _line_for_side(snapshot, is_home):
        if not snapshot:
            return None
        return snapshot.get('home_ml') if bool(is_home) else snapshot.get('away_ml')

    def _build_ml_entry(b, is_consensus=False):
        gid = b.get('game_id')
        opening_snap = opening_by_game.get(gid)
        closing_snap = closing_by_game.get(gid)

        opening_ml = _line_for_side(opening_snap, b.get('is_home'))
        closing_ml = b.get('closing_ml')
        if closing_ml is None:
            closing_ml = _line_for_side(closing_snap, b.get('is_home'))

        entry = {
            'date': b['date'],
            'game': f"{'vs ' if b['is_home'] else '@ '}{b['opponent_name']}",
            'pick': b['pick_team_name'],
            'moneyline': b['moneyline'],
            'opening_ml': opening_ml,
            'closing_ml': closing_ml,
            'clv_implied': b.get('clv_implied'),
            'clv_cents': b.get('clv_cents'),
            'clv_source': (closing_snap.get('snapshot_type') if closing_snap else None),
            'clv_captured_at': (closing_snap.get('captured_at') if closing_snap else None),
            'won': b['won'],
            'profit': b.get('profit', 0) or 0,
        }

        if is_consensus:
            entry['models_agree'] = b.get('models_agree', 0)
            ml = b.get('moneyline')
            implied = (
                abs(ml) / (abs(ml) + 100)
                if ml is not None and ml < 0
                else (100 / (100 + ml) if ml is not None and ml > 0 else 0.5)
            )
            entry['edge'] = round((b.get('avg_prob', 0.5) - implied) * 100, 1) if ml is not None else 0
            entry['model_prob'] = b.get('avg_prob', 0)
        else:
            entry['edge'] = round(b['edge'], 1)
            entry['model_prob'] = b['model_prob']
            entry['dk_implied'] = b['dk_implied']

        return entry

    consensus_bets = []
    pending_consensus = []
    for b in all_consensus:
        entry = _build_ml_entry(b, is_consensus=True)
        if b['won'] is not None:
            consensus_bets.append(entry)
        else:
            pending_consensus.append(entry)

    ev_bets = []
    pending_ev = []
    for b in all_ev:
        entry = _build_ml_entry(b, is_consensus=False)
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
    # Fill in missing dates so the chart doesn't skip days
    all_dates = sorted(set(b['date'] for b in all_completed)) if all_completed else []

    def build_running_pl(bet_list):
        result = []
        cum = 0
        sorted_bets = sorted(bet_list, key=lambda x: x['date'])
        bet_dates = set(b['date'] for b in sorted_bets)
        # Find the first date this bet type starts
        if not sorted_bets:
            return result
        first_date = sorted_bets[0]['date']
        started = False
        bet_idx = 0
        for d in all_dates:
            if d < first_date:
                continue
            started = True
            # Process all bets for this date
            day_bets = []
            while bet_idx < len(sorted_bets) and sorted_bets[bet_idx]['date'] == d:
                cum += sorted_bets[bet_idx]['profit']
                day_bets.append(sorted_bets[bet_idx])
                bet_idx += 1
            if day_bets:
                # Use last bet of the day for the data point
                result.append({'date': d, 'pl': round(cum, 2),
                              'pick': day_bets[-1]['pick'], 'profit': sum(b['profit'] for b in day_bets)})
            else:
                # No bets this day — carry forward cumulative (flat line)
                result.append({'date': d, 'pl': round(cum, 2),
                              'pick': '—', 'profit': 0})
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

    # --- CLAUDE'S PARLAYS (longshot) ---
    conn3 = get_connection()
    c3 = conn3.cursor()
    c3.execute('SELECT * FROM tracked_longshot_parlays ORDER BY date')
    all_longshots = [dict(r) for r in c3.fetchall()]
    conn3.close()

    longshot_bets = []
    pending_longshots = []
    for p in all_longshots:
        legs = _json.loads(p['legs_json'])
        entry = {
            'date': p['date'],
            'legs': legs,
            'num_legs': p['num_legs'],
            'american_odds': p['american_odds'],
            'model_prob': p['model_prob'],
            'bet_amount': p['bet_amount'],
            'payout': p['payout'],
            'won': p['won'],
            'profit': (p['payout'] - p['bet_amount']) if p.get('won') == 1 else (-p['bet_amount'] if p.get('won') == 0 else 0),
            'legs_won': p.get('legs_won', 0) or 0,
            'legs_lost': p.get('legs_lost', 0) or 0,
        }
        if p['won'] is not None:
            longshot_bets.append(entry)
        else:
            pending_longshots.append(entry)

    longshot_stats = calc_stats(longshot_bets)

    # --- CLV SUMMARY ---
    clv_summary = get_clv_summary()

    # --- STRATEGY BREAKDOWN ---
    try:
        strategy_stats, _, _ = generate_strategy_report()
    except Exception:
        strategy_stats = {}

    return render_template('tracker.html',
                          strategy_breakdown=strategy_stats,
                          clv_summary=clv_summary,
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
                          parlay_running_pl=parlay_running_pl,
                          longshot_bets=longshot_bets,
                          pending_longshots=pending_longshots,
                          longshot_stats=longshot_stats)
