"""
Debug Blueprint - Debug tools and bug reporting
"""

import sys
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from flask import Blueprint, render_template, request, jsonify

# Add paths for imports
base_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / "scripts"))

from database import get_connection

debug_bp = Blueprint('debug', __name__)


@debug_bp.route('/debug')
def debug():
    conn = get_connection()
    c = conn.cursor()

    # Teams with more than 5 games in any Mon-Sun week
    # Get all weeks that have final games
    c.execute("SELECT DISTINCT date FROM games WHERE status='final' ORDER BY date")
    all_dates = [row[0] for row in c.fetchall()]

    # Build Monday-Sunday week boundaries
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

    # Teams with fewer than 3 final games (potential data gaps)
    c.execute('''
        SELECT t.id, t.name, t.conference,
            COUNT(g.id) as game_count,
            SUM(CASE WHEN g.winner_id = t.id THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN g.winner_id != t.id AND g.winner_id IS NOT NULL THEN 1 ELSE 0 END) as losses,
            MIN(g.date) as first_game,
            MAX(g.date) as last_game
        FROM teams t
        LEFT JOIN games g ON (g.home_team_id = t.id OR g.away_team_id = t.id) AND g.status = 'final'
        GROUP BY t.id
        HAVING game_count < 3
        ORDER BY game_count ASC, t.name
    ''')
    low_game_teams = [dict(row) for row in c.fetchall()]

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

    # Failed scrapes — teams with D1BB slugs but no recent stats
    slugs_path = base_dir / 'config' / 'd1bb_slugs.json'
    failed_scrapes = []
    if slugs_path.exists():
        slug_data = json.loads(slugs_path.read_text())
        slug_map = slug_data.get('team_id_to_d1bb_slug', {})
        cutoff_dt = (datetime.now() - timedelta(hours=36)).strftime('%Y-%m-%d %H:%M')
        conn2 = get_connection()
        c2 = conn2.cursor()
        updated_teams = set(r[0] for r in c2.execute(
            'SELECT DISTINCT team_id FROM player_stats WHERE updated_at > ?', (cutoff_dt,)).fetchall())
        for tid, slug in sorted(slug_map.items()):
            if tid not in updated_teams:
                team_row = c2.execute('SELECT name, conference FROM teams WHERE id=?', (tid,)).fetchone()
                last_update = c2.execute('SELECT MAX(updated_at) FROM player_stats WHERE team_id=?', (tid,)).fetchone()
                failed_scrapes.append({
                    'id': tid,
                    'name': team_row[0] if team_row else tid,
                    'conference': team_row[1] if team_row else '?',
                    'slug': slug,
                    'last_updated': last_update[0] if last_update and last_update[0] else 'Never',
                    'd1bb_url': f'https://d1baseball.com/team/{slug}/'
                })
        conn2.close()

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
        bug_reports=bug_reports,
        failed_scrapes=failed_scrapes,
        low_game_teams=low_game_teams
    )


@debug_bp.route('/api/debug/flag', methods=['POST'])
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


@debug_bp.route('/api/bug-report', methods=['POST'])
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


@debug_bp.route('/api/bug-report/<int:bug_id>', methods=['PATCH'])
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


@debug_bp.route('/debug/model-testing')
def model_testing():
    """Live model testing — re-run models against today's (or any date's) final results."""
    from models.predictor_db import Predictor

    test_date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    model_filter = request.args.get('model', '')  # empty = all

    conn = get_connection()
    conn.row_factory = sqlite3.Row

    # Get final games for the date
    games = conn.execute('''
        SELECT g.id, g.home_team_id, g.away_team_id, h.name as home_name, a.name as away_name,
               g.home_score, g.away_score, g.is_neutral_site,
               h.conference as home_conf, a.conference as away_conf
        FROM games g
        JOIN teams h ON g.home_team_id = h.id
        JOIN teams a ON g.away_team_id = a.id
        WHERE g.date = ? AND g.status = 'final'
        ORDER BY g.id
    ''', (test_date,)).fetchall()

    # Get stored predictions for comparison
    stored = {}
    for row in conn.execute('''
        SELECT game_id, model_name, predicted_home_prob
        FROM model_predictions
        WHERE game_id IN (SELECT id FROM games WHERE date = ? AND status = 'final')
    ''', (test_date,)).fetchall():
        stored.setdefault(row['game_id'], {})[row['model_name']] = row['predicted_home_prob']

    # Available model names from stored predictions
    all_model_names = sorted(set(
        m for preds in stored.values() for m in preds.keys()
    ))

    # Run live predictions
    predictor = Predictor()
    results = []
    live_totals = {}  # model_name -> {correct, total}
    stored_totals = {}

    for g in games:
        home_won = g['home_score'] > g['away_score']

        # Live prediction from current model code
        try:
            pred = predictor.predict_game(g['home_team_id'], g['away_team_id'])
            components = pred.get('component_predictions', {}) if pred else {}
            live_ensemble_hp = pred.get('home_win_probability', 0.5) if pred else 0.5
        except Exception as e:
            components = {}
            live_ensemble_hp = 0.5

        game_result = {
            'id': g['id'],
            'home_name': g['home_name'],
            'away_name': g['away_name'],
            'home_score': g['home_score'],
            'away_score': g['away_score'],
            'home_conf': g['home_conf'],
            'away_conf': g['away_conf'],
            'home_won': home_won,
            'models': {}
        }

        game_stored = stored.get(g['id'], {})

        # Build per-model comparison
        model_names_for_game = set(list(components.keys()) + list(game_stored.keys()) + ['ensemble'])
        for mname in model_names_for_game:
            if model_filter and mname != model_filter:
                continue

            # Live probability
            if mname == 'ensemble':
                live_hp = live_ensemble_hp
            elif mname in components:
                c = components[mname]
                live_hp = c.get('home_win_probability', c.get('home_prob', 0.5))
            else:
                live_hp = None

            # Stored probability
            stored_hp = game_stored.get(mname)

            live_correct = ((live_hp > 0.5) == home_won) if live_hp is not None else None
            stored_correct = ((stored_hp > 0.5) == home_won) if stored_hp is not None else None

            game_result['models'][mname] = {
                'live_prob': live_hp,
                'stored_prob': stored_hp,
                'live_correct': live_correct,
                'stored_correct': stored_correct,
                'changed': (live_correct != stored_correct) if (live_correct is not None and stored_correct is not None) else False,
            }

            # Accumulate totals
            if live_hp is not None:
                lt = live_totals.setdefault(mname, {'correct': 0, 'total': 0})
                lt['total'] += 1
                if live_correct:
                    lt['correct'] += 1
            if stored_hp is not None:
                st = stored_totals.setdefault(mname, {'correct': 0, 'total': 0})
                st['total'] += 1
                if stored_correct:
                    st['correct'] += 1

        results.append(game_result)

    # Build summary table
    summary = []
    all_names = sorted(set(list(live_totals.keys()) + list(stored_totals.keys())))
    for mname in all_names:
        lt = live_totals.get(mname, {'correct': 0, 'total': 0})
        st = stored_totals.get(mname, {'correct': 0, 'total': 0})
        summary.append({
            'model': mname,
            'live_correct': lt['correct'],
            'live_total': lt['total'],
            'live_pct': round(100 * lt['correct'] / lt['total'], 1) if lt['total'] else 0,
            'stored_correct': st['correct'],
            'stored_total': st['total'],
            'stored_pct': round(100 * st['correct'] / st['total'], 1) if st['total'] else 0,
            'delta': lt['correct'] - st['correct'] if lt['total'] and st['total'] else None,
        })
    summary.sort(key=lambda x: x['live_pct'], reverse=True)

    conn.close()

    return render_template('model_testing.html',
        test_date=test_date,
        model_filter=model_filter,
        results=results,
        summary=summary,
        all_model_names=all_model_names,
        total_games=len(results)
    )
