"""
Debug Blueprint - Debug tools and bug reporting
"""

import sys
import json
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
        failed_scrapes=failed_scrapes
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
