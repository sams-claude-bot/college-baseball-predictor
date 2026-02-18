"""
Rankings Blueprint - Rankings and standings pages
"""

import sys
from pathlib import Path
from flask import Blueprint, render_template, request, current_app

# Add paths for imports
base_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / "scripts"))

from database import get_connection, get_team_record, get_team_runs, get_current_top_25

from web.helpers import get_all_conferences, get_rankings_history, get_rankings_for_date

rankings_bp = Blueprint('rankings', __name__)


@rankings_bp.route('/rankings')
def rankings():
    """Rankings page"""
    conference = request.args.get('conference', '')
    selected_date = request.args.get('week', '')
    featured_team_id = request.args.get('team', 'mississippi-state')

    cache = current_app.cache
    cache_key = f'rankings:{conference}:{selected_date}:{featured_team_id}'
    cached = cache.get(cache_key)
    if cached:
        return cached

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

    # Model Power Rankings
    power_rankings = []
    power_rankings_date = None
    try:
        conn3 = get_connection()
        c3 = conn3.cursor()
        # Get latest power rankings
        c3.execute('SELECT MAX(date) FROM power_rankings')
        pr_date_row = c3.fetchone()
        if pr_date_row and pr_date_row[0]:
            power_rankings_date = pr_date_row[0]
            pr_query = '''
                SELECT pr.*, t.name as team_name, t.conference
                FROM power_rankings pr
                JOIN teams t ON pr.team_id = t.id
                WHERE pr.date = ?
                ORDER BY pr.rank
            '''
            pr_params = [power_rankings_date]
            c3.execute(pr_query, pr_params)
            power_rankings = [dict(r) for r in c3.fetchall()]

            # Add records, AP rank, and Elo rank
            # Build lookup maps
            ap_rank_map = {t['id']: t.get('rank') or t.get('current_rank') for t in top_25}
            elo_rank_map = {t['team_id']: t['elo_rank'] for t in elo_top_25}

            for pr in power_rankings:
                record = get_team_record(pr['team_id'])
                pr['wins'] = record['wins']
                pr['losses'] = record['losses']
                pr['ap_rank'] = ap_rank_map.get(pr['team_id'])
                pr['elo_rank'] = elo_rank_map.get(pr['team_id'])

            # Filter by conference if specified
            if conference:
                power_rankings = [pr for pr in power_rankings if pr.get('conference') == conference]
                # Re-rank within conference
                for i, pr in enumerate(power_rankings):
                    pass  # Keep overall rank visible
        conn3.close()
    except Exception:
        pass  # Table might not exist yet

    result = render_template('rankings.html',
                          top_25=top_25,
                          elo_top_25=elo_top_25,
                          ap_ids=ap_ids,
                          elo_ids=elo_ids,
                          history_dates=history_dates,
                          selected_date=selected_date,
                          historical=historical,
                          conferences=conferences,
                          selected_conference=conference,
                          power_rankings=power_rankings,
                          power_rankings_date=power_rankings_date,
                          featured_team_id=featured_team_id)
    cache.set(cache_key, result, timeout=600)
    return result


@rankings_bp.route('/standings')
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
