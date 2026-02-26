"""
Teams Blueprint - Team listing and detail pages
"""

from flask import Blueprint, render_template, request, current_app

from web.helpers import get_all_teams, get_team_detail
from web.services.team_percentiles import get_team_percentiles

teams_bp = Blueprint('teams', __name__)


@teams_bp.route('/teams')
def teams():
    """Teams listing page"""
    sort_by = request.args.get('sort', 'name')
    cache = current_app.cache
    cache_key = f'teams:{sort_by}'
    cached = cache.get(cache_key)
    if cached:
        return cached

    all_teams = get_all_teams()

    # Get unique conferences
    conferences = sorted(set(t['conference'] for t in all_teams if t['conference']))

    if sort_by == 'elo':
        all_teams.sort(key=lambda x: x.get('elo_rating') or 0, reverse=True)
    elif sort_by == 'rank':
        all_teams.sort(key=lambda x: (x.get('current_rank') or 999, x['name']))
    elif sort_by == 'win_pct':
        all_teams.sort(key=lambda x: x.get('win_pct', 0), reverse=True)
    elif sort_by == 'rpi':
        all_teams.sort(key=lambda x: (x.get('sams_rank') or 9999, x['name']))
    elif sort_by == 'conference':
        all_teams.sort(key=lambda x: (x.get('conference') or 'ZZZ', x['name']))
    elif sort_by == 'sos':
        all_teams.sort(key=lambda x: x.get('overall_sos') or -9999, reverse=True)
    else:
        all_teams.sort(key=lambda x: x['name'])

    result = render_template('teams.html',
                          teams=all_teams,
                          conferences=conferences,
                          sort_by=sort_by)
    cache.set(cache_key, result, timeout=600)
    return result


@teams_bp.route('/team/<team_id>')
def team_detail(team_id):
    """Individual team detail page"""
    team = get_team_detail(team_id)

    if not team:
        return render_template('404.html', message="Team not found"), 404

    # Split roster into batters and pitchers
    pitcher_positions = ('P', 'RHP', 'LHP')
    def is_pitcher(p):
        pos = (p.get('position') or '')
        if pos in pitcher_positions or pos.startswith(('RHP', 'LHP')):
            return True
        # If no position set but has pitching stats, treat as pitcher
        if not pos and (p.get('innings_pitched') or 0) > 0 and (p.get('at_bats') or 0) == 0:
            return True
        return False
    pitchers = [p for p in team['roster'] if is_pitcher(p)]
    batters = [p for p in team['roster'] if not is_pitcher(p)]

    # Get recent form (last 10 games)
    completed_games = [g for g in team['schedule'] if g['status'] == 'final']
    recent_form = completed_games[-10:] if completed_games else []

    percentiles = get_team_percentiles(team_id)

    return render_template('team_detail.html',
                          team=team,
                          batters=batters,
                          pitchers=pitchers,
                          recent_form=recent_form,
                          percentiles=percentiles)
