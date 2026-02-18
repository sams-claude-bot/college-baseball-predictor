"""
Dashboard Blueprint - Main dashboard page
"""

from datetime import datetime, timedelta
from flask import Blueprint, render_template, request

from web.helpers import (
    get_todays_games, get_value_picks, get_quick_stats,
    get_featured_team_info, get_model_accuracy_summary,
    get_model_accuracy, get_recent_results,
    get_games_for_date_with_predictions
)

dashboard_bp = Blueprint('dashboard', __name__)


@dashboard_bp.route('/')
def dashboard():
    """Main dashboard page"""
    featured_team = request.args.get('team', 'mississippi-state')
    todays_games = get_todays_games()
    value_picks = get_value_picks(5)
    stats = get_quick_stats()
    featured = get_featured_team_info(featured_team)
    accuracy = get_model_accuracy_summary()

    # Model snapshot - all models
    all_accuracy = get_model_accuracy()

    # Recent results (last 3 days)
    recent_results = get_recent_results(days_back=3)

    # Tomorrow's games preview
    tomorrow_str = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    tomorrow_games, _, _ = get_games_for_date_with_predictions(tomorrow_str)
    # Only show scheduled games for tomorrow
    tomorrow_games = [g for g in tomorrow_games if g['status'] == 'scheduled'][:10]

    return render_template('dashboard.html',
                          todays_games=todays_games,
                          value_picks=value_picks,
                          stats=stats,
                          featured=featured,
                          accuracy=accuracy,
                          all_accuracy=all_accuracy,
                          recent_results=recent_results,
                          tomorrow_games=tomorrow_games,
                          tomorrow_date=tomorrow_str,
                          today=datetime.now().strftime('%B %d, %Y'))
