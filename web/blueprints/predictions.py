"""
Predictions Blueprint - Prediction tool page
"""

from flask import Blueprint, render_template

from web.helpers import get_all_teams, get_all_conferences

predictions_bp = Blueprint('predictions', __name__)


@predictions_bp.route('/predict')
def predict():
    """Prediction tool page"""
    all_teams = get_all_teams()
    all_teams.sort(key=lambda x: x['name'])
    conferences = get_all_conferences()

    return render_template('predict.html', teams=all_teams, conferences=conferences)
