#!/usr/bin/env python3
"""
College Baseball Predictor - Web Dashboard

Flask web interface for exploring predictions, teams, rankings, and betting data.
Refactored into blueprints for maintainability.
"""

import sys
from pathlib import Path

# Add project root to path for imports
base_dir = Path(__file__).parent.parent
sys.path.insert(0, str(base_dir))

from flask import Flask, render_template
from flask_caching import Cache

# Import blueprints
from web.blueprints import (
    dashboard_bp,
    teams_bp,
    scores_bp,
    betting_bp,
    predictions_bp,
    rankings_bp,
    models_bp,
    api_bp,
    debug_bp
)


def create_app():
    """Application factory for creating the Flask app."""
    app = Flask(__name__)

    # Configure caching (10-minute TTL, per-worker simple cache)
    app.config['CACHE_TYPE'] = 'SimpleCache'
    app.config['CACHE_DEFAULT_TIMEOUT'] = 600  # 10 minutes
    cache = Cache(app)
    app.cache = cache  # Make accessible to blueprints

    # Register blueprints
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(teams_bp)
    app.register_blueprint(scores_bp)
    app.register_blueprint(betting_bp)
    app.register_blueprint(predictions_bp)
    app.register_blueprint(rankings_bp)
    app.register_blueprint(models_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(debug_bp)

    # Register template filters
    register_filters(app)

    # Register error handlers
    register_error_handlers(app)

    return app


def register_filters(app):
    """Register Jinja2 template filters."""

    @app.template_filter('format_odds')
    def format_odds(value):
        """Format American odds with + prefix for positive"""
        if value is None:
            return 'N/A'
        return f"+{value}" if value > 0 else str(value)

    @app.template_filter('format_pct')
    def format_pct(value):
        """Format decimal as percentage"""
        if value is None:
            return 'N/A'
        return f"{value * 100:.1f}%"

    @app.template_filter('format_edge')
    def format_edge(value):
        """Format edge with sign"""
        if value is None:
            return 'N/A'
        return f"+{value:.1f}%" if value > 0 else f"{value:.1f}%"


def register_error_handlers(app):
    """Register error handlers."""

    @app.errorhandler(404)
    def not_found(e):
        return render_template('404.html'), 404

    @app.errorhandler(500)
    def server_error(e):
        return render_template('500.html', error=str(e)), 500


# Create the app instance for WSGI/CLI compatibility
app = create_app()


if __name__ == '__main__':
    print("🏟️  College Baseball Predictor Dashboard")
    print("=" * 40)
    print(f"Running on http://0.0.0.0:5000")
    print("Press Ctrl+C to stop")
    print()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
