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

from flask import Flask, render_template, make_response
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
    debug_bp,
    account_bp
)
from web.blueprints.alerts import alerts_bp


def create_app():
    """Application factory for creating the Flask app."""
    app = Flask(__name__)

    # Configure caching (10-minute TTL, per-worker simple cache)
    app.config['CACHE_TYPE'] = 'SimpleCache'
    app.config['CACHE_DEFAULT_TIMEOUT'] = 600  # 10 minutes
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    cache = Cache(app)
    app.cache = cache  # Make accessible to blueprints

    @app.after_request
    def add_cache_headers(response):
        """Prevent Cloudflare/browser from caching HTML pages"""
        if 'text/html' in response.content_type:
            response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            response.headers['Pragma'] = 'no-cache'
        return response

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
    app.register_blueprint(account_bp)
    app.register_blueprint(alerts_bp)

    # Serve service worker from root (required for push scope)
    @app.route('/sw.js')
    def service_worker():
        response = make_response(app.send_static_file('sw.js'))
        response.headers['Content-Type'] = 'application/javascript'
        response.headers['Service-Worker-Allowed'] = '/'
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response

    # Register template filters
    register_filters(app)

    # Register error handlers
    register_error_handlers(app)

    return app


def register_filters(app):
    """Register Jinja2 template filters."""
    
    # Cache for team colors (populated on first use)
    _team_colors_cache = {}
    # Cache for logo variant existence checks (key: "{variant}:{team_id}")
    _logo_variant_exists_cache = {}
    
    def _load_team_colors():
        """Load all team colors from DB into cache."""
        if _team_colors_cache:
            return
        try:
            from database import get_connection
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute('SELECT id, primary_color, secondary_color FROM teams WHERE primary_color IS NOT NULL')
            for row in cursor.fetchall():
                _team_colors_cache[row['id']] = {
                    'primary': row['primary_color'],
                    'secondary': row['secondary_color']
                }
            conn.close()
        except Exception:
            pass

    # Template global for team logos and colors
    @app.context_processor
    def inject_team_helpers():
        def team_logo(team_id, size=None):
            """Return logo path for a team, with optional optimized size variant."""
            if not team_id:
                return ''

            # Use generated lightweight variants when available.
            if size is not None:
                try:
                    px = int(size)
                except (TypeError, ValueError):
                    px = None

                variant = None
                if px is not None:
                    if px <= 24:
                        variant = '24'
                    elif px <= 48:
                        variant = '48'

                if variant:
                    rel_variant = f'logos/{variant}/{team_id}.png'
                    cache_key = f'{variant}:{team_id}'
                    exists = _logo_variant_exists_cache.get(cache_key)
                    if exists is None:
                        exists = (Path(app.static_folder) / rel_variant).exists()
                        _logo_variant_exists_cache[cache_key] = exists
                    if exists:
                        return f'/static/{rel_variant}'

            return f'/static/logos/{team_id}.png'

        def team_color(team_id, which='primary'):
            """Get team color by ID. Returns hex color or default."""
            _load_team_colors()
            if team_id in _team_colors_cache:
                return _team_colors_cache[team_id].get(which, '#5E6A71')
            return '#5E6A71'  # Default gray

        return dict(team_logo=team_logo, team_color=team_color)

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
