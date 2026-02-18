#!/usr/bin/env python3
"""
Shared pytest fixtures for college baseball predictor tests.

Provides database connections, sample data, and common utilities.
"""

import sys
import sqlite3
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "data" / "baseball.db"


@pytest.fixture(scope="session")
def db_connection():
    """
    Read-only database connection for testing.
    Uses the actual production database - tests should NOT modify data.
    """
    if not DB_PATH.exists():
        pytest.skip(f"Database not found at {DB_PATH}")
    
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


@pytest.fixture(scope="session")
def sample_team_ids(db_connection):
    """
    Get sample team IDs for testing.
    Returns dict with specific known teams and a random pair.
    """
    c = db_connection.cursor()
    
    # Get Mississippi State and Auburn (featured teams)
    teams = {}
    for name, team_id in [("MSU", "mississippi-state"), ("Auburn", "auburn")]:
        c.execute("SELECT id FROM teams WHERE id = ?", (team_id,))
        row = c.fetchone()
        if row:
            teams[name] = row['id']
    
    # Get any two teams with completed games (for feature testing)
    c.execute("""
        SELECT DISTINCT home_team_id, away_team_id FROM games 
        WHERE status = 'final' AND home_score IS NOT NULL
        LIMIT 1
    """)
    row = c.fetchone()
    if row:
        teams['sample_home'] = row['home_team_id']
        teams['sample_away'] = row['away_team_id']
    
    return teams


@pytest.fixture(scope="session")
def sample_game_id(db_connection):
    """Get a sample completed game ID for testing."""
    c = db_connection.cursor()
    c.execute("""
        SELECT id FROM games 
        WHERE status = 'final' AND home_score IS NOT NULL
        ORDER BY date DESC LIMIT 1
    """)
    row = c.fetchone()
    return row['id'] if row else None


@pytest.fixture(scope="session")
def sample_upcoming_game(db_connection):
    """Get a sample upcoming/scheduled game for prediction testing."""
    c = db_connection.cursor()
    c.execute("""
        SELECT id, home_team_id, away_team_id FROM games 
        WHERE status = 'scheduled'
        ORDER BY date ASC LIMIT 1
    """)
    row = c.fetchone()
    if row:
        return {
            'id': row['id'],
            'home_team_id': row['home_team_id'],
            'away_team_id': row['away_team_id']
        }
    return None


# Expected tables in the database
EXPECTED_TABLES = [
    'games',
    'teams', 
    'team_aliases',
    'model_predictions',
    'elo_ratings',
    'player_stats',
    'betting_lines',
    'game_weather',
    'venues',
    'tracked_bets',
    'tracked_confident_bets',
]

# Required columns in games table
GAMES_REQUIRED_COLUMNS = [
    'id',
    'date', 
    'home_team_id',
    'away_team_id',
    'status',
    'home_score',
    'away_score',
]

# Valid game status values
VALID_GAME_STATUSES = ['scheduled', 'final', 'postponed', 'cancelled']
