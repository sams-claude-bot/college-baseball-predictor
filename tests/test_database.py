#!/usr/bin/env python3
"""
Database Schema and Data Integrity Tests

Tests to catch:
- Missing tables
- Missing required columns
- Orphan predictions (reference non-existent game_ids)
- Duplicate game IDs
- Invalid team_aliases
- DB schema drift from expected structure

Reference: CONTEXT.md describes expected schema and relationships.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# Expected tables in the database
EXPECTED_TABLES = [
    'games',
    'teams', 
    'team_aliases',
    'model_predictions',
    'totals_predictions',
    'elo_ratings',
    'player_stats',
    'betting_lines',
    'game_weather',
    'venues',
    'tracked_bets',
    'tracked_confident_bets',
    'team_pitching_quality',
    'team_batting_quality',
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
    'inning_text',
]

# Valid game status values
VALID_GAME_STATUSES = ['scheduled', 'final', 'postponed', 'cancelled', 'canceled', 'in-progress']


class TestSchemaIntegrity:
    """Test that database schema matches expectations."""
    
    def test_all_expected_tables_exist(self, db_connection):
        """All expected tables should exist in the database."""
        c = db_connection.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = {row['name'] for row in c.fetchall()}
        
        missing = []
        for table in EXPECTED_TABLES:
            if table not in existing_tables:
                missing.append(table)
        
        assert len(missing) == 0, (
            f"Missing tables: {missing}. "
            f"Existing tables: {sorted(existing_tables)}"
        )
    
    def test_games_has_required_columns(self, db_connection):
        """games table should have all required columns."""
        c = db_connection.cursor()
        c.execute("PRAGMA table_info(games)")
        columns = {row['name'] for row in c.fetchall()}
        
        missing = []
        for col in GAMES_REQUIRED_COLUMNS:
            if col not in columns:
                missing.append(col)
        
        assert len(missing) == 0, (
            f"games table missing columns: {missing}"
        )
    
    def test_model_predictions_has_required_columns(self, db_connection):
        """model_predictions table should have core columns."""
        c = db_connection.cursor()
        c.execute("PRAGMA table_info(model_predictions)")
        columns = {row['name'] for row in c.fetchall()}
        
        required = ['game_id', 'model_name', 'predicted_home_prob']
        missing = [col for col in required if col not in columns]
        
        assert len(missing) == 0, (
            f"model_predictions table missing columns: {missing}"
        )
    
    def test_team_aliases_has_required_columns(self, db_connection):
        """team_aliases table should have core columns."""
        c = db_connection.cursor()
        c.execute("PRAGMA table_info(team_aliases)")
        columns = {row['name'] for row in c.fetchall()}
        
        required = ['alias', 'team_id', 'source']
        missing = [col for col in required if col not in columns]
        
        assert len(missing) == 0, (
            f"team_aliases table missing columns: {missing}"
        )
    
    def test_totals_predictions_has_required_columns(self, db_connection):
        """totals_predictions table should have core columns."""
        c = db_connection.cursor()
        c.execute("PRAGMA table_info(totals_predictions)")
        columns = {row['name'] for row in c.fetchall()}
        
        required = ['game_id', 'over_under_line', 'model_name', 'projected_total']
        missing = [col for col in required if col not in columns]
        
        assert len(missing) == 0, (
            f"totals_predictions table missing columns: {missing}"
        )
    
    def test_team_pitching_quality_has_required_columns(self, db_connection):
        """team_pitching_quality table should have core columns."""
        c = db_connection.cursor()
        c.execute("PRAGMA table_info(team_pitching_quality)")
        columns = {row['name'] for row in c.fetchall()}
        
        required = ['team_id', 'ace_era', 'rotation_era', 'bullpen_era']
        missing = [col for col in required if col not in columns]
        
        assert len(missing) == 0, (
            f"team_pitching_quality table missing columns: {missing}"
        )
    
    def test_team_batting_quality_has_required_columns(self, db_connection):
        """team_batting_quality table should have core columns."""
        c = db_connection.cursor()
        c.execute("PRAGMA table_info(team_batting_quality)")
        columns = {row['name'] for row in c.fetchall()}
        
        required = ['team_id', 'lineup_ops', 'lineup_woba']
        missing = [col for col in required if col not in columns]
        
        assert len(missing) == 0, (
            f"team_batting_quality table missing columns: {missing}"
        )


class TestDataIntegrity:
    """Test data integrity and relationships."""
    
    def test_no_duplicate_game_ids(self, db_connection):
        """Game IDs should be unique."""
        c = db_connection.cursor()
        c.execute("""
            SELECT id, COUNT(*) as cnt 
            FROM games 
            GROUP BY id 
            HAVING cnt > 1
        """)
        duplicates = c.fetchall()
        
        if duplicates:
            dup_list = [f"{row['id']} ({row['cnt']}x)" for row in duplicates]
            pytest.fail(f"Duplicate game IDs found: {dup_list[:10]}")
    
    def test_game_status_values_valid(self, db_connection):
        """All game status values should be valid."""
        c = db_connection.cursor()
        c.execute("SELECT DISTINCT status FROM games")
        statuses = {row['status'] for row in c.fetchall()}
        
        invalid = statuses - set(VALID_GAME_STATUSES) - {None}
        
        assert len(invalid) == 0, (
            f"Invalid game status values found: {invalid}. "
            f"Valid values: {VALID_GAME_STATUSES}"
        )
    
    def test_no_orphan_predictions(self, db_connection):
        """model_predictions should reference valid game_ids."""
        c = db_connection.cursor()
        c.execute("""
            SELECT COUNT(*) as cnt
            FROM model_predictions mp
            LEFT JOIN games g ON mp.game_id = g.id
            WHERE g.id IS NULL
        """)
        row = c.fetchone()
        orphan_count = row['cnt']
        
        # Live DB tolerance: games may be rescheduled/removed after predictions made
        c.execute("SELECT COUNT(*) as total FROM model_predictions")
        total = c.fetchone()['total']
        orphan_pct = (orphan_count / total * 100) if total > 0 else 0
        
        assert orphan_pct < 1.0, (
            f"Found {orphan_count} orphan predictions ({orphan_pct:.1f}% of {total}). "
            f"Exceeds 1% tolerance."
        )
    
    def test_team_aliases_resolve_to_valid_teams(self, db_connection):
        """All team_aliases should reference existing team_ids."""
        c = db_connection.cursor()
        c.execute("""
            SELECT COUNT(*) as cnt
            FROM team_aliases ta
            LEFT JOIN teams t ON ta.team_id = t.id
            WHERE t.id IS NULL
        """)
        row = c.fetchone()
        orphan_count = row['cnt']
        
        # Live DB tolerance: aliases may reference non-D1 teams
        c.execute("SELECT COUNT(*) as total FROM team_aliases")
        total = c.fetchone()['total']
        orphan_pct = (orphan_count / total * 100) if total > 0 else 0
        
        assert orphan_pct < 15.0, (
            f"Found {orphan_count} orphan aliases ({orphan_pct:.1f}% of {total}). "
            f"Exceeds 15% tolerance."
        )
    
    def test_completed_games_have_scores(self, db_connection):
        """Games with status='final' should have scores."""
        c = db_connection.cursor()
        c.execute("""
            SELECT COUNT(*) as cnt
            FROM games
            WHERE status = 'final'
            AND (home_score IS NULL OR away_score IS NULL)
        """)
        row = c.fetchone()
        missing_scores = row['cnt']
        
        assert missing_scores == 0, (
            f"{missing_scores} final games missing scores"
        )
    
    def test_games_reference_valid_teams(self, db_connection):
        """Games should reference existing teams."""
        c = db_connection.cursor()
        
        # Check home_team_id
        c.execute("""
            SELECT COUNT(*) as cnt
            FROM games g
            LEFT JOIN teams t ON g.home_team_id = t.id
            WHERE t.id IS NULL
        """)
        home_orphans = c.fetchone()['cnt']
        
        # Check away_team_id
        c.execute("""
            SELECT COUNT(*) as cnt
            FROM games g
            LEFT JOIN teams t ON g.away_team_id = t.id
            WHERE t.id IS NULL
        """)
        away_orphans = c.fetchone()['cnt']
        
        total_orphans = home_orphans + away_orphans
        # Live DB tolerance: games may include non-D1 opponents
        c.execute("SELECT COUNT(*) as total FROM games")
        total_games = c.fetchone()['total']
        orphan_pct = (total_orphans / (total_games * 2) * 100) if total_games > 0 else 0
        
        assert orphan_pct < 2.0, (
            f"Games reference non-existent teams: "
            f"{home_orphans} home, {away_orphans} away ({orphan_pct:.1f}%). "
            f"Exceeds 2% tolerance."
        )


class TestDataContent:
    """Test that database has expected content."""
    
    def test_teams_table_not_empty(self, db_connection):
        """teams table should have data."""
        c = db_connection.cursor()
        c.execute("SELECT COUNT(*) as cnt FROM teams")
        count = c.fetchone()['cnt']
        
        # CONTEXT.md says 407 D1 teams
        assert count >= 300, (
            f"teams table has only {count} rows, expected 300+"
        )
    
    def test_games_table_not_empty(self, db_connection):
        """games table should have data."""
        c = db_connection.cursor()
        c.execute("SELECT COUNT(*) as cnt FROM games")
        count = c.fetchone()['cnt']
        
        assert count > 0, "games table is empty"
    
    def test_team_aliases_has_entries(self, db_connection):
        """team_aliases should have mappings."""
        c = db_connection.cursor()
        c.execute("SELECT COUNT(*) as cnt FROM team_aliases")
        count = c.fetchone()['cnt']
        
        # CONTEXT.md says 704 aliases
        assert count >= 100, (
            f"team_aliases has only {count} rows, expected 100+"
        )
    
    def test_elo_ratings_populated(self, db_connection):
        """elo_ratings should have entries."""
        c = db_connection.cursor()
        c.execute("SELECT COUNT(*) as cnt FROM elo_ratings")
        count = c.fetchone()['cnt']
        
        assert count > 0, "elo_ratings table is empty"
    
    def test_model_predictions_exist(self, db_connection):
        """model_predictions should have entries if games are final."""
        c = db_connection.cursor()
        
        # Check if we have final games
        c.execute("SELECT COUNT(*) FROM games WHERE status = 'final'")
        final_games = c.fetchone()[0]
        
        if final_games == 0:
            pytest.skip("No final games yet")
        
        c.execute("SELECT COUNT(*) as cnt FROM model_predictions")
        pred_count = c.fetchone()['cnt']
        
        assert pred_count > 0, (
            f"model_predictions is empty but {final_games} final games exist"
        )


class TestSilentFailures:
    """
    Tests to catch silent failures (jobs succeed but insert 0 rows).
    From lessons.md: "Silent failures are the worst failures"
    """
    
    def test_recent_predictions_exist(self, db_connection):
        """
        If we have recent scheduled games, we should have predictions.
        This catches silent prediction failures.
        """
        c = db_connection.cursor()
        
        # Get count of scheduled games in next 3 days
        c.execute("""
            SELECT COUNT(*) as cnt FROM games 
            WHERE status = 'scheduled'
            AND date >= date('now')
            AND date <= date('now', '+3 days')
        """)
        upcoming = c.fetchone()['cnt']
        
        if upcoming == 0:
            pytest.skip("No upcoming games in next 3 days")
        
        # Check if predictions exist for these games
        c.execute("""
            SELECT COUNT(DISTINCT g.id) as cnt
            FROM games g
            INNER JOIN model_predictions mp ON g.id = mp.game_id
            WHERE g.status = 'scheduled'
            AND g.date >= date('now')
            AND g.date <= date('now', '+3 days')
        """)
        predicted = c.fetchone()['cnt']
        
        # At least some games should have predictions
        coverage = predicted / upcoming if upcoming > 0 else 0
        assert coverage >= 0.5, (
            f"Only {predicted}/{upcoming} upcoming games have predictions. "
            f"Check if predict_and_track.py is running."
        )
