#!/usr/bin/env python3
"""
SQLite database for college baseball tracking

Core tables (created by init_database):
- teams: Team info, conference, rankings
- players: Roster with stats
- games: Individual game results
- tournaments: Multi-team events
- team_stats: Season aggregates
- predictions: Legacy prediction tracking
- model_predictions: Per-model prediction tracking
- betting_lines: DraftKings odds
- totals_predictions: O/U prediction tracking

Additional tables created on-demand by their respective modules:
- elo_ratings, elo_history: Created by models/elo_model.py
- team_pitching_quality, team_batting_quality: Created by compute scripts
- player_stats: Created by stats scrapers
- game_weather: Created by weather.py
- See sqlite3 .tables for full list (~40 tables)
"""

import sqlite3
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DB_PATH = BASE_DIR / "data" / "baseball.db"

def get_connection():
    """Get database connection"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row  # Dict-like access
    return conn

def init_database():
    """Initialize database schema"""
    conn = get_connection()
    c = conn.cursor()
    
    # Teams table
    c.execute('''
        CREATE TABLE IF NOT EXISTS teams (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            nickname TEXT,
            conference TEXT,
            division TEXT,
            athletics_url TEXT,
            preseason_rank INTEGER,
            current_rank INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Players table
    c.execute('''
        CREATE TABLE IF NOT EXISTS players (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_id TEXT NOT NULL,
            name TEXT NOT NULL,
            number INTEGER,
            position TEXT,
            year TEXT,
            height TEXT,
            weight INTEGER,
            bats TEXT,
            throws TEXT,
            hometown TEXT,
            -- Batting stats
            games_played INTEGER DEFAULT 0,
            at_bats INTEGER DEFAULT 0,
            hits INTEGER DEFAULT 0,
            doubles INTEGER DEFAULT 0,
            triples INTEGER DEFAULT 0,
            home_runs INTEGER DEFAULT 0,
            rbis INTEGER DEFAULT 0,
            walks INTEGER DEFAULT 0,
            strikeouts INTEGER DEFAULT 0,
            stolen_bases INTEGER DEFAULT 0,
            batting_avg REAL DEFAULT 0,
            -- Pitching stats
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            saves INTEGER DEFAULT 0,
            innings_pitched REAL DEFAULT 0,
            earned_runs INTEGER DEFAULT 0,
            strikeouts_pitched INTEGER DEFAULT 0,
            walks_pitched INTEGER DEFAULT 0,
            era REAL DEFAULT 0,
            whip REAL DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (team_id) REFERENCES teams(id)
        )
    ''')
    
    # Tournaments table
    c.execute('''
        CREATE TABLE IF NOT EXISTS tournaments (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            location TEXT,
            venue TEXT,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            teams TEXT,  -- JSON array of team IDs
            notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Games table
    c.execute('''
        CREATE TABLE IF NOT EXISTS games (
            id TEXT PRIMARY KEY,
            date TEXT NOT NULL,
            time TEXT,
            home_team_id TEXT NOT NULL,
            away_team_id TEXT NOT NULL,
            home_score INTEGER,
            away_score INTEGER,
            winner_id TEXT,
            innings INTEGER DEFAULT 9,
            is_conference_game INTEGER DEFAULT 0,
            is_neutral_site INTEGER DEFAULT 0,
            tournament_id TEXT,
            venue TEXT,
            attendance INTEGER,
            notes TEXT,
            status TEXT DEFAULT 'scheduled',  -- scheduled, final, postponed, cancelled
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (home_team_id) REFERENCES teams(id),
            FOREIGN KEY (away_team_id) REFERENCES teams(id),
            FOREIGN KEY (tournament_id) REFERENCES tournaments(id)
        )
    ''')
    
    # Team season stats (aggregated)
    c.execute('''
        CREATE TABLE IF NOT EXISTS team_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_id TEXT NOT NULL,
            season INTEGER NOT NULL,
            games_played INTEGER DEFAULT 0,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            conference_wins INTEGER DEFAULT 0,
            conference_losses INTEGER DEFAULT 0,
            runs_scored INTEGER DEFAULT 0,
            runs_allowed INTEGER DEFAULT 0,
            batting_avg REAL DEFAULT 0,
            era REAL DEFAULT 0,
            fielding_pct REAL DEFAULT 0,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(team_id, season),
            FOREIGN KEY (team_id) REFERENCES teams(id)
        )
    ''')
    
    # Predictions table (for tracking model accuracy)
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            predicted_winner_id TEXT,
            home_win_prob REAL,
            away_win_prob REAL,
            projected_home_runs REAL,
            projected_away_runs REAL,
            projected_total REAL,
            home_run_line_prob REAL,  -- P(home -1.5)
            away_run_line_prob REAL,  -- P(away +1.5)
            series_home_prob REAL,
            model_version TEXT DEFAULT 'v1',
            predicted_at TEXT DEFAULT CURRENT_TIMESTAMP,
            -- Actual results (filled in after game)
            actual_winner_id TEXT,
            actual_home_score INTEGER,
            actual_away_score INTEGER,
            moneyline_correct INTEGER,  -- 1 if prediction was right
            run_line_correct INTEGER,
            FOREIGN KEY (game_id) REFERENCES games(id)
        )
    ''')
    
    # Model predictions (per-model tracking)
    c.execute('''
        CREATE TABLE IF NOT EXISTS model_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            predicted_home_prob REAL,
            predicted_home_runs REAL,
            predicted_away_runs REAL,
            predicted_at TEXT DEFAULT CURRENT_TIMESTAMP,
            was_correct INTEGER,
            UNIQUE(game_id, model_name),
            FOREIGN KEY (game_id) REFERENCES games(id)
        )
    ''')
    
    # Betting lines (DraftKings odds)
    c.execute('''
        CREATE TABLE IF NOT EXISTS betting_lines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT,
            date TEXT NOT NULL,
            home_team_id TEXT NOT NULL,
            away_team_id TEXT NOT NULL,
            book TEXT DEFAULT 'draftkings',
            home_ml INTEGER,
            away_ml INTEGER,
            home_spread REAL,
            home_spread_odds INTEGER,
            away_spread REAL,
            away_spread_odds INTEGER,
            over_under REAL,
            over_odds INTEGER,
            under_odds INTEGER,
            captured_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (home_team_id) REFERENCES teams(id),
            FOREIGN KEY (away_team_id) REFERENCES teams(id)
        )
    ''')
    
    # Totals predictions (O/U tracking)
    c.execute('''
        CREATE TABLE IF NOT EXISTS totals_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            betting_line_id TEXT,
            over_under_line REAL NOT NULL,
            projected_total REAL NOT NULL,
            prediction TEXT NOT NULL,
            edge_pct REAL,
            confidence REAL,
            predicted_at TEXT DEFAULT CURRENT_TIMESTAMP,
            actual_total INTEGER,
            was_correct INTEGER,
            model_name TEXT DEFAULT 'runs_ensemble',
            UNIQUE(game_id, over_under_line, model_name),
            FOREIGN KEY (game_id) REFERENCES games(id)
        )
    ''')
    
    # Create indexes for common queries
    c.execute('CREATE INDEX IF NOT EXISTS idx_games_date ON games(date)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_games_teams ON games(home_team_id, away_team_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_games_tournament ON games(tournament_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_players_team ON players(team_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_predictions_game ON predictions(game_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_model_pred_game ON model_predictions(game_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_model_pred_model ON model_predictions(model_name)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_betting_lines_game ON betting_lines(game_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_betting_lines_date ON betting_lines(date)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_totals_game ON totals_predictions(game_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_totals_model ON totals_predictions(model_name)')
    
    conn.commit()
    conn.close()
    print(f"✓ Database initialized at {DB_PATH}")

def add_team(team_id, name, nickname=None, conference=None, division=None, 
             athletics_url=None, preseason_rank=None):
    """Add or update a team"""
    conn = get_connection()
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO teams (id, name, nickname, conference, division, athletics_url, preseason_rank)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            name=excluded.name,
            nickname=excluded.nickname,
            conference=excluded.conference,
            division=excluded.division,
            athletics_url=excluded.athletics_url,
            preseason_rank=excluded.preseason_rank,
            updated_at=CURRENT_TIMESTAMP
    ''', (team_id, name, nickname, conference, division, athletics_url, preseason_rank))
    
    conn.commit()
    conn.close()
    return team_id

def add_tournament(tournament_id, name, location, start_date, end_date, 
                   venue=None, teams=None, notes=None):
    """Add a tournament"""
    import json
    conn = get_connection()
    c = conn.cursor()
    
    teams_json = json.dumps(teams) if teams else None
    
    c.execute('''
        INSERT INTO tournaments (id, name, location, venue, start_date, end_date, teams, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            name=excluded.name,
            location=excluded.location,
            venue=excluded.venue,
            start_date=excluded.start_date,
            end_date=excluded.end_date,
            teams=excluded.teams,
            notes=excluded.notes
    ''', (tournament_id, name, location, venue, start_date, end_date, teams_json, notes))
    
    conn.commit()
    conn.close()
    return tournament_id

def add_game(date, home_team_id, away_team_id, home_score=None, away_score=None,
             time=None, tournament_id=None, is_neutral_site=False, 
             is_conference_game=False, venue=None, notes=None, status='scheduled'):
    """Add a game"""
    conn = get_connection()
    c = conn.cursor()
    
    game_id = f"{date}_{away_team_id}_{home_team_id}".lower().replace(" ", "-")
    
    winner_id = None
    if home_score is not None and away_score is not None:
        winner_id = home_team_id if home_score > away_score else away_team_id
        status = 'final'
    
    c.execute('''
        INSERT INTO games (id, date, time, home_team_id, away_team_id, home_score, away_score,
                          winner_id, is_conference_game, is_neutral_site, tournament_id, 
                          venue, notes, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            home_score=excluded.home_score,
            away_score=excluded.away_score,
            winner_id=excluded.winner_id,
            status=excluded.status,
            updated_at=CURRENT_TIMESTAMP
    ''', (game_id, date, time, home_team_id, away_team_id, home_score, away_score,
          winner_id, int(is_conference_game), int(is_neutral_site), tournament_id,
          venue, notes, status))
    
    conn.commit()
    conn.close()
    return game_id

def get_team_record(team_id, conference_only=False):
    """Get team's win-loss record"""
    conn = get_connection()
    c = conn.cursor()
    
    if conference_only:
        c.execute('''
            SELECT 
                COUNT(CASE WHEN winner_id = ? THEN 1 END) as wins,
                COUNT(CASE WHEN winner_id != ? AND winner_id IS NOT NULL THEN 1 END) as losses
            FROM games
            WHERE (home_team_id = ? OR away_team_id = ?)
            AND is_conference_game = 1
            AND status = 'final'
        ''', (team_id, team_id, team_id, team_id))
    else:
        c.execute('''
            SELECT 
                COUNT(CASE WHEN winner_id = ? THEN 1 END) as wins,
                COUNT(CASE WHEN winner_id != ? AND winner_id IS NOT NULL THEN 1 END) as losses
            FROM games
            WHERE (home_team_id = ? OR away_team_id = ?)
            AND status = 'final'
        ''', (team_id, team_id, team_id, team_id))
    
    row = c.fetchone()
    conn.close()
    
    wins, losses = row['wins'], row['losses']
    return {
        'team_id': team_id,
        'wins': wins,
        'losses': losses,
        'pct': wins / (wins + losses) if (wins + losses) > 0 else 0
    }

def get_team_runs(team_id):
    """Get team's runs scored and allowed"""
    conn = get_connection()
    c = conn.cursor()
    
    c.execute('''
        SELECT 
            SUM(CASE WHEN home_team_id = ? THEN home_score ELSE away_score END) as runs_scored,
            SUM(CASE WHEN home_team_id = ? THEN away_score ELSE home_score END) as runs_allowed,
            COUNT(*) as games
        FROM games
        WHERE (home_team_id = ? OR away_team_id = ?)
        AND status = 'final'
    ''', (team_id, team_id, team_id, team_id))
    
    row = c.fetchone()
    conn.close()
    
    return {
        'team_id': team_id,
        'runs_scored': row['runs_scored'] or 0,
        'runs_allowed': row['runs_allowed'] or 0,
        'games': row['games'] or 0
    }

def get_recent_games(team_id=None, limit=10):
    """Get recent games, optionally filtered by team"""
    conn = get_connection()
    c = conn.cursor()
    
    if team_id:
        c.execute('''
            SELECT g.*, 
                   ht.name as home_team_name, 
                   at.name as away_team_name,
                   t.name as tournament_name
            FROM games g
            LEFT JOIN teams ht ON g.home_team_id = ht.id
            LEFT JOIN teams at ON g.away_team_id = at.id
            LEFT JOIN tournaments t ON g.tournament_id = t.id
            WHERE (g.home_team_id = ? OR g.away_team_id = ?)
            ORDER BY g.date DESC, g.time DESC
            LIMIT ?
        ''', (team_id, team_id, limit))
    else:
        c.execute('''
            SELECT g.*, 
                   ht.name as home_team_name, 
                   at.name as away_team_name,
                   t.name as tournament_name
            FROM games g
            LEFT JOIN teams ht ON g.home_team_id = ht.id
            LEFT JOIN teams at ON g.away_team_id = at.id
            LEFT JOIN tournaments t ON g.tournament_id = t.id
            ORDER BY g.date DESC, g.time DESC
            LIMIT ?
        ''', (limit,))
    
    rows = c.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

def get_upcoming_games(team_id=None, days=7):
    """Get upcoming scheduled games"""
    conn = get_connection()
    c = conn.cursor()
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    if team_id:
        c.execute('''
            SELECT g.*, 
                   ht.name as home_team_name, 
                   at.name as away_team_name,
                   t.name as tournament_name
            FROM games g
            LEFT JOIN teams ht ON g.home_team_id = ht.id
            LEFT JOIN teams at ON g.away_team_id = at.id
            LEFT JOIN tournaments t ON g.tournament_id = t.id
            WHERE (g.home_team_id = ? OR g.away_team_id = ?)
            AND g.date >= ?
            AND g.status = 'scheduled'
            ORDER BY g.date, g.time
        ''', (team_id, team_id, today))
    else:
        c.execute('''
            SELECT g.*, 
                   ht.name as home_team_name, 
                   at.name as away_team_name,
                   t.name as tournament_name
            FROM games g
            LEFT JOIN teams ht ON g.home_team_id = ht.id
            LEFT JOIN teams at ON g.away_team_id = at.id
            LEFT JOIN tournaments t ON g.tournament_id = t.id
            WHERE g.date >= ?
            AND g.status = 'scheduled'
            ORDER BY g.date, g.time
        ''', (today,))
    
    rows = c.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

def get_tournament_games(tournament_id):
    """Get all games in a tournament"""
    conn = get_connection()
    c = conn.cursor()
    
    c.execute('''
        SELECT g.*, 
               ht.name as home_team_name, 
               at.name as away_team_name
        FROM games g
        LEFT JOIN teams ht ON g.home_team_id = ht.id
        LEFT JOIN teams at ON g.away_team_id = at.id
        WHERE g.tournament_id = ?
        ORDER BY g.date, g.time
    ''', (tournament_id,))
    
    rows = c.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

def main():
    import sys
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        
        if cmd == "init":
            init_database()
        elif cmd == "stats":
            print(f"\nDatabase: {DB_PATH}")
            conn = get_connection()
            c = conn.cursor()
            
            c.execute("SELECT COUNT(*) FROM teams")
            print(f"Teams: {c.fetchone()[0]}")
            
            c.execute("SELECT COUNT(*) FROM games")
            print(f"Games: {c.fetchone()[0]}")
            
            c.execute("SELECT COUNT(*) FROM games WHERE status='final'")
            print(f"Completed: {c.fetchone()[0]}")
            
            c.execute("SELECT COUNT(*) FROM tournaments")
            print(f"Tournaments: {c.fetchone()[0]}")
            
            c.execute("SELECT COUNT(*) FROM players")
            print(f"Players: {c.fetchone()[0]}")
            
            conn.close()
        else:
            print(f"Unknown command: {cmd}")
    else:
        print("Usage:")
        print("  python database.py init   - Initialize database")
        print("  python database.py stats  - Show database stats")

if __name__ == "__main__":
    main()


# ============================================
# Rankings Functions
# ============================================

def add_ranking(team_id, rank, poll="d1baseball", week=None, date=None):
    """Add or update a team's ranking"""
    from datetime import datetime
    
    conn = get_connection()
    c = conn.cursor()
    
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    # Ensure team exists
    c.execute("SELECT id FROM teams WHERE id = ?", (team_id,))
    if not c.fetchone():
        # Auto-add team if not exists
        c.execute('''
            INSERT INTO teams (id, name) VALUES (?, ?)
        ''', (team_id, team_id.replace("-", " ").title()))
    
    # Update team's current rank
    c.execute('''
        UPDATE teams SET current_rank = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    ''', (rank, team_id))
    
    # Insert ranking history
    c.execute('''
        INSERT INTO rankings_history (team_id, rank, poll, week, date)
        VALUES (?, ?, ?, ?, ?)
    ''', (team_id, rank, poll, week, date))
    
    conn.commit()
    conn.close()
    
    return team_id

def get_current_top_25(poll="d1baseball"):
    """Get current Top 25 teams"""
    conn = get_connection()
    c = conn.cursor()
    
    c.execute('''
        SELECT t.id, t.name, t.nickname, t.conference, t.current_rank
        FROM teams t
        WHERE t.current_rank IS NOT NULL AND t.current_rank <= 25
        ORDER BY t.current_rank
    ''')
    
    rows = c.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

def get_ranking_history(team_id):
    """Get ranking history for a team"""
    conn = get_connection()
    c = conn.cursor()
    
    c.execute('''
        SELECT rank, poll, week, date
        FROM rankings_history
        WHERE team_id = ?
        ORDER BY date DESC
    ''', (team_id,))
    
    rows = c.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

def init_rankings_table():
    """Add rankings history table if not exists"""
    conn = get_connection()
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS rankings_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_id TEXT NOT NULL,
            rank INTEGER NOT NULL,
            poll TEXT DEFAULT 'd1baseball',
            week INTEGER,
            date TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (team_id) REFERENCES teams(id)
        )
    ''')
    
    c.execute('CREATE INDEX IF NOT EXISTS idx_rankings_team ON rankings_history(team_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_rankings_date ON rankings_history(date)')
    
    conn.commit()
    conn.close()
    print("✓ Rankings table initialized")
