#!/usr/bin/env python3
"""
Database Migration Script v2

Adds tables/columns needed for:
- Pitching matchup model
- Conference strength model  
- Preseason prior model
- Dynamic ensemble weights

Run this once after updating the models.
"""

import sys
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from database import get_connection, DB_PATH


def run_migration():
    """Run all migrations"""
    print(f"Running migrations on {DB_PATH}")
    print("-" * 50)
    
    conn = get_connection()
    c = conn.cursor()
    
    migrations_run = 0
    
    # ===== Migration 1: Model predictions tracking table =====
    c.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='model_predictions'
    """)
    if not c.fetchone():
        print("Creating model_predictions table...")
        c.execute('''
            CREATE TABLE model_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                predicted_home_prob REAL,
                predicted_home_runs REAL,
                predicted_away_runs REAL,
                predicted_at TEXT DEFAULT CURRENT_TIMESTAMP,
                -- Filled in after game
                was_correct INTEGER,
                UNIQUE(game_id, model_name),
                FOREIGN KEY (game_id) REFERENCES games(id)
            )
        ''')
        c.execute('CREATE INDEX IF NOT EXISTS idx_model_pred_game ON model_predictions(game_id)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_model_pred_model ON model_predictions(model_name)')
        migrations_run += 1
        print("  ✓ Created model_predictions table")
    else:
        print("  ✓ model_predictions table already exists")
    
    # ===== Migration 2: Conference strength ratings table =====
    c.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='conference_ratings'
    """)
    if not c.fetchone():
        print("Creating conference_ratings table...")
        c.execute('''
            CREATE TABLE conference_ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conference TEXT NOT NULL,
                season INTEGER NOT NULL,
                strength_rating REAL DEFAULT 1.0,
                rpi_rank INTEGER,
                tournament_teams INTEGER,
                notes TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(conference, season)
            )
        ''')
        c.execute('CREATE INDEX IF NOT EXISTS idx_conf_rating ON conference_ratings(conference, season)')
        migrations_run += 1
        print("  ✓ Created conference_ratings table")
    else:
        print("  ✓ conference_ratings table already exists")
    
    # ===== Migration 3: Preseason priors table =====
    c.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='preseason_priors'
    """)
    if not c.fetchone():
        print("Creating preseason_priors table...")
        c.execute('''
            CREATE TABLE preseason_priors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_id TEXT NOT NULL,
                season INTEGER NOT NULL,
                preseason_rank INTEGER,
                preseason_elo REAL,
                projected_win_pct REAL,
                returning_war REAL,
                returning_starters INTEGER,
                poll_source TEXT DEFAULT 'd1baseball',
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(team_id, season),
                FOREIGN KEY (team_id) REFERENCES teams(id)
            )
        ''')
        c.execute('CREATE INDEX IF NOT EXISTS idx_preseason_team ON preseason_priors(team_id, season)')
        migrations_run += 1
        print("  ✓ Created preseason_priors table")
    else:
        print("  ✓ preseason_priors table already exists")
    
    # ===== Migration 4: Pitcher game log table =====
    c.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='pitcher_game_log'
    """)
    if not c.fetchone():
        print("Creating pitcher_game_log table...")
        c.execute('''
            CREATE TABLE pitcher_game_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                player_id INTEGER NOT NULL,
                team_id TEXT NOT NULL,
                was_starter INTEGER DEFAULT 0,
                innings_pitched REAL,
                hits_allowed INTEGER,
                runs_allowed INTEGER,
                earned_runs INTEGER,
                walks INTEGER,
                strikeouts INTEGER,
                pitches INTEGER,
                strikes INTEGER,
                decision TEXT,  -- W, L, S, H, or NULL
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (game_id) REFERENCES games(id),
                FOREIGN KEY (player_id) REFERENCES player_stats(id),
                FOREIGN KEY (team_id) REFERENCES teams(id)
            )
        ''')
        c.execute('CREATE INDEX IF NOT EXISTS idx_pitcher_log_game ON pitcher_game_log(game_id)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_pitcher_log_player ON pitcher_game_log(player_id)')
        migrations_run += 1
        print("  ✓ Created pitcher_game_log table")
    else:
        print("  ✓ pitcher_game_log table already exists")
    
    # ===== Migration 5: Add pitch count columns to player_stats =====
    c.execute("PRAGMA table_info(player_stats)")
    columns = [row[1] for row in c.fetchall()]
    
    new_columns = [
        ("season_pitches", "INTEGER DEFAULT 0"),
        ("avg_pitch_count", "REAL DEFAULT 0"),
        ("last_start_date", "TEXT"),
        ("last_start_pitches", "INTEGER")
    ]
    
    for col_name, col_def in new_columns:
        if col_name not in columns:
            print(f"Adding column {col_name} to player_stats...")
            c.execute(f"ALTER TABLE player_stats ADD COLUMN {col_name} {col_def}")
            migrations_run += 1
            print(f"  ✓ Added {col_name} column")
        else:
            print(f"  ✓ {col_name} column already exists")
    
    # ===== Migration 6: Add conference column to teams if missing =====
    c.execute("PRAGMA table_info(teams)")
    team_columns = [row[1] for row in c.fetchall()]
    
    if "conference" not in team_columns:
        print("Adding conference column to teams...")
        c.execute("ALTER TABLE teams ADD COLUMN conference TEXT")
        migrations_run += 1
        print("  ✓ Added conference column to teams")
    else:
        print("  ✓ conference column already exists in teams")
    
    # ===== Migration 7: Ensemble weights history =====
    c.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='ensemble_weights_history'
    """)
    if not c.fetchone():
        print("Creating ensemble_weights_history table...")
        c.execute('''
            CREATE TABLE ensemble_weights_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                weights_json TEXT NOT NULL,
                accuracy_json TEXT,
                total_predictions INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        migrations_run += 1
        print("  ✓ Created ensemble_weights_history table")
    else:
        print("  ✓ ensemble_weights_history table already exists")
    
    conn.commit()
    conn.close()
    
    print("-" * 50)
    print(f"Migration complete. {migrations_run} changes made.")
    return migrations_run


def verify_schema():
    """Verify all required tables and columns exist"""
    print("\nVerifying schema...")
    
    conn = get_connection()
    c = conn.cursor()
    
    # Check all required tables
    required_tables = [
        "teams", "games", "players", "player_stats", "pitching_matchups",
        "predictions", "elo_ratings", "rankings_history",
        "model_predictions", "conference_ratings", "preseason_priors",
        "pitcher_game_log", "ensemble_weights_history"
    ]
    
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    existing_tables = {row[0] for row in c.fetchall()}
    
    all_good = True
    for table in required_tables:
        if table in existing_tables:
            print(f"  ✓ {table}")
        else:
            print(f"  ✗ {table} MISSING")
            all_good = False
    
    conn.close()
    
    if all_good:
        print("\n✓ All required tables present")
    else:
        print("\n✗ Some tables missing - run migration")
    
    return all_good


def seed_conference_ratings(season=None):
    """Seed initial conference ratings"""
    sys.path.insert(0, str(BASE_DIR / "models"))
    from conference_model import CONFERENCE_RATINGS
    
    if season is None:
        season = datetime.now().year
    
    conn = get_connection()
    c = conn.cursor()
    
    print(f"\nSeeding conference ratings for {season}...")
    
    for conf, rating in CONFERENCE_RATINGS.items():
        c.execute('''
            INSERT INTO conference_ratings (conference, season, strength_rating)
            VALUES (?, ?, ?)
            ON CONFLICT(conference, season) DO UPDATE SET
                strength_rating = excluded.strength_rating,
                updated_at = CURRENT_TIMESTAMP
        ''', (conf, season, rating))
    
    conn.commit()
    conn.close()
    
    print(f"  ✓ Seeded {len(CONFERENCE_RATINGS)} conference ratings")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        verify_schema()
    elif len(sys.argv) > 1 and sys.argv[1] == "seed":
        seed_conference_ratings()
    else:
        run_migration()
        verify_schema()
        seed_conference_ratings()
