#!/usr/bin/env python3
"""Migration: Add advanced stats columns to player_stats and create player_stats_snapshots."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / 'data' / 'baseball.db'

ADVANCED_COLUMNS = [
    # Advanced batting
    ('k_pct', 'REAL'), ('bb_pct', 'REAL'), ('k_bb_ratio', 'REAL'),
    ('iso', 'REAL'), ('babip', 'REAL'), ('woba', 'REAL'),
    ('wrc', 'REAL'), ('wraa', 'REAL'), ('wrc_plus', 'REAL'),
    # Batted ball batting
    ('gb_pct', 'REAL'), ('ld_pct', 'REAL'), ('fb_pct', 'REAL'),
    ('pu_pct', 'REAL'), ('hr_fb_pct', 'REAL'),
    # Advanced pitching
    ('fip', 'REAL'), ('xfip', 'REAL'), ('siera', 'REAL'), ('lob_pct', 'REAL'),
    ('k_pct_pitch', 'REAL'), ('bb_pct_pitch', 'REAL'), ('k_bb_pct_pitch', 'REAL'),
    ('k_bb_ratio_pitch', 'REAL'), ('babip_pitch', 'REAL'),
    ('obp_against', 'REAL'), ('slg_against', 'REAL'), ('ops_against', 'REAL'),
    # Batted ball pitching
    ('gb_pct_pitch', 'REAL'), ('ld_pct_pitch', 'REAL'), ('fb_pct_pitch', 'REAL'),
    ('pu_pct_pitch', 'REAL'), ('hr_fb_pct_pitch', 'REAL'),
]

def migrate():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    
    # 1. Add advanced columns to player_stats
    existing = {row[1] for row in conn.execute("PRAGMA table_info(player_stats)").fetchall()}
    for col, typ in ADVANCED_COLUMNS:
        if col not in existing:
            conn.execute(f"ALTER TABLE player_stats ADD COLUMN {col} {typ}")
            print(f"  Added {col} to player_stats")
    
    # 2. Create player_stats_snapshots
    conn.execute("""
        CREATE TABLE IF NOT EXISTS player_stats_snapshots (
            snapshot_date TEXT NOT NULL,
            team_id TEXT NOT NULL,
            player_name TEXT NOT NULL,
            player_class TEXT,
            position TEXT,
            -- Standard batting
            games INTEGER, pa INTEGER, at_bats INTEGER, runs INTEGER, hits INTEGER,
            doubles INTEGER, triples INTEGER, home_runs INTEGER, rbi INTEGER,
            hbp INTEGER, walks INTEGER, strikeouts INTEGER, stolen_bases INTEGER, caught_stealing INTEGER,
            batting_avg REAL, obp REAL, slg REAL, ops REAL,
            -- Advanced batting
            k_pct REAL, bb_pct REAL, k_bb_ratio REAL, iso REAL, babip REAL,
            woba REAL, wrc REAL, wraa REAL, wrc_plus REAL,
            -- Batted ball batting
            gb_pct REAL, ld_pct REAL, fb_pct REAL, pu_pct REAL, hr_fb_pct REAL,
            -- Standard pitching
            wins INTEGER, losses INTEGER, era REAL, appearances INTEGER, games_started INTEGER,
            complete_games INTEGER, shutouts INTEGER, saves INTEGER,
            innings_pitched REAL, hits_allowed INTEGER, runs_allowed INTEGER, earned_runs INTEGER,
            walks_allowed INTEGER, strikeouts_pitched INTEGER, hbp_pitch INTEGER, ba_against REAL,
            -- Advanced pitching
            obp_against REAL, slg_against REAL, ops_against REAL, babip_pitch REAL,
            bb_pct_pitch REAL, k_pct_pitch REAL, k_bb_pct_pitch REAL, k_bb_ratio_pitch REAL,
            whip REAL, fip REAL, xfip REAL, siera REAL, lob_pct REAL,
            -- Batted ball pitching
            gb_pct_pitch REAL, ld_pct_pitch REAL, fb_pct_pitch REAL,
            pu_pct_pitch REAL, hr_fb_pct_pitch REAL,
            PRIMARY KEY (snapshot_date, team_id, player_name)
        )
    """)
    print("  Created player_stats_snapshots table")
    
    conn.commit()
    conn.close()
    print("Migration complete!")

if __name__ == '__main__':
    migrate()
