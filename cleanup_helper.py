#!/usr/bin/env python3

import sqlite3
import json
from datetime import datetime

def examine_database():
    """Examine the current database structure and contents"""
    conn = sqlite3.connect('data/baseball.db')
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    print("=== DATABASE STRUCTURE ===")
    for table in tables:
        table_name = table[0]
        print(f"\n--- {table_name} ---")
        
        # Get table schema
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        for col in columns:
            print(f"  {col[1]} ({col[2]})")
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"  Rows: {count}")
        
        if table_name == 'games' and count < 50:
            # Show some sample games if the count is reasonable
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 10")
            rows = cursor.fetchall()
            print("  Sample rows:")
            for row in rows:
                print(f"    {row}")
    
    conn.close()

def find_tournament_games():
    """Find games that appear to be tournament games"""
    conn = sqlite3.connect('data/baseball.db')
    cursor = conn.cursor()
    
    print("\n=== POTENTIAL TOURNAMENT GAMES ===")
    
    # Look for games with neutral site indicators
    tournament_keywords = [
        'shriners', 'globe life', 'puerto rico', 'desert invitational', 
        'surprise', 'showdown', 'classic', 'tournament', 'invitational'
    ]
    
    for keyword in tournament_keywords:
        cursor.execute("""
            SELECT id, home_team, away_team, game_date, venue, notes 
            FROM games 
            WHERE LOWER(venue) LIKE ? OR LOWER(notes) LIKE ?
            ORDER BY game_date
        """, (f'%{keyword}%', f'%{keyword}%'))
        
        results = cursor.fetchall()
        if results:
            print(f"\n--- Games matching '{keyword}' ---")
            for game in results:
                print(f"  ID: {game[0]} | {game[2]} @ {game[1]} | {game[3]} | {game[4]} | {game[5]}")
    
    # Also look for games with unusual venues (not typical home team venues)
    cursor.execute("""
        SELECT id, home_team, away_team, game_date, venue, notes 
        FROM games 
        WHERE venue IS NOT NULL 
        AND venue NOT LIKE '%Stadium' 
        AND venue NOT LIKE '%Field' 
        AND venue NOT LIKE '%Park'
        AND venue NOT LIKE 'Dudy Noble%'
        ORDER BY game_date
    """)
    
    results = cursor.fetchall()
    if results:
        print(f"\n--- Games with unusual venues ---")
        for game in results:
            print(f"  ID: {game[0]} | {game[2]} @ {game[1]} | {game[3]} | {game[4]} | {game[5]}")
    
    conn.close()

if __name__ == "__main__":
    examine_database()
    find_tournament_games()