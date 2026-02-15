#!/usr/bin/env python3

import sqlite3
import json
from datetime import datetime

def find_tournament_games():
    """Find games that appear to be tournament games"""
    conn = sqlite3.connect('data/baseball.db')
    cursor = conn.cursor()
    
    print("\n=== CURRENT TOURNAMENTS ===")
    cursor.execute("SELECT * FROM tournaments")
    tournaments = cursor.fetchall()
    for tournament in tournaments:
        print(f"ID: {tournament[0]} | Name: {tournament[1]} | Location: {tournament[2]} | Venue: {tournament[3]}")
        print(f"  Dates: {tournament[4]} to {tournament[5]} | Notes: {tournament[7]}")
    
    print("\n=== TOURNAMENT GAMES ===")
    cursor.execute("""
        SELECT id, home_team_id, away_team_id, date, venue, tournament_id, notes, is_neutral_site 
        FROM games 
        WHERE tournament_id IS NOT NULL
        ORDER BY date
    """)
    
    tournament_games = cursor.fetchall()
    print(f"Found {len(tournament_games)} games with tournament_id")
    for game in tournament_games:
        print(f"  ID: {game[0]} | {game[2]} @ {game[1]} | {game[3]} | Tournament: {game[5]} | Neutral: {game[7]}")
    
    print("\n=== NEUTRAL SITE GAMES ===")
    cursor.execute("""
        SELECT id, home_team_id, away_team_id, date, venue, tournament_id, notes, is_neutral_site 
        FROM games 
        WHERE is_neutral_site = 1
        ORDER BY date
    """)
    
    neutral_games = cursor.fetchall()
    print(f"Found {len(neutral_games)} neutral site games")
    for game in neutral_games:
        print(f"  ID: {game[0]} | {game[2]} @ {game[1]} | {game[3]} | Venue: {game[4]} | Tournament: {game[5]}")
    
    # Look for games with suspicious venue names
    print("\n=== GAMES WITH SUSPICIOUS VENUES ===")
    tournament_keywords = [
        'shriners', 'globe life', 'puerto rico', 'desert invitational', 
        'surprise', 'showdown', 'classic', 'tournament', 'invitational',
        'arlington', 'arizona'
    ]
    
    for keyword in tournament_keywords:
        cursor.execute("""
            SELECT id, home_team_id, away_team_id, date, venue, notes 
            FROM games 
            WHERE LOWER(venue) LIKE ? OR LOWER(notes) LIKE ?
            ORDER BY date
        """, (f'%{keyword}%', f'%{keyword}%'))
        
        results = cursor.fetchall()
        if results:
            print(f"\n--- Games matching '{keyword}' ---")
            for game in results:
                print(f"  ID: {game[0]} | {game[2]} @ {game[1]} | {game[3]} | {game[4]} | {game[5]}")
    
    conn.close()

def examine_specific_tournaments():
    """Look at specific tournaments we need to remove"""
    conn = sqlite3.connect('data/baseball.db')
    cursor = conn.cursor()
    
    target_tournaments = [
        'shriners', 'globe life', 'puerto rico', 'desert', 'surprise'
    ]
    
    print("\n=== TOURNAMENTS TO REMOVE ===")
    
    for target in target_tournaments:
        # Check tournaments table
        cursor.execute("""
            SELECT id, name, location, venue FROM tournaments 
            WHERE LOWER(name) LIKE ? OR LOWER(location) LIKE ?
        """, (f'%{target}%', f'%{target}%'))
        
        tournaments = cursor.fetchall()
        if tournaments:
            print(f"\n--- Tournaments matching '{target}' ---")
            for t in tournaments:
                print(f"  Tournament: {t[0]} | {t[1]} | {t[2]} | {t[3]}")
                
                # Count games in this tournament
                cursor.execute("SELECT COUNT(*) FROM games WHERE tournament_id = ?", (t[0],))
                game_count = cursor.fetchone()[0]
                print(f"    Games in this tournament: {game_count}")
                
                if game_count < 20:  # Show games if reasonable count
                    cursor.execute("""
                        SELECT id, home_team_id, away_team_id, date 
                        FROM games WHERE tournament_id = ?
                        ORDER BY date
                    """, (t[0],))
                    games = cursor.fetchall()
                    for g in games:
                        print(f"      Game: {g[0]} | {g[2]} @ {g[1]} | {g[3]}")
    
    conn.close()

if __name__ == "__main__":
    find_tournament_games()
    examine_specific_tournaments()