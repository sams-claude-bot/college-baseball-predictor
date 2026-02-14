#!/usr/bin/env python3

import sqlite3
import json
from datetime import datetime

def identify_tournament_games_to_remove():
    """Identify all tournament games that need to be removed"""
    conn = sqlite3.connect('data/baseball.db')
    cursor = conn.cursor()
    
    games_to_remove = []
    tournaments_to_remove = []
    
    # Target tournament names/keywords
    tournament_targets = [
        'amegy-bank-2026',  # Amegy Bank College Baseball Series (Globe Life Field)
        'las-vegas-classic-2026',  # Las Vegas Classic
    ]
    
    # Target venue patterns
    venue_targets = [
        'globe life field',
        'surprise, az', 
        'terry park, fort myers',  # Likely spring training
        'tony gwynn stadium, san diego',  # Neutral site
    ]
    
    print("=== IDENTIFYING GAMES TO REMOVE ===")
    
    # 1. Remove specific tournaments
    for tournament_id in tournament_targets:
        cursor.execute("SELECT * FROM tournaments WHERE id = ?", (tournament_id,))
        tournament = cursor.fetchone()
        if tournament:
            print(f"\nTournament to remove: {tournament[1]} ({tournament[0]})")
            tournaments_to_remove.append(tournament[0])
            
            # Find all games in this tournament
            cursor.execute("""
                SELECT id, home_team_id, away_team_id, date, venue
                FROM games WHERE tournament_id = ?
            """, (tournament_id,))
            
            tournament_games = cursor.fetchall()
            for game in tournament_games:
                games_to_remove.append({
                    'id': game[0],
                    'reason': f'Tournament: {tournament[1]}',
                    'home': game[1],
                    'away': game[2], 
                    'date': game[3],
                    'venue': game[4]
                })
                print(f"  Game: {game[0]} | {game[2]} @ {game[1]} | {game[3]}")
    
    # 2. Remove neutral site games at target venues
    for venue_pattern in venue_targets:
        cursor.execute("""
            SELECT id, home_team_id, away_team_id, date, venue, tournament_id
            FROM games 
            WHERE LOWER(venue) LIKE ? AND tournament_id IS NULL
        """, (f'%{venue_pattern}%',))
        
        venue_games = cursor.fetchall()
        if venue_games:
            print(f"\nNeutral site games at '{venue_pattern}':")
            for game in venue_games:
                games_to_remove.append({
                    'id': game[0],
                    'reason': f'Neutral site: {venue_pattern}',
                    'home': game[1],
                    'away': game[2], 
                    'date': game[3],
                    'venue': game[4]
                })
                print(f"  Game: {game[0]} | {game[2]} @ {game[1]} | {game[3]} | {game[4]}")
    
    # 3. Find any other suspicious neutral site games
    cursor.execute("""
        SELECT id, home_team_id, away_team_id, date, venue
        FROM games 
        WHERE is_neutral_site = 1 AND tournament_id IS NULL
        AND venue NOT LIKE '%Stadium' 
        AND venue NOT LIKE '%Field'
        AND venue NOT LIKE '%Park'
        AND venue IS NOT NULL
    """)
    
    other_neutral = cursor.fetchall()
    if other_neutral:
        print(f"\nOther suspicious neutral site games:")
        for game in other_neutral:
            # Skip if already in removal list
            if not any(g['id'] == game[0] for g in games_to_remove):
                print(f"  Review: {game[0]} | {game[2]} @ {game[1]} | {game[3]} | {game[4]}")
    
    conn.close()
    
    print(f"\n=== SUMMARY ===")
    print(f"Tournaments to remove: {len(tournaments_to_remove)}")
    print(f"Games to remove: {len(games_to_remove)}")
    
    return games_to_remove, tournaments_to_remove

def remove_tournament_games(games_to_remove, tournaments_to_remove, dry_run=True):
    """Remove the identified games and tournaments"""
    conn = sqlite3.connect('data/baseball.db')
    cursor = conn.cursor()
    
    print(f"\n=== {'DRY RUN' if dry_run else 'REMOVING'} TOURNAMENT DATA ===")
    
    removed_games = 0
    removed_tournaments = 0
    
    # Remove games
    for game in games_to_remove:
        if dry_run:
            print(f"Would remove game: {game['id']} ({game['reason']})")
        else:
            # Remove from related tables
            tables_with_game_id = ['game_predictions', 'betting_lines', 'pitching_matchups', 'pitcher_game_log']
            
            for table in tables_with_game_id:
                cursor.execute(f"DELETE FROM {table} WHERE game_id = ?", (game['id'],))
                affected = cursor.rowcount
                if affected > 0:
                    print(f"  Removed {affected} records from {table}")
            
            # Remove the main game record
            cursor.execute("DELETE FROM games WHERE id = ?", (game['id'],))
            if cursor.rowcount > 0:
                print(f"Removed game: {game['id']} ({game['reason']})")
                removed_games += 1
    
    # Remove tournaments
    for tournament_id in tournaments_to_remove:
        if dry_run:
            print(f"Would remove tournament: {tournament_id}")
        else:
            cursor.execute("DELETE FROM tournaments WHERE id = ?", (tournament_id,))
            if cursor.rowcount > 0:
                print(f"Removed tournament: {tournament_id}")
                removed_tournaments += 1
    
    if not dry_run:
        conn.commit()
        print(f"\nCommitted changes: {removed_games} games, {removed_tournaments} tournaments removed")
    
    conn.close()
    
    return removed_games, removed_tournaments

def clean_orphaned_data():
    """Clean up any orphaned records after game removal"""
    conn = sqlite3.connect('data/baseball.db')
    cursor = conn.cursor()
    
    print("\n=== CLEANING ORPHANED DATA ===")
    
    # Find orphaned betting lines
    cursor.execute("""
        DELETE FROM betting_lines 
        WHERE game_id NOT IN (SELECT id FROM games)
    """)
    orphaned_betting = cursor.rowcount
    if orphaned_betting > 0:
        print(f"Removed {orphaned_betting} orphaned betting lines")
    
    # Find orphaned predictions
    cursor.execute("""
        DELETE FROM game_predictions 
        WHERE game_id NOT IN (SELECT id FROM games)
    """)
    orphaned_predictions = cursor.rowcount
    if orphaned_predictions > 0:
        print(f"Removed {orphaned_predictions} orphaned predictions")
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    # First, identify what needs to be removed
    games_to_remove, tournaments_to_remove = identify_tournament_games_to_remove()
    
    # Show dry run first
    print("\n" + "="*50)
    print("DRY RUN - REVIEW BEFORE ACTUAL REMOVAL")
    print("="*50)
    remove_tournament_games(games_to_remove, tournaments_to_remove, dry_run=True)
    
    # Ask for confirmation (in real script, would wait for input)
    print("\n" + "="*50)
    print("PROCEEDING WITH ACTUAL REMOVAL...")
    print("="*50)
    
    # Actually remove
    removed_games, removed_tournaments = remove_tournament_games(games_to_remove, tournaments_to_remove, dry_run=False)
    
    # Clean up orphaned data
    clean_orphaned_data()
    
    # Save removal log
    removal_log = {
        'timestamp': datetime.now().isoformat(),
        'games_removed': len(games_to_remove),
        'tournaments_removed': len(tournaments_to_remove),
        'games': [g for g in games_to_remove],
        'tournaments': tournaments_to_remove
    }
    
    with open('data/tournament_removal_log.json', 'w') as f:
        json.dump(removal_log, f, indent=2)
    
    print(f"\nCleaned up {removed_games} tournament games and {removed_tournaments} tournaments")
    print("Removal log saved to data/tournament_removal_log.json")