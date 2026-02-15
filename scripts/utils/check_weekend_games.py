#!/usr/bin/env python3

import sqlite3

def check_weekend_games():
    conn = sqlite3.connect('data/baseball.db')
    cursor = conn.cursor()
    
    weekend_dates = ['2026-02-14', '2026-02-15', '2026-02-16']
    
    print("=== CHECKING WEEKEND GAMES ===")
    
    for date in weekend_dates:
        cursor.execute("""
            SELECT id, home_team_id, away_team_id, status, venue
            FROM games 
            WHERE date = ?
            ORDER BY id
        """, (date,))
        
        games = cursor.fetchall()
        print(f"\n{date}: {len(games)} games")
        
        for game in games[:10]:  # Show first 10
            game_id, home_team, away_team, status, venue = game
            print(f"  {game_id}: {away_team} @ {home_team} | Status: {status} | Venue: {venue}")
    
    # Also check next few days
    other_dates = ['2026-02-17', '2026-02-18', '2026-02-19']
    
    print("\n=== CHECKING NEXT WEEK ===")
    for date in other_dates:
        cursor.execute("SELECT COUNT(*) FROM games WHERE date = ?", (date,))
        count = cursor.fetchone()[0]
        print(f"{date}: {count} games")
    
    conn.close()

if __name__ == "__main__":
    check_weekend_games()