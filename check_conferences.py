#!/usr/bin/env python3

import sqlite3

def check_conferences():
    conn = sqlite3.connect('data/baseball.db')
    cursor = conn.cursor()
    
    # Get conference distribution
    cursor.execute("""
        SELECT conference, COUNT(*) as team_count
        FROM teams 
        GROUP BY conference
        ORDER BY team_count DESC
    """)
    
    conferences = cursor.fetchall()
    print("=== CONFERENCE DISTRIBUTION ===")
    for conf, count in conferences:
        print(f"{conf or 'NULL'}: {count} teams")
    
    # Show some sample teams
    print("\n=== SAMPLE TEAMS ===")
    cursor.execute("""
        SELECT id, name, nickname, conference 
        FROM teams 
        LIMIT 20
    """)
    
    sample_teams = cursor.fetchall()
    for team in sample_teams:
        print(f"{team[0]} | {team[1]} {team[2]} | Conference: {team[3] or 'None'}")
    
    # Check for SEC teams specifically
    print("\n=== SEARCHING FOR SEC-LIKE TEAMS ===")
    cursor.execute("""
        SELECT id, name, nickname, conference 
        FROM teams 
        WHERE LOWER(name) LIKE '%mississippi%' 
        OR LOWER(name) LIKE '%alabama%'
        OR LOWER(name) LIKE '%georgia%'
        OR LOWER(name) LIKE '%florida%'
        OR LOWER(name) LIKE '%tennessee%'
        OR LOWER(name) LIKE '%arkansas%'
        OR LOWER(name) LIKE '%auburn%'
        OR LOWER(name) LIKE '%vanderbilt%'
        OR LOWER(name) LIKE '%texas a%'
        OR LOWER(name) LIKE '%south carolina%'
        OR LOWER(name) LIKE '%kentucky%'
        OR LOWER(name) LIKE '%missouri%'
        OR LOWER(name) LIKE '%lsu%'
        OR LOWER(name) LIKE '%ole miss%'
        OR LOWER(name) LIKE '%oklahoma%'
    """)
    
    sec_like = cursor.fetchall()
    for team in sec_like:
        print(f"{team[0]} | {team[1]} {team[2]} | Conference: {team[3] or 'None'}")
    
    conn.close()

if __name__ == "__main__":
    check_conferences()