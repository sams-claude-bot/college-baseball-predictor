#!/usr/bin/env python3
"""
Infer Probable Starters

Analyzes pitcher_game_log to identify rotation patterns and predicts starters
for upcoming games. Inserts predictions into pitching_matchups table.

College baseball rotation patterns:
- Friday/Saturday/Sunday weekend series: Ace/Game2/Game3 starters
- Midweek games: Usually a different starter (4th guy or bullpen day)
- Most teams have 3-4 regular starters
"""

import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

DB_PATH = Path(__file__).parent.parent / "data" / "baseball.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_team_rotation_history(conn, team_id: str, lookback_days: int = 21):
    """
    Get recent starting pitching history for a team.
    Returns list of (date, player_id, player_name, day_of_week, innings_pitched)
    """
    c = conn.cursor()
    
    cutoff_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    
    c.execute('''
        SELECT g.date, pgl.player_id, ps.name as player_name, pgl.innings_pitched,
               pgl.earned_runs, pgl.strikeouts
        FROM pitcher_game_log pgl
        JOIN games g ON pgl.game_id = g.id
        LEFT JOIN player_stats ps ON pgl.player_id = ps.id
        WHERE pgl.team_id = ?
        AND pgl.was_starter = 1
        AND g.date >= ?
        ORDER BY g.date DESC
    ''', (team_id, cutoff_date))
    
    results = []
    for row in c.fetchall():
        date = datetime.strptime(row['date'], '%Y-%m-%d')
        day_of_week = date.weekday()  # 0=Mon, 4=Fri, 5=Sat, 6=Sun
        results.append({
            'date': row['date'],
            'player_id': row['player_id'],
            'player_name': row['player_name'] or f"Unknown ({row['player_id']})",
            'day_of_week': day_of_week,
            'innings_pitched': row['innings_pitched'],
            'earned_runs': row['earned_runs'],
            'strikeouts': row['strikeouts']
        })
    
    return results


def analyze_rotation_pattern(starts: list):
    """
    Analyze starts to identify rotation pattern.
    Returns:
    {
        'friday_starter': (player_id, confidence),
        'saturday_starter': (player_id, confidence),
        'sunday_starter': (player_id, confidence),
        'midweek_starter': (player_id, confidence),
    }
    """
    if not starts:
        return {}
    
    # Group starts by day of week
    by_day = defaultdict(list)
    for start in starts:
        by_day[start['day_of_week']].append(start['player_id'])
    
    rotation = {}
    
    # Analyze each day
    day_names = {4: 'friday_starter', 5: 'saturday_starter', 6: 'sunday_starter'}
    midweek_days = [0, 1, 2, 3]  # Mon-Thu
    
    # Weekend starters (Fri/Sat/Sun)
    for dow, key in day_names.items():
        day_starts = by_day.get(dow, [])
        if day_starts:
            # Count occurrences
            counts = defaultdict(int)
            for pid in day_starts:
                counts[pid] += 1
            
            # Most frequent starter
            most_common = max(counts.items(), key=lambda x: x[1])
            player_id, count = most_common
            
            # Confidence based on consistency
            # 2+ starts on same day = high, 1 start = medium
            if count >= 2:
                confidence = 'high'
            elif len(day_starts) == 1:
                confidence = 'medium'
            else:
                confidence = 'low'
            
            rotation[key] = (player_id, confidence)
    
    # Midweek starters
    midweek_starts = []
    for dow in midweek_days:
        midweek_starts.extend(by_day.get(dow, []))
    
    if midweek_starts:
        counts = defaultdict(int)
        for pid in midweek_starts:
            counts[pid] += 1
        
        most_common = max(counts.items(), key=lambda x: x[1])
        player_id, count = most_common
        
        confidence = 'high' if count >= 2 else 'medium' if count == 1 else 'low'
        rotation['midweek_starter'] = (player_id, confidence)
    
    return rotation


def get_upcoming_games(conn, days_ahead: int = 7):
    """Get upcoming scheduled games."""
    c = conn.cursor()
    
    today = datetime.now().strftime('%Y-%m-%d')
    cutoff = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
    
    c.execute('''
        SELECT id, date, home_team_id, away_team_id
        FROM games
        WHERE date >= ? AND date <= ?
        AND status = 'scheduled'
        ORDER BY date
    ''', (today, cutoff))
    
    return c.fetchall()


def get_player_name(conn, player_id: int):
    """Get player name from player_stats."""
    if player_id is None or player_id < 0:
        return None
    
    c = conn.cursor()
    c.execute('SELECT name FROM player_stats WHERE id = ?', (player_id,))
    row = c.fetchone()
    return row['name'] if row else None


def infer_starter_for_game(conn, team_id: str, game_date: str, rotation: dict):
    """
    Infer the probable starter for a team on a specific date.
    Returns (player_id, player_name, confidence) or (None, None, 'none')
    """
    date = datetime.strptime(game_date, '%Y-%m-%d')
    dow = date.weekday()
    
    # Map day of week to rotation key
    if dow == 4:
        key = 'friday_starter'
    elif dow == 5:
        key = 'saturday_starter'
    elif dow == 6:
        key = 'sunday_starter'
    else:
        key = 'midweek_starter'
    
    if key in rotation:
        player_id, confidence = rotation[key]
        player_name = get_player_name(conn, player_id)
        return player_id, player_name, confidence
    
    # Fallback: use any starter we know about
    for fallback_key in ['friday_starter', 'saturday_starter', 'sunday_starter', 'midweek_starter']:
        if fallback_key in rotation:
            player_id, _ = rotation[fallback_key]
            player_name = get_player_name(conn, player_id)
            return player_id, player_name, 'low'
    
    return None, None, 'none'


def populate_pitching_matchups(conn, dry_run: bool = False):
    """
    Populate pitching_matchups table with inferred starters.
    """
    c = conn.cursor()
    
    # Get all teams with recent starts
    c.execute('''
        SELECT DISTINCT team_id FROM pitcher_game_log
        WHERE was_starter = 1
    ''')
    teams_with_data = [row['team_id'] for row in c.fetchall()]
    print(f"Found {len(teams_with_data)} teams with starter data")
    
    # Build rotation patterns for all teams
    rotations = {}
    for team_id in teams_with_data:
        starts = get_team_rotation_history(conn, team_id)
        if starts:
            rotations[team_id] = analyze_rotation_pattern(starts)
    
    print(f"Built rotation patterns for {len(rotations)} teams")
    
    # Get upcoming games
    upcoming = get_upcoming_games(conn)
    print(f"Found {len(upcoming)} upcoming games")
    
    # Clear existing matchups for upcoming games
    if not dry_run:
        for game in upcoming:
            c.execute('DELETE FROM pitching_matchups WHERE game_id = ?', (game['id'],))
    
    inserted = 0
    high_confidence = 0
    
    for game in upcoming:
        game_id = game['id']
        game_date = game['date']
        home_team = game['home_team_id']
        away_team = game['away_team_id']
        
        # Get rotation patterns
        home_rotation = rotations.get(home_team, {})
        away_rotation = rotations.get(away_team, {})
        
        # Infer starters
        home_pid, home_name, home_conf = infer_starter_for_game(conn, home_team, game_date, home_rotation)
        away_pid, away_name, away_conf = infer_starter_for_game(conn, away_team, game_date, away_rotation)
        
        # Only insert if we have at least one starter
        if home_pid or away_pid:
            notes = f"home_conf={home_conf}, away_conf={away_conf}"
            
            if not dry_run:
                c.execute('''
                    INSERT INTO pitching_matchups (
                        game_id, home_starter_id, away_starter_id,
                        home_starter_name, away_starter_name, notes
                    ) VALUES (?, ?, ?, ?, ?, ?)
                ''', (game_id, home_pid, away_pid, home_name, away_name, notes))
            
            inserted += 1
            if home_conf == 'high' or away_conf == 'high':
                high_confidence += 1
            
            if dry_run:
                date_str = datetime.strptime(game_date, '%Y-%m-%d').strftime('%a %m/%d')
                print(f"  {date_str}: {away_name or 'TBD'} ({away_conf}) @ {home_name or 'TBD'} ({home_conf})")
    
    if not dry_run:
        conn.commit()
    
    print(f"\nInserted {inserted} pitching matchups")
    print(f"  {high_confidence} with at least one high-confidence starter")


def show_rotation_summary(conn):
    """Display rotation analysis for top teams."""
    print("\n" + "="*60)
    print("ROTATION ANALYSIS - TOP TEAMS")
    print("="*60)
    
    # Get teams with most starts recorded
    c = conn.cursor()
    c.execute('''
        SELECT team_id, COUNT(*) as starts
        FROM pitcher_game_log
        WHERE was_starter = 1
        GROUP BY team_id
        ORDER BY starts DESC
        LIMIT 10
    ''')
    top_teams = c.fetchall()
    
    for team in top_teams:
        team_id = team['team_id']
        starts = get_team_rotation_history(conn, team_id)
        rotation = analyze_rotation_pattern(starts)
        
        print(f"\n{team_id.upper()} ({team['starts']} starts):")
        
        for key in ['friday_starter', 'saturday_starter', 'sunday_starter', 'midweek_starter']:
            if key in rotation:
                pid, conf = rotation[key]
                name = get_player_name(conn, pid) or f"ID:{pid}"
                day = key.replace('_starter', '').title()
                print(f"  {day:10} {name:25} ({conf})")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Infer probable starters')
    parser.add_argument('--dry-run', action='store_true', help='Show predictions without inserting')
    parser.add_argument('--summary', action='store_true', help='Show rotation analysis summary')
    parser.add_argument('--days', type=int, default=7, help='Days ahead to predict')
    args = parser.parse_args()
    
    print("Infer Probable Starters")
    print("="*50)
    
    conn = get_connection()
    
    try:
        if args.summary:
            show_rotation_summary(conn)
        
        populate_pitching_matchups(conn, dry_run=args.dry_run)
        
    finally:
        conn.close()
    
    print("\n✓ Starter inference complete")


if __name__ == "__main__":
    main()
