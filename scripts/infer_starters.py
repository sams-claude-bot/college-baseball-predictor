#!/usr/bin/env python3
"""
Infer Probable Starters - Series-Based Rotation Logic

College baseball rotations are based on SERIES POSITION, not day of week:
- Game 1 of a weekend series → ace (usually Friday starter)
- Game 2 → second starter
- Game 3 → third starter
- Midweek games → separate rotation (4th starter / bullpen day)

A "series" = consecutive games against the same opponent.
"""

import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

DB_PATH = Path(__file__).parent.parent / "data" / "baseball.db"

# Days that are typically "weekend series" days
WEEKEND_DAYS = {4, 5, 6}  # Fri, Sat, Sun
MIDWEEK_DAYS = {0, 1, 2, 3}  # Mon-Thu


def get_connection():
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def group_into_series(games):
    """
    Group games into series based on opponent.
    A series = consecutive games vs the same opponent (possibly with gaps of 1 day).
    
    Returns list of series, each series is a list of games with 'series_position' added.
    """
    if not games:
        return []
    
    series_list = []
    current_series = [games[0]]
    
    for i in range(1, len(games)):
        prev = current_series[-1]
        curr = games[i]
        
        # Same opponent and within 2 days = same series
        prev_date = datetime.strptime(prev['game_date'], '%Y-%m-%d')
        curr_date = datetime.strptime(curr['game_date'], '%Y-%m-%d')
        same_opp = prev['opponent'] == curr['opponent']
        close_dates = (curr_date - prev_date).days <= 2
        
        if same_opp and close_dates:
            current_series.append(curr)
        else:
            series_list.append(current_series)
            current_series = [curr]
    
    series_list.append(current_series)
    
    # Add series_position to each game
    for series in series_list:
        for i, game in enumerate(series):
            game['series_position'] = i + 1
    
    return series_list


def is_weekend_series(series):
    """Check if a series overlaps with Fri/Sat/Sun."""
    for game in series:
        date = datetime.strptime(game['game_date'], '%Y-%m-%d')
        if date.weekday() in WEEKEND_DAYS:
            return True
    return False


def get_team_rotation(conn, team_id, lookback_days=21):
    """
    Analyze a team's rotation based on series position.
    
    Returns dict:
    {
        'weekend': {1: (name, confidence), 2: (name, conf), 3: (name, conf)},
        'midweek': [(name, confidence), ...]
    }
    """
    c = conn.cursor()
    cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    
    c.execute('''
        SELECT game_date, game_num, day_of_week, opponent, starting_pitcher
        FROM lineup_history
        WHERE team_id = ? AND game_date >= ?
        ORDER BY game_date, game_num
    ''', (team_id, cutoff))
    
    games = [dict(r) for r in c.fetchall()]
    if not games:
        return {'weekend': {}, 'midweek': []}
    
    series_list = group_into_series(games)
    
    # Track starters by series position
    weekend_starters = defaultdict(list)  # position -> [names]
    midweek_starters = []
    
    for series in series_list:
        if is_weekend_series(series):
            for game in series:
                pos = game['series_position']
                weekend_starters[pos].append(game['starting_pitcher'])
        else:
            for game in series:
                midweek_starters.append(game['starting_pitcher'])
    
    # Build rotation with confidence
    rotation = {'weekend': {}, 'midweek': []}
    
    for pos, names in weekend_starters.items():
        counts = defaultdict(int)
        for n in names:
            counts[n] += 1
        best = max(counts.items(), key=lambda x: x[1])
        name, count = best
        conf = 'high' if count >= 2 else 'medium'
        rotation['weekend'][pos] = (name, conf)
    
    if midweek_starters:
        counts = defaultdict(int)
        for n in midweek_starters:
            counts[n] += 1
        # Return all midweek starters sorted by frequency
        rotation['midweek'] = sorted(counts.items(), key=lambda x: -x[1])
    
    return rotation


def group_upcoming_into_series(games):
    """Group upcoming games (from games table) into series."""
    if not games:
        return []
    
    series_list = []
    current_series = [games[0]]
    
    for i in range(1, len(games)):
        prev = current_series[-1]
        curr = games[i]
        
        prev_date = datetime.strptime(prev['date'], '%Y-%m-%d')
        curr_date = datetime.strptime(curr['date'], '%Y-%m-%d')
        
        # Determine opponent from perspective of the team
        prev_opp = prev['_opponent']
        curr_opp = curr['_opponent']
        
        same_opp = prev_opp == curr_opp
        close_dates = (curr_date - prev_date).days <= 2
        
        if same_opp and close_dates:
            current_series.append(curr)
        else:
            series_list.append(current_series)
            current_series = [curr]
    
    series_list.append(current_series)
    return series_list


def find_player_id(conn, team_id, player_name):
    """Look up player_stats ID by name and team."""
    if not player_name:
        return None
    c = conn.cursor()
    c.execute('''
        SELECT id FROM player_stats 
        WHERE team_id = ? AND LOWER(name) = LOWER(?)
        LIMIT 1
    ''', (team_id, player_name))
    row = c.fetchone()
    if row:
        return row['id']
    
    # Partial match on last name
    parts = player_name.split()
    if parts:
        c.execute('''
            SELECT id FROM player_stats 
            WHERE team_id = ? AND LOWER(name) LIKE ?
            LIMIT 1
        ''', (team_id, f'%{parts[-1].lower()}%'))
        row = c.fetchone()
        if row:
            return row['id']
    return None


def populate_pitching_matchups(conn, days_ahead=7, dry_run=False):
    """
    Populate pitching_matchups for upcoming games using series-based rotation logic.
    """
    c = conn.cursor()
    today = datetime.now().strftime('%Y-%m-%d')
    cutoff = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
    
    # Get all teams with lineup history
    c.execute('SELECT DISTINCT team_id FROM lineup_history')
    teams_with_data = {r['team_id'] for r in c.fetchall()}
    print(f"Teams with starter data: {len(teams_with_data)}")
    
    # Build rotations
    rotations = {}
    for team_id in teams_with_data:
        rot = get_team_rotation(conn, team_id)
        if rot['weekend'] or rot['midweek']:
            rotations[team_id] = rot
    print(f"Teams with rotation patterns: {len(rotations)}")
    
    # Get upcoming games
    c.execute('''
        SELECT id, date, home_team_id, away_team_id
        FROM games
        WHERE date >= ? AND date <= ?
        AND status = 'scheduled'
        ORDER BY date
    ''', (today, cutoff))
    upcoming = [dict(r) for r in c.fetchall()]
    print(f"Upcoming games: {len(upcoming)}")
    
    # For each team, get their upcoming games and group into series
    team_games = defaultdict(list)
    for game in upcoming:
        home = game['home_team_id']
        away = game['away_team_id']
        
        home_game = dict(game)
        home_game['_team'] = home
        home_game['_opponent'] = away
        team_games[home].append(home_game)
        
        away_game = dict(game)
        away_game['_team'] = away
        away_game['_opponent'] = home
        team_games[away].append(away_game)
    
    # Predict starters per team per game
    predictions = {}  # game_id -> {home_starter, away_starter, ...}
    
    for team_id, games in team_games.items():
        if team_id not in rotations:
            continue
        
        rot = rotations[team_id]
        series_list = group_upcoming_into_series(games)

        # Tournament weekends can have 1-game "series" vs different opponents each day.
        # In those cases, keep a rolling weekend slot (Game 1/2/3) across singleton sets
        # so we don't assign the Game 1 starter every day.
        weekend_singleton_slot = 0

        for series in series_list:
            first_date = datetime.strptime(series[0]['date'], '%Y-%m-%d')
            is_weekend = first_date.weekday() in WEEKEND_DAYS or (
                len(series) >= 3 and any(
                    datetime.strptime(g['date'], '%Y-%m-%d').weekday() in WEEKEND_DAYS
                    for g in series
                )
            )

            singleton_weekend_series = is_weekend and len(series) == 1

            for i, game in enumerate(series):
                game_id = game['id']
                pos = i + 1
                if singleton_weekend_series:
                    pos = (weekend_singleton_slot % 3) + 1
                    weekend_singleton_slot += 1
                
                if game_id not in predictions:
                    predictions[game_id] = {
                        'game_id': game_id, 'date': game['date'],
                        'home_team': game['home_team_id'], 'away_team': game['away_team_id'],
                        'home_starter': None, 'home_starter_id': None, 'home_conf': 'none',
                        'away_starter': None, 'away_starter_id': None, 'away_conf': 'none',
                    }
                
                pred = predictions[game_id]
                is_home = game['_team'] == game['home_team_id']
                
                if is_weekend and pos in rot['weekend']:
                    name, conf = rot['weekend'][pos]
                elif not is_weekend and rot['midweek']:
                    # Use midweek rotation (pick by position in midweek series)
                    idx = min(i, len(rot['midweek']) - 1)
                    name, _ = rot['midweek'][idx]
                    conf = 'medium' if len(rot['midweek']) > 1 else 'low'
                else:
                    name, conf = None, 'none'
                
                if name:
                    pid = find_player_id(conn, team_id, name)
                    if is_home:
                        pred['home_starter'] = name
                        pred['home_starter_id'] = pid
                        pred['home_conf'] = conf
                    else:
                        pred['away_starter'] = name
                        pred['away_starter_id'] = pid
                        pred['away_conf'] = conf
    
    # Write to DB
    if not dry_run:
        for game_id in predictions:
            c.execute('DELETE FROM pitching_matchups WHERE game_id = ?', (game_id,))
    
    inserted = 0
    for pred in predictions.values():
        if pred['home_starter'] or pred['away_starter']:
            notes = f"home_conf={pred['home_conf']}, away_conf={pred['away_conf']}"
            
            if dry_run:
                date = datetime.strptime(pred['date'], '%Y-%m-%d')
                dow = date.strftime('%a')
                print(f"  {dow} {pred['date']}: {pred['away_starter'] or 'TBD'} ({pred['away_conf']}) @ {pred['home_starter'] or 'TBD'} ({pred['home_conf']})")
            else:
                c.execute('''
                    INSERT INTO pitching_matchups (
                        game_id, home_starter_id, away_starter_id,
                        home_starter_name, away_starter_name, notes
                    ) VALUES (?, ?, ?, ?, ?, ?)
                ''', (pred['game_id'], pred['home_starter_id'], pred['away_starter_id'],
                      pred['home_starter'], pred['away_starter'], notes))
            inserted += 1
    
    if not dry_run:
        conn.commit()
    
    print(f"\n{'Would insert' if dry_run else 'Inserted'} {inserted} pitching matchups")


def show_team_rotation(conn, team_id):
    """Show rotation analysis for a specific team."""
    rot = get_team_rotation(conn, team_id)
    
    c = conn.cursor()
    c.execute('SELECT name FROM teams WHERE id = ?', (team_id,))
    row = c.fetchone()
    name = row['name'] if row else team_id
    
    print(f"\n{'='*50}")
    print(f"ROTATION: {name}")
    print(f"{'='*50}")
    
    if rot['weekend']:
        print("\nWeekend Series:")
        for pos in sorted(rot['weekend'].keys()):
            starter, conf = rot['weekend'][pos]
            print(f"  Game {pos}: {starter:25s} ({conf})")
    
    if rot['midweek']:
        print("\nMidweek:")
        for name, count in rot['midweek']:
            print(f"  {name:25s} ({count} start{'s' if count > 1 else ''})")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Infer probable starters (series-based)')
    parser.add_argument('--dry-run', action='store_true', help='Show predictions without inserting')
    parser.add_argument('--team', type=str, help='Show rotation for specific team')
    parser.add_argument('--days', type=int, default=7, help='Days ahead to predict')
    args = parser.parse_args()
    
    print("Infer Probable Starters (Series-Based)")
    print("=" * 50)
    
    conn = get_connection()
    
    try:
        if args.team:
            show_team_rotation(conn, args.team)
        
        populate_pitching_matchups(conn, days_ahead=args.days, dry_run=args.dry_run)
    finally:
        conn.close()
    
    print("\n✓ Done")


if __name__ == "__main__":
    main()
