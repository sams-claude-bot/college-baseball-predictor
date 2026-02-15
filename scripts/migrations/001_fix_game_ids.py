#!/usr/bin/env python3
"""
Migration: Fix game ID mismatches in model_predictions and totals_predictions tables.

The predictions tables use inconsistent formats:
  - Some: 2026-02-14-byu-western-kentucky (hyphens, variable team order)
  - Games: 2026-02-14_western-kentucky_byu (underscores, away_home order)

This script matches predictions to games by date + team names, then updates the game_id.
"""

import sqlite3
import re
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent.parent.parent / "data" / "baseball.db"

def normalize_team_name(name):
    """Normalize team name for matching"""
    return name.lower().replace('-', '').replace('_', '').replace(' ', '')

def extract_date_and_teams(game_id):
    """Extract date and team names from a game_id in any format"""
    # Try to extract date (YYYY-MM-DD)
    date_match = re.match(r'(\d{4}-\d{2}-\d{2})', game_id)
    if not date_match:
        return None, set()
    
    date = date_match.group(1)
    
    # Remove date and separators, get team parts
    remainder = game_id[len(date):]
    # Remove _g1, _g2 suffixes
    remainder = re.sub(r'_g\d+$', '', remainder)
    # Split by underscore or hyphen
    parts = re.split(r'[-_]', remainder)
    # Filter empty strings and join multi-word team names
    parts = [p for p in parts if p]
    
    # Build team name set (normalized)
    teams = set()
    if len(parts) >= 2:
        # Try to reconstruct team names
        # Common patterns: team1-team2 or team1_team2
        teams.add(normalize_team_name(parts[0] if len(parts) == 2 else '-'.join(parts[:len(parts)//2])))
        teams.add(normalize_team_name(parts[-1] if len(parts) == 2 else '-'.join(parts[len(parts)//2:])))
    
    return date, teams

def find_matching_game(conn, pred_game_id, all_games_by_date):
    """Find the actual game_id that matches a prediction's game_id"""
    date, pred_teams = extract_date_and_teams(pred_game_id)
    if not date:
        return None
    
    # Get candidate games for this date
    candidates = all_games_by_date.get(date, [])
    
    for game_id, home_team, away_team in candidates:
        game_teams = {normalize_team_name(home_team), normalize_team_name(away_team)}
        
        # Check if both teams match
        if pred_teams == game_teams:
            return game_id
        
        # More flexible matching - check if teams are subsets (for abbreviated names)
        pred_teams_list = list(pred_teams)
        game_teams_list = list(game_teams)
        
        matches = 0
        for pt in pred_teams_list:
            for gt in game_teams_list:
                if pt in gt or gt in pt:
                    matches += 1
                    break
        
        if matches >= 2:
            return game_id
    
    return None

def run_migration():
    print(f"Opening database: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # Build a lookup of all games by date
    cur.execute("SELECT id, home_team_id, away_team_id, date FROM games")
    all_games_by_date = {}
    for game_id, home, away, date in cur.fetchall():
        if date not in all_games_by_date:
            all_games_by_date[date] = []
        all_games_by_date[date].append((game_id, home, away))
    
    print(f"Loaded {sum(len(v) for v in all_games_by_date.values())} games")
    
    # Process model_predictions
    print("\n=== Fixing model_predictions ===")
    cur.execute("""
        SELECT DISTINCT game_id FROM model_predictions 
        WHERE game_id NOT IN (SELECT id FROM games)
    """)
    orphan_pred_ids = [row[0] for row in cur.fetchall()]
    print(f"Found {len(orphan_pred_ids)} distinct orphaned prediction game_ids")
    
    fixed = 0
    unfixable = []
    for pred_id in orphan_pred_ids:
        actual_id = find_matching_game(conn, pred_id, all_games_by_date)
        if actual_id:
            # Check if target already has predictions
            cur.execute("SELECT COUNT(*) FROM model_predictions WHERE game_id = ?", (actual_id,))
            existing = cur.fetchone()[0]
            
            if existing > 0:
                # Delete the orphan predictions (duplicates)
                cur.execute("DELETE FROM model_predictions WHERE game_id = ?", (pred_id,))
                print(f"  DELETED {pred_id} (duplicate of {actual_id})")
            else:
                cur.execute("UPDATE model_predictions SET game_id = ? WHERE game_id = ?", 
                           (actual_id, pred_id))
                print(f"  FIXED: {pred_id} -> {actual_id}")
            fixed += 1
        else:
            unfixable.append(pred_id)
    
    print(f"\nFixed/cleaned: {fixed}, Could not match: {len(unfixable)}")
    if unfixable:
        print("Unfixable IDs (will delete):")
        for uid in unfixable:
            print(f"  - {uid}")
            cur.execute("DELETE FROM model_predictions WHERE game_id = ?", (uid,))
    
    # Process totals_predictions
    print("\n=== Fixing totals_predictions ===")
    cur.execute("""
        SELECT DISTINCT game_id FROM totals_predictions 
        WHERE game_id NOT IN (SELECT id FROM games)
    """)
    orphan_totals_ids = [row[0] for row in cur.fetchall()]
    print(f"Found {len(orphan_totals_ids)} distinct orphaned totals game_ids")
    
    fixed_totals = 0
    unfixable_totals = []
    for pred_id in orphan_totals_ids:
        actual_id = find_matching_game(conn, pred_id, all_games_by_date)
        if actual_id:
            cur.execute("SELECT COUNT(*) FROM totals_predictions WHERE game_id = ?", (actual_id,))
            existing = cur.fetchone()[0]
            
            if existing > 0:
                cur.execute("DELETE FROM totals_predictions WHERE game_id = ?", (pred_id,))
                print(f"  DELETED {pred_id} (duplicate of {actual_id})")
            else:
                cur.execute("UPDATE totals_predictions SET game_id = ? WHERE game_id = ?", 
                           (actual_id, pred_id))
                print(f"  FIXED: {pred_id} -> {actual_id}")
            fixed_totals += 1
        else:
            unfixable_totals.append(pred_id)
    
    print(f"\nFixed/cleaned: {fixed_totals}, Could not match: {len(unfixable_totals)}")
    if unfixable_totals:
        print("Unfixable totals IDs (will delete):")
        for uid in unfixable_totals:
            print(f"  - {uid}")
            cur.execute("DELETE FROM totals_predictions WHERE game_id = ?", (uid,))
    
    conn.commit()
    
    # Verify
    print("\n=== Verification ===")
    cur.execute("SELECT COUNT(*) FROM model_predictions WHERE game_id NOT IN (SELECT id FROM games)")
    remaining_orphans = cur.fetchone()[0]
    print(f"Remaining orphaned model_predictions: {remaining_orphans}")
    
    cur.execute("SELECT COUNT(*) FROM totals_predictions WHERE game_id NOT IN (SELECT id FROM games)")
    remaining_totals_orphans = cur.fetchone()[0]
    print(f"Remaining orphaned totals_predictions: {remaining_totals_orphans}")
    
    conn.close()
    print("\n✅ Migration complete!")

if __name__ == "__main__":
    run_migration()
