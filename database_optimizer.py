#!/usr/bin/env python3

import sqlite3
import json
from datetime import datetime

class DatabaseOptimizer:
    def __init__(self, db_path='data/baseball.db'):
        self.db_path = db_path
        self.optimization_log = []
        
    def log_action(self, action, details, rows_affected=0):
        """Log an optimization action"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details,
            'rows_affected': rows_affected
        }
        self.optimization_log.append(entry)
        print(f"[OPTIMIZE] {action}: {details} ({rows_affected} rows)")
    
    def analyze_database_structure(self):
        """Analyze current database structure and identify issues"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        print("=== DATABASE ANALYSIS ===")
        
        analysis = {}
        for table in tables:
            # Get table info
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            row_count = cursor.fetchone()[0]
            
            # Check for indexes
            cursor.execute(f"PRAGMA index_list({table})")
            indexes = cursor.fetchall()
            
            analysis[table] = {
                'row_count': row_count,
                'columns': len(columns),
                'indexes': len(indexes),
                'column_details': columns
            }
            
            print(f"\n{table}: {row_count} rows, {len(columns)} columns, {len(indexes)} indexes")
        
        conn.close()
        return analysis
    
    def find_orphaned_records(self):
        """Find and remove orphaned records"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        print("\n=== FINDING ORPHANED RECORDS ===")
        
        # Check for betting lines without corresponding games
        cursor.execute("""
            SELECT COUNT(*) FROM betting_lines 
            WHERE game_id NOT IN (SELECT id FROM games)
        """)
        orphaned_betting = cursor.fetchone()[0]
        
        if orphaned_betting > 0:
            cursor.execute("""
                DELETE FROM betting_lines 
                WHERE game_id NOT IN (SELECT id FROM games)
            """)
            self.log_action("REMOVE_ORPHANED", "betting_lines records", orphaned_betting)
        
        # Check for predictions without corresponding games
        cursor.execute("""
            SELECT COUNT(*) FROM game_predictions 
            WHERE game_id NOT IN (SELECT id FROM games)
        """)
        orphaned_predictions = cursor.fetchone()[0]
        
        if orphaned_predictions > 0:
            cursor.execute("""
                DELETE FROM game_predictions 
                WHERE game_id NOT IN (SELECT id FROM games)
            """)
            self.log_action("REMOVE_ORPHANED", "game_predictions records", orphaned_predictions)
        
        # Check for player stats without corresponding teams
        cursor.execute("""
            SELECT COUNT(*) FROM player_stats 
            WHERE team_id NOT IN (SELECT id FROM teams)
        """)
        orphaned_player_stats = cursor.fetchone()[0]
        
        if orphaned_player_stats > 0:
            cursor.execute("""
                DELETE FROM player_stats 
                WHERE team_id NOT IN (SELECT id FROM teams)
            """)
            self.log_action("REMOVE_ORPHANED", "player_stats records", orphaned_player_stats)
        
        # Check for elo ratings without corresponding teams
        cursor.execute("""
            SELECT COUNT(*) FROM elo_ratings 
            WHERE team_id NOT IN (SELECT id FROM teams)
        """)
        orphaned_elo = cursor.fetchone()[0]
        
        if orphaned_elo > 0:
            cursor.execute("""
                DELETE FROM elo_ratings 
                WHERE team_id NOT IN (SELECT id FROM teams)
            """)
            self.log_action("REMOVE_ORPHANED", "elo_ratings records", orphaned_elo)
        
        conn.commit()
        conn.close()
        
        total_orphaned = orphaned_betting + orphaned_predictions + orphaned_player_stats + orphaned_elo
        print(f"Removed {total_orphaned} total orphaned records")
        return total_orphaned
    
    def clean_inconsistent_team_ids(self):
        """Clean up inconsistent team IDs"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        print("\n=== CLEANING INCONSISTENT TEAM IDs ===")
        
        # Get all team IDs from teams table
        cursor.execute("SELECT id FROM teams ORDER BY id")
        valid_team_ids = set(row[0] for row in cursor.fetchall())
        print(f"Valid team IDs: {len(valid_team_ids)}")
        
        # Check games table
        cursor.execute("""
            SELECT DISTINCT home_team_id FROM games 
            WHERE home_team_id NOT IN (SELECT id FROM teams)
        """)
        invalid_home_teams = cursor.fetchall()
        
        cursor.execute("""
            SELECT DISTINCT away_team_id FROM games 
            WHERE away_team_id NOT IN (SELECT id FROM teams)
        """)
        invalid_away_teams = cursor.fetchall()
        
        if invalid_home_teams or invalid_away_teams:
            print(f"Found {len(invalid_home_teams)} invalid home teams, {len(invalid_away_teams)} invalid away teams")
            # For now, just log - would need manual review to fix
            self.log_action("IDENTIFY_INCONSISTENT", f"team IDs in games table", 
                          len(invalid_home_teams) + len(invalid_away_teams))
        
        conn.close()
    
    def add_missing_indexes(self):
        """Add missing database indexes for performance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        print("\n=== ADDING MISSING INDEXES ===")
        
        # Important indexes for performance
        indexes_to_add = [
            # Games table - most queried
            ("idx_games_date", "games", "date"),
            ("idx_games_home_team", "games", "home_team_id"),
            ("idx_games_away_team", "games", "away_team_id"),
            ("idx_games_status", "games", "status"),
            
            # Betting lines
            ("idx_betting_lines_game", "betting_lines", "game_id"),
            ("idx_betting_lines_date", "betting_lines", "date"),
            
            # Game predictions  
            ("idx_game_predictions_game", "game_predictions", "game_id"),
            ("idx_game_predictions_model", "game_predictions", "model"),
            
            # Player stats
            ("idx_player_stats_team", "player_stats", "team_id"),
            ("idx_player_stats_position", "player_stats", "position"),
            
            # Rankings history
            ("idx_rankings_team", "rankings_history", "team_id"),
            ("idx_rankings_date", "rankings_history", "date"),
            
            # ELO ratings
            ("idx_elo_team", "elo_ratings", "team_id"),
        ]
        
        for index_name, table, column in indexes_to_add:
            try:
                # Check if index already exists
                cursor.execute(f"PRAGMA index_info({index_name})")
                existing = cursor.fetchall()
                
                if not existing:
                    cursor.execute(f"CREATE INDEX {index_name} ON {table}({column})")
                    self.log_action("CREATE_INDEX", f"{index_name} on {table}.{column}", 0)
                else:
                    print(f"Index {index_name} already exists")
            
            except sqlite3.Error as e:
                print(f"Error creating index {index_name}: {e}")
        
        conn.commit()
        conn.close()
    
    def ensure_referential_integrity(self):
        """Ensure referential integrity without foreign keys"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        print("\n=== CHECKING REFERENTIAL INTEGRITY ===")
        
        integrity_issues = []
        
        # Check games -> teams relationship
        cursor.execute("""
            SELECT COUNT(*) FROM games 
            WHERE home_team_id NOT IN (SELECT id FROM teams)
            OR away_team_id NOT IN (SELECT id FROM teams)
        """)
        invalid_games = cursor.fetchone()[0]
        if invalid_games > 0:
            integrity_issues.append(f"Games with invalid team references: {invalid_games}")
        
        # Check betting_lines -> games relationship
        cursor.execute("""
            SELECT COUNT(*) FROM betting_lines 
            WHERE game_id NOT IN (SELECT id FROM games)
        """)
        invalid_betting = cursor.fetchone()[0]
        if invalid_betting > 0:
            integrity_issues.append(f"Betting lines with invalid game references: {invalid_betting}")
        
        # Check game_predictions -> games relationship  
        cursor.execute("""
            SELECT COUNT(*) FROM game_predictions 
            WHERE game_id NOT IN (SELECT id FROM games)
        """)
        invalid_predictions = cursor.fetchone()[0]
        if invalid_predictions > 0:
            integrity_issues.append(f"Predictions with invalid game references: {invalid_predictions}")
        
        if integrity_issues:
            print("Referential integrity issues found:")
            for issue in integrity_issues:
                print(f"  - {issue}")
            self.log_action("INTEGRITY_ISSUES", f"Found {len(integrity_issues)} types of issues", 
                          invalid_games + invalid_betting + invalid_predictions)
        else:
            print("✅ Referential integrity is good")
            self.log_action("INTEGRITY_CHECK", "All relationships valid", 0)
        
        conn.close()
        return integrity_issues
    
    def vacuum_database(self):
        """Vacuum the database to reclaim space and optimize"""
        conn = sqlite3.connect(self.db_path)
        
        # Get size before vacuum
        cursor = conn.cursor()
        cursor.execute("PRAGMA page_count")
        pages_before = cursor.fetchone()[0]
        cursor.execute("PRAGMA page_size")
        page_size = cursor.fetchone()[0]
        size_before = pages_before * page_size
        
        print(f"\n=== VACUUMING DATABASE ===")
        print(f"Size before vacuum: {size_before / 1024 / 1024:.2f} MB")
        
        # Vacuum
        conn.execute("VACUUM")
        
        # Get size after vacuum
        cursor.execute("PRAGMA page_count")
        pages_after = cursor.fetchone()[0]
        size_after = pages_after * page_size
        
        print(f"Size after vacuum: {size_after / 1024 / 1024:.2f} MB")
        space_saved = size_before - size_after
        print(f"Space saved: {space_saved / 1024 / 1024:.2f} MB")
        
        self.log_action("VACUUM", f"Reclaimed {space_saved / 1024 / 1024:.2f} MB", 0)
        
        conn.close()
        return space_saved
    
    def run_optimization(self):
        """Run complete database optimization"""
        print("=== DATABASE OPTIMIZATION STARTED ===")
        start_time = datetime.now()
        
        # Step 1: Analyze current state
        analysis = self.analyze_database_structure()
        
        # Step 2: Remove orphaned records
        orphaned_removed = self.find_orphaned_records()
        
        # Step 3: Clean inconsistent team IDs
        self.clean_inconsistent_team_ids()
        
        # Step 4: Add missing indexes
        self.add_missing_indexes()
        
        # Step 5: Check referential integrity
        integrity_issues = self.ensure_referential_integrity()
        
        # Step 6: Vacuum database
        space_saved = self.vacuum_database()
        
        # Create summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        summary = {
            'optimization_time': start_time.isoformat(),
            'duration_seconds': duration,
            'orphaned_records_removed': orphaned_removed,
            'space_saved_mb': space_saved / 1024 / 1024,
            'integrity_issues_count': len(integrity_issues),
            'actions': self.optimization_log
        }
        
        # Save optimization log
        with open('data/database_optimization_log.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n=== OPTIMIZATION COMPLETE ===")
        print(f"Duration: {duration:.1f} seconds")
        print(f"Orphaned records removed: {orphaned_removed}")
        print(f"Space saved: {space_saved / 1024 / 1024:.2f} MB")
        print(f"Actions performed: {len(self.optimization_log)}")
        print("Log saved to data/database_optimization_log.json")
        
        return summary

if __name__ == "__main__":
    optimizer = DatabaseOptimizer()
    result = optimizer.run_optimization()