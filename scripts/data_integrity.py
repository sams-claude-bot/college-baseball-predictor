#!/usr/bin/env python3
"""
Data Integrity Check Script

Checks for common data issues:
- Orphaned predictions (game_id not in games)
- Duplicate teams (same name, different IDs)
- Games with scores but no predictions
- Predictions needing evaluation (game has score, was_correct IS NULL)
- Orphan teams (no games)
- Game ID format consistency
- Missing team references in games

Run this periodically to catch problems early.
"""

import sqlite3
from pathlib import Path
from collections import defaultdict

DB_PATH = Path(__file__).parent.parent / "data" / "baseball.db"

class IntegrityChecker:
    def __init__(self, db_path=DB_PATH):
        self.conn = sqlite3.connect(db_path)
        self.cur = self.conn.cursor()
        self.issues = []
        self.warnings = []
        self.stats = {}
    
    def check_all(self):
        """Run all integrity checks"""
        print("🔍 Running data integrity checks...\n")
        
        self.check_orphaned_predictions()
        self.check_duplicate_teams()
        self.check_games_without_predictions()
        self.check_unevaluated_predictions()
        self.check_orphan_teams()
        self.check_game_id_format()
        self.check_missing_team_references()
        
        self.print_summary()
        return len(self.issues) == 0
    
    def check_orphaned_predictions(self):
        """Check for predictions that reference non-existent games"""
        print("📋 Checking orphaned predictions...")
        
        # model_predictions
        self.cur.execute("""
            SELECT COUNT(*) FROM model_predictions 
            WHERE game_id NOT IN (SELECT id FROM games)
        """)
        orphan_model = self.cur.fetchone()[0]
        
        # totals_predictions
        self.cur.execute("""
            SELECT COUNT(*) FROM totals_predictions 
            WHERE game_id NOT IN (SELECT id FROM games)
        """)
        orphan_totals = self.cur.fetchone()[0]
        
        if orphan_model > 0:
            self.issues.append(f"❌ {orphan_model} orphaned model_predictions")
        else:
            print("   ✓ No orphaned model_predictions")
        
        if orphan_totals > 0:
            self.issues.append(f"❌ {orphan_totals} orphaned totals_predictions")
        else:
            print("   ✓ No orphaned totals_predictions")
        
        self.stats['orphan_model_predictions'] = orphan_model
        self.stats['orphan_totals_predictions'] = orphan_totals
    
    def check_duplicate_teams(self):
        """Check for teams with same name but different IDs"""
        print("\n📋 Checking duplicate teams...")
        
        self.cur.execute("""
            SELECT name, GROUP_CONCAT(id, ', ') as ids, COUNT(*) as count
            FROM teams
            GROUP BY LOWER(name)
            HAVING count > 1
        """)
        
        duplicates = self.cur.fetchall()
        if duplicates:
            for name, ids, count in duplicates:
                self.issues.append(f"❌ Duplicate team '{name}': {ids}")
        else:
            print("   ✓ No duplicate teams found")
        
        self.stats['duplicate_teams'] = len(duplicates)
    
    def check_games_without_predictions(self):
        """Check for completed games that have no predictions"""
        print("\n📋 Checking games without predictions...")
        
        self.cur.execute("""
            SELECT COUNT(*) FROM games g
            WHERE g.home_score IS NOT NULL
            AND g.id NOT IN (SELECT DISTINCT game_id FROM model_predictions)
        """)
        
        count = self.cur.fetchone()[0]
        if count > 0:
            self.warnings.append(f"⚠️  {count} completed games have no predictions")
        else:
            print("   ✓ All completed games have predictions")
        
        self.stats['games_without_predictions'] = count
    
    def check_unevaluated_predictions(self):
        """Check for predictions that can be evaluated but haven't been"""
        print("\n📋 Checking unevaluated predictions...")
        
        self.cur.execute("""
            SELECT COUNT(DISTINCT mp.game_id) FROM model_predictions mp
            JOIN games g ON mp.game_id = g.id
            WHERE g.home_score IS NOT NULL
            AND mp.was_correct IS NULL
        """)
        
        count = self.cur.fetchone()[0]
        if count > 0:
            self.warnings.append(f"⚠️  {count} games have unevaluated predictions (run: predict_and_track.py evaluate)")
        else:
            print("   ✓ All predictions with results have been evaluated")
        
        self.stats['unevaluated_predictions'] = count
    
    def check_orphan_teams(self):
        """Check for teams with no games"""
        print("\n📋 Checking orphan teams...")
        
        self.cur.execute("""
            SELECT t.id, t.name, t.conference
            FROM teams t
            LEFT JOIN games g ON t.id = g.home_team_id OR t.id = g.away_team_id
            GROUP BY t.id
            HAVING COUNT(g.id) = 0
        """)
        
        orphans = self.cur.fetchall()
        if orphans:
            for team_id, name, conf in orphans:
                conf_str = f" ({conf})" if conf else ""
                self.warnings.append(f"⚠️  Orphan team: {team_id} - {name}{conf_str}")
        else:
            print("   ✓ No orphan teams (all teams have at least one game)")
        
        self.stats['orphan_teams'] = len(orphans)
    
    def check_game_id_format(self):
        """Check game ID format consistency"""
        print("\n📋 Checking game ID format...")
        
        # Expected format: YYYY-MM-DD_away-team_home-team (with optional _g1, _g2 suffix)
        self.cur.execute("SELECT id FROM games LIMIT 1000")
        games = self.cur.fetchall()
        
        bad_format = []
        for (game_id,) in games:
            parts = game_id.split('_')
            if len(parts) < 3:
                bad_format.append(game_id)
            elif not parts[0].count('-') == 2:  # Date should have 2 hyphens
                bad_format.append(game_id)
        
        if bad_format:
            for gid in bad_format[:5]:  # Show first 5
                self.issues.append(f"❌ Bad game ID format: {gid}")
            if len(bad_format) > 5:
                self.issues.append(f"❌ ... and {len(bad_format) - 5} more")
        else:
            print("   ✓ All game IDs follow expected format")
        
        self.stats['bad_game_ids'] = len(bad_format)
    
    def check_missing_team_references(self):
        """Check for games that reference teams not in teams table"""
        print("\n📋 Checking team references...")
        
        self.cur.execute("""
            SELECT g.id, g.home_team_id, g.away_team_id
            FROM games g
            LEFT JOIN teams h ON g.home_team_id = h.id
            LEFT JOIN teams a ON g.away_team_id = a.id
            WHERE h.id IS NULL OR a.id IS NULL
        """)
        
        missing = self.cur.fetchall()
        if missing:
            teams_missing = set()
            for game_id, home, away in missing:
                self.cur.execute("SELECT id FROM teams WHERE id = ?", (home,))
                if not self.cur.fetchone():
                    teams_missing.add(home)
                self.cur.execute("SELECT id FROM teams WHERE id = ?", (away,))
                if not self.cur.fetchone():
                    teams_missing.add(away)
            
            for team in list(teams_missing)[:10]:
                self.issues.append(f"❌ Missing team: {team}")
            if len(teams_missing) > 10:
                self.issues.append(f"❌ ... and {len(teams_missing) - 10} more missing teams")
        else:
            print("   ✓ All game team references exist in teams table")
        
        self.stats['missing_teams'] = len(missing)
    
    def print_summary(self):
        """Print summary of all checks"""
        print("\n" + "=" * 60)
        print("📊 INTEGRITY CHECK SUMMARY")
        print("=" * 60)
        
        print(f"\nStats:")
        for key, value in self.stats.items():
            print(f"  • {key}: {value}")
        
        if self.issues:
            print(f"\n❌ ISSUES FOUND ({len(self.issues)}):")
            for issue in self.issues:
                print(f"  {issue}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  {warning}")
        
        if not self.issues and not self.warnings:
            print("\n✅ All checks passed!")
        elif not self.issues:
            print(f"\n✅ No critical issues, {len(self.warnings)} warning(s)")
        else:
            print(f"\n❌ {len(self.issues)} issue(s) need attention")
        
        print()
    
    def close(self):
        self.conn.close()


def main():
    checker = IntegrityChecker()
    success = checker.check_all()
    checker.close()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
