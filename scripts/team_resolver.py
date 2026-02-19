#!/usr/bin/env python3
"""
Team ID Resolver - Single source of truth for team name → team_id mapping.

Uses the team_aliases table in the database. All scrapers should use this
instead of hardcoding aliases.

Usage:
    from team_resolver import resolve_team, add_alias, TeamResolver
    
    # Simple lookup
    team_id = resolve_team("Florida International")  # → "florida-international"
    team_id = resolve_team("FIU")                    # → "florida-international"
    
    # Add new alias (e.g., when scraper finds unknown name)
    add_alias("gators", "florida", source="espn")
    
    # Batch operations
    resolver = TeamResolver("data/baseball.db")
    resolver.resolve("ETSU")  # → "east-tennessee-state"
"""

import sqlite3
import re
from functools import lru_cache
from pathlib import Path

# Default database path (relative to project root)
DEFAULT_DB = Path(__file__).parent.parent / "data" / "baseball.db"


class TeamResolver:
    """Resolve team names to canonical team IDs using the database."""
    
    def __init__(self, db_path=None):
        self.db_path = db_path or DEFAULT_DB
        self._cache = {}
        self._load_aliases()
    
    def _load_aliases(self):
        """Load all aliases into memory for fast lookup."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        c = conn.cursor()
        c.execute("SELECT alias, team_id FROM team_aliases")
        for alias, team_id in c.fetchall():
            self._cache[alias.lower()] = team_id
        conn.close()
    
    def resolve(self, name: str):
        """
        Resolve a team name to its canonical team_id.
        
        Returns None if not found (caller should handle unknown teams).
        """
        if not name:
            return None
        
        key = name.lower().strip()
        
        # Direct lookup
        if key in self._cache:
            return self._cache[key]
        
        # Try slugified version
        slug = self._slugify(key)
        if slug in self._cache:
            return self._cache[slug]
        
        return None
    
    def _slugify(self, name: str) -> str:
        """Convert name to slug format (lowercase, hyphens)."""
        # Remove special chars, replace spaces with hyphens
        slug = re.sub(r"[^\w\s-]", "", name.lower())
        slug = re.sub(r"[-\s]+", "-", slug).strip("-")
        return slug
    
    def add_alias(self, alias: str, team_id: str, source: str = "auto"):
        """Add a new alias to the database."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        c = conn.cursor()
        try:
            c.execute(
                "INSERT OR IGNORE INTO team_aliases (alias, team_id, source) VALUES (?, ?, ?)",
                (alias.lower().strip(), team_id, source)
            )
            conn.commit()
            # Update cache
            self._cache[alias.lower().strip()] = team_id
        finally:
            conn.close()
    
    def get_aliases(self, team_id: str):
        """Get all aliases for a team."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        c = conn.cursor()
        c.execute("SELECT alias FROM team_aliases WHERE team_id = ?", (team_id,))
        aliases = [row[0] for row in c.fetchall()]
        conn.close()
        return aliases
    
    def find_similar(self, name: str, limit: int = 5):
        """Find similar team names (for fuzzy matching / suggestions)."""
        key = name.lower().strip()
        matches = []
        for alias, team_id in self._cache.items():
            if key in alias or alias in key:
                matches.append((alias, team_id))
        return matches[:limit]


# Global resolver instance (lazy loaded)
_resolver = None

def _get_resolver():
    global _resolver
    if _resolver is None:
        _resolver = TeamResolver()
    return _resolver


def resolve_team(name: str):
    """Resolve a team name to its canonical team_id."""
    return _get_resolver().resolve(name)


def add_alias(alias: str, team_id: str, source: str = "auto"):
    """Add a new alias to the database."""
    _get_resolver().add_alias(alias, team_id, source)


def get_aliases(team_id: str):
    """Get all aliases for a team."""
    return _get_resolver().get_aliases(team_id)


if __name__ == "__main__":
    # CLI for testing
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python team_resolver.py <team_name>")
        print("       python team_resolver.py --add <alias> <team_id> [source]")
        print("       python team_resolver.py --list <team_id>")
        sys.exit(1)
    
    if sys.argv[1] == "--add" and len(sys.argv) >= 4:
        alias, team_id = sys.argv[2], sys.argv[3]
        source = sys.argv[4] if len(sys.argv) > 4 else "manual"
        add_alias(alias, team_id, source)
        print(f"Added: {alias} → {team_id} (source: {source})")
    
    elif sys.argv[1] == "--list" and len(sys.argv) >= 3:
        team_id = sys.argv[2]
        aliases = get_aliases(team_id)
        print(f"Aliases for {team_id}:")
        for a in aliases:
            print(f"  {a}")
    
    else:
        name = " ".join(sys.argv[1:])
        result = resolve_team(name)
        if result:
            print(f"{name} → {result}")
        else:
            print(f"{name} → NOT FOUND")
            resolver = _get_resolver()
            similar = resolver.find_similar(name)
            if similar:
                print("Similar:")
                for alias, tid in similar:
                    print(f"  {alias} → {tid}")
