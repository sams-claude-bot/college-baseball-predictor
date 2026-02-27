#!/usr/bin/env python3
"""
Schedule Gateway — single write path to the games table.

All schedule scripts should route through this module instead of writing
directly.  Provides:
  - Deterministic canonical game-ID generation
  - Multi-strategy dedup (exact, legacy suffix, swapped H/A, fuzzy)
  - Status-hierarchy-aware score updates
  - Ghost/duplicate replacement with FK migration
  - Structured audit logging
"""

import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ensure the scripts directory is importable
_SCRIPTS_DIR = Path(__file__).parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if str(_SCRIPTS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR.parent))

from team_resolver import TeamResolver


# Status hierarchy — higher number wins
_STATUS_RANK = {
    "cancelled": 0,
    "postponed": 1,
    "scheduled": 2,
    "in-progress": 3,
    "final": 4,
}

# Tables with a game_id column that must be migrated on ghost replacement
FK_TABLES = [
    "model_predictions",
    "betting_lines",
    "game_weather",
    "tracked_bets",
    "tracked_bets_spreads",
    "tracked_confident_bets",
    "totals_predictions",
    "spread_predictions",
    "game_predictions",
    "pitching_matchups",
    "game_boxscores",
    "game_batting_stats",
    "game_pitching_stats",
    "player_boxscore_batting",
    "player_boxscore_pitching",
    "statbroadcast_boxscores",
]


class ScheduleGateway:
    """Single write path to the games table."""

    def __init__(self, db_connection):
        self.db = db_connection
        self.resolver = TeamResolver()

    # ------------------------------------------------------------------
    # ID Generation
    # ------------------------------------------------------------------

    @staticmethod
    def canonical_game_id(date: str, away_id: str, home_id: str,
                          game_num: int = 1) -> str:
        """Generate canonical game ID.

        Format: ``YYYY-MM-DD_away-id_home-id`` (game 1, no suffix)
                ``YYYY-MM-DD_away-id_home-id_gm2`` (game 2)

        ``game_num`` must be 1 or 2 (doubleheader limit).  Values > 2
        are **clamped** to 2 with a warning.
        """
        if game_num > 2:
            print(f"[ScheduleGateway] WARNING game_num={game_num} exceeds "
                  f"doubleheader limit; clamping to 2")
            game_num = 2
        suffix = f"_gm{game_num}" if game_num >= 2 else ""
        return f"{date}_{away_id}_{home_id}{suffix}"

    # ------------------------------------------------------------------
    # Team Resolution
    # ------------------------------------------------------------------

    def resolve_team(self, name: str, slug: str = None,
                     source: str = None) -> Optional[str]:
        """Resolve *name* (or *slug*) to a canonical ``team_id``.

        Delegates entirely to :class:`TeamResolver` — no substring
        matching, ever.
        """
        # Try slug first (it's more specific), fall back to name
        if slug:
            result = self.resolver.resolve(slug)
            if result:
                return result
        if name:
            return self.resolver.resolve(name)
        return None

    # ------------------------------------------------------------------
    # Finding existing games (dedup)
    # ------------------------------------------------------------------

    def find_existing_game(self, date: str, away_id: str, home_id: str,
                           game_num: int = 1):
        """Find an existing game row matching this matchup.

        Search order:
        1. Exact canonical ID
        2. Legacy suffix variants (``_g1``/``_gm1`` for game 1, ``_g2`` for game 2)
        3. Swapped home/away (same date, reversed teams)
        4. Fuzzy — same date + same two teams in either order
        """
        c = self.db.cursor()
        canon = self.canonical_game_id(date, away_id, home_id, game_num)

        # --- 1. Exact canonical ID ---
        row = c.execute("SELECT * FROM games WHERE id = ?",
                        (canon,)).fetchone()
        if row:
            return row

        # --- 2. Legacy suffix variants ---
        legacy_ids = set()
        if game_num == 1:
            # game-1 was sometimes stored as _g1 or _gm1
            base = f"{date}_{away_id}_{home_id}"
            legacy_ids.update([f"{base}_g1", f"{base}_gm1"])
        else:
            base = f"{date}_{away_id}_{home_id}"
            legacy_ids.add(f"{base}_g{game_num}")
        for lid in legacy_ids:
            row = c.execute("SELECT * FROM games WHERE id = ?",
                            (lid,)).fetchone()
            if row:
                return row

        # --- 3. Swapped home/away (same date, reversed teams) ---
        swapped = self.canonical_game_id(date, home_id, away_id, game_num)
        row = c.execute("SELECT * FROM games WHERE id = ?",
                        (swapped,)).fetchone()
        if row:
            return row
        # Also check swapped legacy
        if game_num == 1:
            sbase = f"{date}_{home_id}_{away_id}"
            for sid in [sbase, f"{sbase}_g1", f"{sbase}_gm1"]:
                row = c.execute("SELECT * FROM games WHERE id = ?",
                                (sid,)).fetchone()
                if row:
                    return row
        else:
            sbase = f"{date}_{home_id}_{away_id}"
            for sid in [f"{sbase}_gm{game_num}", f"{sbase}_g{game_num}"]:
                row = c.execute("SELECT * FROM games WHERE id = ?",
                                (sid,)).fetchone()
                if row:
                    return row

        # --- 4. Fuzzy: same date + same two teams in any order ---
        if game_num == 1:
            row = c.execute("""
                SELECT * FROM games
                WHERE date = ?
                  AND id NOT LIKE '%\\_gm2' ESCAPE '\\'
                  AND id NOT LIKE '%\\_g2' ESCAPE '\\'
                  AND (
                    (home_team_id = ? AND away_team_id = ?) OR
                    (home_team_id = ? AND away_team_id = ?)
                  )
                LIMIT 1
            """, (date, home_id, away_id, away_id, home_id)).fetchone()
        else:
            row = c.execute("""
                SELECT * FROM games
                WHERE date = ?
                  AND (id LIKE ? OR id LIKE ?)
                  AND (
                    (home_team_id = ? AND away_team_id = ?) OR
                    (home_team_id = ? AND away_team_id = ?)
                  )
                LIMIT 1
            """, (date, f"%_gm{game_num}", f"%_g{game_num}",
                  home_id, away_id, away_id, home_id)).fetchone()

        return row  # may be None

    # ------------------------------------------------------------------
    # FK migration
    # ------------------------------------------------------------------

    def migrate_fk_rows(self, old_game_id: str, new_game_id: str) -> dict:
        """Migrate foreign-key references from *old_game_id* to *new_game_id*.

        Returns ``{"migrated": int, "deleted": int}`` counts.
        """
        migrated = 0
        deleted = 0
        for table in FK_TABLES:
            try:
                n = self.db.execute(
                    f"UPDATE {table} SET game_id = ? WHERE game_id = ?",
                    (new_game_id, old_game_id),
                ).rowcount
                migrated += n
            except Exception:
                # Table may not exist or unique-constraint conflict
                try:
                    n = self.db.execute(
                        f"DELETE FROM {table} WHERE game_id = ?",
                        (old_game_id,),
                    ).rowcount
                    deleted += n
                except Exception:
                    pass
        return {"migrated": migrated, "deleted": deleted}

    # ------------------------------------------------------------------
    # Core upsert
    # ------------------------------------------------------------------

    def upsert_game(self, date, away_id, home_id, game_num=1, *,
                    time=None, home_score=None, away_score=None,
                    status=None, inning_text=None, source=None) -> str:
        """Insert or update a game.  Returns action taken:
        ``'created'``, ``'updated'``, ``'unchanged'``, or ``'replaced'``.
        """
        if game_num > 2:
            print(f"[ScheduleGateway] WARNING game_num={game_num} rejected "
                  f"(max 2). source={source}")
            return "rejected"

        canon = self.canonical_game_id(date, away_id, home_id, game_num)
        incoming_status = status or ("final" if home_score is not None and away_score is not None else "scheduled")

        existing = self.find_existing_game(date, away_id, home_id, game_num)

        if existing is None:
            # --- Brand-new game ---
            winner = self._compute_winner(home_id, away_id, home_score,
                                          away_score)
            self.db.execute("""
                INSERT INTO games
                    (id, date, time, home_team_id, away_team_id,
                     home_score, away_score, winner_id, status,
                     inning_text, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (canon, date, time, home_id, away_id, home_score,
                  away_score, winner, incoming_status, inning_text,
                  datetime.utcnow().isoformat()))
            self.db.commit()
            self._log("created", canon, source)
            return "created"

        # --- Existing game found ---
        old_id = existing["id"]

        # Status-hierarchy check
        old_status = existing["status"] or "scheduled"
        old_rank = _STATUS_RANK.get(old_status, 2)
        new_rank = _STATUS_RANK.get(incoming_status, 2)

        if new_rank < old_rank:
            # Lower-status update cannot overwrite higher-status game
            self._log("unchanged", old_id, source)
            return "unchanged"

        # Score-update rules
        new_home = home_score
        new_away = away_score
        if new_home is None and existing["home_score"] is not None:
            new_home = existing["home_score"]
        if new_away is None and existing["away_score"] is not None:
            new_away = existing["away_score"]

        # Determine whether the row actually needs a change
        same_scores = (
            (new_home == existing["home_score"] or
             (new_home is None and existing["home_score"] is None)) and
            (new_away == existing["away_score"] or
             (new_away is None and existing["away_score"] is None))
        )
        final_status = incoming_status if new_rank >= old_rank else old_status
        same_status = (final_status == old_status)
        if same_scores and same_status and old_id == canon:
            self._log("unchanged", old_id, source)
            return "unchanged"

        winner = self._compute_winner(home_id, away_id, new_home, new_away)

        # Ghost replacement: mismatched ID → migrate FKs, delete old, insert new
        if old_id != canon:
            fk = self.migrate_fk_rows(old_id, canon)
            self.db.execute("DELETE FROM games WHERE id = ?", (old_id,))
            self.db.execute("""
                INSERT INTO games
                    (id, date, time, home_team_id, away_team_id,
                     home_score, away_score, winner_id, status,
                     inning_text, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (canon, date, time or existing["time"], home_id, away_id,
                  new_home, new_away, winner, final_status,
                  inning_text, datetime.utcnow().isoformat()))
            self.db.commit()
            self._log("replaced", f"{old_id} -> {canon}", source,
                      extra=f"FK {fk}")
            return "replaced"

        # Normal in-place update
        self.db.execute("""
            UPDATE games
               SET home_score  = ?,
                   away_score  = ?,
                   winner_id   = ?,
                   status      = ?,
                   inning_text = COALESCE(?, inning_text),
                   time        = COALESCE(?, time),
                   updated_at  = ?
             WHERE id = ?
        """, (new_home, new_away, winner, final_status, inning_text,
              time, datetime.utcnow().isoformat(), old_id))
        self.db.commit()
        self._log("updated", old_id, source)
        return "updated"

    # ------------------------------------------------------------------
    # Convenience mutators
    # ------------------------------------------------------------------

    def finalize_game(self, game_id: str, home_score: int,
                      away_score: int, innings: int = None) -> bool:
        """Mark *game_id* as final with scores.  Computes ``winner_id``,
        clears ``inning_text``.  Sets ``innings`` only for extra-inning
        games; clears it for regulation (9-inning) games so the UI
        doesn't show a stale mid-game inning count."""
        c = self.db.cursor()
        row = c.execute("SELECT home_team_id, away_team_id FROM games "
                        "WHERE id = ?", (game_id,)).fetchone()
        if not row:
            return False
        winner = self._compute_winner(
            row["home_team_id"], row["away_team_id"], home_score, away_score)
        # Only store innings if it's extra innings (>9); clear otherwise
        final_innings = innings if innings and innings > 9 else None
        self.db.execute("""
            UPDATE games
               SET home_score = ?, away_score = ?, winner_id = ?,
                   status = 'final', inning_text = NULL,
                   innings = ?,
                   updated_at = ?
             WHERE id = ?
        """, (home_score, away_score, winner, final_innings,
              datetime.utcnow().isoformat(), game_id))
        self.db.commit()
        self._log("finalized", game_id, None)
        return True

    def mark_postponed(self, game_id: str, reason: str = None) -> bool:
        """Mark game as postponed."""
        n = self.db.execute("""
            UPDATE games SET status = 'postponed', notes = COALESCE(?, notes),
                             updated_at = ?
            WHERE id = ?
        """, (reason, datetime.utcnow().isoformat(), game_id)).rowcount
        self.db.commit()
        self._log("postponed", game_id, None)
        return n > 0

    def mark_cancelled(self, game_id: str, reason: str = None) -> bool:
        """Mark game as cancelled."""
        n = self.db.execute("""
            UPDATE games SET status = 'cancelled', notes = COALESCE(?, notes),
                             updated_at = ?
            WHERE id = ?
        """, (reason, datetime.utcnow().isoformat(), game_id)).rowcount
        self.db.commit()
        self._log("cancelled", game_id, None)
        return n > 0

    def update_live_score(self, game_id: str, home_score: int,
                          away_score: int, inning_text: str,
                          innings: int = None) -> bool:
        """Update in-progress game scores and inning text."""
        if innings is not None:
            n = self.db.execute("""
                UPDATE games
                   SET home_score = ?, away_score = ?, inning_text = ?,
                       innings = ?, status = 'in-progress', updated_at = ?
                 WHERE id = ?
            """, (home_score, away_score, inning_text, innings,
                  datetime.utcnow().isoformat(), game_id)).rowcount
        else:
            n = self.db.execute("""
                UPDATE games
                   SET home_score = ?, away_score = ?, inning_text = ?,
                       status = 'in-progress', updated_at = ?
                 WHERE id = ?
            """, (home_score, away_score, inning_text,
                  datetime.utcnow().isoformat(), game_id)).rowcount
        self.db.commit()
        self._log("live-update", game_id, None)
        return n > 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_winner(home_id, away_id, home_score, away_score):
        if home_score is not None and away_score is not None:
            if home_score > away_score:
                return home_id
            if away_score > home_score:
                return away_id
        return None

    @staticmethod
    def _log(action, game_id, source, extra=""):
        msg = f"[ScheduleGateway] {action} {game_id} (source={source})"
        if extra:
            msg += f" {extra}"
        print(msg)
