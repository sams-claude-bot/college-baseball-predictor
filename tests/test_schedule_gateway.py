#!/usr/bin/env python3
"""
Tests for scripts/schedule_gateway.py

Unit tests (1-18) use an in-memory SQLite database.
Integration test (19) hits D1Baseball.com and compares against our DB.
"""

import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.schedule_gateway import ScheduleGateway, FK_TABLES


# ---------------------------------------------------------------------------
# Helpers: in-memory DB scaffolding
# ---------------------------------------------------------------------------

def _make_db():
    """Create an in-memory SQLite connection with the games table + FK tables."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("""
        CREATE TABLE games (
            id TEXT PRIMARY KEY,
            date TEXT NOT NULL,
            time TEXT,
            home_team_id TEXT NOT NULL,
            away_team_id TEXT NOT NULL,
            home_score INTEGER,
            away_score INTEGER,
            winner_id TEXT,
            innings INTEGER DEFAULT 9,
            is_conference_game INTEGER DEFAULT 0,
            is_neutral_site INTEGER DEFAULT 0,
            tournament_id TEXT,
            venue TEXT,
            attendance INTEGER,
            notes TEXT,
            status TEXT DEFAULT 'scheduled',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            inning_text TEXT
        )
    """)

    # Minimal FK tables used by migration tests
    c.execute("""
        CREATE TABLE model_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            predicted_home_prob REAL,
            UNIQUE(game_id, model_name)
        )
    """)
    c.execute("""
        CREATE TABLE betting_lines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT,
            date TEXT,
            home_team_id TEXT,
            away_team_id TEXT,
            book TEXT DEFAULT 'draftkings',
            home_ml INTEGER,
            away_ml INTEGER
        )
    """)
    c.execute("""
        CREATE TABLE totals_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            over_under_line REAL NOT NULL,
            projected_total REAL NOT NULL,
            prediction TEXT NOT NULL,
            model_name TEXT DEFAULT 'runs_ensemble',
            UNIQUE(game_id, over_under_line, model_name)
        )
    """)

    conn.commit()
    return conn


def _gw(conn=None):
    """Create a ScheduleGateway backed by an in-memory DB, with a mocked
    TeamResolver so we don't need the real database for alias lookups."""
    if conn is None:
        conn = _make_db()
    gw = ScheduleGateway.__new__(ScheduleGateway)
    gw.db = conn

    # Build a minimal mock TeamResolver
    aliases = {
        "fiu": "florida-international",
        "florida international": "florida-international",
        "florida-international": "florida-international",
        "kansas": "kansas",
        "arkansas": "arkansas",
        "mississippi-state": "mississippi-state",
        "auburn": "auburn",
        "florida": "florida",
        "miami-ohio": "miami-ohio",
        "miami-fl": "miami-fl",
    }
    mock_resolver = MagicMock()
    mock_resolver.resolve = lambda name: aliases.get(name.lower().strip()) if name else None
    gw.resolver = mock_resolver

    return gw, conn


# =========================================================================
# Unit Tests
# =========================================================================


class TestCanonicalGameId:
    """1. test_canonical_game_id — format, suffix logic, max game_num=2."""

    def test_game1_no_suffix(self):
        gid = ScheduleGateway.canonical_game_id(
            "2025-03-01", "auburn", "mississippi-state", 1)
        assert gid == "2025-03-01_auburn_mississippi-state"

    def test_game2_suffix(self):
        gid = ScheduleGateway.canonical_game_id(
            "2025-03-01", "auburn", "mississippi-state", 2)
        assert gid == "2025-03-01_auburn_mississippi-state_gm2"

    def test_game3_clamped_to_2(self, capsys):
        gid = ScheduleGateway.canonical_game_id(
            "2025-03-01", "auburn", "mississippi-state", 3)
        assert gid == "2025-03-01_auburn_mississippi-state_gm2"
        assert "WARNING" in capsys.readouterr().out


class TestUpsertCreateNewGame:
    """2. test_upsert_create_new_game — basic insert."""

    def test_creates_game(self):
        gw, conn = _gw()
        action = gw.upsert_game("2025-03-01", "auburn", "mississippi-state",
                                source="test")
        assert action == "created"
        row = conn.execute("SELECT * FROM games").fetchone()
        assert row["id"] == "2025-03-01_auburn_mississippi-state"
        assert row["status"] == "scheduled"


class TestUpsertUpdateScores:
    """3. test_upsert_update_scores — update existing game with scores."""

    def test_scores_applied(self):
        gw, conn = _gw()
        gw.upsert_game("2025-03-01", "auburn", "mississippi-state")
        action = gw.upsert_game("2025-03-01", "auburn", "mississippi-state",
                                home_score=5, away_score=3, status="final",
                                source="test")
        assert action == "updated"
        row = conn.execute("SELECT * FROM games").fetchone()
        assert row["home_score"] == 5
        assert row["away_score"] == 3
        assert row["winner_id"] == "mississippi-state"
        assert row["status"] == "final"


class TestUpsertNoOverwriteFinal:
    """4. test_upsert_no_overwrite_final — scheduled update can't overwrite final."""

    def test_no_downgrade(self):
        gw, conn = _gw()
        gw.upsert_game("2025-03-01", "auburn", "mississippi-state",
                        home_score=5, away_score=3, status="final")
        action = gw.upsert_game("2025-03-01", "auburn", "mississippi-state",
                                status="scheduled", source="stale-feed")
        assert action == "unchanged"
        row = conn.execute("SELECT * FROM games").fetchone()
        assert row["status"] == "final"
        assert row["home_score"] == 5


class TestUpsertLegacySuffixMatch:
    """5. test_upsert_legacy_suffix_match — finds _g1 for game 1."""

    def test_finds_legacy_g1(self):
        gw, conn = _gw()
        # Manually insert with legacy _g1 suffix
        conn.execute("""
            INSERT INTO games (id, date, home_team_id, away_team_id, status)
            VALUES ('2025-03-01_auburn_mississippi-state_g1', '2025-03-01',
                    'mississippi-state', 'auburn', 'scheduled')
        """)
        conn.commit()
        action = gw.upsert_game("2025-03-01", "auburn", "mississippi-state",
                                game_num=1, home_score=4, away_score=2,
                                status="final", source="test")
        # Should have replaced the legacy ID with canonical ID
        assert action == "replaced"
        row = conn.execute("SELECT * FROM games").fetchone()
        assert row["id"] == "2025-03-01_auburn_mississippi-state"


class TestUpsertSwappedHomeAway:
    """6. test_upsert_swapped_home_away — finds game with teams reversed."""

    def test_finds_swapped(self):
        gw, conn = _gw()
        # Insert with home/away swapped relative to what we'll upsert
        conn.execute("""
            INSERT INTO games (id, date, home_team_id, away_team_id, status)
            VALUES ('2025-03-01_mississippi-state_auburn', '2025-03-01',
                    'auburn', 'mississippi-state', 'scheduled')
        """)
        conn.commit()
        action = gw.upsert_game("2025-03-01", "auburn", "mississippi-state",
                                home_score=7, away_score=1, status="final",
                                source="test")
        assert action in ("updated", "replaced")
        rows = conn.execute("SELECT * FROM games").fetchall()
        assert len(rows) == 1
        assert rows[0]["home_score"] == 7


class TestUpsertFuzzySameDateTeams:
    """7. test_upsert_fuzzy_same_date_teams — finds by date+teams when ID differs."""

    def test_fuzzy_match(self):
        gw, conn = _gw()
        # Some legacy/ESPN-style ID that doesn't match our canonical format
        conn.execute("""
            INSERT INTO games (id, date, home_team_id, away_team_id, status)
            VALUES ('espn-12345', '2025-03-01',
                    'mississippi-state', 'auburn', 'scheduled')
        """)
        conn.commit()
        action = gw.upsert_game("2025-03-01", "auburn", "mississippi-state",
                                home_score=3, away_score=2, status="final",
                                source="d1bb")
        assert action == "replaced"
        rows = conn.execute("SELECT * FROM games").fetchall()
        assert len(rows) == 1
        assert rows[0]["id"] == "2025-03-01_auburn_mississippi-state"


class TestUpsertGhostReplacement:
    """8. test_upsert_ghost_replacement — ESPN ghost replaced, FKs migrated."""

    def test_ghost_replaced_and_fks_migrated(self):
        gw, conn = _gw()
        old_id = "2025-03-01_msst_aub"  # ESPN-style ghost
        conn.execute("""
            INSERT INTO games (id, date, home_team_id, away_team_id, status)
            VALUES (?, '2025-03-01', 'mississippi-state', 'auburn', 'scheduled')
        """, (old_id,))
        conn.execute("""
            INSERT INTO model_predictions (game_id, model_name, predicted_home_prob)
            VALUES (?, 'elo', 0.65)
        """, (old_id,))
        conn.execute("""
            INSERT INTO betting_lines (game_id, date, home_team_id, away_team_id, home_ml)
            VALUES (?, '2025-03-01', 'mississippi-state', 'auburn', -150)
        """, (old_id,))
        conn.commit()

        action = gw.upsert_game("2025-03-01", "auburn", "mississippi-state",
                                home_score=6, away_score=4, status="final",
                                source="d1bb")
        assert action == "replaced"
        canon = "2025-03-01_auburn_mississippi-state"
        # Old game gone
        assert conn.execute("SELECT * FROM games WHERE id = ?",
                            (old_id,)).fetchone() is None
        # New game present
        row = conn.execute("SELECT * FROM games WHERE id = ?",
                           (canon,)).fetchone()
        assert row["home_score"] == 6
        # FK rows migrated
        mp = conn.execute("SELECT * FROM model_predictions WHERE game_id = ?",
                          (canon,)).fetchone()
        assert mp is not None
        bl = conn.execute("SELECT * FROM betting_lines WHERE game_id = ?",
                          (canon,)).fetchone()
        assert bl is not None


class TestStatusHierarchy:
    """9. test_status_hierarchy — final > in-progress > scheduled."""

    def test_final_beats_scheduled(self):
        gw, conn = _gw()
        gw.upsert_game("2025-03-01", "auburn", "mississippi-state",
                        status="scheduled")
        action = gw.upsert_game("2025-03-01", "auburn", "mississippi-state",
                                home_score=3, away_score=1, status="final")
        assert action == "updated"
        assert conn.execute("SELECT status FROM games").fetchone()["status"] == "final"

    def test_inprogress_beats_scheduled(self):
        gw, conn = _gw()
        gw.upsert_game("2025-03-01", "auburn", "mississippi-state",
                        status="scheduled")
        action = gw.upsert_game("2025-03-01", "auburn", "mississippi-state",
                                home_score=2, away_score=0,
                                status="in-progress")
        assert action == "updated"
        assert conn.execute("SELECT status FROM games").fetchone()["status"] == "in-progress"

    def test_scheduled_cannot_overwrite_inprogress(self):
        gw, conn = _gw()
        gw.upsert_game("2025-03-01", "auburn", "mississippi-state",
                        home_score=2, away_score=0, status="in-progress")
        action = gw.upsert_game("2025-03-01", "auburn", "mississippi-state",
                                status="scheduled")
        assert action == "unchanged"
        assert conn.execute("SELECT status FROM games").fetchone()["status"] == "in-progress"

    def test_scheduled_cannot_overwrite_final(self):
        gw, conn = _gw()
        gw.upsert_game("2025-03-01", "auburn", "mississippi-state",
                        home_score=5, away_score=3, status="final")
        action = gw.upsert_game("2025-03-01", "auburn", "mississippi-state",
                                status="scheduled")
        assert action == "unchanged"
        assert conn.execute("SELECT status FROM games").fetchone()["status"] == "final"


class TestDoubleheader:
    """10. test_doubleheader_game_1_and_2."""

    def test_two_games_same_day(self):
        gw, conn = _gw()
        a1 = gw.upsert_game("2025-03-01", "auburn", "mississippi-state",
                             game_num=1, source="test")
        a2 = gw.upsert_game("2025-03-01", "auburn", "mississippi-state",
                             game_num=2, source="test")
        assert a1 == "created"
        assert a2 == "created"
        rows = conn.execute("SELECT id FROM games ORDER BY id").fetchall()
        assert len(rows) == 2
        ids = {r["id"] for r in rows}
        assert "2025-03-01_auburn_mississippi-state" in ids
        assert "2025-03-01_auburn_mississippi-state_gm2" in ids


class TestRejectGameNum3:
    """11. test_reject_game_num_3."""

    def test_game_num_3_rejected(self, capsys):
        gw, _ = _gw()
        result = gw.upsert_game("2025-03-01", "auburn", "mississippi-state",
                                game_num=3, source="test")
        assert result == "rejected"
        assert "WARNING" in capsys.readouterr().out


class TestFinalizeGame:
    """12. test_finalize_game — sets status=final, computes winner, clears inning_text."""

    def test_finalize(self):
        gw, conn = _gw()
        gw.upsert_game("2025-03-01", "auburn", "mississippi-state",
                        home_score=2, away_score=1, status="in-progress",
                        inning_text="Top 7th")
        ok = gw.finalize_game("2025-03-01_auburn_mississippi-state", 5, 3)
        assert ok is True
        row = conn.execute("SELECT * FROM games").fetchone()
        assert row["status"] == "final"
        assert row["home_score"] == 5
        assert row["away_score"] == 3
        assert row["winner_id"] == "mississippi-state"
        assert row["inning_text"] is None


class TestMarkPostponed:
    """13. test_mark_postponed."""

    def test_postpone(self):
        gw, conn = _gw()
        gw.upsert_game("2025-03-01", "auburn", "mississippi-state")
        ok = gw.mark_postponed("2025-03-01_auburn_mississippi-state",
                               reason="Rain")
        assert ok is True
        row = conn.execute("SELECT * FROM games").fetchone()
        assert row["status"] == "postponed"
        assert row["notes"] == "Rain"


class TestMarkCancelled:
    """14. test_mark_cancelled."""

    def test_cancel(self):
        gw, conn = _gw()
        gw.upsert_game("2025-03-01", "auburn", "mississippi-state")
        ok = gw.mark_cancelled("2025-03-01_auburn_mississippi-state")
        assert ok is True
        row = conn.execute("SELECT * FROM games").fetchone()
        assert row["status"] == "cancelled"


class TestUpdateLiveScore:
    """15. test_update_live_score."""

    def test_live_update(self):
        gw, conn = _gw()
        gw.upsert_game("2025-03-01", "auburn", "mississippi-state")
        ok = gw.update_live_score(
            "2025-03-01_auburn_mississippi-state", 3, 2, "Bot 5th")
        assert ok is True
        row = conn.execute("SELECT * FROM games").fetchone()
        assert row["status"] == "in-progress"
        assert row["home_score"] == 3
        assert row["away_score"] == 2
        assert row["inning_text"] == "Bot 5th"


class TestFkMigration:
    """16. test_fk_migration — model_predictions, betting_lines migrate to new ID."""

    def test_migrate_fk_rows(self):
        gw, conn = _gw()
        old = "old-id"
        new = "new-id"
        conn.execute("""
            INSERT INTO model_predictions (game_id, model_name, predicted_home_prob)
            VALUES (?, 'elo', 0.55)
        """, (old,))
        conn.execute("""
            INSERT INTO betting_lines (game_id, date, home_team_id, away_team_id)
            VALUES (?, '2025-03-01', 'a', 'b')
        """, (old,))
        conn.commit()
        result = gw.migrate_fk_rows(old, new)
        assert result["migrated"] >= 2
        # Old rows gone
        assert conn.execute(
            "SELECT COUNT(*) FROM model_predictions WHERE game_id = ?",
            (old,)).fetchone()[0] == 0
        # New rows present
        assert conn.execute(
            "SELECT COUNT(*) FROM model_predictions WHERE game_id = ?",
            (new,)).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM betting_lines WHERE game_id = ?",
            (new,)).fetchone()[0] == 1


class TestResolveTeamNoSubstring:
    """17. test_resolve_team_no_substring — 'kansas' does NOT resolve to 'arkansas'."""

    def test_kansas_is_not_arkansas(self):
        gw, _ = _gw()
        result = gw.resolve_team("kansas")
        assert result == "kansas"
        assert result != "arkansas"


class TestResolveTeamExactAlias:
    """18. test_resolve_team_exact_alias — 'FIU' → 'florida-international'."""

    def test_fiu(self):
        gw, _ = _gw()
        result = gw.resolve_team("FIU")
        assert result == "florida-international"


# =========================================================================
# Integration Test
# =========================================================================


@pytest.mark.integration
def test_win_loss_records_match_d1baseball():
    """19. Compare W/L records from our DB against D1Baseball for selected teams.

    This test makes HTTP calls to d1baseball.com.

    Run standalone:
        pytest tests/test_schedule_gateway.py::test_win_loss_records_match_d1baseball -v

    Mismatches between our DB and D1Baseball are EXPECTED and VALUABLE —
    they surface ghost games, missing games, and score disagreements.
    The test documents them rather than skipping them.
    """
    import json
    import time as _time

    from scripts.verify_team_schedule import (
        fetch_d1bb_schedule,
        load_d1bb_slugs,
    )
    from scripts.database import get_connection

    slugs = load_d1bb_slugs()

    test_teams = [
        "mississippi-state",
        "auburn",
        "florida",
        "wichita-state",
        "dallas-baptist",
        "hofstra",
        "omaha",
    ]

    conn = get_connection()
    all_ok = True
    mismatch_details = []

    for team_id in test_teams:
        d1bb_slug = slugs.get(team_id)
        if not d1bb_slug:
            mismatch_details.append(f"{team_id}: no D1BB slug configured")
            all_ok = False
            continue

        # Fetch D1BB schedule with retry
        d1bb_games = None
        for attempt in range(3):
            try:
                d1bb_games = fetch_d1bb_schedule(d1bb_slug)
                break
            except Exception as exc:
                if attempt < 2:
                    _time.sleep(2 * (attempt + 1))
                else:
                    mismatch_details.append(
                        f"{team_id}: failed to fetch D1BB schedule: {exc}")
                    all_ok = False

        if d1bb_games is None:
            continue

        # D1BB W/L from parsed results
        d1bb_wins = sum(
            1 for g in d1bb_games
            if g.get("result") and g["result"].get("outcome") == "W"
        )
        d1bb_losses = sum(
            1 for g in d1bb_games
            if g.get("result") and g["result"].get("outcome") == "L"
        )

        # DB W/L
        c = conn.cursor()
        c.execute("""
            SELECT
                COUNT(CASE WHEN winner_id = ? THEN 1 END) AS wins,
                COUNT(CASE WHEN winner_id IS NOT NULL AND winner_id != ? THEN 1 END) AS losses
            FROM games
            WHERE (home_team_id = ? OR away_team_id = ?)
              AND status = 'final'
        """, (team_id, team_id, team_id, team_id))
        row = c.fetchone()
        db_wins, db_losses = row["wins"], row["losses"]

        status = "OK" if (db_wins == d1bb_wins and db_losses == d1bb_losses) else "MISMATCH"

        detail = (f"{team_id}: DB {db_wins}-{db_losses} vs "
                  f"D1BB {d1bb_wins}-{d1bb_losses} [{status}]")
        print(detail)

        if status == "MISMATCH":
            all_ok = False

            # Build lookup sets for detailed diff
            d1bb_final_set = {}
            for g in d1bb_games:
                if g.get("result") and g["result"].get("outcome") in ("W", "L"):
                    opp = g["opponent_team_id"]
                    key = (g["date"], opp, g.get("game_num", 1))
                    d1bb_final_set[key] = g

            db_final_rows = conn.execute("""
                SELECT id, date, home_team_id, away_team_id,
                       home_score, away_score, winner_id
                FROM games
                WHERE (home_team_id = ? OR away_team_id = ?)
                  AND status = 'final'
                ORDER BY date
            """, (team_id, team_id)).fetchall()

            db_final_set = {}
            for r in db_final_rows:
                opp = (r["away_team_id"] if r["home_team_id"] == team_id
                       else r["home_team_id"])
                gid = r["id"]
                gn = 2 if "_gm2" in gid or "_g2" in gid else 1
                key = (r["date"], opp, gn)
                db_final_set[key] = dict(r)

            extra_in_db = set(db_final_set) - set(d1bb_final_set)
            missing_from_db = set(d1bb_final_set) - set(db_final_set)
            common = set(db_final_set) & set(d1bb_final_set)

            score_mismatches = []
            for k in common:
                db_r = db_final_set[k]
                d1_g = d1bb_final_set[k]
                d1_r = d1_g["result"]
                # Determine D1BB scores from team perspective
                if d1_g["is_home"]:
                    d1_hs, d1_as = d1_r["team_score"], d1_r["opp_score"]
                else:
                    d1_hs, d1_as = d1_r["opp_score"], d1_r["team_score"]
                if db_r["home_score"] != d1_hs or db_r["away_score"] != d1_as:
                    score_mismatches.append(
                        f"  score: {k} DB={db_r['home_score']}-{db_r['away_score']} "
                        f"D1BB={d1_hs}-{d1_as}")

            if extra_in_db:
                detail += f"\n  Extra in DB: {sorted(extra_in_db)}"
            if missing_from_db:
                detail += f"\n  Missing from DB: {sorted(missing_from_db)}"
            if score_mismatches:
                detail += "\n" + "\n".join(score_mismatches)

            mismatch_details.append(detail)

        # Polite delay between teams
        _time.sleep(1.5)

    conn.close()

    # NOTE: Mismatches are expected and valuable — they surface data quality
    # issues between our DB and D1Baseball.  The assertion below will fail
    # when there are mismatches, which is BY DESIGN so that CI surfaces them.
    #
    # Known mismatch sources (document as discovered):
    # - Ghost ESPN games not yet cleaned up
    # - D1BB neutral-site home/away labeling differs from ours
    # - Games added by one scraper but not yet reconciled
    # - Doubleheader game-number disagreements
    if not all_ok:
        msg = "W/L mismatches found (expected — documents data gaps):\n"
        msg += "\n".join(mismatch_details)
        pytest.fail(msg)
