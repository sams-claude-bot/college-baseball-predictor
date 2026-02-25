#!/usr/bin/env python3
"""Run safe, additive schema migrations in a fixed order."""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.database import DB_PATH  # noqa: E402

MIGRATIONS = [
    {
        "name": "2026_02_24_risk_engine_columns",
        "path": PROJECT_ROOT / "scripts" / "migrations" / "2026_02_24_risk_engine_columns.py",
        "description": "Add risk engine columns to tracked bet tables",
    },
]


def _ensure_registry_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migration_runs (
            name TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL,
            notes TEXT
        )
        """
    )
    conn.commit()


def _already_applied(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM schema_migration_runs WHERE name = ?",
        (name,),
    ).fetchone()
    return row is not None


def _record_applied(conn: sqlite3.Connection, name: str, notes: str) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO schema_migration_runs (name, applied_at, notes)
        VALUES (?, ?, ?)
        """,
        (name, datetime.now(timezone.utc).isoformat(), notes),
    )
    conn.commit()


def _format_notes(result: object) -> str:
    if isinstance(result, dict):
        cols = result.get("columns_added")
        changed = result.get("changed")
        return f"changed={changed}; columns_added={cols}"
    return "result=ok"


def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(f"migration_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load migration spec: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    print(f"Migration runner DB: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    try:
        _ensure_registry_table(conn)

        applied_count = 0
        skipped_count = 0

        for item in MIGRATIONS:
            name = item["name"]
            if _already_applied(conn, name):
                print(f"[SKIPPED] {name} (already recorded)")
                skipped_count += 1
                continue

            print(f"[RUN   ] {name} - {item['description']}")
            module = _load_module_from_path(name, item["path"])
            if not hasattr(module, "run_migration"):
                raise RuntimeError(f"{name} missing run_migration()")

            result = module.run_migration()
            notes = _format_notes(result)
            _record_applied(conn, name, notes)
            print(f"[APPLIED] {name} ({notes})")
            applied_count += 1

        print(
            f"Migration runner summary: applied={applied_count} skipped={skipped_count} "
            f"total={len(MIGRATIONS)}"
        )
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
