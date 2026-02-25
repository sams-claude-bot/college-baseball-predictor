#!/usr/bin/env python3
"""Idempotent schema migration for risk engine tracking columns."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.database import DB_PATH  # noqa: E402

TARGET_TABLES = (
    "tracked_bets",
    "tracked_confident_bets",
    "tracked_bets_spreads",
)

TARGET_COLUMNS = (
    ("risk_mode", "TEXT"),
    ("risk_score", "REAL"),
    ("kelly_fraction_used", "REAL"),
    ("suggested_stake", "REAL"),
)


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _table_exists(cur: sqlite3.Cursor, table_name: str) -> bool:
    cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    )
    return cur.fetchone() is not None


def _table_columns(cur: sqlite3.Cursor, table_name: str) -> set[str]:
    cur.execute(f"PRAGMA table_info({_quote_ident(table_name)})")
    return {row[1] for row in cur.fetchall()}


def run_migration() -> dict[str, object]:
    print(f"Opening database: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    table_summaries: list[dict[str, object]] = []
    total_columns_added = 0

    try:
        for table_name in TARGET_TABLES:
            if not _table_exists(cur, table_name):
                print(f"[SKIP] Table missing: {table_name}")
                table_summaries.append(
                    {"table": table_name, "status": "missing_table", "columns_added": 0}
                )
                continue

            existing = _table_columns(cur, table_name)
            added_here = 0

            for column_name, column_type in TARGET_COLUMNS:
                if column_name in existing:
                    print(f"[SKIP] {table_name}.{column_name} already exists")
                    continue

                sql = (
                    f"ALTER TABLE {_quote_ident(table_name)} "
                    f"ADD COLUMN {_quote_ident(column_name)} {column_type}"
                )
                cur.execute(sql)
                print(f"[ADD ] {table_name}.{column_name} {column_type}")
                added_here += 1
                total_columns_added += 1

            table_summaries.append(
                {
                    "table": table_name,
                    "status": "updated" if added_here else "unchanged",
                    "columns_added": added_here,
                }
            )

        conn.commit()
    finally:
        conn.close()

    changed = total_columns_added > 0
    print(
        f"Migration summary: columns_added={total_columns_added} "
        f"tables={len(TARGET_TABLES)} changed={changed}"
    )
    return {
        "name": "2026_02_24_risk_engine_columns",
        "changed": changed,
        "columns_added": total_columns_added,
        "tables": table_summaries,
    }


if __name__ == "__main__":
    run_migration()
