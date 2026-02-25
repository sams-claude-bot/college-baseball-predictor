#!/usr/bin/env python3
"""Scan repo Python files for non-canonical sqlite DB path usage."""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.database import DB_PATH  # noqa: E402

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "data",
    "logs",
    "artifacts",
}


@dataclass
class Finding:
    path: Path
    line: int
    kind: str
    detail: str


@dataclass
class FileScan:
    imports_canonical_db_path: bool = False
    db_path_assigned_locally: bool = False
    findings: list[Finding] = field(default_factory=list)


class Visitor(ast.NodeVisitor):
    def __init__(self, path: Path, source: str) -> None:
        self.path = path
        self.source = source
        self.scan = FileScan()

    def _src(self, node: ast.AST | None) -> str:
        if node is None:
            return ""
        segment = ast.get_source_segment(self.source, node)
        if segment:
            return segment
        try:
            return ast.unparse(node)
        except Exception:
            return "<expr>"

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "scripts.database":
            for alias in node.names:
                if alias.name == "DB_PATH":
                    self.scan.imports_canonical_db_path = True
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if any(isinstance(t, ast.Name) and t.id == "DB_PATH" for t in node.targets):
            src = self._src(node.value)
            if "baseball.db" in src:
                self.scan.db_path_assigned_locally = True
                self.scan.findings.append(
                    Finding(
                        self.path,
                        node.lineno,
                        "local-db-path",
                        f"Local DB_PATH assignment: {src}",
                    )
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        is_sqlite_connect = (
            isinstance(func, ast.Attribute)
            and func.attr == "connect"
            and isinstance(func.value, ast.Name)
            and func.value.id == "sqlite3"
        )
        if not is_sqlite_connect:
            return self.generic_visit(node)

        arg0 = node.args[0] if node.args else None
        arg_src = self._src(arg0)
        canonical = False
        if arg0 is not None:
            if isinstance(arg0, ast.Name) and arg0.id == "DB_PATH":
                canonical = self.scan.imports_canonical_db_path and not self.scan.db_path_assigned_locally
            elif (
                isinstance(arg0, ast.Call)
                and isinstance(arg0.func, ast.Name)
                and arg0.func.id == "str"
                and arg0.args
                and isinstance(arg0.args[0], ast.Name)
                and arg0.args[0].id == "DB_PATH"
            ):
                canonical = self.scan.imports_canonical_db_path and not self.scan.db_path_assigned_locally

        suspicious = (
            "baseball.db" in arg_src
            or ("DB_PATH" in arg_src and not canonical)
            or arg_src in {"'data/baseball.db'", '"data/baseball.db"'}
        )
        if suspicious and not canonical:
            self.scan.findings.append(
                Finding(
                    self.path,
                    node.lineno,
                    "sqlite-connect",
                    f"sqlite3.connect uses non-canonical path expression: {arg_src}",
                )
            )

        self.generic_visit(node)


def iter_python_files(root: Path):
    for path in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def main() -> int:
    findings: list[Finding] = []
    scanned = 0

    for path in iter_python_files(PROJECT_ROOT):
        if path == (PROJECT_ROOT / "scripts" / "database.py"):
            continue
        scanned += 1
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            continue

        visitor = Visitor(path, source)
        visitor.visit(tree)
        findings.extend(visitor.scan.findings)

    print(f"Canonical DB path: {DB_PATH}")
    print("Preferred usage: `from scripts.database import DB_PATH` (or `get_connection()`).")
    print(f"Scanned {scanned} Python files under {PROJECT_ROOT}")

    if not findings:
        print("No non-canonical sqlite DB path usages found.")
        return 0

    print(f"Found {len(findings)} non-canonical usage(s):")
    for f in sorted(findings, key=lambda x: (str(x.path), x.line, x.kind)):
        rel = f.path.relative_to(PROJECT_ROOT)
        print(f"- {rel}:{f.line} [{f.kind}] {f.detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
