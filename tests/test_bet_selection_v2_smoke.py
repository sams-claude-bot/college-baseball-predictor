#!/usr/bin/env python3
"""Smoke checks for refactored bet selection modules and entrypoint."""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_core_imports_and_wrapper_exports():
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))

    import bet_selection_v2
    from scripts.betting import cli, record, risk, selection

    assert callable(bet_selection_v2.main)
    assert callable(bet_selection_v2.analyze_games)
    assert callable(bet_selection_v2.record_bets)
    assert callable(bet_selection_v2.kelly_fraction)
    assert callable(risk.apply_correlation_caps)
    assert callable(selection.analyze_games)
    assert callable(record.record_bets)
    assert callable(cli.main)


def test_old_entrypoint_usage_runs():
    proc = subprocess.run(
        [sys.executable, "scripts/bet_selection_v2.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 1
    assert "Usage: python3 bet_selection_v2.py [analyze|record]" in proc.stdout

