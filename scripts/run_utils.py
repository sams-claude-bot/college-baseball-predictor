#!/usr/bin/env python3
"""
Shared utilities for cron-run scripts.

Provides consistent logging, timing, error handling, and structured summaries.

Usage:
    from run_utils import ScriptRunner

    runner = ScriptRunner("d1b_schedule")

    runner.info("Scraping 7 days of schedule")
    runner.add_stat("games_found", 42)
    runner.add_stat("teams_unresolved", 3)
    runner.add_error("Could not resolve team: xyz-university")

    # At the end:
    runner.finish()  # prints summary, exits with appropriate code
"""

import logging
import sys
import time
from datetime import datetime
from pathlib import Path


class ScriptRunner:
    """Wrapper for consistent script execution logging."""

    def __init__(self, name: str):
        self.name = name
        self.start_time = time.time()
        self.stats: dict = {}
        self.errors: list = []
        self.warnings: list = []

        # Configure logger
        self.log = logging.getLogger(name)
        if not self.log.handlers:
            self.log.setLevel(logging.INFO)
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            self.log.addHandler(handler)

        self.log.info(f"=== {name} started ===")

    def info(self, msg: str):
        self.log.info(msg)

    def warn(self, msg: str):
        self.log.warning(msg)
        self.warnings.append(msg)

    def error(self, msg: str):
        self.log.error(msg)
        self.errors.append(msg)

    def add_stat(self, key: str, value):
        """Track a numeric or string stat for the summary."""
        self.stats[key] = value

    def inc_stat(self, key: str, amount: int = 1):
        """Increment a counter stat."""
        self.stats[key] = self.stats.get(key, 0) + amount

    def finish(self, exit_on_error: bool = True):
        """Print structured summary and exit with appropriate code.
        
        Returns exit code (0=ok, 1=errors) if exit_on_error is False.
        """
        elapsed = time.time() - self.start_time
        minutes = int(elapsed // 60)
        seconds = elapsed % 60

        self.log.info("")
        self.log.info(f"{'=' * 50}")
        self.log.info(f"SUMMARY: {self.name}")
        self.log.info(f"{'=' * 50}")

        if minutes > 0:
            self.log.info(f"Duration: {minutes}m {seconds:.0f}s")
        else:
            self.log.info(f"Duration: {seconds:.1f}s")

        for key, val in self.stats.items():
            self.log.info(f"  {key}: {val}")

        if self.warnings:
            self.log.info(f"Warnings: {len(self.warnings)}")
            for w in self.warnings:
                self.log.info(f"  ⚠ {w}")

        if self.errors:
            self.log.info(f"Errors: {len(self.errors)}")
            for e in self.errors:
                self.log.info(f"  ✗ {e}")
            self.log.info(f"STATUS: FAILED")
        else:
            self.log.info(f"STATUS: OK")

        self.log.info(f"{'=' * 50}")

        code = 1 if self.errors else 0
        if exit_on_error:
            sys.exit(code)
        return code
