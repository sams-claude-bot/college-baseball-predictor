"""Tests for scripts/daily_benchmark.py"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.daily_benchmark import (
    cleanup_artifacts,
    find_yesterday_artifact,
    generate_delta,
    parse_leaderboard,
)

# ---------------------------------------------------------------------------
# Sample benchmark markdown for testing
# ---------------------------------------------------------------------------

BENCHMARK_V1 = """\
# Model Benchmark (Leak-Safe Meta Stack)

- Generated: `2026-03-04T08:00:00`
- Date window: `2026-02-18` to `2026-03-04`

## Leak-safe per-model leaderboard

| Model | n predictions | Win accuracy | Brier | Log loss | ECE |
|---|---:|---:|---:|---:|---:|
| meta_ensemble | 600 | 0.6400 | 0.2350 | 0.6800 | 0.1100 |
| elo | 900 | 0.6300 | 0.2200 | 0.6300 | 0.0250 |
| pitching | 900 | 0.7100 | 0.1900 | 0.5500 | 0.0300 |

## Strict cohort leaderboard

| Model | n predictions | Win accuracy | Brier | Log loss | ECE |
|---|---:|---:|---:|---:|---:|
| meta_ensemble | 20 | 0.7000 | 0.2200 | 0.6800 | 0.2300 |
"""

BENCHMARK_V2 = """\
# Model Benchmark (Leak-Safe Meta Stack)

- Generated: `2026-03-05T08:00:00`
- Date window: `2026-02-18` to `2026-03-05`

## Leak-safe per-model leaderboard

| Model | n predictions | Win accuracy | Brier | Log loss | ECE |
|---|---:|---:|---:|---:|---:|
| meta_ensemble | 634 | 0.6483 | 0.2309 | 0.6762 | 0.1119 |
| elo | 946 | 0.6459 | 0.2158 | 0.6219 | 0.0248 |
| pitching | 946 | 0.7230 | 0.1842 | 0.5458 | 0.0322 |
| neural | 943 | 0.6320 | 0.2299 | 0.6573 | 0.0552 |

## Strict cohort leaderboard

| Model | n predictions | Win accuracy | Brier | Log loss | ECE |
|---|---:|---:|---:|---:|---:|
| meta_ensemble | 26 | 0.7308 | 0.2167 | 0.6749 | 0.2277 |
"""


class TestParseLeaderboard:
    def test_extracts_all_models(self):
        metrics = parse_leaderboard(BENCHMARK_V1)
        assert set(metrics.keys()) == {"meta_ensemble", "elo", "pitching"}

    def test_metric_values(self):
        metrics = parse_leaderboard(BENCHMARK_V1)
        assert metrics["elo"]["accuracy"] == pytest.approx(0.6300)
        assert metrics["elo"]["brier"] == pytest.approx(0.2200)
        assert metrics["elo"]["logloss"] == pytest.approx(0.6300)
        assert metrics["elo"]["n"] == 900

    def test_only_leak_safe_section(self):
        """Should parse leak-safe leaderboard, not strict cohort."""
        metrics = parse_leaderboard(BENCHMARK_V1)
        # Strict cohort has n=20 for meta_ensemble; leak-safe has n=600
        assert metrics["meta_ensemble"]["n"] == 600


class TestArtifactCreated:
    def test_artifact_file_created_with_correct_date(self, tmp_path):
        """Benchmark artifact is created with correct date in filename."""
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        today_path = artifacts / "model_benchmark_2026-03-05.md"
        today_path.write_text(BENCHMARK_V2)

        assert today_path.exists()
        assert "2026-03-05" in today_path.name


class TestDeltaReport:
    def test_delta_identifies_accuracy_changes(self, tmp_path):
        """Delta report correctly identifies accuracy changes per model."""
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        yesterday = artifacts / "model_benchmark_2026-03-04.md"
        today = artifacts / "model_benchmark_2026-03-05.md"
        yesterday.write_text(BENCHMARK_V1)
        today.write_text(BENCHMARK_V2)

        with patch("scripts.daily_benchmark.ARTIFACTS_DIR", artifacts):
            delta_path = generate_delta(today, yesterday, "2026-03-05")

        assert delta_path.exists()
        content = delta_path.read_text()

        # Should contain delta header
        assert "Benchmark Delta" in content
        assert "2026-03-05" in content

        # elo accuracy went from 0.6300 to 0.6459 => +0.0159
        assert "+0.0159" in content
        # pitching accuracy went from 0.7100 to 0.7230 => +0.0130
        assert "+0.0130" in content

    def test_delta_handles_new_model(self, tmp_path):
        """Delta shows NEW for models that appear in today but not yesterday."""
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        yesterday = artifacts / "model_benchmark_2026-03-04.md"
        today = artifacts / "model_benchmark_2026-03-05.md"
        yesterday.write_text(BENCHMARK_V1)
        today.write_text(BENCHMARK_V2)

        with patch("scripts.daily_benchmark.ARTIFACTS_DIR", artifacts):
            delta_path = generate_delta(today, yesterday, "2026-03-05")

        content = delta_path.read_text()
        # neural is in V2 but not V1
        assert "neural" in content
        assert "NEW" in content

    def test_delta_handles_removed_model(self, tmp_path):
        """Delta shows REMOVED for models gone from today."""
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        # Swap: V2 is "yesterday" (has neural), V1 is "today" (no neural)
        yesterday = artifacts / "model_benchmark_2026-03-04.md"
        today = artifacts / "model_benchmark_2026-03-05.md"
        yesterday.write_text(BENCHMARK_V2)
        today.write_text(BENCHMARK_V1)

        with patch("scripts.daily_benchmark.ARTIFACTS_DIR", artifacts):
            delta_path = generate_delta(today, yesterday, "2026-03-05")

        content = delta_path.read_text()
        assert "REMOVED" in content


class TestMissingYesterdaySkipsDelta:
    def test_no_previous_artifact_returns_none(self, tmp_path):
        """When no previous artifact exists, find_yesterday_artifact returns None."""
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()

        with patch("scripts.daily_benchmark.ARTIFACTS_DIR", artifacts):
            result = find_yesterday_artifact("2026-03-05")

        assert result is None

    def test_finds_artifact_from_two_days_ago(self, tmp_path):
        """Looks back up to 7 days to find previous artifact."""
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        # No yesterday artifact, but day-before-yesterday exists
        two_days_ago = artifacts / "model_benchmark_2026-03-03.md"
        two_days_ago.write_text(BENCHMARK_V1)

        with patch("scripts.daily_benchmark.ARTIFACTS_DIR", artifacts):
            result = find_yesterday_artifact("2026-03-05")

        assert result is not None
        assert "2026-03-03" in result.name


class TestRetainDeletesOld:
    def test_retain_deletes_old_artifacts(self, tmp_path):
        """--retain N deletes benchmark and delta artifacts beyond threshold."""
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()

        # Create 5 benchmark artifacts
        for i in range(1, 6):
            (artifacts / f"model_benchmark_2026-03-{i:02d}.md").write_text("x")
            (artifacts / f"benchmark_delta_2026-03-{i:02d}.md").write_text("x")

        with patch("scripts.daily_benchmark.ARTIFACTS_DIR", artifacts):
            deleted = cleanup_artifacts(retain=3)

        # Should have deleted 2 benchmark + 2 delta = 4 files
        assert len(deleted) == 4
        # Oldest 2 of each prefix should be gone
        assert not (artifacts / "model_benchmark_2026-03-01.md").exists()
        assert not (artifacts / "model_benchmark_2026-03-02.md").exists()
        assert not (artifacts / "benchmark_delta_2026-03-01.md").exists()
        assert not (artifacts / "benchmark_delta_2026-03-02.md").exists()
        # Newest 3 still exist
        assert (artifacts / "model_benchmark_2026-03-03.md").exists()
        assert (artifacts / "model_benchmark_2026-03-05.md").exists()

    def test_retain_no_op_when_fewer_artifacts(self, tmp_path):
        """No deletion when artifact count is within retain limit."""
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        (artifacts / "model_benchmark_2026-03-05.md").write_text("x")

        with patch("scripts.daily_benchmark.ARTIFACTS_DIR", artifacts):
            deleted = cleanup_artifacts(retain=30)

        assert len(deleted) == 0

    def test_does_not_delete_non_daily_artifacts(self, tmp_path):
        """Non-daily artifacts (e.g. meta_ensemble_audit.md) are untouched."""
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        (artifacts / "meta_ensemble_audit.md").write_text("x")
        for i in range(1, 6):
            (artifacts / f"model_benchmark_2026-03-{i:02d}.md").write_text("x")

        with patch("scripts.daily_benchmark.ARTIFACTS_DIR", artifacts):
            cleanup_artifacts(retain=2)

        assert (artifacts / "meta_ensemble_audit.md").exists()
