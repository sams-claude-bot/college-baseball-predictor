#!/usr/bin/env python3
"""
Nightly snapshot of player stats + team quality tables.
Preserves daily time series for tracking player trends, hot streaks, slumps, etc.

Run nightly after stats collection:
    python3 scripts/snapshot_stats.py

Tables created/populated:
    - player_stats_snapshots (already exists)
    - team_batting_quality_snapshots
    - team_pitching_quality_snapshots
    - team_stats_snapshots
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.database import get_connection


def ensure_tables(conn):
    """Create snapshot tables if they don't exist."""
    c = conn.cursor()

    # player_stats_snapshots already exists, but ensure it's there
    c.execute('''
        CREATE TABLE IF NOT EXISTS player_stats_snapshots (
            snapshot_date TEXT NOT NULL,
            team_id TEXT NOT NULL,
            player_name TEXT NOT NULL,
            player_class TEXT,
            position TEXT,
            games INTEGER DEFAULT 0,
            pa INTEGER DEFAULT 0,
            at_bats INTEGER DEFAULT 0,
            runs INTEGER DEFAULT 0,
            hits INTEGER DEFAULT 0,
            doubles INTEGER DEFAULT 0,
            triples INTEGER DEFAULT 0,
            home_runs INTEGER DEFAULT 0,
            rbi INTEGER DEFAULT 0,
            hbp INTEGER DEFAULT 0,
            walks INTEGER DEFAULT 0,
            strikeouts INTEGER DEFAULT 0,
            stolen_bases INTEGER DEFAULT 0,
            caught_stealing INTEGER DEFAULT 0,
            batting_avg REAL DEFAULT 0,
            obp REAL DEFAULT 0,
            slg REAL DEFAULT 0,
            ops REAL DEFAULT 0,
            k_pct REAL, bb_pct REAL, k_bb_ratio REAL,
            iso REAL, babip REAL, woba REAL, wrc REAL, wraa REAL, wrc_plus REAL,
            gb_pct REAL, ld_pct REAL, fb_pct REAL, pu_pct REAL, hr_fb_pct REAL,
            -- pitching
            wins INTEGER DEFAULT 0, losses INTEGER DEFAULT 0,
            era REAL DEFAULT 0, appearances INTEGER DEFAULT 0,
            games_started INTEGER DEFAULT 0, complete_games INTEGER DEFAULT 0,
            shutouts INTEGER DEFAULT 0, saves INTEGER DEFAULT 0,
            innings_pitched REAL DEFAULT 0,
            hits_allowed INTEGER DEFAULT 0, runs_allowed INTEGER DEFAULT 0,
            earned_runs INTEGER DEFAULT 0, walks_allowed INTEGER DEFAULT 0,
            strikeouts_pitched INTEGER DEFAULT 0, hbp_pitch INTEGER DEFAULT 0,
            ba_against REAL, obp_against REAL, slg_against REAL, ops_against REAL,
            babip_pitch REAL, bb_pct_pitch REAL, k_pct_pitch REAL,
            k_bb_pct_pitch REAL, k_bb_ratio_pitch REAL,
            whip REAL, fip REAL, xfip REAL, siera REAL, lob_pct REAL,
            gb_pct_pitch REAL, ld_pct_pitch REAL, fb_pct_pitch REAL,
            pu_pct_pitch REAL, hr_fb_pct_pitch REAL,
            PRIMARY KEY (snapshot_date, team_id, player_name)
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS team_batting_quality_snapshots (
            snapshot_date TEXT NOT NULL,
            team_id TEXT NOT NULL,
            lineup_avg REAL, lineup_obp REAL, lineup_slg REAL, lineup_ops REAL,
            lineup_woba REAL, lineup_wrc_plus REAL, lineup_iso REAL, lineup_babip REAL,
            lineup_k_pct REAL, lineup_bb_pct REAL,
            hr_per_game REAL, extra_base_hit_pct REAL,
            bench_avg REAL, bench_ops REAL, bench_depth INTEGER,
            top3_ab_pct REAL, lineup_hhi REAL,
            runs_per_game REAL, total_abs INTEGER, total_hits INTEGER, total_hrs INTEGER,
            elite_bats INTEGER, solid_bats INTEGER, weak_bats INTEGER,
            PRIMARY KEY (snapshot_date, team_id)
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS team_pitching_quality_snapshots (
            snapshot_date TEXT NOT NULL,
            team_id TEXT NOT NULL,
            ace_era REAL, ace_whip REAL, ace_k_per_9 REAL, ace_bb_per_9 REAL,
            ace_fip REAL, ace_innings REAL,
            rotation_era REAL, rotation_whip REAL, rotation_k_per_9 REAL,
            rotation_bb_per_9 REAL, rotation_fip REAL, rotation_innings REAL,
            bullpen_era REAL, bullpen_whip REAL, bullpen_k_per_9 REAL,
            bullpen_bb_per_9 REAL, bullpen_fip REAL, bullpen_innings REAL,
            staff_size INTEGER, starter_count INTEGER,
            ace_ip_pct REAL, top3_ip_pct REAL, innings_hhi REAL,
            staff_era REAL, staff_whip REAL, staff_k_per_9 REAL,
            staff_bb_per_9 REAL, staff_fip REAL, staff_total_ip REAL,
            quality_arms INTEGER, shutdown_arms INTEGER, liability_arms INTEGER,
            PRIMARY KEY (snapshot_date, team_id)
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS team_stats_snapshots (
            snapshot_date TEXT NOT NULL,
            team_id TEXT NOT NULL,
            season INTEGER,
            games_played INTEGER, wins INTEGER, losses INTEGER,
            conference_wins INTEGER, conference_losses INTEGER,
            runs_scored INTEGER, runs_allowed INTEGER,
            batting_avg REAL, era REAL, fielding_pct REAL,
            PRIMARY KEY (snapshot_date, team_id)
        )
    ''')

    conn.commit()


def snapshot_player_stats(conn, date_str):
    """Snapshot player_stats → player_stats_snapshots."""
    c = conn.cursor()

    # Check if already snapshotted today
    existing = c.execute(
        'SELECT COUNT(*) FROM player_stats_snapshots WHERE snapshot_date = ?', (date_str,)
    ).fetchone()[0]
    if existing > 0:
        print(f"Player stats already snapshotted for {date_str} ({existing} rows), skipping")
        return existing

    c.execute('''
        INSERT OR IGNORE INTO player_stats_snapshots (
            snapshot_date, team_id, player_name, player_class, position,
            games, pa, at_bats, runs, hits, doubles, triples, home_runs,
            rbi, hbp, walks, strikeouts, stolen_bases, caught_stealing,
            batting_avg, obp, slg, ops,
            k_pct, bb_pct, k_bb_ratio, iso, babip, woba, wrc, wraa, wrc_plus,
            gb_pct, ld_pct, fb_pct, pu_pct, hr_fb_pct,
            wins, losses, era, appearances, games_started, complete_games,
            shutouts, saves, innings_pitched,
            hits_allowed, runs_allowed, earned_runs, walks_allowed,
            strikeouts_pitched, hbp_pitch,
            ba_against, obp_against, slg_against, ops_against,
            babip_pitch, bb_pct_pitch, k_pct_pitch, k_bb_pct_pitch, k_bb_ratio_pitch,
            whip, fip, xfip, siera, lob_pct,
            gb_pct_pitch, ld_pct_pitch, fb_pct_pitch, pu_pct_pitch, hr_fb_pct_pitch
        )
        SELECT
            ?, team_id, name, year, position,
            games, 0, at_bats, runs, hits, doubles, triples, home_runs,
            rbi, 0, walks, strikeouts, stolen_bases, caught_stealing,
            batting_avg, obp, slg, ops,
            k_pct, bb_pct, k_bb_ratio, iso, babip, woba, wrc, wraa, wrc_plus,
            gb_pct, ld_pct, fb_pct, pu_pct, hr_fb_pct,
            wins, losses, era, games_pitched, games_started, 0,
            0, saves, innings_pitched,
            hits_allowed, runs_allowed, earned_runs, walks_allowed,
            strikeouts_pitched, 0,
            NULL, obp_against, slg_against, ops_against,
            babip_pitch, bb_pct_pitch, k_pct_pitch, k_bb_pct_pitch, k_bb_ratio_pitch,
            whip, fip, xfip, siera, lob_pct,
            gb_pct_pitch, ld_pct_pitch, fb_pct_pitch, pu_pct_pitch, hr_fb_pct_pitch
        FROM player_stats
    ''', (date_str,))

    count = c.rowcount
    conn.commit()
    print(f"Player stats snapshot: {count} rows for {date_str}")
    return count


def snapshot_team_batting_quality(conn, date_str):
    """Snapshot team_batting_quality → team_batting_quality_snapshots."""
    c = conn.cursor()

    existing = c.execute(
        'SELECT COUNT(*) FROM team_batting_quality_snapshots WHERE snapshot_date = ?', (date_str,)
    ).fetchone()[0]
    if existing > 0:
        print(f"Team batting quality already snapshotted for {date_str} ({existing} rows), skipping")
        return existing

    c.execute('''
        INSERT OR IGNORE INTO team_batting_quality_snapshots (
            snapshot_date, team_id,
            lineup_avg, lineup_obp, lineup_slg, lineup_ops,
            lineup_woba, lineup_wrc_plus, lineup_iso, lineup_babip,
            lineup_k_pct, lineup_bb_pct,
            hr_per_game, extra_base_hit_pct,
            bench_avg, bench_ops, bench_depth,
            top3_ab_pct, lineup_hhi,
            runs_per_game, total_abs, total_hits, total_hrs,
            elite_bats, solid_bats, weak_bats
        )
        SELECT
            ?, team_id,
            lineup_avg, lineup_obp, lineup_slg, lineup_ops,
            lineup_woba, lineup_wrc_plus, lineup_iso, lineup_babip,
            lineup_k_pct, lineup_bb_pct,
            hr_per_game, extra_base_hit_pct,
            bench_avg, bench_ops, bench_depth,
            top3_ab_pct, lineup_hhi,
            runs_per_game, total_abs, total_hits, total_hrs,
            elite_bats, solid_bats, weak_bats
        FROM team_batting_quality
    ''', (date_str,))

    count = c.rowcount
    conn.commit()
    print(f"Team batting quality snapshot: {count} rows for {date_str}")
    return count


def snapshot_team_pitching_quality(conn, date_str):
    """Snapshot team_pitching_quality → team_pitching_quality_snapshots."""
    c = conn.cursor()

    existing = c.execute(
        'SELECT COUNT(*) FROM team_pitching_quality_snapshots WHERE snapshot_date = ?', (date_str,)
    ).fetchone()[0]
    if existing > 0:
        print(f"Team pitching quality already snapshotted for {date_str} ({existing} rows), skipping")
        return existing

    c.execute('''
        INSERT OR IGNORE INTO team_pitching_quality_snapshots (
            snapshot_date, team_id,
            ace_era, ace_whip, ace_k_per_9, ace_bb_per_9, ace_fip, ace_innings,
            rotation_era, rotation_whip, rotation_k_per_9, rotation_bb_per_9, rotation_fip, rotation_innings,
            bullpen_era, bullpen_whip, bullpen_k_per_9, bullpen_bb_per_9, bullpen_fip, bullpen_innings,
            staff_size, starter_count,
            ace_ip_pct, top3_ip_pct, innings_hhi,
            staff_era, staff_whip, staff_k_per_9, staff_bb_per_9, staff_fip, staff_total_ip,
            quality_arms, shutdown_arms, liability_arms
        )
        SELECT
            ?, team_id,
            ace_era, ace_whip, ace_k_per_9, ace_bb_per_9, ace_fip, ace_innings,
            rotation_era, rotation_whip, rotation_k_per_9, rotation_bb_per_9, rotation_fip, rotation_innings,
            bullpen_era, bullpen_whip, bullpen_k_per_9, bullpen_bb_per_9, bullpen_fip, bullpen_innings,
            staff_size, starter_count,
            ace_ip_pct, top3_ip_pct, innings_hhi,
            staff_era, staff_whip, staff_k_per_9, staff_bb_per_9, staff_fip, staff_total_ip,
            quality_arms, shutdown_arms, liability_arms
        FROM team_pitching_quality
    ''', (date_str,))

    count = c.rowcount
    conn.commit()
    print(f"Team pitching quality snapshot: {count} rows for {date_str}")
    return count


def snapshot_team_stats(conn, date_str):
    """Snapshot team_stats → team_stats_snapshots."""
    c = conn.cursor()

    existing = c.execute(
        'SELECT COUNT(*) FROM team_stats_snapshots WHERE snapshot_date = ?', (date_str,)
    ).fetchone()[0]
    if existing > 0:
        print(f"Team stats already snapshotted for {date_str} ({existing} rows), skipping")
        return existing

    c.execute('''
        INSERT OR IGNORE INTO team_stats_snapshots (
            snapshot_date, team_id, season,
            games_played, wins, losses, conference_wins, conference_losses,
            runs_scored, runs_allowed, batting_avg, era, fielding_pct
        )
        SELECT
            ?, team_id, season,
            games_played, wins, losses, conference_wins, conference_losses,
            runs_scored, runs_allowed, batting_avg, era, fielding_pct
        FROM team_stats
    ''', (date_str,))

    count = c.rowcount
    conn.commit()
    print(f"Team stats snapshot: {count} rows for {date_str}")
    return count


def main():
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"=== Stats Snapshot for {today} ===")

    conn = get_connection()
    ensure_tables(conn)

    snapshot_player_stats(conn, today)
    snapshot_team_batting_quality(conn, today)
    snapshot_team_pitching_quality(conn, today)
    snapshot_team_stats(conn, today)

    # Summary
    c = conn.cursor()
    for table in ['player_stats_snapshots', 'team_batting_quality_snapshots',
                   'team_pitching_quality_snapshots', 'team_stats_snapshots']:
        dates = c.execute(f'SELECT COUNT(DISTINCT snapshot_date) FROM {table}').fetchone()[0]
        rows = c.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
        print(f"  {table}: {dates} dates, {rows} total rows")

    conn.close()
    print("Done!")


if __name__ == '__main__':
    main()
