#!/usr/bin/env python3
"""Tests for the SIDEARM Live Stats Poller."""

import json
import os
import sqlite3
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from sidearm_poller import (
    extract_school_code,
    extract_situation,
    is_game_complete,
    merge_sidearm_situation,
    ordinal,
    player_name,
    SidearmPoller,
    ensure_school_code_column,
    ensure_live_events_table,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_GAME_DATA = {
    "Game": {
        "HasStarted": True,
        "IsComplete": False,
        "Period": 7,
        "Context": "0 Ball 0 Strike 0 Out",
        "HomeTeam": {
            "Id": "UTAH TEC", "Name": "Utah Tech", "Score": 13,
            "PeriodScores": [2, 0, 0, 5, 4, 2, 0]
        },
        "VisitingTeam": {
            "Id": "AUSTIN P", "Name": "Austin Peay", "Score": 11,
            "PeriodScores": [0, 1, 0, 4, 1, 2, 3]
        },
        "Situation": {
            "PitchingTeam": "VisitingTeam",
            "BattingTeam": "HomeTeam",
            "Pitcher": {"FirstName": "Colin", "LastName": "Carney", "UniformNumber": "24"},
            "PitcherPitchCount": 16,
            "PitcherHandedness": "R",
            "Batter": {"FirstName": "Kyle", "LastName": "McDaniel", "UniformNumber": "14"},
            "BatterHandedness": "L",
            "OnDeck": {"FirstName": "Kace", "LastName": "Naone", "UniformNumber": "8"},
            "OnFirst": {"FirstName": "Petey", "LastName": "Soto Jr.", "UniformNumber": "4"},
            "OnSecond": None,
            "OnThird": None,
            "Balls": 0, "Strikes": 0, "Outs": 0,
            "Inning": 7.5
        }
    },
    "Plays": [
        {
            "Narrative": "Kyler Proctor grounded out to 3b (1-1 FB).",
            "Context": "P: J. PRICE; B: K. PROCTOR",
            "Type": "Out", "Action": "Ground", "Period": 1,
            "Score": None,
            "Player": {"FirstName": "Kyler", "LastName": "Proctor"},
            "Team": "VisitingTeam"
        },
        {
            "Narrative": "John Smith singled to left field.",
            "Context": "P: C. CARNEY; B: J. SMITH",
            "Type": "Hit", "Action": "Single", "Period": 3,
            "Score": None,
            "Player": {"FirstName": "John", "LastName": "Smith"},
            "Team": "HomeTeam"
        }
    ]
}


@pytest.fixture
def db():
    """In-memory SQLite DB with required tables."""
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE sidearm_links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            domain TEXT NOT NULL,
            url TEXT NOT NULL,
            game_date TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(game_id, domain)
        )
    """)
    conn.execute("""
        CREATE TABLE games (
            id TEXT PRIMARY KEY,
            home_team_id TEXT,
            away_team_id TEXT,
            home_score INTEGER,
            away_score INTEGER,
            inning_text TEXT,
            innings INTEGER,
            status TEXT DEFAULT 'scheduled',
            winner_id TEXT,
            situation_json TEXT,
            updated_at TEXT,
            date TEXT
        )
    """)
    ensure_school_code_column(conn)
    ensure_live_events_table(conn)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Tests: school code extraction
# ---------------------------------------------------------------------------

class TestSchoolCodeExtraction:
    def test_extract_from_link(self):
        assert extract_school_code("https://sidearmstats.com/dixie/baseball/") == "dixie"

    def test_extract_from_link_no_trailing_slash(self):
        assert extract_school_code("https://sidearmstats.com/utahtech/baseball") == "utahtech"

    def test_extract_from_complex_url(self):
        assert extract_school_code("https://sidearmstats.com/olemiss/baseball/game.json") == "olemiss"

    def test_no_match(self):
        assert extract_school_code("https://example.com/foo") is None

    def test_empty(self):
        assert extract_school_code("") is None


# ---------------------------------------------------------------------------
# Tests: situation field mapping
# ---------------------------------------------------------------------------

class TestSituationExtraction:
    def test_basic_fields(self):
        sa = extract_situation(SAMPLE_GAME_DATA)
        assert sa is not None
        assert sa['sa_home_score'] == 13
        assert sa['sa_visitor_score'] == 11
        assert sa['sa_home_name'] == 'Utah Tech'
        assert sa['sa_visitor_name'] == 'Austin Peay'
        assert sa['sa_batter'] == 'Kyle McDaniel'
        assert sa['sa_batter_number'] == '14'
        assert sa['sa_pitcher'] == 'Colin Carney'
        assert sa['sa_pitcher_number'] == '24'
        assert sa['sa_pitcher_pitch_count'] == 16
        assert sa['sa_pitcher_hand'] == 'R'
        assert sa['sa_batter_hand'] == 'L'
        assert sa['sa_count'] == '0-0'
        assert sa['sa_outs'] == 0
        assert sa['sa_on_deck'] == 'Kace Naone'

    def test_runners(self):
        sa = extract_situation(SAMPLE_GAME_DATA)
        assert sa['sa_on_first'] is True
        assert sa['sa_on_second'] is False
        assert sa['sa_on_third'] is False
        assert sa['sa_runner_first'] == 'Petey Soto Jr.'
        assert sa['sa_runner_second'] is None
        assert sa['sa_runner_third'] is None

    def test_innings_scores(self):
        sa = extract_situation(SAMPLE_GAME_DATA)
        assert sa['sa_home_innings'] == [2, 0, 0, 5, 4, 2, 0]
        assert sa['sa_visitor_innings'] == [0, 1, 0, 4, 1, 2, 3]

    def test_plays(self):
        sa = extract_situation(SAMPLE_GAME_DATA)
        plays = sa['sa_plays']
        assert len(plays) == 2
        # Most recent first
        assert plays[0]['text'] == 'John Smith singled to left field.'
        assert plays[0]['half'] == 'bottom'
        assert plays[1]['text'] == 'Kyler Proctor grounded out to 3b (1-1 FB).'
        assert plays[1]['half'] == 'top'

    def test_not_started(self):
        data = {"Game": {"HasStarted": False, "IsComplete": False}}
        assert extract_situation(data) is None

    def test_no_game(self):
        assert extract_situation({}) is None

    def test_updated_at_present(self):
        sa = extract_situation(SAMPLE_GAME_DATA)
        assert 'sa_updated_at' in sa


# ---------------------------------------------------------------------------
# Tests: inning half detection
# ---------------------------------------------------------------------------

class TestInningHalf:
    def test_top_of_first(self):
        data = dict(SAMPLE_GAME_DATA)
        data = json.loads(json.dumps(data))
        data['Game']['Situation']['Inning'] = 1.0
        sa = extract_situation(data)
        assert sa['sa_inning'] == 1
        assert sa['sa_inning_half'] == 'top'
        assert sa['sa_inning_display'] == 'Top 1st'

    def test_bottom_of_first(self):
        data = json.loads(json.dumps(SAMPLE_GAME_DATA))
        data['Game']['Situation']['Inning'] = 1.5
        sa = extract_situation(data)
        assert sa['sa_inning'] == 1
        assert sa['sa_inning_half'] == 'bottom'
        assert sa['sa_inning_display'] == 'Bot 1st'

    def test_top_of_seventh(self):
        data = json.loads(json.dumps(SAMPLE_GAME_DATA))
        data['Game']['Situation']['Inning'] = 7.0
        sa = extract_situation(data)
        assert sa['sa_inning'] == 7
        assert sa['sa_inning_half'] == 'top'
        assert sa['sa_inning_display'] == 'Top 7th'

    def test_bottom_of_seventh(self):
        sa = extract_situation(SAMPLE_GAME_DATA)  # Inning = 7.5
        assert sa['sa_inning'] == 7
        assert sa['sa_inning_half'] == 'bottom'
        assert sa['sa_inning_display'] == 'Bot 7th'

    def test_top_of_ninth(self):
        data = json.loads(json.dumps(SAMPLE_GAME_DATA))
        data['Game']['Situation']['Inning'] = 9.0
        sa = extract_situation(data)
        assert sa['sa_inning_display'] == 'Top 9th'

    def test_extras(self):
        data = json.loads(json.dumps(SAMPLE_GAME_DATA))
        data['Game']['Situation']['Inning'] = 12.5
        sa = extract_situation(data)
        assert sa['sa_inning'] == 12
        assert sa['sa_inning_half'] == 'bottom'
        assert sa['sa_inning_display'] == 'Bot 12th'


# ---------------------------------------------------------------------------
# Tests: ordinal helper
# ---------------------------------------------------------------------------

class TestOrdinal:
    def test_first(self):
        assert ordinal(1) == '1st'

    def test_second(self):
        assert ordinal(2) == '2nd'

    def test_third(self):
        assert ordinal(3) == '3rd'

    def test_fourth(self):
        assert ordinal(4) == '4th'

    def test_eleventh(self):
        assert ordinal(11) == '11th'

    def test_twelfth(self):
        assert ordinal(12) == '12th'

    def test_twenty_first(self):
        assert ordinal(21) == '21st'


# ---------------------------------------------------------------------------
# Tests: merge function
# ---------------------------------------------------------------------------

class TestMerge:
    def test_merge_into_empty(self):
        result = merge_sidearm_situation(None, {'sa_home_score': 5})
        parsed = json.loads(result)
        assert parsed['sa_home_score'] == 5

    def test_preserves_sb_fields(self):
        existing = json.dumps({'sb_home_score': 3, 'sb_batter': 'Joe Smith'})
        result = merge_sidearm_situation(existing, {'sa_home_score': 5, 'sa_batter': 'Kyle McDaniel'})
        parsed = json.loads(result)
        assert parsed['sb_home_score'] == 3
        assert parsed['sb_batter'] == 'Joe Smith'
        assert parsed['sa_home_score'] == 5
        assert parsed['sa_batter'] == 'Kyle McDaniel'

    def test_overwrites_sa_fields(self):
        existing = json.dumps({'sa_home_score': 3, 'sb_home_score': 3})
        result = merge_sidearm_situation(existing, {'sa_home_score': 7})
        parsed = json.loads(result)
        assert parsed['sa_home_score'] == 7
        assert parsed['sb_home_score'] == 3

    def test_skips_none_values(self):
        existing = json.dumps({'sa_batter': 'Old Batter'})
        result = merge_sidearm_situation(existing, {'sa_batter': None, 'sa_home_score': 2})
        parsed = json.loads(result)
        assert parsed['sa_batter'] == 'Old Batter'
        assert parsed['sa_home_score'] == 2

    def test_handles_corrupt_json(self):
        result = merge_sidearm_situation("not json{", {'sa_home_score': 1})
        parsed = json.loads(result)
        assert parsed['sa_home_score'] == 1


# ---------------------------------------------------------------------------
# Tests: game complete detection
# ---------------------------------------------------------------------------

class TestGameComplete:
    def test_not_complete(self):
        assert is_game_complete(SAMPLE_GAME_DATA) is False

    def test_complete(self):
        data = json.loads(json.dumps(SAMPLE_GAME_DATA))
        data['Game']['IsComplete'] = True
        assert is_game_complete(data) is True

    def test_empty_data(self):
        assert is_game_complete({}) is False


# ---------------------------------------------------------------------------
# Tests: score change detection
# ---------------------------------------------------------------------------

class TestScoreChangeDetection:
    def test_detects_score_change(self, db):
        db.execute("INSERT INTO games (id, status) VALUES ('g1', 'scheduled')")
        db.commit()

        poller = SidearmPoller(db)

        # First call seeds state
        sa1 = {'sa_home_score': 0, 'sa_visitor_score': 0, 'sa_home_name': 'A', 'sa_visitor_name': 'B', 'sa_inning_display': 'Top 1st'}
        poller._check_score_change('g1', sa1)

        # No event yet (first sighting)
        events = db.execute("SELECT * FROM live_events WHERE game_id = 'g1'").fetchall()
        assert len(events) == 0

        # Score changes
        sa2 = dict(sa1)
        sa2['sa_home_score'] = 2
        poller._check_score_change('g1', sa2)

        events = db.execute("SELECT * FROM live_events WHERE game_id = 'g1'").fetchall()
        assert len(events) == 1
        data = json.loads(events[0]['data_json'])
        assert data['home_score'] == 2
        assert data['prev_home_score'] == 0
        assert data['source'] == 'sidearm'

    def test_no_event_on_same_score(self, db):
        db.execute("INSERT INTO games (id, status) VALUES ('g2', 'scheduled')")
        db.commit()

        poller = SidearmPoller(db)
        sa = {'sa_home_score': 3, 'sa_visitor_score': 1, 'sa_home_name': 'A', 'sa_visitor_name': 'B', 'sa_inning_display': 'Top 3rd'}

        poller._check_score_change('g2', sa)
        poller._check_score_change('g2', sa)

        events = db.execute("SELECT * FROM live_events WHERE game_id = 'g2'").fetchall()
        assert len(events) == 0


# ---------------------------------------------------------------------------
# Tests: DB migration
# ---------------------------------------------------------------------------

class TestDBMigration:
    def test_school_code_column_added(self):
        conn = sqlite3.connect(':memory:')
        conn.execute("""
            CREATE TABLE sidearm_links (
                id INTEGER PRIMARY KEY, game_id TEXT, domain TEXT, url TEXT, game_date TEXT
            )
        """)
        ensure_school_code_column(conn)
        cols = [row[1] for row in conn.execute("PRAGMA table_info(sidearm_links)").fetchall()]
        assert 'school_code' in cols

    def test_idempotent(self):
        conn = sqlite3.connect(':memory:')
        conn.execute("""
            CREATE TABLE sidearm_links (
                id INTEGER PRIMARY KEY, game_id TEXT, domain TEXT, url TEXT, game_date TEXT
            )
        """)
        ensure_school_code_column(conn)
        ensure_school_code_column(conn)  # Should not raise
