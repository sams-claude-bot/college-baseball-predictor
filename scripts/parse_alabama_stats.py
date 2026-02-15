#!/usr/bin/env python3

import sqlite3
import json
import re
from datetime import datetime

def clean_name(name):
    """Remove asterisk and clean up player name"""
    return name.replace("* ", "").strip()

def parse_gp_gs(gp_gs_str):
    """Parse GP-GS format like '3 - 3' into games and games_started"""
    if ' - ' in gp_gs_str:
        parts = gp_gs_str.split(' - ')
        return int(parts[0]), int(parts[1])
    return int(gp_gs_str), 0

def parse_sb_att(sb_att_str):
    """Parse SB-ATT format like '1 - 1' into stolen_bases and caught_stealing"""
    if ' - ' in sb_att_str:
        parts = sb_att_str.split(' - ')
        stolen = int(parts[0])
        attempts = int(parts[1])
        caught = attempts - stolen
        return stolen, caught
    return int(sb_att_str), 0

def parse_w_l(w_l_str):
    """Parse W-L format like '1 - 0' into wins and losses"""
    if ' - ' in w_l_str:
        parts = w_l_str.split(' - ')
        return int(parts[0]), int(parts[1])
    return 0, 0

def safe_float(value, default=0.0):
    """Safely convert to float"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value, default=0):
    """Safely convert to int"""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

# Alabama batting stats (from first snapshot)
batting_stats = [
    ["9", "Fowler, Bryce", ".500", "1.192", "3 - 3", "8", "5", "4", "0", "0", "0", "0", "4", ".500", "5", "0", "2", "0", ".692", "0", "0", "1 - 1"],
    ["10", "Neal, Brady", ".500", "1.000", "3 - 1", "4", "0", "2", "0", "0", "0", "4", "2", ".500", "0", "1", "0", "0", ".500", "1", "0", "0 - 0"],
    ["32", "Torres, Jason", ".400", ".938", "3 - 3", "10", "1", "4", "0", "0", "0", "4", "4", ".400", "3", "0", "3", "0", ".538", "0", "0", "0 - 0"],
    ["1", "Lebron, Justin", ".273", "1.247", "3 - 3", "11", "6", "3", "0", "0", "2", "3", "9", ".818", "3", "0", "0", "1", ".429", "0", "0", "3 - 3"],
    ["33", "Lemm, John", ".250", ".712", "3 - 3", "8", "2", "2", "0", "0", "0", "2", "2", ".250", "3", "1", "0", "0", ".462", "1", "0", "0 - 0"],
    ["4", "Holt, Brennan", ".222", ".829", "3 - 3", "9", "3", "2", "0", "0", "1", "2", "5", ".556", "1", "0", "3", "0", ".273", "1", "0", "0 - 0"],
    ["14", "Steele, Peyton", ".182", ".546", "3 - 3", "11", "0", "2", "2", "0", "0", "2", "4", ".364", "0", "0", "3", "0", ".182", "0", "0", "0 - 0"],
    ["3", "Plattner, Will", ".167", ".453", "3 - 2", "6", "0", "1", "0", "0", "0", "0", "1", ".167", "1", "0", "1", "0", ".286", "0", "0", "0 - 0"],
    ["18", "Osterhouse, Justin", ".000", ".273", "3 - 3", "8", "3", "0", "0", "0", "0", "0", "0", ".000", "3", "0", "5", "0", ".273", "0", "0", "0 - 0"],
    ["21", "* Purdy, Andrew", ".250", ".750", "2 - 1", "4", "2", "1", "0", "0", "0", "3", "1", ".250", "1", "1", "2", "1", ".500", "0", "0", "0 - 0"],
    ["13", "* Vaughn, Luke", ".167", ".953", "2 - 2", "6", "1", "1", "0", "0", "1", "2", "4", ".667", "0", "1", "4", "0", ".286", "0", "0", "0 - 0"],
    ["42", "* Hines, Eric", ".000", ".000", "1 - 0", "2", "0", "0", "0", "0", "0", "0", "0", ".000", "0", "0", "1", "0", ".000", "0", "0", "0 - 0"],
    ["6", "* Taylor, Evan", ".000", ".000", "1 - 0", "1", "0", "0", "0", "0", "0", "0", "0", ".000", "0", "0", "0", "0", ".000", "0", "0", "0 - 0"],
    ["19", "* Barnett, Caleb", ".000", "1.000", "1 - 0", "0", "0", "0", "0", "0", "0", "0", "0", ".000", "1", "0", "0", "0", "1.000", "0", "0", "0 - 0"]
]

# Alabama pitching stats (from second snapshot)  
pitching_stats = [
    ["11", "Upchurch, Myles", "0.00", "1.00", "1 - 0", "1 - 1", "0", "0 - 0", "0", "4.0", "2", "0", "0", "2", "9", "0", "0", "0", "13", ".154", "1", "0", "0", "0", "0"],
    ["20", "Adams, Zane", "0.00", ".60", "1 - 0", "1 - 1", "0", "0 - 0", "0", "5.0", "3", "0", "0", "0", "9", "1", "0", "0", "19", ".158", "0", "0", "0", "0", "0"],
    ["7", "Heiberger, Matthew", "3.00", "1.67", "0 - 0", "1 - 0", "0", "0 - 0", "0", "3.0", "3", "1", "1", "2", "3", "0", "0", "0", "11", ".273", "0", "0", "0", "0", "1"],
    ["8", "Fay, Tyler", "12.27", "1.64", "0 - 1", "1 - 1", "0", "0 - 0", "0", "3.2", "5", "5", "5", "1", "7", "0", "0", "1", "15", ".333", "0", "1", "1", "0", "0"],
    ["24", "* Robertson, Tate", "0.00", "1.00", "0 - 0", "1 - 0", "0", "0 - 0", "0", "1.0", "0", "0", "0", "1", "2", "0", "0", "0", "3", ".000", "0", "0", "0", "0", "0"],
    ["27", "* Smyers, Luke", "0.00", "3.00", "0 - 0", "1 - 0", "0", "0 - 0", "0", "0.1", "0", "0", "0", "1", "1", "0", "0", "0", "1", ".000", "0", "0", "0", "0", "0"],
    ["28", "* Mitchell, Sam", "0.00", ".50", "0 - 0", "1 - 0", "0", "0 - 0", "0", "2.0", "1", "0", "0", "0", "2", "1", "0", "0", "7", ".143", "0", "0", "0", "0", "0"],
    ["44", "* Steckmesser, Evan", "0.00", ".00", "0 - 0", "1 - 0", "0", "0 - 0", "0", "1.0", "0", "0", "0", "0", "2", "0", "0", "0", "3", ".000", "0", "0", "0", "0", "0"],
    ["45", "* Chiarodo, Joe", "0.00", "1.50", "0 - 0", "1 - 0", "0", "0 - 0", "0", "0.2", "1", "0", "0", "0", "0", "0", "0", "0", "3", ".333", "0", "0", "0", "0", "0"],
    ["99", "* Morris, Austin", "4.50", "1.50", "0 - 0", "1 - 0", "0", "0 - 0", "0", "2.0", "1", "1", "1", "2", "2", "0", "0", "0", "6", ".167", "0", "0", "0", "0", "0"],
    ["12", "* Blackwood, JT", "6.75", "2.25", "0 - 0", "1 - 0", "0", "0 - 0", "0", "1.1", "3", "2", "1", "0", "1", "1", "0", "1", "8", ".375", "0", "0", "0", "0", "0"],
    ["48", "* Crowther, Ashton", "9.00", "2.00", "0 - 0", "1 - 0", "0", "0 - 0", "0", "1.0", "2", "1", "1", "0", "0", "0", "0", "0", "5", ".400", "1", "0", "0", "0", "0"]
]

def insert_alabama_stats():
    conn = sqlite3.connect('/home/sam/college-baseball-predictor/data/baseball.db')
    cursor = conn.cursor()
    
    # Clear existing Alabama data
    cursor.execute("DELETE FROM player_stats WHERE team_id = 'alabama'")
    
    batting_count = 0
    pitching_count = 0
    
    # Insert batting stats
    for row in batting_stats:
        number, name, avg, ops, gp_gs, ab, r, h, doubles, triples, hr, rbi, tb, slg, bb, hbp, so, gdp, obp, sf, sh, sb_att = row
        
        # Parse compound fields
        games, games_started = parse_gp_gs(gp_gs)
        stolen_bases, caught_stealing = parse_sb_att(sb_att)
        
        # Calculate SLG from TB and AB
        at_bats = safe_int(ab)
        total_bases = safe_int(tb)
        slg_calc = total_bases / at_bats if at_bats > 0 else 0.0
        
        cursor.execute("""
            INSERT OR REPLACE INTO player_stats (
                team_id, name, number, games, at_bats, runs, hits, doubles, triples, 
                home_runs, rbi, walks, strikeouts, stolen_bases, caught_stealing,
                batting_avg, obp, slg, ops, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            'alabama', clean_name(name), safe_int(number), games, safe_int(ab),
            safe_int(r), safe_int(h), safe_int(doubles), safe_int(triples),
            safe_int(hr), safe_int(rbi), safe_int(bb), safe_int(so),
            stolen_bases, caught_stealing, safe_float(avg), safe_float(obp),
            safe_float(slg), safe_float(ops), datetime.now().isoformat()
        ))
        batting_count += 1
    
    # Insert/Update pitching stats
    for row in pitching_stats:
        number, name, era, whip, w_l, app_gs, cg, sho_sv, sv, ip, h_allowed, r_allowed, er, bb_allowed, so_pitched, doubles_allowed, triples_allowed, hr_allowed, ab_against, b_avg, wp, hbp_allowed, bk, sfa, sha = row
        
        # Parse compound fields
        wins, losses = parse_w_l(w_l)
        appearances, games_started_pitch = parse_gp_gs(app_gs)
        
        # Clean name for matching
        clean_player_name = clean_name(name)
        
        # Try to update existing batting record, or insert new pitching-only record
        cursor.execute("""
            INSERT OR REPLACE INTO player_stats (
                team_id, name, number, wins, losses, era, games_pitched, games_started,
                saves, innings_pitched, hits_allowed, runs_allowed, earned_runs,
                walks_allowed, strikeouts_pitched, whip, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(team_id, name) DO UPDATE SET
                wins = excluded.wins,
                losses = excluded.losses,
                era = excluded.era,
                games_pitched = excluded.games_pitched,
                games_started = excluded.games_started,
                saves = excluded.saves,
                innings_pitched = excluded.innings_pitched,
                hits_allowed = excluded.hits_allowed,
                runs_allowed = excluded.runs_allowed,
                earned_runs = excluded.earned_runs,
                walks_allowed = excluded.walks_allowed,
                strikeouts_pitched = excluded.strikeouts_pitched,
                whip = excluded.whip,
                updated_at = excluded.updated_at
        """, (
            'alabama', clean_player_name, safe_int(number), wins, losses,
            safe_float(era), appearances, games_started_pitch, safe_int(sv),
            safe_float(ip), safe_int(h_allowed), safe_int(r_allowed),
            safe_int(er), safe_int(bb_allowed), safe_int(so_pitched),
            safe_float(whip), datetime.now().isoformat()
        ))
        pitching_count += 1
    
    conn.commit()
    conn.close()
    
    print(f"Alabama stats inserted:")
    print(f"- Batting records: {batting_count}")
    print(f"- Pitching records: {pitching_count}")
    return batting_count, pitching_count

if __name__ == "__main__":
    insert_alabama_stats()