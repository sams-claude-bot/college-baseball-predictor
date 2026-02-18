#!/usr/bin/env python3
"""
WMT (Web Management Tool) stats scraper for teams using WordPress + WMT iframes.
Used as a fallback for teams like Miami that don't use SIDEARM Nuxt 3.

Usage:
    from wmt_scraper import scrape_wmt_stats
    batting, pitching = scrape_wmt_stats('miami-fl', 'https://wmt.games/miamihurricanes/stats/season/614661')
"""

import re
import logging

log = logging.getLogger('wmt_scraper')

# WMT iframe URLs for teams that use WordPress + WMT
# These need to be discovered by visiting the team's cumestats page and finding the iframe src
WMT_URLS = {
    'miami-fl': 'https://wmt.games/miamihurricanes/stats/season/614661',
}


def parse_wmt_batting_text(text):
    """Parse WMT batting stats from page text (tab-separated format)."""
    players = []
    lines = text.strip().split('\n')
    
    # Find the batting data section
    in_batting = False
    header_found = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Detect header line
        if 'NAME' in line and 'AVG' in line and 'AB' in line:
            header_found = True
            continue
            
        if not header_found:
            continue
            
        # Skip totals/opponents
        if line.startswith('Totals') or line.startswith('Opponents'):
            break
            
        # Parse player line: number name avg ops gp gs ab r h 2b 3b hr rbi tb slg bb hbp k gdp obp sf sh sb-att
        parts = line.split('\t')
        if len(parts) < 15:
            # Try splitting on multiple spaces
            parts = re.split(r'\s{2,}', line)
        
        if len(parts) < 15:
            continue
            
        try:
            number = parts[0].strip()
            name = parts[1].strip()
            avg = parts[2].strip()
            ops = parts[3].strip()
            gp = parts[4].strip()
            gs = parts[5].strip()
            ab = parts[6].strip()
            r = parts[7].strip()
            h = parts[8].strip()
            doubles = parts[9].strip()
            triples = parts[10].strip()
            hr = parts[11].strip()
            rbi = parts[12].strip()
            # tb = parts[13]
            slg = parts[14].strip()
            bb = parts[15].strip()
            # hbp = parts[16]
            k = parts[17].strip()
            # gdp = parts[18]
            obp = parts[19].strip()
            
            # Parse SB from "SB - ATT" format  
            sb = 0
            cs = 0
            if len(parts) > 21:
                sb_att = parts[21].strip()
                m = re.match(r'(\d+)\s*-\s*(\d+)', sb_att)
                if m:
                    sb = int(m.group(1))
                    cs = int(m.group(2)) - sb  # ATT = SB + CS
                    if cs < 0:
                        cs = 0
            
            players.append({
                'name': name,
                'number': _safe_int(number),
                'games': _safe_int(gp),
                'at_bats': _safe_int(ab),
                'runs': _safe_int(r),
                'hits': _safe_int(h),
                'doubles': _safe_int(doubles),
                'triples': _safe_int(triples),
                'home_runs': _safe_int(hr),
                'rbi': _safe_int(rbi),
                'walks': _safe_int(bb),
                'strikeouts': _safe_int(k),
                'stolen_bases': sb,
                'caught_stealing': cs,
            })
        except (ValueError, IndexError) as e:
            log.debug(f"Failed to parse batting line: {line[:80]}: {e}")
            continue
    
    return players


def parse_wmt_pitching_text(text):
    """Parse WMT pitching stats from page text."""
    players = []
    lines = text.strip().split('\n')
    
    header_found = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if 'NAME' in line and 'ERA' in line and 'IP' in line:
            header_found = True
            continue
            
        if not header_found:
            continue
            
        if line.startswith('Totals') or line.startswith('Opponents'):
            break
            
        parts = line.split('\t')
        if len(parts) < 15:
            parts = re.split(r'\s{2,}', line)
        
        if len(parts) < 15:
            continue
            
        try:
            # #  NAME  ERA  WHIP  W  L  APP  GS  CG  SHO  SV  IP  H  R  ER  BB  SO
            number = parts[0].strip()
            name = parts[1].strip()
            # era = parts[2]  # We'll calculate ourselves
            # whip = parts[3]
            w = parts[4].strip()
            l = parts[5].strip()
            app = parts[6].strip()
            gs = parts[7].strip()
            # cg = parts[8]
            # sho = parts[9]
            sv = parts[10].strip()
            ip = parts[11].strip()
            h = parts[12].strip()
            r = parts[13].strip()
            er = parts[14].strip()
            bb = parts[15].strip()
            so = parts[16].strip()
            
            players.append({
                'name': name,
                'number': _safe_int(number),
                'wins': _safe_int(w),
                'losses': _safe_int(l),
                'games_pitched': _safe_int(app),
                'games_started': _safe_int(gs),
                'saves': _safe_int(sv),
                'innings_pitched': _safe_float(ip),
                'hits_allowed': _safe_int(h),
                'runs_allowed': _safe_int(r),
                'earned_runs': _safe_int(er),
                'walks_allowed': _safe_int(bb),
                'strikeouts_pitched': _safe_int(so),
            })
        except (ValueError, IndexError) as e:
            log.debug(f"Failed to parse pitching line: {line[:80]}: {e}")
            continue
    
    return players


def _safe_int(val, default=0):
    if val is None or val == '' or val == '—':
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _safe_float(val, default=0.0):
    if val is None or val == '' or val == '—':
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default
