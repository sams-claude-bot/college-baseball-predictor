#!/usr/bin/env python3
"""
Arkansas Razorbacks PDF Stats Scraper

Scrapes player stats from Arkansas' PDF stats page.
PDF URL: https://arkansasrazorbacks.com/stats/baseball/2026/overall.pdf

Usage: python3 scripts/custom_scrapers/arkansas.py
"""

import re
import sys
import logging
from pathlib import Path

# Add project root to path
PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from scripts.custom_scrapers.pdf_base import PDFStatsScraper
from scripts.stats_collector import normalize_player_name

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

class ArkansasScraper(PDFStatsScraper):
    """Arkansas Razorbacks PDF stats scraper."""
    
    def __init__(self):
        super().__init__(
            team_id='arkansas',
            pdf_url='https://arkansasrazorbacks.com/stats/baseball/2026/overall.pdf'
        )
    
    def parse_batting_line(self, line):
        """
        Parse Arkansas batting line format:
        Player                  avg gp-gs        ab       r    h     2b 3b hr          rbi    tb slg%      bb hp      so gdp     ob% sf sh sb-att            po    a           e fld%
        42 Damian Ruiz         .600     4-4      10   5        6      0    0       0     0  6       .600   6     1    2     0    .765   0    0       2-2      7        0       0 1.000
        """
        line = line.strip()
        if not line or 'Totals' in line or 'Opponents' in line:
            return None
        
        # Skip separator lines
        if line.startswith('----') or len(line) < 30:
            return None
        
        # Arkansas format: number, name, then stats separated by spaces
        # The tricky part is that names can have spaces, so we need to be careful
        
        # Look for a pattern starting with number + name
        match = re.match(r'^(\d+)\s+(.+?)\s+(\.\d+|\d+\.\d+)(\s+.+)$', line)
        if not match:
            # Try without number (some players might not have numbers)
            match = re.match(r'^([A-Za-z][A-Za-z\s]+?)\s+(\.\d+|\d+\.\d+)(\s+.+)$', line)
            if not match:
                log.debug(f"Could not parse batting line: {line}")
                return None
            number = 0
            name = match.group(1).strip()
            avg = match.group(2)
            rest = match.group(3)
        else:
            number = int(match.group(1))
            name = match.group(2).strip()
            avg = match.group(3)
            rest = match.group(4)
        
        # Parse the rest of the stats - they should be space-separated
        # Format after avg: gp-gs ab r h 2b 3b hr rbi tb slg% bb hp so gdp ob% sf sh sb-att po a e fld%
        parts = rest.strip().split()
        
        if len(parts) < 15:  # Need at least the core batting stats
            log.debug(f"Not enough stat fields in line: {line}")
            return None
        
        try:
            i = 0
            gp_gs = parts[i]; i += 1
            ab = parts[i]; i += 1
            r = parts[i]; i += 1
            h = parts[i]; i += 1
            doubles = parts[i]; i += 1
            triples = parts[i]; i += 1
            hr = parts[i]; i += 1
            rbi = parts[i]; i += 1
            tb = parts[i]; i += 1  # total bases (calculated field, skip)
            slg = parts[i]; i += 1  # (calculated field, skip)
            bb = parts[i]; i += 1
            hp = parts[i]; i += 1  # hit by pitch (not in our schema)
            so = parts[i]; i += 1
            gdp = parts[i]; i += 1  # grounded into double play (not in our schema)
            obp = parts[i]; i += 1  # (calculated field, skip)
            sf = parts[i]; i += 1  # sacrifice flies (not in our schema)
            sh = parts[i]; i += 1  # sacrifice hits (not in our schema)
            sb_att = parts[i]; i += 1
            # po, a, e, fld% are fielding stats (skip)
            
            # Parse compound fields
            games, games_started = self.parse_gp_gs(gp_gs)
            stolen_bases, caught_stealing = self.parse_sb_att(sb_att)
            
            name = normalize_player_name(name)
            
            return {
                'team_id': self.team_id,
                'name': name,
                'number': self._safe_int(number),
                'games': games,
                'at_bats': self._safe_int(ab),
                'runs': self._safe_int(r),
                'hits': self._safe_int(h),
                'doubles': self._safe_int(doubles),
                'triples': self._safe_int(triples),
                'home_runs': self._safe_int(hr),
                'rbi': self._safe_int(rbi),
                'walks': self._safe_int(bb),
                'strikeouts': self._safe_int(so),
                'stolen_bases': stolen_bases,
                'caught_stealing': caught_stealing,
            }
            
        except (IndexError, ValueError) as e:
            log.debug(f"Error parsing batting stats from line: {line} - {e}")
            return None
    
    def parse_pitching_line(self, line):
        """
        Parse Arkansas pitching line format:
        Player                   era     w-l    app gs        cg    sho    sv           ip     h       r    er       bb    so    2b     3b   hr b/avg wp hp bk sfa sha
        26 Tate McGuire         0.00    0-0       1   0       0     0/0    0           5.0     0      0     0        0      4     0     0    0       .000     0    0       0   0    0
        """
        line = line.strip()
        if not line or 'Totals' in line or 'Opponents' in line:
            return None
        
        # Skip separator lines
        if line.startswith('----') or len(line) < 30:
            return None
        
        # Parse number + name + era pattern
        match = re.match(r'^(\d+)\s+(.+?)\s+(\d+\.\d+)(\s+.+)$', line)
        if not match:
            # Try without number
            match = re.match(r'^([A-Za-z][A-Za-z\s]+?)\s+(\d+\.\d+)(\s+.+)$', line)
            if not match:
                log.debug(f"Could not parse pitching line: {line}")
                return None
            number = 0
            name = match.group(1).strip()
            era = match.group(2)
            rest = match.group(3)
        else:
            number = int(match.group(1))
            name = match.group(2).strip()
            era = match.group(3)
            rest = match.group(4)
        
        # Parse the rest: w-l app gs cg sho sv ip h r er bb so 2b 3b hr b/avg wp hp bk sfa sha
        parts = rest.strip().split()
        
        if len(parts) < 10:  # Need at least the core pitching stats
            log.debug(f"Not enough pitching stat fields in line: {line}")
            return None
        
        try:
            i = 0
            w_l = parts[i]; i += 1
            app = parts[i]; i += 1  # appearances
            gs = parts[i]; i += 1   # games started
            cg = parts[i]; i += 1   # complete games (skip)
            sho = parts[i]; i += 1  # shutouts (skip) - might be "0/0" format
            sv = parts[i]; i += 1   # saves
            ip = parts[i]; i += 1   # innings pitched
            h = parts[i]; i += 1    # hits allowed
            r = parts[i]; i += 1    # runs allowed
            er = parts[i]; i += 1   # earned runs
            bb = parts[i]; i += 1   # walks allowed
            so = parts[i]; i += 1   # strikeouts
            # Rest are advanced stats we don't need
            
            # Parse compound fields
            wins, losses = self.parse_w_l(w_l)
            
            name = normalize_player_name(name)
            
            return {
                'name': name,
                'number': self._safe_int(number),
                'wins': wins,
                'losses': losses,
                'games_pitched': self._safe_int(app),
                'games_started': self._safe_int(gs),
                'saves': self._safe_int(sv),
                'innings_pitched': self._safe_float(ip),
                'hits_allowed': self._safe_int(h),
                'runs_allowed': self._safe_int(r),
                'earned_runs': self._safe_int(er),
                'walks_allowed': self._safe_int(bb),
                'strikeouts_pitched': self._safe_int(so),
            }
            
        except (IndexError, ValueError) as e:
            log.debug(f"Error parsing pitching stats from line: {line} - {e}")
            return None


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Scrape Arkansas baseball stats from PDF')
    parser.add_argument('--dry-run', action='store_true', help='Parse but do not save to database')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    scraper = ArkansasScraper()
    
    try:
        result = scraper.scrape_stats(dry_run=args.dry_run)
        
        if result['status'] == 'ok':
            log.info(f"✓ SUCCESS: {result['batting']} batters, {result['pitching']} pitchers")
        else:
            log.error(f"✗ FAILED: {result}")
            sys.exit(1)
            
    except Exception as e:
        log.error(f"✗ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()