#!/usr/bin/env python3
"""
Generic PDF Stats Scraper Base Class

Base class for scraping college baseball stats from PDF files.
Other PDF-based scrapers (Georgia Tech, Arkansas, etc.) can inherit from this.

Usage:
    class MyTeamScraper(PDFStatsScraper):
        def parse_batting_line(self, line):
            # Override to parse team-specific batting format
            pass
        
        def parse_pitching_line(self, line):
            # Override to parse team-specific pitching format
            pass
"""

import os
import re
import sys
import subprocess
import tempfile
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from scripts.stats_collector import (
    get_db, calculate_derived_stats, upsert_player_stats, 
    normalize_player_name, PITCHING_DEFAULTS
)

log = logging.getLogger(__name__)

class PDFStatsScraper:
    """Base class for PDF-based team stats scrapers."""
    
    def __init__(self, team_id, pdf_url):
        self.team_id = team_id
        self.pdf_url = pdf_url
    
    def download_pdf(self, output_path=None):
        """Download PDF to temporary file or specified path."""
        if output_path is None:
            output_path = f"/tmp/{self.team_id}_stats.pdf"
        
        cmd = ['curl', '-s', '-L', self.pdf_url, '-o', output_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Failed to download PDF: {result.stderr}")
        
        # Verify file was downloaded and has content
        if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
            raise Exception(f"Downloaded PDF is empty or invalid")
        
        return output_path
    
    def extract_text(self, pdf_path):
        """Extract text from PDF using pdftotext with layout preserved."""
        cmd = ['pdftotext', '-layout', pdf_path, '-']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Failed to extract text: {result.stderr}")
        
        return result.stdout
    
    def find_batting_section(self, text):
        """Find the batting stats section in the text. Override if needed."""
        lines = text.split('\n')
        batting_lines = []
        in_batting = False
        
        for line in lines:
            # Look for batting header
            if 'Player' in line and 'avg' in line and ('ab' in line or 'gp' in line):
                in_batting = True
                continue
            
            # Stop at pitching section or other indicators
            if in_batting and any(phrase in line.lower() for phrase in 
                                ['earned run avg', 'era', 'pitching', '----']):
                if 'era' in line and 'w-l' in line:  # Pitching header
                    break
                if line.strip() == '--------------------':  # Common separator
                    continue
            
            if in_batting and line.strip():
                batting_lines.append(line)
        
        return batting_lines
    
    def find_pitching_section(self, text):
        """Find the pitching stats section in the text. Override if needed."""
        lines = text.split('\n')
        pitching_lines = []
        in_pitching = False
        
        for line in lines:
            # Look for pitching header
            if 'Player' in line and 'era' in line and ('w-l' in line or 'ip' in line):
                in_pitching = True
                continue
            
            # Stop at end markers
            if in_pitching and any(phrase in line.lower() for phrase in 
                                ['totals', 'opponents', 'lob -', 'dps turned']):
                if line.lower().strip().startswith(('totals', 'opponents')):
                    break
            
            if in_pitching and line.strip():
                pitching_lines.append(line)
        
        return pitching_lines
    
    def _safe_int(self, val, default=0):
        """Safely convert to int, handling dashes and empty strings."""
        if val is None or val == '' or val == '—' or val == '-':
            return default
        try:
            return int(float(str(val).replace('%', '').strip()))
        except (ValueError, TypeError):
            return default
    
    def _safe_float(self, val, default=0.0):
        """Safely convert to float."""
        if val is None or val == '' or val == '—' or val == '-':
            return default
        try:
            return float(str(val).replace('%', '').strip())
        except (ValueError, TypeError):
            return default
    
    def parse_sb_att(self, sb_att_str):
        """Parse stolen base attempts string like '2-2' into (stolen_bases, caught_stealing)."""
        if not sb_att_str or sb_att_str.strip() in ('', '-', '—', '0-0'):
            return 0, 0
        
        try:
            if '-' in sb_att_str:
                parts = sb_att_str.split('-')
                stolen = self._safe_int(parts[0])
                attempts = self._safe_int(parts[1])
                caught = max(0, attempts - stolen)  # caught = attempts - successful
                return stolen, caught
            else:
                # Just stolen bases, no attempts
                return self._safe_int(sb_att_str), 0
        except:
            return 0, 0
    
    def parse_w_l(self, w_l_str):
        """Parse win-loss string like '1-0' into (wins, losses)."""
        if not w_l_str or w_l_str.strip() in ('', '-', '—'):
            return 0, 0
        
        try:
            if '-' in w_l_str:
                parts = w_l_str.split('-')
                return self._safe_int(parts[0]), self._safe_int(parts[1])
            else:
                return self._safe_int(w_l_str), 0
        except:
            return 0, 0
    
    def parse_gp_gs(self, gp_gs_str):
        """Parse games played/started string like '4-4' into (games, games_started)."""
        if not gp_gs_str or gp_gs_str.strip() in ('', '-', '—'):
            return 0, 0
        
        try:
            if '-' in gp_gs_str:
                parts = gp_gs_str.split('-')
                return self._safe_int(parts[0]), self._safe_int(parts[1])
            else:
                return self._safe_int(gp_gs_str), 0
        except:
            return 0, 0
    
    def parse_batting_line(self, line):
        """
        Parse a batting stats line. MUST be overridden by subclasses.
        Should return a dict with player stats or None if invalid.
        """
        raise NotImplementedError("Subclasses must implement parse_batting_line")
    
    def parse_pitching_line(self, line):
        """
        Parse a pitching stats line. MUST be overridden by subclasses.
        Should return a dict with player stats or None if invalid.
        """
        raise NotImplementedError("Subclasses must implement parse_pitching_line")
    
    def scrape_stats(self, dry_run=False):
        """Main scraping method. Downloads PDF, extracts text, and parses stats."""
        log.info(f"Scraping {self.team_id} stats from PDF...")
        
        # Download and extract
        pdf_path = None
        try:
            pdf_path = self.download_pdf()
            text = self.extract_text(pdf_path)
            
            # Parse sections
            batting_lines = self.find_batting_section(text)
            pitching_lines = self.find_pitching_section(text)
            
            log.info(f"Found {len(batting_lines)} batting lines, {len(pitching_lines)} pitching lines")
            
            # Parse batting players
            batting_players = []
            for line in batting_lines:
                player = self.parse_batting_line(line)
                if player:
                    batting_players.append(player)
            
            # Parse pitching players
            pitching_players = []
            for line in pitching_lines:
                player = self.parse_pitching_line(line)
                if player:
                    pitching_players.append(player)
            
            log.info(f"Parsed {len(batting_players)} batters, {len(pitching_players)} pitchers")
            
            if not batting_players:
                log.error("No batting stats found - check parsing logic")
                return {'status': 'failed', 'batting': 0, 'pitching': 0}
            
            # Save to database
            if not dry_run:
                self.save_to_database(batting_players, pitching_players)
            
            return {
                'status': 'ok',
                'batting': len(batting_players),
                'pitching': len(pitching_players),
                'source': 'pdf'
            }
            
        finally:
            # Cleanup
            if pdf_path and os.path.exists(pdf_path):
                os.unlink(pdf_path)
    
    def save_to_database(self, batting_players, pitching_players):
        """Save parsed player stats to database using the same pattern as stats_collector.py."""
        db = get_db()
        
        try:
            # Create pitching lookup
            pitching_by_name = {p['name']: p for p in pitching_players}
            
            batting_count = 0
            for bp in batting_players:
                # Add pitching defaults
                for k, v in PITCHING_DEFAULTS.items():
                    bp.setdefault(k, v)
                
                # Merge pitching stats if this player also pitches
                if bp['name'] in pitching_by_name:
                    ps = pitching_by_name.pop(bp['name'])
                    for k in PITCHING_DEFAULTS:
                        bp[k] = ps.get(k, 0)
                
                # Calculate derived stats
                calculate_derived_stats(bp)
                
                # Insert/update in database
                upsert_player_stats(db, bp)
                batting_count += 1
            
            # Handle pitchers who didn't bat
            pitching_only = 0
            for name, ps in pitching_by_name.items():
                player = {
                    'team_id': self.team_id,
                    'name': name,
                    'number': ps.get('number', 0),
                    'games': 0, 'at_bats': 0, 'runs': 0, 'hits': 0, 'doubles': 0,
                    'triples': 0, 'home_runs': 0, 'rbi': 0, 'walks': 0, 'strikeouts': 0,
                    'stolen_bases': 0, 'caught_stealing': 0,
                }
                for k in PITCHING_DEFAULTS:
                    player[k] = ps.get(k, 0)
                
                calculate_derived_stats(player)
                upsert_player_stats(db, player)
                pitching_only += 1
            
            db.commit()
            log.info(f"Saved {batting_count} batters, {len(pitching_players)} pitchers ({pitching_only} pitch-only)")
            
        finally:
            db.close()