#!/usr/bin/env python3
"""Backup the baseball database with timestamped copies and rotation."""
import shutil
import os
import sys
import glob
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'scripts'))
from run_utils import ScriptRunner

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'baseball.db')
BACKUP_DIR = os.path.join(os.path.dirname(__file__), '..', 'backups')
MAX_BACKUPS = 14  # Keep 2 weeks of daily backups

def backup():
    runner = ScriptRunner("backup_db")
    
    os.makedirs(BACKUP_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M')
    backup_path = os.path.join(BACKUP_DIR, f'baseball_{timestamp}.db')
    
    # Use sqlite3 backup API for consistency (no partial writes)
    import sqlite3
    try:
        src = sqlite3.connect(DB_PATH)
        dst = sqlite3.connect(backup_path)
        src.backup(dst)
        dst.close()
        src.close()
    except Exception as e:
        runner.error(f"Backup failed: {e}")
        runner.finish()
    
    size_mb = os.path.getsize(backup_path) / (1024 * 1024)
    runner.info(f"Backup created: {backup_path}")
    
    runner.add_stat("backup_path", os.path.basename(backup_path))
    runner.add_stat("size_mb", f"{size_mb:.1f}")
    
    # Rotate old backups
    backups = sorted(glob.glob(os.path.join(BACKUP_DIR, 'baseball_*.db')))
    rotated = 0
    if len(backups) > MAX_BACKUPS:
        for old in backups[:-MAX_BACKUPS]:
            os.remove(old)
            rotated += 1
            runner.info(f"Rotated out: {os.path.basename(old)}")
    
    backups_on_disk = len(glob.glob(os.path.join(BACKUP_DIR, 'baseball_*.db')))
    runner.add_stat("backups_on_disk", backups_on_disk)
    
    runner.finish()

if __name__ == '__main__':
    backup()
