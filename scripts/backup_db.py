#!/usr/bin/env python3
"""Backup the baseball database with timestamped copies and rotation."""
import shutil
import os
import glob
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'baseball.db')
BACKUP_DIR = os.path.join(os.path.dirname(__file__), '..', 'backups')
MAX_BACKUPS = 14  # Keep 2 weeks of daily backups

def backup():
    os.makedirs(BACKUP_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M')
    backup_path = os.path.join(BACKUP_DIR, f'baseball_{timestamp}.db')
    
    # Use sqlite3 backup API for consistency (no partial writes)
    import sqlite3
    src = sqlite3.connect(DB_PATH)
    dst = sqlite3.connect(backup_path)
    src.backup(dst)
    dst.close()
    src.close()
    
    size_mb = os.path.getsize(backup_path) / (1024 * 1024)
    print(f"✅ Backup created: {backup_path} ({size_mb:.1f} MB)")
    
    # Rotate old backups
    backups = sorted(glob.glob(os.path.join(BACKUP_DIR, 'baseball_*.db')))
    if len(backups) > MAX_BACKUPS:
        for old in backups[:-MAX_BACKUPS]:
            os.remove(old)
            print(f"🗑️  Rotated out: {os.path.basename(old)}")
    
    print(f"📦 {len(glob.glob(os.path.join(BACKUP_DIR, 'baseball_*.db')))} backups on disk")

if __name__ == '__main__':
    backup()
