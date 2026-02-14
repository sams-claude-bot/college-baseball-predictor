#!/usr/bin/env python3
"""
One-time script to remove the verification cron job
Set to run after the season or when no longer needed
"""

import subprocess
import os
from datetime import datetime

def remove_verification_cron():
    try:
        # Get current crontab
        current_cron = subprocess.check_output(['crontab', '-l'], 
                                             stderr=subprocess.DEVNULL).decode('utf-8')
        
        # Remove lines containing verification_job.py
        new_lines = []
        removed_count = 0
        
        for line in current_cron.split('\n'):
            if 'verification_job.py' not in line:
                new_lines.append(line)
            else:
                removed_count += 1
                print(f"Removing: {line}")
        
        if removed_count == 0:
            print("No verification cron jobs found to remove")
            return
        
        # Write new crontab
        new_cron = '\n'.join(new_lines)
        temp_file = '/tmp/cleaned_crontab'
        
        with open(temp_file, 'w') as f:
            f.write(new_cron)
        
        subprocess.run(['crontab', temp_file], check=True)
        os.remove(temp_file)
        
        print(f"✅ Removed {removed_count} verification cron job(s)")
        
        # Log the removal
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': 'removed_verification_cron',
            'jobs_removed': removed_count
        }
        
        import json
        with open('/home/sam/college-baseball-predictor/data/cron_removal_log.json', 'w') as f:
            json.dump(log_entry, f, indent=2)
        
    except Exception as e:
        print(f"Error removing cron job: {e}")

if __name__ == "__main__":
    remove_verification_cron()
    print("Verification cron job removal complete")
