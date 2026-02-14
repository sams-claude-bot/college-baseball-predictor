#!/usr/bin/env python3
"""
Setup 3 AM verification cron job
"""

import os
import subprocess
from datetime import datetime

def setup_verification_cron():
    """Set up the 3 AM verification cron job"""
    
    # Get the full path to our verification script
    base_dir = os.path.abspath(os.path.dirname(__file__))
    verification_script = os.path.join(base_dir, 'verification_job.py')
    python_path = '/usr/bin/python3'
    
    # Cron job command - 3 AM CST every day
    cron_line = f"0 3 * * * cd {base_dir} && {python_path} {verification_script} >> data/logs/verification_cron.log 2>&1"
    
    print("=== SETTING UP VERIFICATION CRON JOB ===")
    print(f"Script: {verification_script}")
    print(f"Schedule: Daily at 3:00 AM CST")
    print(f"Cron line: {cron_line}")
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(base_dir, 'data', 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    try:
        # Get current crontab
        try:
            current_cron = subprocess.check_output(['crontab', '-l'], 
                                                 stderr=subprocess.DEVNULL).decode('utf-8')
        except subprocess.CalledProcessError:
            current_cron = ""
        
        # Check if our job already exists
        verification_jobs = [line for line in current_cron.split('\n') 
                           if 'verification_job.py' in line]
        
        if verification_jobs:
            print("Verification cron job already exists:")
            for job in verification_jobs:
                print(f"  {job}")
            print("Skipping setup.")
            return True
        
        # Add our cron job
        new_cron = current_cron.strip() + '\n' + cron_line + '\n'
        
        # Write to temporary file
        temp_cron_file = '/tmp/new_crontab'
        with open(temp_cron_file, 'w') as f:
            f.write(new_cron)
        
        # Install new crontab
        subprocess.run(['crontab', temp_cron_file], check=True)
        
        # Clean up temp file
        os.remove(temp_cron_file)
        
        print("✅ Verification cron job installed successfully!")
        
        # Verify installation
        verification_cron = subprocess.check_output(['crontab', '-l']).decode('utf-8')
        if 'verification_job.py' in verification_cron:
            print("✅ Cron job verified in crontab")
            return True
        else:
            print("❌ Cron job not found after installation")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error setting up cron job: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def create_verification_test():
    """Create a test script to verify the cron job works"""
    
    base_dir = os.path.abspath(os.path.dirname(__file__))
    test_script = os.path.join(base_dir, 'test_verification_job.py')
    
    test_content = f'''#!/usr/bin/env python3
"""
Test the verification job manually
"""

import sys
import os
sys.path.insert(0, "{base_dir}")

from verification_job import GameVerificationJob

if __name__ == "__main__":
    print("=== TESTING VERIFICATION JOB ===")
    
    job = GameVerificationJob()
    results = job.generate_report()
    
    print(f"Test completed successfully!")
    print(f"Games verified: {{results['games_verified']}}")
    print(f"Discrepancies: {{len(results['discrepancies'])}}")
    print(f"Errors: {{len(results['errors'])}}")
'''
    
    with open(test_script, 'w') as f:
        f.write(test_content)
    
    os.chmod(test_script, 0o755)
    
    print(f"Created test script: {test_script}")
    print("Run with: python3 test_verification_job.py")

def setup_one_time_delete():
    """Setup the deleteAfterRun functionality"""
    
    base_dir = os.path.abspath(os.path.dirname(__file__))
    delete_script = os.path.join(base_dir, 'delete_verification_cron.py')
    
    delete_content = f'''#!/usr/bin/env python3
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
        
        for line in current_cron.split('\\n'):
            if 'verification_job.py' not in line:
                new_lines.append(line)
            else:
                removed_count += 1
                print(f"Removing: {{line}}")
        
        if removed_count == 0:
            print("No verification cron jobs found to remove")
            return
        
        # Write new crontab
        new_cron = '\\n'.join(new_lines)
        temp_file = '/tmp/cleaned_crontab'
        
        with open(temp_file, 'w') as f:
            f.write(new_cron)
        
        subprocess.run(['crontab', temp_file], check=True)
        os.remove(temp_file)
        
        print(f"✅ Removed {{removed_count}} verification cron job(s)")
        
        # Log the removal
        log_entry = {{
            'timestamp': datetime.now().isoformat(),
            'action': 'removed_verification_cron',
            'jobs_removed': removed_count
        }}
        
        import json
        with open('{base_dir}/data/cron_removal_log.json', 'w') as f:
            json.dump(log_entry, f, indent=2)
        
    except Exception as e:
        print(f"Error removing cron job: {{e}}")

if __name__ == "__main__":
    remove_verification_cron()
    print("Verification cron job removal complete")
'''
    
    with open(delete_script, 'w') as f:
        f.write(delete_content)
    
    os.chmod(delete_script, 0o755)
    
    print(f"Created deletion script: {delete_script}")
    print("Use this to remove the cron job when no longer needed")

if __name__ == "__main__":
    # Setup the cron job
    success = setup_verification_cron()
    
    # Create helper scripts
    create_verification_test()
    setup_one_time_delete()
    
    if success:
        print("\n=== CRON SETUP COMPLETE ===")
        print("✅ 3 AM verification cron job installed")
        print("✅ Test script created")
        print("✅ Deletion script created for later cleanup")
        print("\nThe cron job will:")
        print("- Run daily at 3:00 AM CST")
        print("- Verify today's game results against alternate sources")
        print("- Log results to data/verification_results.json")
        print("- Report discrepancies for manual review")
    else:
        print("\n❌ CRON SETUP FAILED")
        print("Manual setup may be required")