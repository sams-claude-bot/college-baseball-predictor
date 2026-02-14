#!/usr/bin/env python3

import os
import shutil
import re
from datetime import datetime
import json

class CodebaseCleanup:
    def __init__(self, dry_run=True):
        self.dry_run = dry_run
        self.changes = []
        self.base_dir = os.getcwd()
        
    def log_change(self, action, target, details=""):
        """Log a cleanup action"""
        change = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'target': target,
            'details': details,
            'dry_run': self.dry_run
        }
        self.changes.append(change)
        
        if self.dry_run:
            print(f"[DRY RUN] {action}: {target} - {details}")
        else:
            print(f"[APPLIED] {action}: {target} - {details}")
    
    def archive_migration_scripts(self):
        """Archive one-time migration scripts"""
        migration_scripts = [
            'scripts/migrate_data.py',
            'scripts/migrate_v2.py'
        ]
        
        archive_dir = 'scripts/archived'
        
        if not self.dry_run:
            os.makedirs(archive_dir, exist_ok=True)
        
        for script in migration_scripts:
            if os.path.exists(script):
                archive_path = os.path.join(archive_dir, os.path.basename(script))
                if not self.dry_run:
                    shutil.move(script, archive_path)
                self.log_change('ARCHIVE', script, f'moved to {archive_path}')
    
    def fix_pythonpath_issues(self):
        """Remove PYTHONPATH hacks and create proper __init__.py files"""
        
        # Create __init__.py files to make proper packages
        init_files = [
            'scripts/__init__.py',
            'models/__init__.py'
        ]
        
        for init_file in init_files:
            if not os.path.exists(init_file):
                if not self.dry_run:
                    with open(init_file, 'w') as f:
                        f.write('# Package initialization file\n')
                self.log_change('CREATE', init_file, 'package init file')
        
        # Fix PYTHONPATH hacks in files
        with open('data/codebase_audit.json', 'r') as f:
            audit = json.load(f)
        
        pythonpath_issues = audit['cleanup_candidates']['pythonpath_issues']
        
        for issue in pythonpath_issues:
            file_path = issue['file']
            if os.path.exists(file_path):
                self.fix_pythonpath_in_file(file_path)
    
    def fix_pythonpath_in_file(self, file_path):
        """Fix PYTHONPATH issues in a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Remove sys.path.insert lines
            lines = content.split('\n')
            fixed_lines = []
            
            for line in lines:
                if 'sys.path.insert' not in line:
                    fixed_lines.append(line)
                else:
                    # Comment out instead of removing entirely
                    fixed_lines.append('# ' + line + '  # Removed by cleanup')
            
            # Fix imports to use proper package structure
            fixed_content = '\n'.join(fixed_lines)
            
            # Add proper imports at the top if needed
            if 'from database import' in fixed_content and 'from scripts.database import' not in fixed_content:
                fixed_content = fixed_content.replace('from database import', 'from scripts.database import')
            
            if 'from predictor_db import' in fixed_content and 'from models.predictor_db import' not in fixed_content:
                fixed_content = fixed_content.replace('from predictor_db import', 'from models.predictor_db import')
            
            # Apply other common fixes
            common_fixes = [
                ('from ensemble_model import', 'from models.ensemble_model import'),
                ('from elo_model import', 'from models.elo_model import'),
                ('from log5_model import', 'from models.log5_model import'),
                ('from pythagorean_model import', 'from models.pythagorean_model import'),
                ('from advanced_model import', 'from models.advanced_model import'),
                ('from collect_box_scores import', 'from scripts.collect_box_scores import'),
                ('from collect_all_stats import', 'from scripts.collect_all_stats import'),
            ]
            
            for old_import, new_import in common_fixes:
                if old_import in fixed_content and new_import not in fixed_content:
                    fixed_content = fixed_content.replace(old_import, new_import)
            
            if fixed_content != original_content:
                if not self.dry_run:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(fixed_content)
                
                changes = len([l for l in lines if 'sys.path.insert' in l])
                self.log_change('FIX_PYTHONPATH', file_path, f'removed {changes} sys.path.insert lines')
        
        except Exception as e:
            self.log_change('ERROR', file_path, f'failed to fix PYTHONPATH: {e}')
    
    def fix_hardcoded_paths(self):
        """Fix hardcoded paths"""
        hardcoded_files = [
            'scripts/load_big12_schedules.py',
            'scripts/load_acc_schedules.py', 
            'scripts/load_sec_schedules.py'
        ]
        
        for file_path in hardcoded_files:
            if os.path.exists(file_path):
                self.fix_hardcoded_paths_in_file(file_path)
    
    def fix_hardcoded_paths_in_file(self, file_path):
        """Fix hardcoded paths in a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Replace common hardcoded paths
            path_fixes = [
                (r'/home/sam/', '~/'),
                (r'/home/[^/]+/', '~/'),
            ]
            
            for pattern, replacement in path_fixes:
                content = re.sub(pattern, replacement, content)
            
            # Also suggest using pathlib
            if '/home/' in original_content and 'from pathlib import Path' not in content:
                # Add pathlib import
                lines = content.split('\n')
                import_inserted = False
                for i, line in enumerate(lines):
                    if line.startswith('import ') or line.startswith('from '):
                        if not import_inserted and 'pathlib' not in line:
                            lines.insert(i, 'from pathlib import Path')
                            import_inserted = True
                            break
                content = '\n'.join(lines)
            
            if content != original_content:
                if not self.dry_run:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                
                self.log_change('FIX_HARDCODED_PATHS', file_path, 'replaced hardcoded paths')
        
        except Exception as e:
            self.log_change('ERROR', file_path, f'failed to fix paths: {e}')
    
    def consolidate_duplicate_scripts(self):
        """Identify and consolidate duplicate functionality"""
        # Check for duplicate loading scripts
        loader_scripts = [
            'scripts/load_acc_schedules.py',
            'scripts/load_big12_schedules.py', 
            'scripts/load_sec_schedules.py',
            'scripts/load_big_ten.py'
        ]
        
        self.log_change('ANALYZE', 'loader_scripts', 
                       f'Found {len(loader_scripts)} conference-specific loaders - consider consolidating')
        
        # Check for duplicate scraping scripts  
        scraper_scripts = [
            'scripts/scrape_acc_rosters.py',
            'scripts/scrape_sec_rosters.py'
        ]
        
        self.log_change('ANALYZE', 'scraper_scripts',
                       f'Found {len(scraper_scripts)} conference-specific scrapers - consider consolidating')
    
    def check_daily_collection_dependencies(self):
        """Ensure daily_collection.py works after cleanup"""
        daily_script = 'scripts/daily_collection.py'
        
        if os.path.exists(daily_script):
            try:
                with open(daily_script, 'r') as f:
                    content = f.read()
                
                # Check for potential import issues after our fixes
                imports_to_check = [
                    'from database import',
                    'from predictor_db import',
                    'from collect_box_scores import',
                    'from collect_all_stats import',
                    'from ensemble_model import'
                ]
                
                issues = []
                for imp in imports_to_check:
                    if imp in content and not imp.startswith('from scripts.') and not imp.startswith('from models.'):
                        issues.append(imp)
                
                if issues:
                    self.log_change('WARNING', daily_script, 
                                   f'May have import issues after cleanup: {issues}')
                else:
                    self.log_change('VERIFY', daily_script, 'imports look good after cleanup')
            
            except Exception as e:
                self.log_change('ERROR', daily_script, f'failed to analyze: {e}')
    
    def create_setup_py(self):
        """Create a proper setup.py for the project"""
        setup_content = '''#!/usr/bin/env python3
"""
College Baseball Predictor Package Setup
"""

from setuptools import setup, find_packages

setup(
    name="college-baseball-predictor",
    version="2026.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "requests",
        "beautifulsoup4",
        "sqlite3",  # built-in but listed for clarity
        "pandas",
        "numpy", 
        "scikit-learn",
    ],
    entry_points={
        'console_scripts': [
            'college-baseball-daily=scripts.daily_collection:main',
        ],
    },
    author="College Baseball Predictor",
    description="College baseball game prediction system",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
)
'''
        
        setup_file = 'setup.py'
        if not os.path.exists(setup_file):
            if not self.dry_run:
                with open(setup_file, 'w') as f:
                    f.write(setup_content)
            self.log_change('CREATE', setup_file, 'project setup file')
    
    def run_cleanup(self):
        """Run all cleanup operations"""
        print(f"=== CODEBASE CLEANUP {'(DRY RUN)' if self.dry_run else '(APPLYING CHANGES)'} ===")
        
        # Step 1: Archive migration scripts
        self.archive_migration_scripts()
        
        # Step 2: Fix PYTHONPATH issues
        self.fix_pythonpath_issues()
        
        # Step 3: Fix hardcoded paths  
        self.fix_hardcoded_paths()
        
        # Step 4: Analyze consolidation opportunities
        self.consolidate_duplicate_scripts()
        
        # Step 5: Check daily collection
        self.check_daily_collection_dependencies()
        
        # Step 6: Create proper setup
        self.create_setup_py()
        
        # Save cleanup log
        cleanup_log = {
            'cleanup_time': datetime.now().isoformat(),
            'dry_run': self.dry_run,
            'changes': self.changes
        }
        
        log_file = f'data/cleanup_log_{"dry_run" if self.dry_run else "applied"}.json'
        with open(log_file, 'w') as f:
            json.dump(cleanup_log, f, indent=2)
        
        print(f"\n=== CLEANUP COMPLETE ===")
        print(f"Total actions: {len(self.changes)}")
        print(f"Log saved to: {log_file}")
        
        return self.changes

if __name__ == "__main__":
    # First run dry run
    print("Running dry run first...")
    cleanup = CodebaseCleanup(dry_run=True)
    changes = cleanup.run_cleanup()
    
    print(f"\n{'='*60}")
    print("DRY RUN COMPLETE - REVIEW CHANGES ABOVE")
    print("PROCEEDING WITH ACTUAL CLEANUP...")
    print("="*60)
    
    # Now apply changes
    cleanup_real = CodebaseCleanup(dry_run=False)  
    final_changes = cleanup_real.run_cleanup()
    
    print(f"\nCleanup completed with {len(final_changes)} changes applied.")