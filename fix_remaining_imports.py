#!/usr/bin/env python3

import os
import re

def fix_imports_in_file(file_path):
    """Fix remaining import issues in a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix model imports within models directory
        model_import_fixes = [
            ('from base_model import', 'from models.base_model import'),
            ('from elo_model import', 'from models.elo_model import'),
            ('from log5_model import', 'from models.log5_model import'),
            ('from pythagorean_model import', 'from models.pythagorean_model import'),
            ('from advanced_model import', 'from models.advanced_model import'),
            ('from ensemble_model import', 'from models.ensemble_model import'),
            ('from pitching_model import', 'from models.pitching_model import'),
            ('from conference_model import', 'from models.conference_model import'),
            ('from prior_model import', 'from models.prior_model import'),
        ]
        
        # Script imports within scripts directory  
        script_import_fixes = [
            ('from database import', 'from scripts.database import'),
            ('from collect_box_scores import', 'from scripts.collect_box_scores import'),
            ('from collect_all_stats import', 'from scripts.collect_all_stats import'),
            ('from betting_lines import', 'from scripts.betting_lines import'),
            ('from player_stats import', 'from scripts.player_stats import'),
        ]
        
        # Apply fixes based on file location
        if file_path.startswith('models/'):
            for old_import, new_import in model_import_fixes:
                if old_import in content and new_import not in content:
                    content = content.replace(old_import, new_import)
            
            # Also fix script imports from models
            for old_import, new_import in script_import_fixes:
                if old_import in content and new_import not in content:
                    content = content.replace(old_import, new_import)
        
        elif file_path.startswith('scripts/'):
            for old_import, new_import in script_import_fixes:
                if old_import in content and new_import not in content:
                    content = content.replace(old_import, new_import)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed imports in {file_path}")
            return True
        else:
            print(f"No changes needed in {file_path}")
            return False
    
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

def fix_all_remaining_imports():
    """Fix all remaining import issues"""
    files_to_fix = []
    
    # Get all Python files in models and scripts
    for root, dirs, files in os.walk('models'):
        for file in files:
            if file.endswith('.py'):
                files_to_fix.append(os.path.join(root, file))
    
    for root, dirs, files in os.walk('scripts'):
        for file in files:
            if file.endswith('.py'):
                files_to_fix.append(os.path.join(root, file))
    
    print(f"Fixing imports in {len(files_to_fix)} files...")
    
    fixed_count = 0
    for file_path in files_to_fix:
        if fix_imports_in_file(file_path):
            fixed_count += 1
    
    print(f"Fixed imports in {fixed_count} files")
    return fixed_count

if __name__ == "__main__":
    fix_all_remaining_imports()