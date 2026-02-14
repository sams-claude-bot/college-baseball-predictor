#!/usr/bin/env python3

import os
import re
import ast
from datetime import datetime
import sqlite3

def analyze_script_imports(file_path):
    """Analyze imports and dependencies in a Python script"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse imports
        imports = []
        local_imports = []
        
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                imports.append(line)
                if ('from .' in line or 
                    'import database' in line or 
                    'import models.' in line or 
                    'import scripts.' in line):
                    local_imports.append(line)
        
        # Look for PYTHONPATH workarounds
        pythonpath_hacks = []
        for line in content.split('\n'):
            if 'sys.path' in line or 'PYTHONPATH' in line:
                pythonpath_hacks.append(line.strip())
        
        # Look for hardcoded paths
        hardcoded_paths = []
        path_patterns = [
            r'/home/[^/]+',
            r'C:\\[^\\]+',
            r'/Users/[^/]+',
            r'/tmp/',
            r'/var/'
        ]
        
        for pattern in path_patterns:
            matches = re.findall(pattern, content)
            hardcoded_paths.extend(matches)
        
        # Get file stats
        stat = os.stat(file_path)
        
        return {
            'file': file_path,
            'size': stat.st_size,
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'imports': imports,
            'local_imports': local_imports,
            'pythonpath_hacks': pythonpath_hacks,
            'hardcoded_paths': list(set(hardcoded_paths)),
            'line_count': len(content.split('\n'))
        }
        
    except Exception as e:
        return {
            'file': file_path,
            'error': str(e)
        }

def find_unused_scripts():
    """Try to identify unused or deprecated scripts"""
    scripts_dir = 'scripts'
    models_dir = 'models'
    
    all_files = []
    
    # Get all Python files
    for root, dirs, files in os.walk(scripts_dir):
        for file in files:
            if file.endswith('.py'):
                all_files.append(os.path.join(root, file))
    
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if file.endswith('.py'):
                all_files.append(os.path.join(root, file))
    
    # Analyze each file
    analysis = {}
    for file_path in all_files:
        analysis[file_path] = analyze_script_imports(file_path)
    
    return analysis

def check_database_dependencies():
    """Check which scripts actually interact with the database"""
    conn = sqlite3.connect('data/baseball.db')
    cursor = conn.cursor()
    
    # Get table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    
    return tables

def identify_cleanup_candidates():
    """Identify scripts that might be deprecated or need cleanup"""
    analysis = find_unused_scripts()
    db_tables = check_database_dependencies()
    
    cleanup_candidates = {
        'migration_scripts': [],
        'deprecated_loaders': [],
        'outdated_configs': [],
        'pythonpath_issues': [],
        'hardcoded_paths': [],
        'large_scripts': [],
        'recent_scripts': [],
        'old_scripts': []
    }
    
    for file_path, data in analysis.items():
        if 'error' in data:
            continue
            
        filename = os.path.basename(file_path)
        
        # Migration scripts (usually one-time use)
        if 'migrate' in filename.lower():
            cleanup_candidates['migration_scripts'].append(file_path)
        
        # Loader scripts that might be outdated
        if filename.startswith('load_') and 'schedule' not in filename:
            cleanup_candidates['deprecated_loaders'].append(file_path)
        
        # Scripts with PYTHONPATH hacks
        if data.get('pythonpath_hacks'):
            cleanup_candidates['pythonpath_issues'].append({
                'file': file_path,
                'issues': data['pythonpath_hacks']
            })
        
        # Scripts with hardcoded paths
        if data.get('hardcoded_paths'):
            cleanup_candidates['hardcoded_paths'].append({
                'file': file_path,
                'paths': data['hardcoded_paths']
            })
        
        # Very large scripts (potential for refactoring)
        if data.get('line_count', 0) > 1000:
            cleanup_candidates['large_scripts'].append({
                'file': file_path,
                'lines': data['line_count']
            })
        
        # Very old files
        try:
            mod_time = datetime.fromisoformat(data.get('modified', ''))
            if mod_time < datetime(2026, 2, 1):
                cleanup_candidates['old_scripts'].append({
                    'file': file_path,
                    'modified': data['modified']
                })
        except:
            pass
        
        # Recently modified files (might be active)
        try:
            mod_time = datetime.fromisoformat(data.get('modified', ''))
            if mod_time > datetime(2026, 2, 10):
                cleanup_candidates['recent_scripts'].append({
                    'file': file_path,
                    'modified': data['modified']
                })
        except:
            pass
    
    return cleanup_candidates, analysis

if __name__ == "__main__":
    print("=== CODEBASE AUDIT ===")
    
    candidates, full_analysis = identify_cleanup_candidates()
    
    print(f"\nAnalyzed {len(full_analysis)} Python files")
    
    for category, items in candidates.items():
        if items:
            print(f"\n=== {category.upper().replace('_', ' ')} ===")
            for item in items:
                if isinstance(item, dict):
                    print(f"  {item['file']}")
                    if 'issues' in item:
                        for issue in item['issues']:
                            print(f"    Issue: {issue}")
                    if 'paths' in item:
                        for path in item['paths']:
                            print(f"    Path: {path}")
                    if 'lines' in item:
                        print(f"    Lines: {item['lines']}")
                    if 'modified' in item:
                        print(f"    Modified: {item['modified']}")
                else:
                    print(f"  {item}")
    
    # Check daily_collection.py specifically
    print(f"\n=== DAILY_COLLECTION.PY ANALYSIS ===")
    daily_script = 'scripts/daily_collection.py'
    if daily_script in full_analysis:
        data = full_analysis[daily_script]
        print(f"File size: {data.get('size', 0)} bytes")
        print(f"Line count: {data.get('line_count', 0)}")
        print(f"Last modified: {data.get('modified', 'unknown')}")
        print("Imports:")
        for imp in data.get('imports', []):
            print(f"  {imp}")
        
        if data.get('pythonpath_hacks'):
            print("PYTHONPATH issues found:")
            for hack in data['pythonpath_hacks']:
                print(f"  {hack}")
    
    # Save detailed analysis
    import json
    with open('data/codebase_audit.json', 'w') as f:
        json.dump({
            'audit_time': datetime.now().isoformat(),
            'cleanup_candidates': candidates,
            'full_analysis': full_analysis
        }, f, indent=2)
    
    print(f"\nDetailed analysis saved to data/codebase_audit.json")