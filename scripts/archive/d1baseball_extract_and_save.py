#!/usr/bin/env python3
"""
Quick helper: reads JSON from stdin (the window._d1 string from browser),
saves to /tmp/, and runs d1baseball_advanced.py on it.
Usage: echo '<json>' | python3 scripts/d1baseball_extract_and_save.py
"""
import json, sys, subprocess
from pathlib import Path

data = json.load(sys.stdin)
slug = data.get('team_slug', 'unknown')
path = f'/tmp/d1_{slug}_advanced.json'
with open(path, 'w') as f:
    json.dump(data, f)

result = subprocess.run(
    ['python3', 'scripts/d1baseball_advanced.py', '--file', path],
    capture_output=True, text=True, cwd=str(Path(__file__).parent.parent)
)
print(result.stdout)
if result.stderr:
    print(result.stderr, file=sys.stderr)
sys.exit(result.returncode)
