#!/usr/bin/env python3
"""Save scraped stat data from stdin to JSON file."""
import json
import sys

def main():
    if len(sys.argv) < 3:
        print("Usage: save_stat.py <output_file> <stat_name>")
        sys.exit(1)
    
    output_file = sys.argv[1]
    stat_name = sys.argv[2]
    
    # Read JSON data from stdin
    data = json.load(sys.stdin)
    
    # Load or create output file
    try:
        with open(output_file) as f:
            output = json.load(f)
    except FileNotFoundError:
        output = []
    
    # Remove existing stat if present
    output = [s for s in output if s.get('stat_name') != stat_name]
    
    # Add new stat
    output.append({
        'stat_name': stat_name,
        'season': 2025,
        'teams': data
    })
    
    # Save
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved {stat_name}: {len(data)} teams")

if __name__ == "__main__":
    main()
