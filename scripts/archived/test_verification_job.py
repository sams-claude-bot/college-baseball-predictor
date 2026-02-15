#!/usr/bin/env python3
"""
Test the verification job manually
"""

import sys
import os
sys.path.insert(0, "/home/sam/college-baseball-predictor")

from verification_job import GameVerificationJob

if __name__ == "__main__":
    print("=== TESTING VERIFICATION JOB ===")
    
    job = GameVerificationJob()
    results = job.generate_report()
    
    print(f"Test completed successfully!")
    print(f"Games verified: {results['games_verified']}")
    print(f"Discrepancies: {len(results['discrepancies'])}")
    print(f"Errors: {len(results['errors'])}")
