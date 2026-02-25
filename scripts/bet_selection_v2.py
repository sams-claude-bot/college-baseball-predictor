#!/usr/bin/env python3
"""
Improved Bet Selection Logic (v2)

Backwards-compatible entrypoint and import surface. Implementation lives in
`scripts.betting`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.betting import *  # noqa: F401,F403


if __name__ == '__main__':
    main()
