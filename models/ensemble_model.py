#!/usr/bin/env python3
"""Backward-compatible import shim for the split Ensemble package."""

import sys

if __name__ == "__main__":
    from models.ensemble.predict import main

    raise SystemExit(main())

from models import ensemble as _ensemble  # noqa: E402

sys.modules[__name__] = _ensemble
