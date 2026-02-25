#!/usr/bin/env python3
"""Backward-compatible import shim for the split Poisson package."""

import sys

if __name__ == "__main__":
    raise SystemExit(
        "Run the Poisson model via package imports (models.poisson_model / models.poisson). "
        "Direct script execution for this shim is not supported."
    )

from models import poisson as _poisson  # noqa: E402

sys.modules[__name__] = _poisson
