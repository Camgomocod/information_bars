#!/usr/bin/env python3
"""Compatibility wrapper for the installed healthcheck command."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.healthcheck import main

if __name__ == "__main__":
    raise SystemExit(main())
