"""Puts this repo's `src/` (+ `src/data/`) on `sys.path` for every test --
done once here so individual test files don't each need their own path
dance. Fully self-contained since the 2026-07-31 split from the sibling
research repo -- no other repo's path is added."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "data"))
