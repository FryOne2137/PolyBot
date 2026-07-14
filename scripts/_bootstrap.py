"""Make repository-local ``src/`` imports work for direct script execution."""

from __future__ import annotations

import sys
from pathlib import Path


def add_local_source_root() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    source_root_text = str(source_root)
    if source_root_text not in sys.path:
        sys.path.insert(0, source_root_text)
