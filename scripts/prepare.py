#!/usr/bin/env python3
"""Create a new CPU-only experiment; no automatic launch."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from geo_opsd.prepare import main
if __name__ == "__main__":
    main()
