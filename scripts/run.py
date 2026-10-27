#!/usr/bin/env python3
"""Explicitly prepare, verify and train from a downloaded official PGPS9K directory."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from geo_opsd.workflow import main

if __name__ == '__main__':
    main()
