#!/usr/bin/env python3
"""Compatibility entry point for source-grounded TARA-5 repair.

The former in-place regex/template cleaner could inject one book's text into
another and reverse system/user prompts. Repair now stages source-offset based
candidates and independent review records before explicit publication.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.repair_tara5_data import main

if __name__ == '__main__':
    main()
