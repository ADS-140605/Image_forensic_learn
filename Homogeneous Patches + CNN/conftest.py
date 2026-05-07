"""conftest.py — shared pytest configuration."""
import sys
from pathlib import Path

# Make sure the project root is always importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
