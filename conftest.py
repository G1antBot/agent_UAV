"""pytest configuration – ensures the repo root is on sys.path."""
import sys
import os

# Add repo root to path so `from src.xxx import yyy` works from any directory
sys.path.insert(0, os.path.dirname(__file__))
