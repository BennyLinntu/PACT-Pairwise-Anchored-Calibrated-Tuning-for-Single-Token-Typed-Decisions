"""Shared workspace paths, independent of module location or current directory."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
