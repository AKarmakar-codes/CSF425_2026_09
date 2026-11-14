"""Semantic segmentation package for Assignment II."""
from __future__ import annotations

from src.semantic.prepare_source import (
    get_physical_board_group,
    deterministic_board_split,
    DSLRPhotometricAugmentation,
    extract_tiles_from_board,
    discover_source_boards,
)

__all__ = [
    "get_physical_board_group",
    "deterministic_board_split",
    "DSLRPhotometricAugmentation",
    "extract_tiles_from_board",
    "discover_source_boards",
]
