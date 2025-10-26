"""Custom type aliases for utilities (renamed from types.py to avoid circular import)."""

from typing import Literal

Intensity = Literal["low", "mid", "high"]
AuxTuple = tuple[str, str, str]

__all__ = ["Intensity", "AuxTuple"]
