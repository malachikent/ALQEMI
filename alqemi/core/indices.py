"""Orbital indices and the orbital spaces they range over."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Space(Enum):
    """Orbital space an index ranges over."""

    GENERAL = "general"
    OCC = "occ"
    VIRT = "virt"

    def overlaps(self, other: Space) -> bool:
        """Whether an index in this space can coincide with one in ``other``."""
        if self is Space.GENERAL or other is Space.GENERAL:
            return True
        return self is other

    def contains(self, other: Space) -> bool:
        """Whether every index in ``other`` can also be an index in this space."""
        return self is Space.GENERAL or self is other


class Spin(Enum):
    """Spin of a spin-integrated index (Step 12); spin-orbital indices have none."""

    ALPHA = "alpha"
    BETA = "beta"


@dataclass(frozen=True)
class Index:
    """A named orbital index; its space and spin are explicit, never parsed from the name."""

    name: str
    space: Space = Space.GENERAL
    spin: Optional[Spin] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"Index name must be a non-empty string, got {self.name!r}")
        if not isinstance(self.space, Space):
            raise TypeError(f"Index space must be a Space, got {self.space!r}")
        if self.spin is not None and not isinstance(self.spin, Spin):
            raise TypeError(f"Index spin must be a Spin or None, got {self.spin!r}")

    def __str__(self) -> str:
        return self.name
