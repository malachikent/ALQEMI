"""Fresh dummy indices, issued by an explicit pool object (no global state)."""
from __future__ import annotations

from typing import Dict, Iterable, Set

from alqemi.core.indices import Index, Space

_PREFIX = {Space.GENERAL: "p", Space.OCC: "i", Space.VIRT: "a"}


class IndexPool:
    """Issues indices named p1, p2, ... (GENERAL), i1, ... (OCC), a1, ... (VIRT).

    A pool never issues the same name twice. Separate pools are independent and
    deterministic: each new pool starts its sequences again from 1.
    """

    def __init__(self) -> None:
        self._next: Dict[Space, int] = {space: 1 for space in Space}
        self._issued: Set[str] = set()

    def fresh(self, space: Space, avoid: Iterable[str] = ()) -> Index:
        """Return a new index in ``space`` whose name is not in ``avoid``."""
        if not isinstance(space, Space):
            raise TypeError(f"space must be a Space, got {space!r}")
        avoid = set(avoid)
        n = self._next[space]
        while True:
            name = f"{_PREFIX[space]}{n}"
            n += 1
            if name not in avoid and name not in self._issued:
                break
        self._next[space] = n
        self._issued.add(name)
        return Index(name, space)
