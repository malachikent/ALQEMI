"""Tensors: commuting scalars with upper and lower orbital indices (amplitudes, integrals)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Tuple

from alqemi.core.indices import Index


class Swap(Enum):
    """Symmetry under exchanging the whole upper group with the whole lower group."""

    NONE = "none"
    SYMMETRIC = "symmetric"
    ANTISYMMETRIC = "antisymmetric"


@dataclass(frozen=True)
class Tensor:
    """X^{upper}_{lower}: antisymmetric within each group, plus its ``swap`` symmetry.

    Data only: construction never reorders indices or detects zeros.
    """

    name: str
    upper: Tuple[Index, ...] = ()
    lower: Tuple[Index, ...] = ()
    swap: Swap = Swap.NONE

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"Tensor name must be a non-empty string, got {self.name!r}")
        upper, lower = tuple(self.upper), tuple(self.lower)
        for ix in upper + lower:
            if not isinstance(ix, Index):
                raise TypeError(f"Tensor indices must be Index, got {ix!r}")
        if not isinstance(self.swap, Swap):
            raise TypeError(f"Tensor swap must be a Swap, got {self.swap!r}")
        if self.swap is not Swap.NONE and len(upper) != len(lower):
            raise ValueError("a swap symmetry needs as many upper as lower indices")
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "lower", lower)

    @property
    def indices(self) -> Tuple[Index, ...]:
        return self.upper + self.lower

    def _sort_key(self) -> tuple:
        key = lambda ix: (ix.name, ix.space.value)
        return (2, self.name, self.swap.value, tuple(map(key, self.upper)), tuple(map(key, self.lower)))

    def _rename(self, mapping: Mapping[Index, Index]) -> Tensor:
        sub = lambda ix: mapping.get(ix, ix)
        return Tensor(self.name, tuple(map(sub, self.upper)), tuple(map(sub, self.lower)), self.swap)

    def __str__(self) -> str:
        out = self.name
        if self.upper:
            out += "^{" + "".join(str(ix) for ix in self.upper) + "}"
        if self.lower:
            out += "_{" + "".join(str(ix) for ix in self.lower) + "}"
        return out
