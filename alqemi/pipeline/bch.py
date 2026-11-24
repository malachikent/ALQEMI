"""The Baker-Campbell-Hausdorff expansion e^{-T} H e^{T}."""
from __future__ import annotations

import math
from fractions import Fraction

from alqemi.core.expr import Expr, comm
from alqemi.pipeline.run import run
from alqemi.pipeline.simplify import simplify


def bch(h, t, *, order: int = 4) -> Expr:
    """sum_{k=0}^{order} (1/k!) C_k, C_0 = run(h), C_k = run([C_{k-1}, t]); simplified.

    Each nested commutator is evaluated before the next is formed. For a two-body H
    the series ends after order 4, where it equals e^{-T} H e^{T} exactly.
    """
    if not isinstance(order, int) or order < 0:
        raise ValueError(f"order must be a non-negative integer, got {order!r}")
    c = run(h)
    total = c
    for k in range(1, order + 1):
        c = run(comm(c, t))
        if not len(c):
            break
        total = total + c.scale(Fraction(1, math.factorial(k)))
    return simplify(total)
