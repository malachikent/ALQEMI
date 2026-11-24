"""Builders for the standard operators of coupled-cluster and response theory.

Every builder returns an ordinary Expr: sums of (coefficient x tensor x operator
string) terms whose dummies come from an IndexPool (numbered names, decision 12).
New objects are one ``tensor_operator`` call. Nothing here evaluates anything.
"""
from __future__ import annotations

import math
from collections import Counter
from fractions import Fraction
from typing import List, Optional, Sequence, Tuple

from alqemi.core.expr import Expr, Term
from alqemi.core.indices import Index, Space
from alqemi.core.operators import Kind, Op, OpString, ann, cre
from alqemi.core.pool import IndexPool
from alqemi.core.tensor import Swap, Tensor

O, V, G = Space.OCC, Space.VIRT, Space.GENERAL
_PAIRS = ((O, O), (O, V), (V, V))  # the distinct two-index blocks
_FORMS = ("fock", "core")


def tensor_operator(
    name: str,
    upper: Sequence[Space],
    lower: Sequence[Space],
    *,
    swap: Swap = Swap.NONE,
    coeff=None,
    normal_ordered: bool = False,
    pool: Optional[IndexPool] = None,
) -> Expr:
    """coeff * sum X^{u1..un}_{l1..lm} a+_{u1}..a+_{un} a_{lm}..a_{l1}, one term.

    ``upper``/``lower`` give the spaces of the creator and annihilator indices.
    Default coeff: 1 / prod over both groups of (count of each space)!, which counts
    each distinct operator once for an antisymmetric tensor. normal_ordered=True
    writes the normal-ordered product {...} relative to the Fermi vacuum (OCC/VIRT
    indices only); that differs from the plain product when the string holds an
    occupied creator and an occupied annihilator.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"tensor name must be a non-empty string, got {name!r}")
    upper, lower = tuple(upper), tuple(lower)
    for space in upper + lower:
        if not isinstance(space, Space):
            raise TypeError(f"upper and lower must hold Space values, got {space!r}")
    if not upper and not lower:
        raise ValueError("an operator needs at least one index")
    pool = IndexPool() if pool is None else pool
    up = tuple(pool.fresh(s) for s in upper)
    low = tuple(pool.fresh(s) for s in lower)
    tensor = Tensor(name, up, low, swap)
    if coeff is None:
        coeff = Fraction(1, _multiplicity(upper) * _multiplicity(lower))
    ops = [cre(ix) for ix in up] + [ann(ix) for ix in reversed(low)]
    sign = 1
    if normal_ordered:
        ops, sign = _normal_order(ops)
    # Term checks the coefficient's type (int or Fraction).
    return Expr((Term(coeff * sign, (tensor,), (OpString(tuple(ops)),), frozenset(up + low)),))


def fock_operator(*, normal_ordered: bool = False, pool: Optional[IndexPool] = None) -> Expr:
    """F = sum_pq f^{p}_{q} a+_p a_q; normal_ordered: its 4 normal-ordered blocks + sum_i f^{i}_{i}."""
    pool = IndexPool() if pool is None else pool
    if not normal_ordered:
        return tensor_operator("f", [G], [G], swap=Swap.SYMMETRIC, pool=pool)
    out = Expr()
    for u, l in ((O, O), (O, V), (V, O), (V, V)):
        out = out + tensor_operator("f", [u], [l], swap=Swap.SYMMETRIC, normal_ordered=True, pool=pool)
    m = pool.fresh(O)
    return out + Expr((Term(1, (Tensor("f", (m,), (m,), Swap.SYMMETRIC),), (), frozenset({m})),))


def fluctuation_potential(*, normal_ordered: bool = False, pool: Optional[IndexPool] = None) -> Expr:
    """Phi = V - V_HF; normal_ordered: the 9 normal-ordered blocks of V - 1/2 sum_ij v^{ij}_{ij}."""
    pool = IndexPool() if pool is None else pool
    if not normal_ordered:
        two = tensor_operator("v", [G, G], [G, G], swap=Swap.SYMMETRIC, pool=pool)
        x, y, m = pool.fresh(G), pool.fresh(G), pool.fresh(O)
        v_hf = Term(
            -1, (Tensor("v", (x, m), (y, m), Swap.SYMMETRIC),), (cre(x) * ann(y),), frozenset({x, y, m})
        )
        return two + Expr((v_hf,))
    out = Expr()
    for u in _PAIRS:
        for l in _PAIRS:
            out = out + tensor_operator("v", u, l, swap=Swap.SYMMETRIC, normal_ordered=True, pool=pool)
    m, n = pool.fresh(O), pool.fresh(O)
    const = Term(Fraction(-1, 2), (Tensor("v", (m, n), (m, n), Swap.SYMMETRIC),), (), frozenset({m, n}))
    return out + Expr((const,))


def hamiltonian(
    *, form: str = "fock", normal_ordered: bool = False, pool: Optional[IndexPool] = None
) -> Expr:
    """H = F + Phi (form="fock") or h + V (form="core"); the same operator."""
    if form not in _FORMS:
        raise ValueError(f"form must be one of {_FORMS}, got {form!r}")
    pool = IndexPool() if pool is None else pool
    if form == "fock":
        return fock_operator(normal_ordered=normal_ordered, pool=pool) + fluctuation_potential(
            normal_ordered=normal_ordered, pool=pool
        )
    if normal_ordered:
        raise ValueError('normal_ordered=True is only defined for form="fock"')
    return tensor_operator("h", [G], [G], swap=Swap.SYMMETRIC, pool=pool) + tensor_operator(
        "v", [G, G], [G, G], swap=Swap.SYMMETRIC, pool=pool
    )


def cluster(n: int, *, name: str = "t", pool: Optional[IndexPool] = None) -> Expr:
    """T_n = (1/n!)^2 sum t^{a1..an}_{i1..in} a+_{a1}..a+_{an} a_{in}..a_{i1}."""
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"cluster rank must be a positive integer, got {n!r}")
    return tensor_operator(name, [V] * n, [O] * n, pool=pool)


# ---- helpers ----------------------------------------------------------------------
def _multiplicity(spaces: Tuple[Space, ...]) -> int:
    return math.prod(math.factorial(c) for c in Counter(spaces).values())


def _quasi_creator(op: Op) -> bool:
    if op.index.space is G:
        raise ValueError(f"normal order needs OCC or VIRT indices, got {op}")
    return (op.kind is Kind.CRE) == (op.index.space is V)


def _normal_order(ops: List[Op]) -> Tuple[List[Op], int]:
    """Quasi-creators left of quasi-annihilators, each keeping its order; and the sign."""
    flags = [_quasi_creator(op) for op in ops]
    sign, passed = 1, 0
    for is_creator in flags:  # each quasi-creator moves left past the annihilators before it
        if is_creator:
            sign *= (-1) ** passed
        else:
            passed += 1
    creators = [op for op, f in zip(ops, flags) if f]
    annihilators = [op for op, f in zip(ops, flags) if not f]
    return creators + annihilators, sign
