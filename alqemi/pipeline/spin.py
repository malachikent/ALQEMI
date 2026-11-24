"""Spin integration for UHF: spin-orbital equations to their alpha/beta spin blocks.

``spin_integrate`` works on projected, tensor-only equations (DESIGN_DECISIONS,
Step 12). The free indices get the canonical spin blocks: alpha before beta within
the occupied and within the virtual free indices, with as many alpha occupied as
alpha virtual (Sz conserved), e.g. OOVV, OoVv, oovv; OV, ov; "" with no free
indices. Each summed index becomes an alpha and a beta copy of its term; a copy is
dropped when a tensor's upper spins differ from its lower spins, which is exact
because every operator conserves spin; then Simplify merges the copies. Alpha
indices are written in upper case, beta in lower case (names only: spin is the
Index's ``spin`` field, never parsed from the name). No Wick's theorem, no new algebra.
"""
from __future__ import annotations

import itertools
from collections import Counter
from typing import Dict, List

from alqemi.core.expr import Expr, Term, as_expr
from alqemi.core.indices import Index, Space, Spin
from alqemi.core.pool import IndexPool
from alqemi.core.tensor import Tensor
from alqemi.pipeline.simplify import simplify

A, B = Spin.ALPHA, Spin.BETA
_LETTER = {(Space.OCC, A): "O", (Space.OCC, B): "o", (Space.VIRT, A): "V", (Space.VIRT, B): "v"}


def spin_integrate(x) -> Dict[str, Expr]:
    """{block key: the block's equation}, keys in order k = n, ..., 0 alpha pairs.

    Raises ValueError for operators, deltas, general indices, indices that already
    have a spin, tensors with unequal ranks, unequal numbers of free occupied and
    virtual indices, or free names that clash after the case change.
    """
    expr = as_expr(x)
    for term in expr:
        if term.factors:
            raise ValueError(f"term {term} has operators: project it first")
        for s in term.scalars:
            if not isinstance(s, Tensor):
                raise ValueError(f"{type(s).__name__} {s} is not supported by spin integration")
            if len(s.upper) != len(s.lower):
                raise ValueError(f"tensor {s} has unequal ranks; spin integration needs equal ranks")
        for ix in term.free_indices | term.summed:
            if ix.space is Space.GENERAL:
                raise ValueError(f"general index {ix.name}: spin integration needs OCC or VIRT indices")
            if ix.spin is not None:
                raise ValueError(f"index {ix.name} already has a spin")
    free = expr.free_indices
    occ = sorted((ix for ix in free if ix.space is Space.OCC), key=lambda ix: ix.name)
    virt = sorted((ix for ix in free if ix.space is Space.VIRT), key=lambda ix: ix.name)
    if len(occ) != len(virt):
        raise ValueError("spin integration needs as many free occupied as free virtual indices")
    n = len(occ)
    blocks: Dict[str, Expr] = {}
    for k in range(n, -1, -1):
        spins = [A] * k + [B] * (n - k)
        mapping = {}
        for group in (occ, virt):
            for ix, s in zip(group, spins):
                mapping[ix] = Index(ix.name.upper() if s is A else ix.name.lower(), ix.space, s)
        names = [new.name for new in mapping.values()]
        if len(set(names)) != len(names):
            raise ValueError(f"free index names {sorted(ix.name for ix in free)} clash after spin labelling")
        key = "".join(_LETTER[(mapping[ix].space, mapping[ix].spin)] for ix in occ + virt)
        terms: List[Term] = []
        for term in expr:
            terms.extend(_spin_copies(term, mapping, set(names)))
        blocks[key] = simplify(Expr(tuple(terms)))
    return blocks


def _spin_copies(term: Term, free_map, free_names) -> List[Term]:
    """Every alpha/beta choice for the summed indices of ``term`` that conserves spin."""
    pool = IndexPool()
    avoid = set(term.all_names) | free_names
    dummies = sorted(term.summed, key=lambda ix: ix.name)
    fresh = {}
    for d in dummies:
        new = pool.fresh(d.space, avoid)
        avoid.add(new.name)
        fresh[d] = new.name
    out = []
    for spins in itertools.product((A, B), repeat=len(dummies)):
        mapping = dict(free_map)
        for d, s in zip(dummies, spins):
            mapping[d] = Index(fresh[d], d.space, s)
        copy = term._rename(mapping)
        if all(_conserves_spin(t) for t in copy.scalars):
            out.append(copy)
    return out


def _conserves_spin(t: Tensor) -> bool:
    """Upper and lower indices carry the same spins (as multisets)."""
    return Counter(ix.spin for ix in t.upper) == Counter(ix.spin for ix in t.lower)
