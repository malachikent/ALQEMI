"""Pipeline stage 4: Simplify.

The first stage allowed to merge terms and drop zeros, and the only one that
renames dummies for display (DESIGN_DECISIONS 12). Operators are still never
reordered (decision 8): terms are alike only if their operator strings match
exactly, up to dummy names.
"""
from __future__ import annotations

import itertools
from fractions import Fraction
from typing import Dict, List, Set, Tuple

from alqemi.core.expr import Bracket, ElementaryBracket, Expr, Term, as_expr
from alqemi.core.indices import Index, Space

# Textbook letters for dummies, then a numbered fallback (i1, a1, p1, ...).
LETTERS = {Space.OCC: "ijklmno", Space.VIRT: "abcdefgh", Space.GENERAL: "pqrstuvw"}
FALLBACK_PREFIX = {Space.OCC: "i", Space.VIRT: "a", Space.GENERAL: "p"}


def simplify(x) -> Expr:
    """Merge like terms, drop zero terms, and relabel dummies to letters."""
    expr = _checked(x)
    keys: List[tuple] = []
    first: Dict[tuple, Term] = {}
    total: Dict[tuple, Fraction] = {}
    for term in expr:
        key = _like_key(term)
        if key not in first:
            keys.append(key)
            first[key] = term
            total[key] = Fraction(0)
        total[key] += term.coeff
    merged = tuple(
        Term(total[key], first[key].scalars, first[key].factors, first[key].summed)
        for key in keys
        if total[key] != 0
    )
    return relabel_dummies(Expr(merged))


def relabel_dummies(x) -> Expr:
    """Rename each term's dummies to the first free letters of their space."""
    expr = _checked(x)
    taken_by_free = {ix.name for term in expr for ix in term.free_indices}
    return Expr(tuple(_relabel_term(term, taken_by_free) for term in expr))


# ---- like terms -------------------------------------------------------------------
IndexKey = Tuple[str, str]


def _like_key(term: Term) -> tuple:
    """A key equal for two terms exactly when they are alike.

    Alike: equal after renaming dummies within their spaces and reading deltas as
    symmetric. The coefficient is not part of the key. Every renaming within each
    space is tried and the smallest key kept (n! per space; terms have few dummies).
    """
    by_space: Dict[Space, List[Index]] = {}
    for d in sorted(term.summed, key=lambda ix: (ix.name, ix.space.value)):
        by_space.setdefault(d.space, []).append(d)
    spaces = list(by_space)

    best = None
    for orders in itertools.product(*(itertools.permutations(by_space[s]) for s in spaces)):
        # Placeholder names start with "\0", so they can never equal a real name.
        placeholder = {
            d: ("\0" + str(n), space.value)
            for space, order in zip(spaces, orders)
            for n, d in enumerate(order)
        }

        def name(ix: Index) -> IndexKey:
            return placeholder.get(ix, (ix.name, ix.space.value))

        key = (
            tuple(tuple((op.kind.value, *name(op.index)) for op in string) for string in term.factors),
            tuple(sorted(tuple(sorted((name(d.p), name(d.q)))) for d in term.scalars)),
            tuple(sorted(name(d) for d in term.summed)),
        )
        if best is None or key < best:
            best = key
    return best


# ---- relabelling -------------------------------------------------------------------
def _relabel_term(term: Term, taken_by_free: Set[str]) -> Term:
    order: List[Index] = []
    for string in term.factors:  # operators left to right first
        for op in string:
            if op.index in term.summed and op.index not in order:
                order.append(op.index)
    for d in term.scalars:  # then deltas, in stored order
        for ix in (d.p, d.q):
            if ix in term.summed and ix not in order:
                order.append(ix)

    used = set(taken_by_free)
    mapping: Dict[Index, Index] = {}
    for dummy in order:
        new = _next_name(dummy.space, used)
        used.add(new)
        mapping[dummy] = Index(new, dummy.space)
    return term.rename(mapping) if mapping else term


def _next_name(space: Space, used: Set[str]) -> str:
    for letter in LETTERS[space]:
        if letter not in used:
            return letter
    n = 1
    while f"{FALLBACK_PREFIX[space]}{n}" in used:
        n += 1
    return f"{FALLBACK_PREFIX[space]}{n}"


# ---- input check -------------------------------------------------------------------
def _checked(x) -> Expr:
    expr = as_expr(x)
    for term in expr:
        if any(isinstance(f, Bracket) for f in term.factors):
            raise ValueError(f"brackets remain; run IdentityExpand first: {term}")
        if any(isinstance(s, ElementaryBracket) for s in term.scalars):
            raise ValueError(f"elementary brackets remain; run DeltaEvaluate first: {term}")
    return expr
