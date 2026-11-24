"""Brute-force reference for Simplify (the Step 6/8 algorithm), used only by tests.

Step 10 made Simplify's like-term search faster (it only exchanges dummies with equal
fingerprints). This module keeps the original search, which tries EVERY renaming of
the dummies within each space, as a slow but obviously correct referee: the fast
Simplify must give exactly the same output. Never edit this to make code pass.
"""
from __future__ import annotations

import itertools
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from alqemi.core.expr import Bracket, Delta, ElementaryBracket, Expr, Term, as_expr
from alqemi.core.indices import Index, Space
from alqemi.core.operators import OpString
from alqemi.core.tensor import Swap, Tensor

# Textbook letters for dummies, then a numbered fallback (i1, a1, p1, ...).
LETTERS = {Space.OCC: "ijklmno", Space.VIRT: "abcdefgh", Space.GENERAL: "pqrstuvw"}
FALLBACK_PREFIX = {Space.OCC: "i", Space.VIRT: "a", Space.GENERAL: "p"}


def simplify(x) -> Expr:
    """Merge like terms, drop zero terms, and relabel dummies to letters."""
    expr = _checked(x)
    keys: List[tuple] = []
    first: Dict[tuple, Term] = {}
    first_sign: Dict[tuple, int] = {}
    total: Dict[tuple, Fraction] = {}
    for term in expr:
        like = _like_key(term)
        if like is None:  # the term equals minus itself
            continue
        key, sign = like  # term = sign * coeff * (canonical form)
        if key not in first:
            keys.append(key)
            first[key] = term
            first_sign[key] = sign
            total[key] = Fraction(0)
        total[key] += term.coeff * sign
    # Written in the first occurrence's form: canonical form = first_sign * that form.
    merged = tuple(
        Term(first_sign[key] * total[key], first[key].scalars, first[key].factors, first[key].summed)
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
Namer = Callable[[Index], IndexKey]


def _like_key(term: Term) -> Optional[Tuple[tuple, int]]:
    """(key, sign) with term = sign * coeff * (the canonical form named by key); None if 0.

    The coefficient is not part of the key. Every renaming of dummies within each
    space is tried and the smallest key kept (n! per space; terms have few dummies).
    Operator runs and tensors are put in canonical order for each renaming. If the
    smallest key is reached with both signs, the term equals minus itself.
    """
    by_space: Dict[Space, List[Index]] = {}
    for d in sorted(term.summed, key=lambda ix: (ix.name, ix.space.value)):
        by_space.setdefault(d.space, []).append(d)
    spaces = list(by_space)
    deltas = [s for s in term.scalars if isinstance(s, Delta)]
    tensors = [s for s in term.scalars if isinstance(s, Tensor)]

    best = None
    signs: Set[int] = set()
    for orders in itertools.product(*(itertools.permutations(by_space[s]) for s in spaces)):
        # Placeholder names start with "\0", so they can never equal a real name.
        placeholder = {
            d: ("\0" + str(n), space.value)
            for space, order in zip(spaces, orders)
            for n, d in enumerate(order)
        }

        def name(ix: Index) -> IndexKey:
            return placeholder.get(ix, (ix.name, ix.space.value))

        sign = 1
        string_keys = []
        for string in term.factors:
            canon = _canonical_string(string, name)
            if canon is None:
                return None
            string_keys.append(canon[0])
            sign *= canon[1]
        tensor_keys = []
        for t in tensors:
            canon = _canonical_tensor(t, name)
            if canon is None:
                return None
            tensor_keys.append(canon[0])
            sign *= canon[1]
        key = (
            tuple(string_keys),
            tuple(sorted(tuple(sorted((name(d.p), name(d.q)))) for d in deltas)),
            tuple(sorted(tensor_keys)),
            tuple(sorted(name(d) for d in term.summed)),
        )
        if best is None or key < best:
            best, signs = key, {sign}
        elif key == best:
            signs.add(sign)
    if len(signs) > 1:
        return None
    return best, signs.pop()


def _sorted_with_sign(keys: Sequence[tuple]) -> Optional[Tuple[tuple, int]]:
    """``keys`` sorted, with the parity of the sort; None if a key repeats.

    For anticommuting objects (same-kind operators, an antisymmetric index group) the
    sorted order is the canonical one, the parity its sign, and a repeat means zero.
    """
    if len(set(keys)) < len(keys):
        return None
    order = sorted(range(len(keys)), key=lambda n: keys[n])
    return tuple(keys[n] for n in order), _parity(order)


def _canonical_string(string: OpString, name: Namer) -> Optional[Tuple[tuple, int]]:
    """Each run of adjacent same-kind operators sorted (decision 14), with the sign."""
    out: List[tuple] = []
    sign = 1
    for kind, run in itertools.groupby(string, key=lambda op: op.kind):
        canon = _sorted_with_sign([name(op.index) for op in run])
        if canon is None:
            return None  # a repeated operator: a+_p a+_p = 0
        out.extend((kind.value, *key) for key in canon[0])
        sign *= canon[1]
    return tuple(out), sign


def _canonical_tensor(t: Tensor, name: Namer) -> Optional[Tuple[tuple, int]]:
    """The smallest image of ``t`` under its symmetries, with its sign; None if t = -t."""
    upper = _sorted_with_sign([name(ix) for ix in t.upper])
    lower = _sorted_with_sign([name(ix) for ix in t.lower])
    if upper is None or lower is None:
        return None  # a repeated index within a group
    sign = upper[1] * lower[1]
    images = {(upper[0], lower[0]): {sign}}
    if t.swap is not Swap.NONE:
        swapped = (lower[0], upper[0])
        swap_sign = sign if t.swap is Swap.SYMMETRIC else -sign
        images.setdefault(swapped, set()).add(swap_sign)
    smallest = min(images)
    if len(images[smallest]) > 1:
        return None  # e.g. X^{P}_{P} with an antisymmetric swap
    (sign,) = images[smallest]
    return (t.name, t.swap.value, len(t.upper), len(t.lower), smallest), sign


def _parity(perm: Sequence[int]) -> int:
    inversions = sum(1 for x in range(len(perm)) for y in range(x) if perm[y] > perm[x])
    return -1 if inversions % 2 else 1


# ---- relabelling -------------------------------------------------------------------
def _relabel_term(term: Term, taken_by_free: Set[str]) -> Term:
    order: List[Index] = []
    for string in term.factors:  # operators left to right first
        for op in string:
            if op.index in term.summed and op.index not in order:
                order.append(op.index)
    scalars = [s for s in term.scalars if isinstance(s, Delta)]  # then deltas,
    scalars += [s for s in term.scalars if isinstance(s, Tensor)]  # then tensors
    for s in scalars:  # each in stored order
        for ix in s.indices:
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
