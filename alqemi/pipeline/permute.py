"""Permutation operators (1 - P_ij) and (1 - P_ab) for doubles-type residuals (Step 14).

``group_permutations`` folds terms that are related by swapping two free indices into
``PermutedTerm`` groups; ``expand_permutations`` undoes it exactly. Grouping is a
printing / code-generation aid only: it never changes the value of an expression.

P_pq renames the free indices p <-> q. A group with swaps (s1,) stands for
``t - P_s1 t``; one with swaps (s1, s2) stands for ``t - P_s1 t - P_s2 t + P_s1 P_s2 t``.
Partners are matched with Simplify's canonical like-term keys, so dummy renaming and
tensor (anti)symmetries are accounted for, signs included.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from alqemi.core.expr import Expr, Term, as_expr
from alqemi.core.indices import Index, Space
from alqemi.pipeline.simplify import _like_key

Pair = Tuple[Index, Index]

__all__ = ["PermutedTerm", "group_permutations", "expand_permutations"]


@dataclass(frozen=True)
class PermutedTerm:
    """A term with up to two antisymmetrizers (1 - P_pq) applied to it.

    Each swap is a pair of free indices of the term with the same space and spin,
    sorted by name; pairs are disjoint, and with two swaps the occupied pair comes first.
    """

    term: Term
    swaps: Tuple[Pair, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.term, Term):
            raise TypeError(f"PermutedTerm needs a Term, got {self.term!r}")
        swaps = tuple(tuple(s) for s in self.swaps)
        object.__setattr__(self, "swaps", swaps)
        if len(swaps) > 2:
            raise ValueError("at most two swaps")
        free = self.term.free_indices
        seen = set()
        for p, q in swaps:
            if p not in free or q not in free:
                raise ValueError(f"swap ({p}, {q}) needs free indices of the term")
            if p.space is not q.space or p.spin is not q.spin:
                raise ValueError(f"swap ({p}, {q}) needs indices of the same space and spin")
            if not p.name < q.name:
                raise ValueError(f"swap ({p}, {q}) must be sorted by name")
            if {p, q} & seen:
                raise ValueError("swaps must be disjoint")
            seen |= {p, q}
        if len(swaps) == 2 and not (swaps[0][0].space is Space.OCC and swaps[1][0].space is Space.VIRT):
            raise ValueError("with two swaps the occupied pair comes first")


def _swap(term: Term, pairs) -> Term:
    """The term with each pair (p, q) of free indices exchanged."""
    mapping = {}
    for p, q in pairs:
        mapping[p], mapping[q] = q, p
    return term._rename(mapping)


def expand_permutations(groups) -> Expr:
    """The Expr that a sequence of PermutedTerms stands for (exact inverse of grouping)."""
    out: List[Term] = []
    for g in groups:
        s = g.swaps
        subsets = [()] + [(x,) for x in s] + ([tuple(s)] if len(s) == 2 else [])
        for sub in subsets:
            out.append(_swap(g.term, sub).scale((-1) ** len(sub)))
    return Expr(tuple(out))


def _allowed_swaps(expr: Expr) -> Tuple[Optional[Pair], Optional[Pair]]:
    """The occupied and virtual pairs: a (space, spin) class with exactly two free indices."""
    kinds: Dict[tuple, List[Index]] = {}
    for ix in expr.free_indices:
        kinds.setdefault((ix.space, ix.spin), []).append(ix)
    occ = virt = None
    order = lambda kv: (kv[0][0].value, "" if kv[0][1] is None else kv[0][1].value)
    for (space, _), members in sorted(kinds.items(), key=order):
        if len(members) == 2:
            pair = tuple(sorted(members, key=lambda ix: ix.name))
            if space is Space.OCC and occ is None:
                occ = pair
            elif space is Space.VIRT and virt is None:
                virt = pair
    return occ, virt


def _value(term: Term):
    """(canonical key, signed coefficient), or None if the term has no like-key."""
    like = _like_key(term)
    if like is None:
        return None
    key, sign = like
    return key, term.coeff * sign


def group_permutations(x) -> Tuple[PermutedTerm, ...]:
    """Group the terms of ``x`` under (1 - P_ij), (1 - P_ab) and their product.

    Greedy, in term order: for each ungrouped term the product is tried first, then the
    occupied pair, then the virtual pair. A swap that maps a term to +-itself is never
    used for it. Terms with no partners come back with no swaps. The result satisfies
    ``expand_permutations(group_permutations(x)) == x`` up to term order.
    """
    expr = as_expr(x)
    occ, virt = _allowed_swaps(expr)
    values = [_value(t) for t in expr]
    used = [False] * len(expr)

    def find(target, taken):
        if target is None:
            return None
        for n, val in enumerate(values):
            if not used[n] and n not in taken and val == target:
                return n
        return None

    def wanted(n, pairs, sign):
        """The value a partner needs, or None if the swaps map term n to +-itself."""
        val = _value(_swap(expr.terms[n], pairs))
        if val is None or val[0] == values[n][0]:
            return None
        return (val[0], sign * val[1])

    out: List[PermutedTerm] = []
    for n, term in enumerate(expr):
        if used[n]:
            continue
        used[n] = True
        if values[n] is None:
            out.append(PermutedTerm(term, ()))
            continue
        group = None
        if occ and virt:
            taken = {n}
            ms = []
            for pairs, sign in (((occ,), -1), ((virt,), -1), ((occ, virt), 1)):
                m = find(wanted(n, pairs, sign), taken)
                if m is None:
                    break
                taken.add(m)
                ms.append(m)
            if len(ms) == 3:
                group = ((occ, virt), ms)
        if group is None:
            for pair in (occ, virt):
                if pair is None:
                    continue
                m = find(wanted(n, (pair,), -1), {n})
                if m is not None:
                    group = ((pair,), [m])
                    break
        if group is None:
            out.append(PermutedTerm(term, ()))
        else:
            for m in group[1]:
                used[m] = True
            out.append(PermutedTerm(term, group[0]))
    return tuple(out)
