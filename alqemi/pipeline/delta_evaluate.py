"""Pipeline stage 3: DeltaEvaluate.

Deltas are born here. Each ElementaryBracket left by IdentityExpand is evaluated by
HJO (1.2.27)-(1.2.29); then deltas are evaluated and sums over dummies collapsed,
substituting into operators, other deltas and tensors alike.
Operators are never reordered and like terms are never merged (Simplify, stage 4).
"""
from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Set

from alqemi.core.expr import Bracket, Delta, ElementaryBracket, Expr, Term, as_expr
from alqemi.core.indices import Index
from alqemi.core.operators import Op, OpString
from alqemi.core.tensor import Tensor


class DanglingSum(Exception):
    """A summed index no longer occurs, e.g. sum_k delta_kk a_P = (orbital count) a_P."""

    def __init__(self, term: Term, index: Index) -> None:
        super().__init__(
            f"summed index {index} no longer occurs after delta evaluation of {term}; "
            "its sum would be the dimension of its space, which has no symbol yet"
        )
        self.term = term
        self.index = index


def delta_evaluate(x) -> Expr:
    """Evaluate elementary brackets and deltas in every term of ``x`` (bracket-free)."""
    expr = as_expr(x)
    for term in expr:
        if any(isinstance(f, Bracket) for f in term.factors):
            raise ValueError(f"delta_evaluate needs bracket-free input; run IdentityExpand first: {term}")
    out = (_evaluate_term(term) for term in expr)
    return Expr(tuple(t for t in out if t is not None))


# ---- one term -------------------------------------------------------------------
def _evaluate_term(term: Term) -> Optional[Term]:
    """The evaluated term, or None when it vanishes."""
    # 1. Elementary brackets, (1.2.27)-(1.2.29). The delta keeps the bracket's
    #    order, which is how HJO writes its printed results.
    deltas: List[Delta] = []
    tensors: List[Tensor] = []
    for s in term.scalars:
        if isinstance(s, ElementaryBracket):
            if s.left.kind is s.right.kind:
                return None
            deltas.append(Delta(s.left.index, s.right.index))
        elif isinstance(s, Tensor):
            tensors.append(s)
        else:
            deltas.append(s)

    factors = list(term.factors)  # bracket-free: only OpStrings
    summed: Set[Index] = set(term.summed)

    # 2. Evaluate deltas until nothing changes, first applicable delta first.
    while True:
        deltas = _scalar_order(deltas)
        for n, d in enumerate(deltas):
            if d.p == d.q:  # delta_pp = 1
                del deltas[n]
                break
            if not d.p.space.overlaps(d.q.space):  # e.g. delta_ia = 0
                return None
            victim = _collapsible(d, summed)
            if victim is not None:
                keep = d.q if victim == d.p else d.p
                mapping = {victim: keep}
                deltas = [Delta(mapping.get(x.p, x.p), mapping.get(x.q, x.q)) for x in deltas]
                factors = [_substitute(f, mapping) for f in factors]
                tensors = [t._rename(mapping) for t in tensors]
                summed.discard(victim)
                break
        else:
            break

    # 3. Equal-valued deltas (delta_PQ, delta_QP) count once.
    deltas = _dedupe(deltas)

    # 4. A sum that no longer occurs has no symbolic value.
    occurring = (
        {ix for d in deltas for ix in (d.p, d.q)}
        | {op.index for f in factors for op in f}
        | {ix for t in tensors for ix in t.indices}
    )
    for ix in sorted(summed - occurring, key=lambda ix: (ix.name, ix.space.value)):
        raise DanglingSum(term, ix)

    return Term(term.coeff, tuple(deltas) + tuple(tensors), tuple(factors), frozenset(summed))


def _collapsible(d: Delta, summed: Set[Index]) -> Optional[Index]:
    """The summed index of ``d`` to eliminate, or None.

    A summed index may be eliminated only if its space contains the other index's
    space (sum_p delta_pi -> i, but sum_i delta_iP stays: P may be virtual). If both
    qualify, the wider one goes; on a tie, the right index q goes.
    """
    p_ok = d.p in summed and d.p.space.contains(d.q.space)
    q_ok = d.q in summed and d.q.space.contains(d.p.space)
    if p_ok and q_ok:
        p_wider = not d.q.space.contains(d.p.space)
        return d.p if p_wider else d.q
    if p_ok:
        return d.p
    if q_ok:
        return d.q
    return None


def _substitute(string: OpString, mapping: Dict[Index, Index]) -> OpString:
    # Unlike Term.rename, the target may be in a narrower space (sum_p delta_pi -> i).
    return OpString(tuple(Op(op.kind, mapping.get(op.index, op.index)) for op in string))


def _scalar_order(deltas: List[Delta]) -> List[Delta]:
    """The deltas in the canonical scalar order a Term stores them in."""
    return list(Term(scalars=tuple(deltas)).scalars)


def _dedupe(deltas: List[Delta]) -> List[Delta]:
    seen: Set[FrozenSet[Index]] = set()
    kept = []
    for d in deltas:
        key = frozenset((d.p, d.q))
        if key not in seen:
            seen.add(key)
            kept.append(d)
    return kept
