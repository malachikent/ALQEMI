"""Pipeline stage 1: Linearize.

Uses only the bilinearity of the bracket definitions (1.8.1)/(1.8.2):

    [sum_i c_i s_i A_i, sum_j d_j t_j B_j](+) = sum_ij c_i d_j s_i t_j [A_i, B_j](+)

Coefficients, scalars (deltas and elementary anticommutators are c-numbers) and
summations move out of bracket arguments onto the top-level terms. Nothing is
expanded, evaluated, merged or reordered.
"""
from __future__ import annotations

from typing import List, Set

from alqemi.core.expr import Bracket, Expr, Term, as_expr, freshen_dummies
from alqemi.core.operators import OpString
from alqemi.core.pool import IndexPool


def linearize(x, pool: IndexPool) -> Expr:
    """Return ``x`` in linear form (see ``is_linear``), with the same value."""
    out: List[Term] = []
    for term in as_expr(x):
        # One avoid set per input term: fresh names must miss every name in the
        # whole term (later factors and deeper brackets included).
        avoid = set(term.all_names)
        out.extend(_linearize_term(term, pool, avoid))
    return Expr(tuple(out))


def is_linear(x) -> bool:
    """True iff every bracket, at any depth, has one bare term on each side."""
    return isinstance(x, Expr) and all(_brackets_bare(t) for t in x)


# ---- internals --------------------------------------------------------------
def _linearize_term(term: Term, pool: IndexPool, avoid: Set[str]) -> List[Term]:
    # Combine the factors' expansions left-major, as multiply does. The term's
    # own summed indices are re-attached at the end; no-shadowing guarantees no
    # pulled-out dummy shares their names.
    acc = [Term(term.coeff, term.scalars)]
    for factor in term.factors:
        if isinstance(factor, OpString):
            pieces = [Term(factors=(factor,))]
        else:
            pieces = _expand_bracket(factor, pool, avoid)
        acc = [_join(done, piece, pool, avoid) for done in acc for piece in pieces]
    return [Term(t.coeff, t.scalars, t.factors, t.summed | term.summed) for t in acc]


def _expand_bracket(br: Bracket, pool: IndexPool, avoid: Set[str]) -> List[Term]:
    out = []
    for left in _linearize_expr(br.left, pool, avoid):
        for right in _linearize_expr(br.right, pool, avoid):
            right = freshen_dummies(right, left.all_names, avoid, pool)
            bracket = Bracket(br.kind, Term(factors=left.factors), Term(factors=right.factors))
            out.append(
                Term(
                    left.coeff * right.coeff,
                    left.scalars + right.scalars,
                    (bracket,),
                    left.summed | right.summed,
                )
            )
    return out


def _linearize_expr(expr: Expr, pool: IndexPool, avoid: Set[str]) -> List[Term]:
    out: List[Term] = []
    for term in expr:
        out.extend(_linearize_term(term, pool, avoid))
    return out


def _join(done: Term, piece: Term, pool: IndexPool, avoid: Set[str]) -> Term:
    """done * piece, renaming piece's dummies that clash with anything already in done."""
    piece = freshen_dummies(piece, done.all_names, avoid, pool)
    return Term(
        done.coeff * piece.coeff,
        done.scalars + piece.scalars,
        done.factors + piece.factors,
        done.summed | piece.summed,
    )


def _bare(expr: Expr) -> bool:
    if len(expr) != 1:
        return False
    (t,) = expr.terms
    return t.coeff == 1 and not t.scalars and not t.summed and _brackets_bare(t)


def _brackets_bare(term: Term) -> bool:
    return all(
        _bare(f.left) and _bare(f.right) for f in term.factors if isinstance(f, Bracket)
    )
