"""Pipeline stage 2: IdentityExpand.

Rewrites every bracket with the HJO rule library until none remain. Each [x, y]_+
of two single operators becomes an unevaluated ElementaryBracket scalar
(DESIGN_DECISIONS 10): deltas appear only in stage 3. Nothing is merged or
reordered toward normal order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from alqemi.core.expr import Bracket, ElementaryBracket, Expr, Term, as_expr
from alqemi.core.pool import IndexPool
from alqemi.pipeline.linearize import linearize
from alqemi.rules.hjo import is_elementary, select_rule, string_args


@dataclass(frozen=True)
class TraceStep:
    """One rewrite: a Rule.name (or "elementary" for a leaf) and the bracket it rewrote."""

    rule: str
    bracket: Bracket


def identity_expand(x, pool: IndexPool, trace: Optional[List[TraceStep]] = None) -> Expr:
    """Expand every bracket in ``x``. Raises UnsupportedBracket if one cannot be expanded."""
    expr = linearize(x, pool)
    while True:
        index = next((n for n, t in enumerate(expr.terms) if _first_bracket(t) is not None), None)
        if index is None:
            return expr
        replacement = linearize(_rewrite_first(expr.terms[index], pool, trace), pool)
        expr = Expr(expr.terms[:index] + replacement.terms + expr.terms[index + 1 :])


def _first_bracket(term: Term) -> Optional[int]:
    for n, factor in enumerate(term.factors):
        if isinstance(factor, Bracket):
            return n
    return None


def _rewrite_first(term: Term, pool: IndexPool, trace) -> Expr:
    """Rewrite the term's first bracket once, splicing the result in place."""
    n = _first_bracket(term)
    result = _rewrite(term.factors[n], pool, trace)
    return Expr(
        tuple(
            Term(
                term.coeff * r.coeff,
                term.scalars + r.scalars,
                term.factors[:n] + r.factors + term.factors[n + 1 :],
                term.summed | r.summed,
            )
            for r in result
        )
    )


def _rewrite(br: Bracket, pool: IndexPool, trace) -> Expr:
    """One step on ``br``: innermost first, so its arguments are rewritten before it."""
    # Brackets are in linear form here, so each argument is exactly one bare term.
    for side in ("left", "right"):
        (arg,) = getattr(br, side).terms
        if _first_bracket(arg) is not None:
            new_arg = _rewrite_first(arg, pool, trace)
            if side == "left":
                rebuilt = Bracket(br.kind, new_arg, br.right)
            else:
                rebuilt = Bracket(br.kind, br.left, new_arg)
            # Re-linearize: the argument's new scalars (ElementaryBrackets) move out.
            return linearize(rebuilt, pool)
    if is_elementary(br):
        _record(trace, "elementary", br)
        left, right = string_args(br)
        return as_expr(ElementaryBracket(left[0], right[0]))
    rule = select_rule(br)
    _record(trace, rule.name, br)
    return rule.apply(br)


def _record(trace, rule: str, br: Bracket) -> None:
    if trace is not None:
        trace.append(TraceStep(rule, br))
