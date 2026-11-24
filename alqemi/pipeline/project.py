"""Fermi-vacuum projections: <HF| X |HF> and <mu_n| X |HF>, without Wick's theorem.

|mu> = tau_mu |HF> and <mu| = <HF| tau_mu^dagger (HJO's CC notation), so every
projection is a vacuum expectation value. It is evaluated in four steps:

1. ``run``: brackets are expanded.
2. Split: each summed GENERAL index becomes an OCC copy and a VIRT copy of its term
   (DESIGN_DECISIONS, Step 9). Afterwards every operator kills exactly one side of
   the vacuum: a_a (virtual) and a+_i (occupied) kill |HF>; a+_a and a_i kill <HF|.
3. Absorb killers. For the rightmost |HF>-killer K of a string X K Y,
       <HF| X K Y |HF> = <HF| X [K, Y](+) |HF>
   because K|HF> = 0: an anticommutator if len(Y) is odd, else a commutator, so the
   bracket is always rank-reducing (decision 6). With no |HF>-killer every operator
   kills <HF|, and a killer at the end kills |HF>: both terms are 0. The bracket is
   evaluated by IdentityExpand and DeltaEvaluate, and the step repeats until no
   operators remain.
4. Simplify, once (it is the costly stage, so it is not run inside the loop).
"""
from __future__ import annotations

import itertools
from typing import List, Optional

from alqemi.builders import excitation
from alqemi.core.expr import Bracket, BracketKind, Expr, Term, as_expr, multiply
from alqemi.core.indices import Space
from alqemi.core.operators import Kind, Op, OpString
from alqemi.core.pool import IndexPool
from alqemi.pipeline.delta_evaluate import delta_evaluate
from alqemi.pipeline.identity_expand import identity_expand
from alqemi.pipeline.run import run
from alqemi.pipeline.simplify import simplify


def vacuum_expectation(x) -> Expr:
    """<HF| x |HF> as operator-free, simplified terms.

    Raises ValueError for a free GENERAL index (its value depends on its space);
    UnsupportedBracket and DanglingSum propagate.
    """
    expr = run(x)
    for term in expr:
        general = sorted(ix.name for ix in term.free_indices if ix.space is Space.GENERAL)
        if general:
            raise ValueError(f"free general index {general[0]}: give it the OCC or VIRT space")
    pool = IndexPool()
    pending: List[Term] = [copy for term in expr for copy in _split(term, pool)]
    done: List[Term] = []
    while pending:
        following: List[Term] = []
        for term in pending:
            if not term.factors:
                done.append(term)
                continue
            step = _absorb_killer(term)
            if step is not None:  # None: the term is 0
                following.extend(delta_evaluate(identity_expand(step, pool)).terms)
        pending = following
    return simplify(Expr(tuple(done)))


def projection(n: int, x) -> Expr:
    """<mu_n| x |HF> = <HF| tau_mu^dagger x |HF>; n = 0 gives <HF| x |HF>.

    The result's free indices are those of ``excitation(n)``. Raises ValueError if a
    free index of ``x`` has one of their names.
    """
    if n == 0:
        return vacuum_expectation(x)
    (tau,) = excitation(n).terms  # validates n
    (string,) = tau.factors
    dagger = OpString(tuple(op.adjoint() for op in reversed(string)))
    xe = as_expr(x)
    clash = {ix.name for ix in tau.free_indices} & {ix.name for ix in xe.free_indices}
    if clash:
        raise ValueError(f"free indices {sorted(clash)} of x clash with the projection's own")
    return vacuum_expectation(multiply(as_expr(dagger), xe, IndexPool()))


# ---- internals --------------------------------------------------------------------
def _split(term: Term, pool: IndexPool) -> List[Term]:
    """One copy of ``term`` per OCC/VIRT choice for each summed GENERAL index."""
    general = sorted((ix for ix in term.summed if ix.space is Space.GENERAL), key=lambda ix: ix.name)
    if not general:
        return [term]
    copies = []
    avoid = set(term.all_names)
    for spaces in itertools.product((Space.OCC, Space.VIRT), repeat=len(general)):
        mapping = {}
        for ix, space in zip(general, spaces):
            new = pool.fresh(space, avoid)
            avoid.add(new.name)
            mapping[ix] = new
        # Term.rename keeps spaces; here narrowing the space is the point.
        copies.append(term._rename(mapping))
    return copies


def _kills_ket(op: Op) -> bool:
    """a_a (virtual) and a+_i (occupied) annihilate |HF>."""
    return (op.kind is Kind.ANN) == (op.index.space is Space.VIRT)


def _absorb_killer(term: Term) -> Optional[Term]:
    """The term with its rightmost |HF>-killer absorbed into a bracket, or None if 0."""
    (string,) = term.factors  # bracket-free and merged: one operator string
    ops = list(string)
    killers = [n for n, op in enumerate(ops) if _kills_ket(op)]
    if not killers or killers[-1] == len(ops) - 1:
        return None
    n = killers[-1]
    rest = OpString(tuple(ops[n + 1 :]))
    kind = BracketKind.ACOMM if len(rest) % 2 else BracketKind.COMM
    bracket = Bracket(kind, Term(factors=(OpString((ops[n],)),)), Term(factors=(rest,)))
    return Term(term.coeff, term.scalars, (OpString(tuple(ops[:n])), bracket), term.summed)
