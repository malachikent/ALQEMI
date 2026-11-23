"""Numerical evaluation of ALQEMI expression objects on the oracle's Fock space.

This module turns Step 2 data objects (Delta, ElementaryBracket, Bracket, Term,
Expr) into matrices, so every later step can compare ALQEMI's symbolic output
with brute-force matrix algebra. Like oracle.py, it only READS ALQEMI's data;
it never calls ALQEMI's algebra (no rules, no multiply, no rename).

Conventions
- Free indices take their values from the assignment passed in.
- A Term's summed indices are summed over every mode allowed by their space,
  in a scope local to that Term (nested Terms inside a Bracket get their own).
- Scalars are evaluated as matrices too: Delta -> 0 or 1 times identity,
  ElementaryBracket -> the matrix anticommutator of its two operators, which
  the fermion relations (HJO 1.2.27-1.2.29) make a multiple of the identity.
"""
from __future__ import annotations

import itertools
from typing import Iterable

import numpy as np

from alqemi.core.expr import (
    Bracket,
    BracketKind,
    Delta,
    ElementaryBracket,
    Expr,
    Term,
)
from alqemi.core.indices import Index
from alqemi.core.operators import Op, OpString
from alqemi.tests.oracle import Assignment, FockSpace, anticommutator, commutator


def evaluate(fock: FockSpace, x, assignment: Assignment) -> np.ndarray:
    """Matrix of x for one assignment of its free indices."""
    if isinstance(x, Op):
        return fock.op(x, assignment)
    if isinstance(x, OpString):
        return fock.string(x, assignment)
    if isinstance(x, Delta):
        same = assignment[x.p] == assignment[x.q]
        return (1.0 if same else 0.0) * fock.identity
    if isinstance(x, ElementaryBracket):
        return anticommutator(fock.op(x.left, assignment), fock.op(x.right, assignment))
    if isinstance(x, Bracket):
        left = evaluate(fock, x.left, assignment)
        right = evaluate(fock, x.right, assignment)
        if x.kind is BracketKind.COMM:
            return commutator(left, right)
        return anticommutator(left, right)
    if isinstance(x, Term):
        return _evaluate_term(fock, x, assignment)
    if isinstance(x, Expr):
        out = np.zeros((fock.dim, fock.dim))
        for term in x.terms:
            out = out + _evaluate_term(fock, term, assignment)
        return out
    raise TypeError(f"cannot evaluate {type(x).__name__}")


def _evaluate_term(fock: FockSpace, term: Term, assignment: Assignment) -> np.ndarray:
    dummies = sorted(term.summed, key=lambda ix: (ix.name, ix.space.value))
    out = np.zeros((fock.dim, fock.dim))
    for values in itertools.product(*(fock.modes(ix.space) for ix in dummies)):
        local = dict(assignment)
        local.update(zip(dummies, values))
        mat = float(term.coeff) * fock.identity
        for s in term.scalars:
            mat = mat @ evaluate(fock, s, local)
        for f in term.factors:
            mat = mat @ evaluate(fock, f, local)
        out = out + mat
    return out


def mismatch(fock: FockSpace, lhs, rhs, extra: Iterable[Index] = ()):
    """First assignment of the free indices where lhs and rhs differ, or None.

    lhs and rhs are ALQEMI objects (anything evaluate() accepts) or callables
    taking an Assignment and returning a matrix (e.g. a direct oracle formula).
    The indices iterated over are the free indices of the ALQEMI operands plus
    ``extra`` (needed when one side is a callable).
    """
    indices = list(extra)
    for side in (lhs, rhs):
        if not callable(side):
            indices.extend(sorted(_free(side), key=lambda ix: (ix.name, ix.space.value)))
    lhs_f = lhs if callable(lhs) else (lambda g, x=lhs: evaluate(fock, x, g))
    rhs_f = rhs if callable(rhs) else (lambda g, x=rhs: evaluate(fock, x, g))
    for asg in fock.assignments(indices):
        if not np.allclose(lhs_f(asg), rhs_f(asg)):
            return {str(k): v for k, v in asg.items()}
    return None


def _free(x):
    if isinstance(x, Op):
        return {x.index}
    if isinstance(x, OpString):
        return set(x.indices)
    if isinstance(x, Delta):
        return {x.p, x.q}
    if isinstance(x, ElementaryBracket):
        return {x.left.index, x.right.index}
    return set(x.free_indices)
