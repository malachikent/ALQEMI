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
  the fermion relations (HJO 1.2.27-1.2.29) make a multiple of the identity,
  Tensor -> its numerical value (from a TensorValues) times identity.
- TensorValues gives every tensor (name, ranks, swap) one random real array over
  all modes, with exactly the symmetry the Tensor declares (Step 8): antisymmetric within
  the upper and within the lower indices, plus its Swap symmetry. It reads the
  Tensor's declared data only; it never calls ALQEMI's algebra.
"""
from __future__ import annotations

import itertools
import math
from typing import Dict, Iterable, Optional, Tuple

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
from alqemi.core.tensor import Swap, Tensor
from alqemi.tests.oracle import Assignment, FockSpace, anticommutator, commutator


class TensorValues:
    """Random real values for tensors, fixed by ``seed``.

    One array per signature (name, len(upper), len(lower), swap), made the first
    time it is needed, with shape (n_modes,) * (len(upper) + len(lower)), axes
    ordered upper then lower, and the declared symmetry. So t in T1 and t in T2
    are different arrays. ``set`` fixes an array explicitly.
    """

    def __init__(self, fock: FockSpace, seed: int = 0) -> None:
        self.fock = fock
        self._rng = np.random.default_rng(seed)
        self._arrays: Dict[Tuple[str, int, int, Swap], np.ndarray] = {}

    def array(self, name: str, n_upper: int, n_lower: int, swap: Optional[Swap] = None) -> np.ndarray:
        """The array for this signature; ``swap=None`` means Swap.NONE."""
        swap = Swap.NONE if swap is None else swap
        key = (name, n_upper, n_lower, swap)
        if key in self._arrays:
            return self._arrays[key]
        n = self.fock.n_modes
        raw = self._rng.normal(size=(n,) * (n_upper + n_lower))
        arr = _antisymmetrize(raw, range(n_upper))
        arr = _antisymmetrize(arr, range(n_upper, n_upper + n_lower))
        if swap is not Swap.NONE:
            if n_upper != n_lower:
                raise ValueError("a swap symmetry needs equal upper and lower ranks")
            axes = tuple(range(n_upper, 2 * n_upper)) + tuple(range(n_upper))
            sign = 1.0 if swap is Swap.SYMMETRIC else -1.0
            arr = 0.5 * (arr + sign * arr.transpose(axes))
        self._arrays[key] = arr
        return arr

    def set(self, name: str, n_upper: int, n_lower: int, swap: Swap, arr: np.ndarray) -> None:
        self._arrays[(name, n_upper, n_lower, swap)] = np.asarray(arr, dtype=float)

    def value(self, t: Tensor, assignment: Assignment) -> float:
        arr = self.array(t.name, len(t.upper), len(t.lower), t.swap)
        return float(arr[tuple(assignment[ix] for ix in t.upper + t.lower)])


def _antisymmetrize(arr: np.ndarray, axes) -> np.ndarray:
    axes = list(axes)
    if len(axes) < 2:
        return arr
    out = np.zeros_like(arr)
    for perm in itertools.permutations(range(len(axes))):
        inversions = sum(1 for x in range(len(perm)) for y in range(x) if perm[y] > perm[x])
        order = list(range(arr.ndim))
        for k, src in enumerate(perm):
            order[axes[k]] = axes[src]
        out = out + (-1.0) ** inversions * arr.transpose(order)
    return out / math.factorial(len(axes))


def evaluate(
    fock: FockSpace, x, assignment: Assignment, tensors: Optional[TensorValues] = None
) -> np.ndarray:
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
    if isinstance(x, Tensor):
        if tensors is None:
            raise ValueError(f"tensor {x} needs a TensorValues to evaluate")
        return tensors.value(x, assignment) * fock.identity
    if isinstance(x, Bracket):
        left = evaluate(fock, x.left, assignment, tensors)
        right = evaluate(fock, x.right, assignment, tensors)
        if x.kind is BracketKind.COMM:
            return commutator(left, right)
        return anticommutator(left, right)
    if isinstance(x, Term):
        return _evaluate_term(fock, x, assignment, tensors)
    if isinstance(x, Expr):
        out = np.zeros((fock.dim, fock.dim))
        for term in x.terms:
            out = out + _evaluate_term(fock, term, assignment, tensors)
        return out
    raise TypeError(f"cannot evaluate {type(x).__name__}")


def _evaluate_term(
    fock: FockSpace, term: Term, assignment: Assignment, tensors: Optional[TensorValues]
) -> np.ndarray:
    dummies = sorted(term.summed, key=lambda ix: (ix.name, ix.space.value))
    out = np.zeros((fock.dim, fock.dim))
    for values in itertools.product(*(fock.modes(ix.space) for ix in dummies)):
        local = dict(assignment)
        local.update(zip(dummies, values))
        mat = float(term.coeff) * fock.identity
        for s in term.scalars:
            mat = mat @ evaluate(fock, s, local, tensors)
        for f in term.factors:
            mat = mat @ evaluate(fock, f, local, tensors)
        out = out + mat
    return out


def mismatch(
    fock: FockSpace, lhs, rhs, extra: Iterable[Index] = (), tensors: Optional[TensorValues] = None
):
    """First assignment of the free indices where lhs and rhs differ, or None.

    lhs and rhs are ALQEMI objects (anything evaluate() accepts) or callables
    taking an Assignment and returning a matrix (e.g. a direct oracle formula).
    The indices iterated over are the free indices of the ALQEMI operands plus
    ``extra`` (needed when one side is a callable). ``tensors`` supplies tensor
    values; callables read the same values from it.
    """
    indices = list(extra)
    for side in (lhs, rhs):
        if not callable(side):
            indices.extend(sorted(_free(side), key=lambda ix: (ix.name, ix.space.value)))
    lhs_f = lhs if callable(lhs) else (lambda g, x=lhs: evaluate(fock, x, g, tensors))
    rhs_f = rhs if callable(rhs) else (lambda g, x=rhs: evaluate(fock, x, g, tensors))
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
    if isinstance(x, Tensor):
        return set(x.upper + x.lower)
    return set(x.free_indices)
