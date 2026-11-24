"""Expression layer: scalars, brackets, terms and sums.

Data only, as in operators.py: nothing here expands a bracket, evaluates a delta,
merges like terms or reorders operators. Construction only normalises the
representation (commuting scalars sorted, adjacent operator strings joined).
The one real operation is ``multiply``, which freshens dummies through an
explicit IndexPool so that products never capture indices.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Dict, FrozenSet, Iterable, Iterator, Mapping, Set, Tuple, Union

from alqemi.core.indices import Index
from alqemi.core.operators import Op, OpString
from alqemi.core.pool import IndexPool
from alqemi.core.tensor import Tensor

Coeff = Union[int, Fraction]


def _as_fraction(c: Coeff) -> Fraction:
    if isinstance(c, Fraction):
        return c
    if isinstance(c, int) and not isinstance(c, bool):
        return Fraction(c)
    raise TypeError(f"coefficient must be int or Fraction, got {c!r}")


def _index_key(ix: Index) -> Tuple[str, str]:
    return (ix.name, ix.space.value, "" if ix.spin is None else ix.spin.value)


def _names(indices: Iterable[Index]) -> Set[str]:
    return {ix.name for ix in indices}


def _rename_op(op: Op, mapping: Mapping[Index, Index]) -> Op:
    return Op(op.kind, mapping.get(op.index, op.index))


# =========================================================================
# Scalars
# =========================================================================
@dataclass(frozen=True)
class Delta:
    """Kronecker delta δ_pq. Never evaluated here (that is DeltaEvaluate's job)."""

    p: Index
    q: Index

    def __post_init__(self) -> None:
        if not isinstance(self.p, Index) or not isinstance(self.q, Index):
            raise TypeError(f"Delta needs two Index arguments, got {self.p!r}, {self.q!r}")

    @property
    def indices(self) -> Tuple[Index, Index]:
        return (self.p, self.q)

    def _sort_key(self) -> tuple:
        return (0, *_index_key(self.p), *_index_key(self.q))

    def _rename(self, mapping: Mapping[Index, Index]) -> Delta:
        return Delta(mapping.get(self.p, self.p), mapping.get(self.q, self.q))

    def __str__(self) -> str:
        return f"δ_{self.p}{self.q}"


@dataclass(frozen=True)
class ElementaryBracket:
    """The elementary anticommutator [left, right]_+ of HJO (1.2.27)-(1.2.29).

    Held unevaluated through stages 1-2 (DESIGN_DECISIONS 10): not zeroed for
    two creators or two annihilators, not turned into a delta otherwise.
    """

    left: Op
    right: Op

    def __post_init__(self) -> None:
        if not isinstance(self.left, Op) or not isinstance(self.right, Op):
            raise TypeError(
                f"ElementaryBracket needs two Op arguments, got {self.left!r}, {self.right!r}"
            )

    @property
    def indices(self) -> Tuple[Index, Index]:
        return (self.left.index, self.right.index)

    def _sort_key(self) -> tuple:
        l, r = self.left, self.right
        return (1, l.kind.value, *_index_key(l.index), r.kind.value, *_index_key(r.index))

    def _rename(self, mapping: Mapping[Index, Index]) -> ElementaryBracket:
        return ElementaryBracket(_rename_op(self.left, mapping), _rename_op(self.right, mapping))

    def __str__(self) -> str:
        return f"[{self.left}, {self.right}]_+"


Scalar = Union[Delta, ElementaryBracket, Tensor]


# =========================================================================
# Brackets (operator-valued factors)
# =========================================================================
class BracketKind(Enum):
    COMM = "comm"
    ACOMM = "acomm"


@dataclass(frozen=True)
class Bracket:
    """[left, right] (COMM) or [left, right]_+ (ACOMM) of two expressions, unexpanded."""

    kind: BracketKind
    left: Expr
    right: Expr

    def __post_init__(self) -> None:
        if not isinstance(self.kind, BracketKind):
            raise TypeError(f"Bracket kind must be a BracketKind, got {self.kind!r}")
        object.__setattr__(self, "left", as_expr(self.left))
        object.__setattr__(self, "right", as_expr(self.right))

    @property
    def free_indices(self) -> FrozenSet[Index]:
        return self.left.free_indices | self.right.free_indices

    def _terms(self) -> Tuple[Term, ...]:
        return self.left.terms + self.right.terms

    def _rename(self, mapping: Mapping[Index, Index]) -> Bracket:
        return Bracket(self.kind, self.left._rename(mapping), self.right._rename(mapping))

    def __str__(self) -> str:
        suffix = "_+" if self.kind is BracketKind.ACOMM else ""
        return f"[{self.left}, {self.right}]{suffix}"


def comm(a, b) -> Bracket:
    return Bracket(BracketKind.COMM, a, b)


def acomm(a, b) -> Bracket:
    return Bracket(BracketKind.ACOMM, a, b)


Factor = Union[OpString, Bracket]


# =========================================================================
# Term
# =========================================================================
@dataclass(frozen=True)
class Term:
    """coeff * (product of scalars) * (ordered product of factors), summed over ``summed``."""

    coeff: Fraction = Fraction(1)
    scalars: Tuple[Scalar, ...] = ()
    factors: Tuple[Factor, ...] = ()
    summed: FrozenSet[Index] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "coeff", _as_fraction(self.coeff))
        object.__setattr__(self, "scalars", self._normalise_scalars(self.scalars))
        object.__setattr__(self, "factors", self._normalise_factors(self.factors))
        summed = frozenset(self.summed)
        for ix in summed:
            if not isinstance(ix, Index):
                raise TypeError(f"summed indices must be Index, got {ix!r}")
        object.__setattr__(self, "summed", summed)
        self._validate_scopes()

    @staticmethod
    def _normalise_scalars(scalars) -> Tuple[Scalar, ...]:
        scalars = tuple(scalars)
        for s in scalars:
            if not isinstance(s, (Delta, ElementaryBracket, Tensor)):
                raise TypeError(f"scalars must be Delta, ElementaryBracket or Tensor, got {s!r}")
        # Scalars commute: a canonical order makes permutations compare equal.
        return tuple(sorted(scalars, key=lambda s: s._sort_key()))

    @staticmethod
    def _normalise_factors(factors) -> Tuple[Factor, ...]:
        out = []
        for f in factors:
            if isinstance(f, Op):
                f = OpString((f,))
            if isinstance(f, OpString):
                if len(f) == 0:
                    continue  # identity
                if out and isinstance(out[-1], OpString):
                    out[-1] = out[-1] * f  # concatenation, order preserved
                else:
                    out.append(f)
            elif isinstance(f, Bracket):
                out.append(f)
            else:
                raise TypeError(f"factors must be Op, OpString or Bracket, got {f!r}")
        return tuple(out)

    def _validate_scopes(self) -> None:
        here = self._indices_here()
        missing = self.summed - here
        if missing:
            names = ", ".join(sorted(str(ix) for ix in missing))
            raise ValueError(f"summed index does not occur in the term: {names}")
        outer = _names(here)
        for f in self.factors:
            if isinstance(f, Bracket):
                for t in f._terms():
                    clash = t._bound_names() & outer
                    if clash:
                        raise ValueError(
                            f"index name(s) {sorted(clash)} bound inside a bracket shadow the enclosing term"
                        )

    # ---- index bookkeeping ------------------------------------------------
    def _indices_here(self) -> Set[Index]:
        """Indices at this level: summed ones plus everything free in scalars/factors."""
        out: Set[Index] = set()
        for s in self.scalars:
            out.update(s.indices)
        for f in self.factors:
            out.update(f.indices if isinstance(f, OpString) else f.free_indices)
        return out

    def _bound(self) -> Set[Index]:
        """Indices bound here or in any nested term."""
        out = set(self.summed)
        for f in self.factors:
            if isinstance(f, Bracket):
                for t in f._terms():
                    out |= t._bound()
        return out

    def _bound_names(self) -> Set[str]:
        return _names(self._bound())

    def _all_names(self) -> Set[str]:
        return _names(self._indices_here()) | self._bound_names()

    @property
    def all_names(self) -> FrozenSet[str]:
        """Every index name in the term, free or bound, at any depth."""
        return frozenset(self._all_names())

    @property
    def free_indices(self) -> FrozenSet[Index]:
        return frozenset(self._indices_here() - self.summed)

    # ---- rewriting --------------------------------------------------------
    def scale(self, c: Coeff) -> Term:
        return Term(self.coeff * _as_fraction(c), self.scalars, self.factors, self.summed)

    def rename(self, mapping: Mapping[Index, Index]) -> Term:
        """Replace indices everywhere; each target must keep its source's space."""
        for src, dst in mapping.items():
            if not isinstance(src, Index) or not isinstance(dst, Index):
                raise TypeError(f"rename maps Index to Index, got {src!r} -> {dst!r}")
            if src.space is not dst.space:
                raise ValueError(f"rename must keep the space: {src!r} -> {dst!r}")
            if src.spin is not dst.spin:
                raise ValueError(f"rename must keep the spin: {src!r} -> {dst!r}")
        return self._rename(mapping)

    def _rename(self, mapping: Mapping[Index, Index]) -> Term:
        def factor(f: Factor) -> Factor:
            if isinstance(f, OpString):
                return OpString(tuple(_rename_op(op, mapping) for op in f))
            return f._rename(mapping)

        return Term(
            self.coeff,
            tuple(s._rename(mapping) for s in self.scalars),
            tuple(factor(f) for f in self.factors),
            frozenset(mapping.get(ix, ix) for ix in self.summed),
        )

    def __str__(self) -> str:
        parts = []
        if self.summed:
            parts.append("Σ_{" + ",".join(sorted(str(ix) for ix in self.summed)) + "}")
        parts.extend(str(s) for s in self.scalars)
        parts.extend(str(f) for f in self.factors)
        body = " ".join(parts)
        if self.coeff == 1 and body:
            return body
        if self.coeff == -1 and body:
            return f"-{body}"
        return f"{self.coeff} {body}".rstrip()


# =========================================================================
# Expr
# =========================================================================
@dataclass(frozen=True)
class Expr:
    """A sum of terms, kept verbatim (no merging, no dropping zeros, order kept)."""

    terms: Tuple[Term, ...] = ()

    def __post_init__(self) -> None:
        terms = tuple(self.terms)
        for t in terms:
            if not isinstance(t, Term):
                raise TypeError(f"Expr terms must be Term, got {t!r}")
        object.__setattr__(self, "terms", terms)

    def __len__(self) -> int:
        return len(self.terms)

    def __iter__(self) -> Iterator[Term]:
        return iter(self.terms)

    def __add__(self, other: Expr) -> Expr:
        if not isinstance(other, Expr):
            return NotImplemented
        return Expr(self.terms + other.terms)

    def __neg__(self) -> Expr:
        return self.scale(-1)

    def __sub__(self, other: Expr) -> Expr:
        if not isinstance(other, Expr):
            return NotImplemented
        return self + (-other)

    def scale(self, c: Coeff) -> Expr:
        return Expr(tuple(t.scale(c) for t in self.terms))

    @property
    def free_indices(self) -> FrozenSet[Index]:
        out: FrozenSet[Index] = frozenset()
        for t in self.terms:
            out |= t.free_indices
        return out

    def _rename(self, mapping: Mapping[Index, Index]) -> Expr:
        return Expr(tuple(t._rename(mapping) for t in self.terms))

    def __str__(self) -> str:
        if not self.terms:
            return "0"
        out = str(self.terms[0])
        for t in self.terms[1:]:
            s = str(t)
            out += f" - {s[1:]}" if s.startswith("-") else f" + {s}"
        return out


def as_expr(x) -> Expr:
    """Wrap Op, OpString, Bracket, Delta, ElementaryBracket, Tensor or Term as an Expr."""
    if isinstance(x, Expr):
        return x
    if isinstance(x, Term):
        return Expr((x,))
    if isinstance(x, (Op, OpString, Bracket)):
        return Expr((Term(factors=(x,)),))
    if isinstance(x, (Delta, ElementaryBracket, Tensor)):
        return Expr((Term(scalars=(x,)),))
    raise TypeError(f"cannot make an Expr from {x!r}")


# =========================================================================
# Products
# =========================================================================
def freshen_dummies(
    target: Term, clash: Iterable[str], avoid: Set[str], pool: IndexPool
) -> Term:
    """Rename every bound index of ``target`` (any depth) whose name is in ``clash``.

    New names come from ``pool.fresh(space, avoid)`` and are added to ``avoid``,
    so one ``avoid`` set can be shared across several calls on the same product.
    Free indices are never renamed.
    """
    clash = set(clash)
    mapping: Dict[Index, Index] = {}
    for ix in sorted(target._bound(), key=_index_key):
        if ix.name in clash:
            new = Index(pool.fresh(ix.space, avoid).name, ix.space, ix.spin)
            avoid.add(new.name)
            mapping[ix] = new
    return target.rename(mapping) if mapping else target


def multiply_terms(left: Term, right: Term, pool: IndexPool) -> Term:
    """left * right, with clashing dummies renamed from ``pool``. Free indices never change."""
    avoid = left._all_names() | right._all_names()
    right = freshen_dummies(right, left._all_names(), avoid, pool)
    left = freshen_dummies(left, right._all_names(), avoid, pool)
    return Term(
        left.coeff * right.coeff,
        left.scalars + right.scalars,
        left.factors + right.factors,
        left.summed | right.summed,
    )


def multiply(left: Expr, right: Expr, pool: IndexPool) -> Expr:
    """Distribute: every left term times every right term, left-major order."""
    return Expr(tuple(multiply_terms(l, r, pool) for l in left for r in right))
