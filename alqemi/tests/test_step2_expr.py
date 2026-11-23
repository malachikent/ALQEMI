"""Step 2 tests: the expression layer (scalars, brackets, terms, sums, dummies).

These tests DEFINE the Step 2 API. Still no algebra: nothing here expands a
bracket, evaluates a delta, merges like terms, or reorders operators. Step 2
builds the objects the pipeline (Linearize -> IdentityExpand -> DeltaEvaluate
-> Simplify) will pass between stages, plus the one operation that needs care
from the start: multiplying terms while freshening dummies via an IndexPool.

  alqemi/core/pool.py
    class IndexPool:                      # explicit object, no global state
        .fresh(space, avoid=()) -> Index  # avoid: iterable of NAMES (str)
            names per space: GENERAL p1, p2, ...; OCC i1, ...; VIRT a1, ...
            never returns a name already issued by this pool or listed in avoid;
            two new pools issue the same sequence (deterministic)

  alqemi/core/expr.py
    Coefficients are fractions.Fraction. int is converted; float (or any
    other type) raises TypeError.

    @dataclass(frozen=True) class Delta:          # Kronecker delta, data only
        p: Index ; q: Index
        structural equality (Delta(P, Q) != Delta(Q, P)); delta_PP and
        delta_ia are NOT evaluated here
        __str__ -> "δ_PQ"

    @dataclass(frozen=True) class ElementaryBracket:   # HJO (1.2.27)-(1.2.29)
        left: Op ; right: Op                # the anticommutator [left, right]_+
        held unevaluated (DESIGN_DECISIONS 10): [a†_P, a†_Q]_+ is not zeroed,
        [a_P, a†_Q]_+ is not turned into a delta
        __str__ -> "[a_P, a†_Q]_+"

    class BracketKind(Enum): COMM, ACOMM
    @dataclass(frozen=True) class Bracket:         # operator-valued factor
        kind: BracketKind ; left: Expr ; right: Expr
        left/right accept anything as_expr() accepts and are stored as Expr
        .free_indices -> frozenset[Index]
    def comm(a, b) -> Bracket(COMM, a, b) ; def acomm(a, b) -> Bracket(ACOMM, a, b)

    @dataclass(frozen=True) class Term:            # coeff * scalars * factors, summed
        coeff:   Fraction = 1
        scalars: tuple[Delta | ElementaryBracket, ...] = ()
        factors: tuple[OpString | Bracket, ...] = ()
        summed:  frozenset[Index] = frozenset()
      construction only normalises representation:
        - scalars commute, so they are stored in a canonical order: any
          permutation of the same scalars gives an equal Term; repeats are kept
        - factors keep their order; an Op becomes an OpString; ADJACENT
          OpStrings are concatenated; empty OpStrings (identity) are dropped
        - summed accepts any iterable of Index, stored as frozenset
        - wrong element types raise TypeError
        - every summed index must occur in the term, else ValueError
        - no shadowing: an index name bound in a Term nested inside one of this
          Term's Brackets must not be the name of any index in this Term's
          summed or free_indices, else ValueError
      .free_indices -> frozenset[Index]   # occurring here (incl. free in nested
                                          # brackets) and not summed
      .scale(c) -> Term                   # coefficient times c
      .rename(mapping: dict[Index, Index]) -> Term
          replaces indices everywhere (scalars, factors, summed, nested
          brackets); a target with a different space raises ValueError
      Term * Term is NOT supported (TypeError): products go through multiply.

    @dataclass(frozen=True) class Expr:            # a sum of terms
        terms: tuple[Term, ...] = ()        # list accepted; non-Term -> TypeError
        kept verbatim: no merging, no dropping zero coefficients, order kept
        Expr() is zero ; __len__, __iter__
        __add__ / __sub__ (concatenate, negating the right side), __neg__
        .scale(c) -> Expr ; .free_indices -> frozenset[Index]
    def as_expr(x) -> Expr
        Expr -> itself; Term -> one-term Expr; Op, OpString, Bracket ->
        Term(factors=(x,)); Delta, ElementaryBracket -> Term(scalars=(x,));
        anything else -> TypeError

    def multiply_terms(left: Term, right: Term, pool: IndexPool) -> Term
        coeff = product; scalars = both; factors = left's then right's;
        summed = union. Before combining, dummies are freshened in two passes:
        first each bound index (at any depth) of RIGHT whose name occurs
        anywhere in left, then each bound index of LEFT whose name still
        occurs anywhere in (the renamed) right. Each is renamed to
        pool.fresh(same space, avoid=all names in both terms). Free indices
        are never renamed; non-clashing dummies keep their names.
    def multiply(left: Expr, right: Expr, pool: IndexPool) -> Expr
        [multiply_terms(l, r, pool) for l in left for r in right], in that order

Numerical checks use alqemi/tests/evaluate.py, which reads these objects and
turns them into oracle matrices without using any ALQEMI algebra.
"""
import dataclasses
from fractions import Fraction

import numpy as np
import pytest

from alqemi.core.expr import (
    Bracket,
    BracketKind,
    Delta,
    ElementaryBracket,
    Expr,
    Term,
    acomm,
    as_expr,
    comm,
    multiply,
    multiply_terms,
)
from alqemi.core.indices import Index, Space
from alqemi.core.operators import OpString, ann, cre
from alqemi.core.pool import IndexPool
from alqemi.tests.evaluate import evaluate, mismatch
from alqemi.tests.oracle import FockSpace, commutator, delta

P, Q, R, S, M, N = (Index(x) for x in "PQRSMN")
k, l = Index("k"), Index("l")
i, j = Index("i", Space.OCC), Index("j", Space.OCC)
a, b = Index("a", Space.VIRT), Index("b", Space.VIRT)


def E(p, q):
    return cre(p) * ann(q)


def T(*factors, coeff=1, scalars=(), summed=()):
    return Term(coeff=coeff, scalars=tuple(scalars), factors=tuple(factors), summed=frozenset(summed))


def names(indices):
    return {ix.name for ix in indices}


@pytest.fixture(scope="module")
def fock():
    return FockSpace(n_occ=2, n_virt=2)


# =========================================================================
# IndexPool
# =========================================================================
class TestIndexPool:
    @pytest.mark.parametrize(
        "space, first, second",
        [(Space.GENERAL, "p1", "p2"), (Space.OCC, "i1", "i2"), (Space.VIRT, "a1", "a2")],
    )
    def test_naming_per_space(self, space, first, second):
        pool = IndexPool()
        assert pool.fresh(space) == Index(first, space)
        assert pool.fresh(space) == Index(second, space)

    def test_counters_are_per_space(self):
        pool = IndexPool()
        pool.fresh(Space.OCC)
        assert pool.fresh(Space.VIRT).name == "a1"
        assert pool.fresh(Space.GENERAL).name == "p1"

    def test_avoid_names(self):
        pool = IndexPool()
        assert pool.fresh(Space.OCC, avoid={"i1", "i2"}).name == "i3"

    def test_never_repeats(self):
        pool = IndexPool()
        issued = [pool.fresh(Space.GENERAL).name for _ in range(20)]
        assert len(set(issued)) == 20

    def test_pools_are_independent_and_deterministic(self):
        # No global state: a fresh pool starts over and repeats the sequence.
        p1, p2 = IndexPool(), IndexPool()
        seq1 = [p1.fresh(Space.GENERAL) for _ in range(3)]
        seq2 = [p2.fresh(Space.GENERAL) for _ in range(3)]
        assert seq1 == seq2


# =========================================================================
# Scalars: Delta and ElementaryBracket (data only)
# =========================================================================
class TestDelta:
    def test_fields_and_str(self):
        d = Delta(P, Q)
        assert (d.p, d.q) == (P, Q)
        assert str(d) == "δ_PQ"

    def test_structural_equality(self):
        assert Delta(P, Q) == Delta(P, Q)
        assert Delta(P, Q) != Delta(Q, P)

    def test_not_evaluated(self):
        # delta_PP = 1 and delta_ia = 0 are DeltaEvaluate's job (stage 3).
        assert Delta(P, P).p == P
        assert Delta(i, a).q == a

    def test_immutable_and_hashable(self):
        d = Delta(P, Q)
        assert len({d, Delta(P, Q)}) == 1
        with pytest.raises(dataclasses.FrozenInstanceError):
            d.p = Q

    def test_rejects_non_index(self):
        with pytest.raises(TypeError):
            Delta("P", Q)


class TestElementaryBracket:
    def test_fields_and_str(self):
        eb = ElementaryBracket(ann(P), cre(Q))
        assert (eb.left, eb.right) == (ann(P), cre(Q))
        assert str(eb) == "[a_P, a†_Q]_+"

    def test_vanishing_kinds_are_kept(self):
        # (1.2.27)/(1.2.28) say these are 0, but stages 1-2 keep them unevaluated.
        assert ElementaryBracket(cre(P), cre(Q)).left == cre(P)
        assert ElementaryBracket(ann(P), ann(Q)).right == ann(Q)

    def test_structural_equality_and_hash(self):
        assert ElementaryBracket(ann(P), cre(Q)) == ElementaryBracket(ann(P), cre(Q))
        assert ElementaryBracket(ann(P), cre(Q)) != ElementaryBracket(cre(Q), ann(P))
        assert len({ElementaryBracket(ann(P), cre(Q))} | {ElementaryBracket(ann(P), cre(Q))}) == 1

    def test_rejects_non_op(self):
        with pytest.raises(TypeError):
            ElementaryBracket(P, cre(Q))


# =========================================================================
# Bracket
# =========================================================================
class TestBracket:
    def test_helpers_and_coercion(self):
        br = comm(cre(P), E(Q, R))
        assert br.kind is BracketKind.COMM
        assert isinstance(br.left, Expr) and isinstance(br.right, Expr)
        assert br.left == as_expr(cre(P))
        assert br.right == as_expr(E(Q, R))
        assert acomm(cre(P), ann(Q)).kind is BracketKind.ACOMM

    def test_equals_explicit_construction(self):
        assert comm(E(P, Q), E(R, S)) == Bracket(BracketKind.COMM, as_expr(E(P, Q)), as_expr(E(R, S)))

    def test_not_expanded_or_reordered(self):
        # [A, B] and [B, A] are different objects; antisymmetry is a rule (1.8.1).
        assert comm(E(P, Q), E(R, S)) != comm(E(R, S), E(P, Q))

    def test_nested(self):
        inner = comm(E(R, S), E(M, N))
        outer = comm(E(P, Q), inner)
        assert outer.right == as_expr(inner)

    def test_free_indices(self):
        assert comm(E(P, Q), comm(E(R, S), cre(M))).free_indices == {P, Q, R, S, M}

    def test_free_indices_exclude_inner_dummies(self):
        inner_sum = as_expr(T(E(P, k), summed=[k]))
        assert comm(cre(Q), inner_sum).free_indices == {P, Q}

    def test_immutable_and_hashable(self):
        br = comm(cre(P), ann(Q))
        assert len({br, comm(cre(P), ann(Q))}) == 1
        with pytest.raises(dataclasses.FrozenInstanceError):
            br.kind = BracketKind.ACOMM

    def test_rejects_garbage(self):
        with pytest.raises(TypeError):
            comm("a_P", cre(Q))


# =========================================================================
# Term
# =========================================================================
class TestTermConstruction:
    def test_defaults_are_identity(self):
        t = Term()
        assert t.coeff == Fraction(1)
        assert t.scalars == () and t.factors == () and t.summed == frozenset()

    def test_int_coeff_becomes_fraction(self):
        t = T(cre(P), coeff=-2)
        assert isinstance(t.coeff, Fraction) and t.coeff == -2

    def test_fraction_coeff_kept_exact(self):
        assert T(cre(P), coeff=Fraction(1, 3)).coeff == Fraction(1, 3)

    @pytest.mark.parametrize("bad", [0.5, "1", None])
    def test_rejects_non_exact_coeff(self, bad):
        with pytest.raises(TypeError):
            T(cre(P), coeff=bad)

    def test_op_factor_becomes_opstring(self):
        assert T(cre(P)).factors == (OpString((cre(P),)),)

    def test_adjacent_opstrings_concatenate_in_order(self):
        t = T(ann(Q), cre(P), E(R, S))
        assert t.factors == (OpString((ann(Q), cre(P), cre(R), ann(S))),)

    def test_bracket_separates_opstrings(self):
        br = comm(cre(M), ann(N))
        t = T(E(P, Q), br, cre(R))
        assert t.factors == (E(P, Q), br, OpString((cre(R),)))

    def test_empty_opstrings_dropped(self):
        assert T(OpString(), cre(P), OpString()).factors == (OpString((cre(P),)),)

    def test_no_reordering_toward_normal_order(self):
        # (1S.2.4) contains a_R a†_P; the Term must keep that order.
        assert T(ann(R), cre(P)).factors[0].ops == (ann(R), cre(P))

    def test_scalars_commute(self):
        d1, d2 = Delta(P, Q), Delta(R, S)
        eb = ElementaryBracket(ann(M), cre(N))
        assert T(scalars=[d1, d2, eb]) == T(scalars=[eb, d2, d1])

    def test_repeated_scalars_kept(self):
        assert len(T(scalars=[Delta(P, Q), Delta(P, Q)]).scalars) == 2

    def test_scalars_not_evaluated(self):
        t = T(scalars=[Delta(P, P), Delta(i, a), ElementaryBracket(cre(P), cre(Q))])
        assert len(t.scalars) == 3

    def test_summed_accepts_iterables(self):
        assert T(E(k, P), summed=[k]).summed == frozenset({k})
        assert isinstance(Term(factors=(E(k, P),), summed={k}).summed, frozenset)

    def test_rejects_bad_factor(self):
        with pytest.raises(TypeError):
            T(Delta(P, Q))          # a scalar is not an operator factor

    def test_rejects_bad_scalar(self):
        with pytest.raises(TypeError):
            T(scalars=[cre(P)])

    def test_rejects_bad_summed(self):
        with pytest.raises(TypeError):
            Term(factors=(cre(P),), summed=frozenset({"P"}))

    def test_summed_index_must_occur(self):
        with pytest.raises(ValueError):
            T(cre(P), summed=[k])

    def test_summed_index_may_occur_only_in_scalar(self):
        assert T(cre(P), scalars=[Delta(P, k)], summed=[k]).summed == {k}

    def test_summed_index_may_occur_only_in_bracket(self):
        assert T(comm(cre(k), ann(P)), summed=[k]).summed == {k}

    def test_summed_space_is_part_of_identity(self):
        # Index("k", OCC) is not Index("k"): summing the wrong one is an error.
        with pytest.raises(ValueError):
            T(E(k, P), summed=[Index("k", Space.OCC)])

    def test_shadowing_rejected(self):
        inner = as_expr(T(E(P, k), summed=[k]))
        with pytest.raises(ValueError):          # outer sums k too
            T(cre(k), comm(cre(Q), inner), summed=[k])
        with pytest.raises(ValueError):          # outer has k free
            T(cre(k), comm(cre(Q), inner))

    def test_sibling_scopes_may_reuse_names(self):
        inner = as_expr(T(E(P, k), summed=[k]))
        t = T(comm(cre(Q), inner), comm(cre(R), inner))
        assert t.free_indices == {P, Q, R}

    def test_immutable_and_hashable(self):
        t = T(E(P, Q))
        assert len({t, T(E(P, Q))}) == 1
        with pytest.raises(dataclasses.FrozenInstanceError):
            t.coeff = Fraction(2)


class TestTermMethods:
    def test_free_indices(self):
        t = T(E(P, k), comm(cre(Q), ann(k)), scalars=[Delta(R, k)], summed=[k])
        assert t.free_indices == {P, Q, R}

    def test_scale(self):
        t = T(E(P, Q), coeff=Fraction(1, 2)).scale(-4)
        assert t.coeff == -2
        assert t.factors == (E(P, Q),)

    def test_scale_rejects_float(self):
        with pytest.raises(TypeError):
            T(E(P, Q)).scale(0.5)

    def test_rename_everywhere(self):
        inner = as_expr(T(E(Q, l), summed=[l]))
        t = T(E(P, k), comm(ann(k), inner), scalars=[Delta(k, Q)], summed=[k])
        out = t.rename({k: M, Q: N})
        expected_inner = as_expr(T(E(N, l), summed=[l]))
        assert out == T(E(P, M), comm(ann(M), expected_inner), scalars=[Delta(M, N)], summed=[M])

    def test_rename_must_keep_space(self):
        with pytest.raises(ValueError):
            T(E(i, P)).rename({i: a})

    def test_no_bare_product(self):
        with pytest.raises(TypeError):
            T(cre(P)) * T(ann(Q))


# =========================================================================
# Expr
# =========================================================================
class TestExpr:
    def test_zero(self):
        z = Expr()
        assert len(z) == 0 and list(z) == []

    def test_list_input_stored_as_tuple(self):
        assert isinstance(Expr([T(cre(P))]).terms, tuple)

    def test_rejects_non_term(self):
        with pytest.raises(TypeError):
            Expr([cre(P)])

    def test_add_concatenates_in_order(self):
        x, y = as_expr(T(cre(P))), as_expr(T(ann(Q)))
        assert (x + y).terms == (T(cre(P)), T(ann(Q)))

    def test_neg_and_sub(self):
        x, y = as_expr(T(cre(P), coeff=2)), as_expr(T(ann(Q)))
        assert (-x).terms == (T(cre(P), coeff=-2),)
        assert (x - y).terms == (T(cre(P), coeff=2), T(ann(Q), coeff=-1))

    def test_kept_verbatim(self):
        # No merging of like terms and no dropping of zeros before Simplify.
        t = T(E(P, Q))
        x = as_expr(t) + as_expr(t) + as_expr(t.scale(0))
        assert len(x) == 3

    def test_scale(self):
        x = as_expr(T(cre(P))) + as_expr(T(ann(Q), coeff=3))
        assert [t.coeff for t in x.scale(Fraction(1, 3))] == [Fraction(1, 3), 1]

    def test_free_indices(self):
        x = as_expr(T(E(P, k), summed=[k])) + as_expr(T(E(Q, k)))
        assert x.free_indices == {P, Q, k}

    def test_immutable_and_hashable(self):
        x = as_expr(T(cre(P)))
        assert len({x, as_expr(T(cre(P)))}) == 1
        with pytest.raises(dataclasses.FrozenInstanceError):
            x.terms = ()

    def test_no_bare_product(self):
        with pytest.raises(TypeError):
            as_expr(cre(P)) * as_expr(ann(Q))


class TestAsExpr:
    @pytest.mark.parametrize(
        "x, term",
        [
            (cre(P), Term(factors=(OpString((cre(P),)),))),
            (E(P, Q), Term(factors=(E(P, Q),))),
            (comm(cre(P), ann(Q)), Term(factors=(comm(cre(P), ann(Q)),))),
            (Delta(P, Q), Term(scalars=(Delta(P, Q),))),
            (ElementaryBracket(ann(P), cre(Q)), Term(scalars=(ElementaryBracket(ann(P), cre(Q)),))),
            (T(E(P, Q), coeff=5), T(E(P, Q), coeff=5)),
        ],
    )
    def test_wraps_one_term(self, x, term):
        assert as_expr(x).terms == (term,)

    def test_expr_is_returned_as_is(self):
        x = as_expr(cre(P)) + as_expr(ann(Q))
        assert as_expr(x) == x

    @pytest.mark.parametrize("bad", [P, "a_P", 1, None])
    def test_rejects_others(self, bad):
        with pytest.raises(TypeError):
            as_expr(bad)


# =========================================================================
# multiply / multiply_terms: dummy freshening through an explicit IndexPool
# =========================================================================
class TestMultiplyStructure:
    def test_no_dummies(self):
        out = multiply_terms(T(E(P, Q), coeff=2, scalars=[Delta(P, R)]), T(E(R, S), coeff=Fraction(1, 2)), IndexPool())
        assert out == T(E(P, Q), E(R, S), coeff=1, scalars=[Delta(P, R)])

    def test_factor_order_left_then_right(self):
        br = comm(cre(M), ann(N))
        out = multiply_terms(T(br), T(ann(R), cre(P)), IndexPool())
        assert out.factors == (br, OpString((ann(R), cre(P))))

    def test_non_clashing_dummies_keep_names(self):
        out = multiply_terms(T(E(P, k), summed=[k]), T(E(l, Q), summed=[l]), IndexPool())
        assert out == T(E(P, k), E(l, Q), summed=[k, l])

    def test_clashing_dummies_renamed_from_pool(self):
        out = multiply_terms(T(E(P, k), summed=[k]), T(E(k, Q), summed=[k]), IndexPool())
        assert out == T(E(P, k), E(Index("p1"), Q), summed=[k, Index("p1")])

    def test_renamed_dummy_keeps_space(self):
        out = multiply_terms(T(E(P, i), summed=[i]), T(E(i, Q), summed=[i]), IndexPool())
        assert len(out.summed) == 2
        assert all(ix.space is Space.OCC for ix in out.summed)

    def test_dummy_clashing_with_free_index_is_renamed(self):
        # left sums k; right has k FREE. The free k must survive untouched.
        out = multiply_terms(T(E(P, k), summed=[k]), T(cre(k)), IndexPool())
        assert out.free_indices == {P, k}
        assert len(out.summed) == 1 and k not in out.summed

    def test_fresh_name_avoids_names_in_both_terms(self):
        p1 = Index("p1")
        out = multiply_terms(T(E(p1, k), summed=[k]), T(E(k, Q), summed=[k]), IndexPool())
        assert out.free_indices == {p1, Q}
        assert len(out.summed) == 2

    def test_nested_bound_names_do_not_clash(self):
        # Right term has k bound inside a bracket; left has k free.
        inner = as_expr(T(E(Q, k), summed=[k]))
        out = multiply_terms(T(cre(k)), T(comm(ann(P), inner)), IndexPool())
        assert out.free_indices == {k, P, Q}

    def test_same_pool_keeps_issuing_new_names(self):
        pool = IndexPool()
        x = T(E(P, k), summed=[k])
        once = multiply_terms(x, x, pool)
        twice = multiply_terms(once, x, pool)
        assert len(names(twice.summed)) == 3

    def test_expr_distributes_in_order(self):
        left = as_expr(T(cre(P))) + as_expr(T(cre(Q)))
        right = as_expr(T(ann(R))) + as_expr(T(ann(S)))
        out = multiply(left, right, IndexPool())
        assert out.terms == (T(E(P, R)), T(E(P, S)), T(E(Q, R)), T(E(Q, S)))

    def test_expr_zero_annihilates(self):
        assert len(multiply(Expr(), as_expr(cre(P)), IndexPool())) == 0


# =========================================================================
# Oracle checks: evaluate() is the referee for every later step, so check it
# against direct matrix formulas, and check multiply against matrix products.
# =========================================================================
class TestEvaluateAgainstOracle:
    def test_elementary_bracket_is_delta(self, fock):
        # (1.2.29): [a_P, a†_Q]_+ = delta_PQ
        assert mismatch(fock, ElementaryBracket(ann(P), cre(Q)), Delta(P, Q)) is None

    @pytest.mark.parametrize("left, right", [(cre(P), cre(Q)), (ann(P), ann(Q))])
    def test_elementary_bracket_vanishing_kinds(self, fock, left, right):
        # (1.2.27), (1.2.28)
        assert mismatch(fock, ElementaryBracket(left, right), Expr()) is None

    def test_unevaluated_bracket_stands_in_for_delta(self, fock):
        # DESIGN_DECISIONS 10: [a_P, a†_Q]_+ a_R carries the same value as delta_PQ a_R.
        lhs = T(ann(R), scalars=[ElementaryBracket(ann(P), cre(Q))])
        rhs = T(ann(R), scalars=[Delta(P, Q)])
        assert mismatch(fock, lhs, rhs) is None

    def test_delta_occ_virt_is_zero(self, fock):
        assert mismatch(fock, Delta(i, a), Expr()) is None

    def test_summation_respects_space(self, fock):
        # sum_i a†_i a_i counts occupied modes only
        number_occ = T(E(i, i), summed=[i])
        direct = lambda g: sum(fock.string(E(i, i), {i: m}) for m in fock.modes(Space.OCC))
        assert mismatch(fock, number_occ, direct) is None

    def test_summed_delta_collapses_numerically(self, fock):
        # sum_k delta_Pk a†_k = a†_P (DeltaEvaluate will do this symbolically)
        assert mismatch(fock, T(cre(k), scalars=[Delta(P, k)], summed=[k]), cre(P)) is None

    def test_detects_wrong_answer(self, fock):
        assert mismatch(fock, comm(cre(P), E(Q, R)), T(cre(Q), scalars=[Delta(P, R)])) is not None

    # Golden results: LHS built as Brackets, RHS as Exprs. evaluate() must agree
    # with itself AND with the direct oracle formula.
    def test_golden_1_8_11(self, fock):
        lhs = comm(cre(P), E(Q, R))
        rhs = T(cre(Q), coeff=-1, scalars=[Delta(P, R)])
        assert mismatch(fock, lhs, rhs) is None
        direct = lambda g: commutator(fock.op(cre(P), g), fock.string(E(Q, R), g))
        assert mismatch(fock, lhs, direct) is None

    def test_golden_1_8_12(self, fock):
        assert mismatch(fock, comm(ann(P), E(Q, R)), T(ann(R), scalars=[Delta(P, Q)])) is None

    def test_golden_1_8_14(self, fock):
        lhs = comm(E(P, Q), E(R, S))
        rhs = as_expr(T(E(P, S), scalars=[Delta(Q, R)])) - as_expr(T(E(R, Q), scalars=[Delta(P, S)]))
        assert mismatch(fock, lhs, rhs) is None

    def test_golden_1_8_15(self, fock):
        lhs = comm(E(P, Q), E(R, S) * E(M, N))
        rhs = (
            as_expr(T(E(P, S), E(M, N), scalars=[Delta(Q, R)]))
            - as_expr(T(E(R, Q), E(M, N), scalars=[Delta(P, S)]))
            + as_expr(T(E(R, S), E(P, N), scalars=[Delta(Q, M)]))
            - as_expr(T(E(R, S), E(M, Q), scalars=[Delta(P, N)]))
        )
        assert mismatch(fock, lhs, rhs) is None

    def test_golden_1_8_16_nested(self, fock):
        lhs = comm(E(P, Q), comm(E(R, S), E(M, N)))
        rhs = (
            as_expr(T(E(P, N), scalars=[Delta(S, M), Delta(Q, R)]))
            - as_expr(T(E(R, Q), scalars=[Delta(S, M), Delta(P, N)]))
            - as_expr(T(E(P, S), scalars=[Delta(R, N), Delta(Q, M)]))
            + as_expr(T(E(M, Q), scalars=[Delta(R, N), Delta(P, S)]))
        )
        assert mismatch(fock, lhs, rhs) is None
        st = fock.string
        direct = lambda g: commutator(st(E(P, Q), g), commutator(st(E(R, S), g), st(E(M, N), g)))
        assert mismatch(fock, lhs, direct) is None

    def test_golden_1S_2_4_not_normal_ordered(self, fock):
        lhs = comm(cre(P) * cre(Q), ann(R) * ann(S))
        rhs = (
            as_expr(T(E(P, S), scalars=[Delta(Q, R)]))
            - as_expr(T(E(Q, S), scalars=[Delta(P, R)]))
            + as_expr(T(ann(R), cre(P), scalars=[Delta(Q, S)]))
            - as_expr(T(ann(R), cre(Q), scalars=[Delta(P, S)]))
        )
        assert mismatch(fock, lhs, rhs) is None

    def test_bracket_of_sums(self, fock):
        # [sum_k a†_k a_P, a†_Q] with the dummy living inside the bracket argument
        lhs = comm(T(E(k, P), summed=[k]), cre(Q))
        direct = lambda g: sum(
            commutator(fock.string(E(k, P), {**g, k: m}), fock.op(cre(Q), g)) for m in fock.modes(Space.GENERAL)
        )
        assert mismatch(fock, lhs, direct, extra=[P, Q]) is None


class TestMultiplyAgainstOracle:
    """multiply must equal the matrix product for every assignment of free indices."""

    CASES = {
        "plain": (T(E(P, Q), coeff=2), T(E(R, S), coeff=Fraction(-1, 2))),
        "both sum k": (T(E(P, k), summed=[k]), T(E(k, Q), summed=[k])),
        "left sums k, right has k free": (T(E(P, k), summed=[k]), T(cre(k))),
        "right sums k, left has k free": (T(ann(k)), T(E(k, P), summed=[k])),
        "occ dummies clash": (T(E(i, P), summed=[i]), T(E(Q, i), scalars=[Delta(i, R)], summed=[i])),
        "nested bound vs free": (T(cre(k)), T(comm(ann(P), T(E(Q, k), summed=[k])))),
        "bracket times sum": (T(comm(cre(P), ann(k))), T(E(k, Q), summed=[k])),
    }

    @pytest.mark.parametrize("case", list(CASES))
    def test_matches_matrix_product(self, fock, case):
        left, right = self.CASES[case]
        out = multiply_terms(left, right, IndexPool())
        free = sorted(left.free_indices | right.free_indices, key=lambda ix: ix.name)
        product = lambda g: evaluate(fock, left, g) @ evaluate(fock, right, g)
        assert out.free_indices == left.free_indices | right.free_indices
        assert len(out.summed) == len(left.summed) + len(right.summed)
        assert mismatch(fock, out, product, extra=free) is None

    def test_expr_product(self, fock):
        left = as_expr(T(E(P, k), summed=[k])) + as_expr(T(cre(Q), ann(k)))
        right = as_expr(T(E(k, R), summed=[k], coeff=3)) - as_expr(T(cre(k)))
        out = multiply(left, right, IndexPool())
        product = lambda g: evaluate(fock, left, g) @ evaluate(fock, right, g)
        assert mismatch(fock, out, product, extra=[P, Q, R, k]) is None
