"""Step 5 tests: pipeline stage 3, DeltaEvaluate.

These tests DEFINE the Step 5 API. Deltas are born here (DESIGN_DECISIONS 7, 10):
every ElementaryBracket left by IdentityExpand is evaluated by HJO (1.2.27)-(1.2.29),
then deltas are evaluated and sums over dummies collapsed. Operators are never
reordered and like terms are never merged (that is Simplify, stage 4).

Values are checked against the oracle. Structure is checked only where it is a
representation choice the oracle cannot see: which way round a delta is written,
which dummy a collapse removes, and what is left unevaluated.

  alqemi/core/indices.py (addition)
    Space.contains(other) -> bool
        every index of space ``other`` can be an index of this space:
        GENERAL contains all three; OCC contains only OCC; VIRT only VIRT

  alqemi/pipeline/delta_evaluate.py
    class DanglingSum(Exception)  .term -> the Term ; .index -> the Index
        a summed index no longer occurs after evaluation (e.g. sum_k delta_kk a_P,
        which is (number of orbitals) * a_P). There is no symbol for a space's
        dimension, so this raises instead of guessing.
    def delta_evaluate(x) -> Expr
        x: anything as_expr() accepts, with no Bracket anywhere (ValueError).
        Per term, in this order:
        1. ElementaryBracket [x, y]_+:
             two creators or two annihilators -> 0     (1.2.27), (1.2.28)
             otherwise -> Delta(x.index, y.index)      (1.2.29)
           The delta keeps the order of the bracket (left index first), which is
           how HJO writes every printed result: [a_Q, a†_R]_+ -> delta_QR (1.8.14).
        2. Repeat until nothing changes, always taking the first delta in the
           term's scalar order that one of these applies to:
           a. delta_pp (same Index) -> 1, removed
           b. delta_pq with non-overlapping spaces (OCC vs VIRT) -> 0
           c. delta_pq with a summed index whose space contains the other's:
              that index is removed from the sum and replaced by the other one
              everywhere in the term. If both qualify, the one with the wider
              space goes; if equal, q (the right index) goes.
              Not collapsible: a summed index narrower than the other, e.g.
              sum_i delta_iP with i OCC and P GENERAL free; it stays as is.
        3. Deltas with the same value (same unordered pair) are kept once.
        4. A summed index that no longer occurs raises DanglingSum.
        A term with a zero factor (steps 1, 2b) is removed. A term whose
        coefficient is 0 is kept (Simplify drops it). Term order is kept.
        Free deltas are never used to substitute: delta_QR a†_P a_S stays as is.
        Idempotent.
"""
from fractions import Fraction

import pytest

from alqemi.core.expr import (
    Bracket,
    Delta,
    ElementaryBracket,
    Expr,
    Term,
    as_expr,
    comm,
)
from alqemi.core.indices import Index, Space
from alqemi.core.operators import OpString, ann, cre
from alqemi.core.pool import IndexPool
from alqemi.pipeline.delta_evaluate import DanglingSum, delta_evaluate
from alqemi.pipeline.identity_expand import identity_expand
from alqemi.tests.evaluate import mismatch
from alqemi.tests.oracle import FockSpace, commutator, delta

P, Q, R, S, M, N = (Index(x) for x in "PQRSMN")
k, l = Index("k"), Index("l")
i, j = Index("i", Space.OCC), Index("j", Space.OCC)
a, b = Index("a", Space.VIRT), Index("b", Space.VIRT)
GENERAL = [P, Q, R, S, M, N]


def E(p, q):
    return cre(p) * ann(q)


def T(*factors, coeff=1, scalars=(), summed=()):
    return Term(coeff=coeff, scalars=tuple(scalars), factors=tuple(factors), summed=frozenset(summed))


def EB(x, y):
    return ElementaryBracket(x, y)


def run(x):
    """The pipeline so far: Linearize -> IdentityExpand -> DeltaEvaluate."""
    return delta_evaluate(identity_expand(x, IndexPool()))


def deltas(expr):
    return [s for t in expr for s in t.scalars if isinstance(s, Delta)]


def elementary(expr):
    return [s for t in expr for s in t.scalars if isinstance(s, ElementaryBracket)]


@pytest.fixture(scope="module")
def fock():
    return FockSpace(n_occ=2, n_virt=2)


# =========================================================================
# Space.contains
# =========================================================================
class TestSpaceContains:
    @pytest.mark.parametrize(
        "outer, inner, expected",
        [
            (Space.GENERAL, Space.GENERAL, True),
            (Space.GENERAL, Space.OCC, True),
            (Space.GENERAL, Space.VIRT, True),
            (Space.OCC, Space.OCC, True),
            (Space.VIRT, Space.VIRT, True),
            (Space.OCC, Space.GENERAL, False),
            (Space.VIRT, Space.GENERAL, False),
            (Space.OCC, Space.VIRT, False),
            (Space.VIRT, Space.OCC, False),
        ],
    )
    def test_contains(self, outer, inner, expected):
        assert outer.contains(inner) is expected

    def test_contains_implies_overlaps(self):
        for x in Space:
            for y in Space:
                if x.contains(y):
                    assert x.overlaps(y)


# =========================================================================
# Step 1: elementary brackets, (1.2.27)-(1.2.29)
# =========================================================================
class TestElementary:
    @pytest.mark.parametrize("x, y", [(cre(P), cre(Q)), (ann(P), ann(Q)), (cre(P), cre(P))])
    def test_same_kind_vanishes(self, x, y):
        # (1.2.27), (1.2.28)
        assert delta_evaluate(T(ann(R), scalars=[EB(x, y)])) == Expr()

    def test_creator_first_gives_delta_in_bracket_order(self):
        # (1.2.29): [a†_P, a_Q]_+ = delta_PQ
        out = delta_evaluate(T(ann(R), scalars=[EB(cre(P), ann(Q))]))
        assert deltas(out) == [Delta(P, Q)]

    def test_annihilator_first_keeps_bracket_order(self):
        # [a_Q, a†_R]_+ -> delta_QR, as HJO writes it in (1.8.14)
        out = delta_evaluate(T(cre(P), scalars=[EB(ann(Q), cre(R))]))
        assert deltas(out) == [Delta(Q, R)]

    def test_occ_virt_elementary_vanishes(self):
        assert delta_evaluate(T(cre(P), scalars=[EB(ann(i), cre(a))])) == Expr()

    def test_no_elementary_left(self):
        x = Expr((T(ann(R), scalars=[EB(ann(P), cre(Q))]), T(cre(S), scalars=[EB(cre(P), ann(M))])))
        assert elementary(delta_evaluate(x)) == []

    def test_operators_untouched(self):
        out = delta_evaluate(T(ann(R), cre(S), scalars=[EB(ann(P), cre(Q))]))
        assert out.terms[0].factors == (OpString((ann(R), cre(S))),)


# =========================================================================
# Step 2: deltas
# =========================================================================
class TestDeltas:
    def test_same_index_is_one(self):
        out = delta_evaluate(T(cre(Q), scalars=[Delta(P, P)]))
        assert out == as_expr(T(cre(Q)))

    def test_same_occ_index_is_one(self):
        assert delta_evaluate(T(cre(Q), scalars=[Delta(i, i)])) == as_expr(T(cre(Q)))

    def test_same_name_different_space_is_not_one(self):
        # Index("p") and Index("p", OCC) are different indices.
        p_occ = Index("P", Space.OCC)
        assert deltas(delta_evaluate(T(cre(Q), scalars=[Delta(P, p_occ)]))) == [Delta(P, p_occ)]

    def test_occ_virt_vanishes(self):
        assert delta_evaluate(T(cre(Q), scalars=[Delta(i, a)])) == Expr()

    def test_free_delta_kept_and_not_substituted(self):
        # delta_QR a†_P a_S: free deltas never rename operators.
        x = T(E(P, S), scalars=[Delta(Q, R)])
        assert delta_evaluate(x) == as_expr(x)

    def test_free_general_occ_delta_kept(self):
        x = T(cre(P), scalars=[Delta(P, i)])
        assert delta_evaluate(x) == as_expr(x)

    def test_duplicate_deltas_kept_once(self):
        out = delta_evaluate(T(cre(R), scalars=[Delta(P, Q), Delta(P, Q)]))
        assert deltas(out) == [Delta(P, Q)]

    def test_reversed_duplicate_kept_once(self):
        out = delta_evaluate(T(cre(R), scalars=[Delta(P, Q), Delta(Q, P)]))
        assert len(deltas(out)) == 1


class TestCollapse:
    def test_general_dummy_collapses(self):
        # sum_k delta_kQ a†_k = a†_Q
        out = delta_evaluate(T(cre(k), scalars=[Delta(k, Q)], summed=[k]))
        assert out == as_expr(T(cre(Q)))

    def test_dummy_on_the_right(self):
        out = delta_evaluate(T(cre(k), scalars=[Delta(Q, k)], summed=[k]))
        assert out == as_expr(T(cre(Q)))

    def test_general_dummy_onto_occ_free(self):
        # sum_p delta_pi a†_p = a†_i  (p GENERAL contains i OCC)
        p = Index("p")
        out = delta_evaluate(T(cre(p), scalars=[Delta(p, i)], summed=[p]))
        assert out == as_expr(T(cre(i)))

    def test_narrow_dummy_does_not_collapse(self):
        # sum_i delta_iP a†_i: P may be virtual, so this is not a†_P.
        x = T(cre(i), scalars=[Delta(i, P)], summed=[i])
        assert delta_evaluate(x) == as_expr(x)

    def test_occ_dummy_against_virt_vanishes(self):
        assert delta_evaluate(T(cre(i), scalars=[Delta(i, a)], summed=[i])) == Expr()

    def test_both_dummies_same_space_right_goes(self):
        out = delta_evaluate(T(E(i, j), scalars=[Delta(i, j)], summed=[i, j]))
        assert out == as_expr(T(E(i, i), summed=[i]))

    def test_both_dummies_wider_goes(self):
        p = Index("p")
        out = delta_evaluate(T(E(p, i), scalars=[Delta(i, p)], summed=[i, p]))
        assert out == as_expr(T(E(i, i), summed=[i]))

    def test_substitution_reaches_everything(self, fock):
        # k is replaced in operators, in other deltas, and in elementary brackets.
        # (Which of P, N, S it becomes depends on the scalars' canonical order.)
        x = T(cre(k), ann(M), scalars=[Delta(k, P), Delta(k, N), EB(ann(k), cre(S))], summed=[k])
        out = delta_evaluate(x)
        assert len(out) == 1
        (t,) = out.terms
        assert not t.summed and k not in t.free_indices
        assert len(deltas(out)) == 2 and elementary(out) == []
        assert t.factors[0][0].index in {P, N, S}
        assert mismatch(fock, out, x) is None

    def test_chain_collapses(self):
        # sum_k delta_Pk delta_kQ a†_k = delta_PQ a†_P (or a†_Q): one free delta left
        out = delta_evaluate(T(cre(k), scalars=[Delta(P, k), Delta(k, Q)], summed=[k]))
        (t,) = out.terms
        assert not t.summed and len(deltas(out)) == 1

    def test_collapse_makes_new_delta_one(self):
        # sum_k delta_kP delta_kP a_Q -> delta_PP a_Q -> a_Q
        out = delta_evaluate(T(ann(Q), cre(k), scalars=[Delta(k, P), Delta(P, k)], summed=[k]))
        assert deltas(out) == []

    def test_collapse_makes_new_delta_vanish(self):
        # sum_p delta_pi delta_pa a†_p = delta_ia a†_i = 0
        p = Index("p")
        assert delta_evaluate(T(cre(p), scalars=[Delta(p, i), Delta(p, a)], summed=[p])) == Expr()

    def test_dummy_only_in_delta(self):
        # sum_k delta_kP a_Q = a_Q
        assert delta_evaluate(T(ann(Q), scalars=[Delta(k, P)], summed=[k])) == as_expr(T(ann(Q)))

    def test_dangling_sum_raises(self):
        # sum_k delta_kk a_P = (number of orbitals) a_P: no symbol for that yet.
        x = T(ann(P), scalars=[Delta(k, k)], summed=[k])
        with pytest.raises(DanglingSum) as err:
            delta_evaluate(x)
        assert err.value.index == k

    def test_other_dummies_untouched(self):
        out = delta_evaluate(T(cre(k), ann(l), scalars=[Delta(k, P)], summed=[k, l]))
        assert out == as_expr(T(cre(P), ann(l), summed=[l]))


# =========================================================================
# Whole-expression behaviour
# =========================================================================
class TestExpr:
    def test_vanishing_terms_removed_order_kept(self):
        x = Expr((
            T(cre(P), coeff=2),
            T(cre(Q), scalars=[EB(cre(R), cre(S))]),
            T(cre(R), coeff=3),
        ))
        assert delta_evaluate(x) == Expr((T(cre(P), coeff=2), T(cre(R), coeff=3)))

    def test_zero_coefficient_kept(self):
        x = T(cre(P), coeff=0)
        assert delta_evaluate(x) == as_expr(x)

    def test_no_merging(self):
        x = Expr((T(cre(P), scalars=[Delta(Q, Q)]), T(cre(P))))
        assert len(delta_evaluate(x)) == 2

    def test_rejects_brackets(self):
        with pytest.raises(ValueError):
            delta_evaluate(comm(cre(P), ann(Q)))

    def test_rejects_garbage(self):
        with pytest.raises(TypeError):
            delta_evaluate("a_P")

    def test_terms_without_scalars_pass_through(self):
        x = Expr((T(E(P, Q), coeff=Fraction(1, 2)), T(E(k, k), summed=[k])))
        assert delta_evaluate(x) == x

    def test_idempotent(self):
        x = run(comm(E(P, Q), comm(E(R, S), E(M, N))))
        assert delta_evaluate(x) == x


# =========================================================================
# Oracle: DeltaEvaluate never changes the value
# =========================================================================
p = Index("p")
ORACLE_CASES = {
    "elementary kinds": Expr((
        T(ann(R), scalars=[EB(ann(P), cre(Q))]),
        T(cre(S), scalars=[EB(cre(P), ann(Q))]),
        T(cre(S), scalars=[EB(cre(P), cre(Q))]),
    )),
    "general dummy": T(cre(k), ann(R), scalars=[Delta(k, Q)], summed=[k]),
    "general dummy onto occ": T(cre(p), scalars=[Delta(p, i)], summed=[p]),
    "narrow dummy kept": T(cre(i), scalars=[Delta(i, P)], summed=[i]),
    "two dummies": T(E(i, j), scalars=[Delta(i, j)], summed=[i, j]),
    "wider dummy": T(E(p, i), scalars=[Delta(i, p)], summed=[i, p]),
    "chain": T(cre(k), scalars=[Delta(P, k), Delta(k, Q)], summed=[k]),
    "duplicates": T(cre(R), scalars=[Delta(P, Q), Delta(Q, P)]),
    "elementary with dummy": T(cre(M), scalars=[EB(ann(k), cre(P)), Delta(k, Q)], summed=[k]),
}


@pytest.mark.parametrize("case", list(ORACLE_CASES))
def test_value_unchanged(fock, case):
    x = ORACLE_CASES[case]
    assert mismatch(fock, delta_evaluate(x), x) is None


# =========================================================================
# The pipeline through stage 3 on the golden results
# =========================================================================
class TestGolden:
    """Values against the oracle; deltas as HJO writes them; term counts as printed."""

    def test_1_8_11(self, fock):
        out = run(comm(cre(P), E(Q, R)))
        golden = lambda g: -delta(P, R, g) * fock.op(cre(Q), g)
        assert mismatch(fock, out, golden, extra=[P, Q, R]) is None
        assert len(out) == 1 and deltas(out) == [Delta(P, R)]

    def test_1_8_12(self, fock):
        out = run(comm(ann(P), E(Q, R)))
        golden = lambda g: delta(P, Q, g) * fock.op(ann(R), g)
        assert mismatch(fock, out, golden, extra=[P, Q, R]) is None
        assert len(out) == 1 and deltas(out) == [Delta(P, Q)]

    def test_1_8_14(self, fock):
        out = run(comm(E(P, Q), E(R, S)))
        st = fock.string
        golden = lambda g: delta(Q, R, g) * st(E(P, S), g) - delta(P, S, g) * st(E(R, Q), g)
        assert mismatch(fock, out, golden, extra=[P, Q, R, S]) is None
        assert len(out) == 2 and deltas(out) == [Delta(Q, R), Delta(P, S)]

    def test_1_8_15(self, fock):
        out = run(comm(E(P, Q), E(R, S) * E(M, N)))
        st = fock.string
        golden = lambda g: (
            delta(Q, R, g) * st(E(P, S) * E(M, N), g)
            - delta(P, S, g) * st(E(R, Q) * E(M, N), g)
            + delta(Q, M, g) * st(E(R, S) * E(P, N), g)
            - delta(P, N, g) * st(E(R, S) * E(M, Q), g)
        )
        assert mismatch(fock, out, golden, extra=GENERAL) is None
        assert len(out) == 4
        assert deltas(out) == [Delta(Q, R), Delta(P, S), Delta(Q, M), Delta(P, N)]

    def test_1_8_16(self, fock):
        out = run(comm(E(P, Q), comm(E(R, S), E(M, N))))
        st = fock.string
        golden = lambda g: (
            delta(S, M, g) * delta(Q, R, g) * st(E(P, N), g)
            - delta(S, M, g) * delta(P, N, g) * st(E(R, Q), g)
            - delta(R, N, g) * delta(Q, M, g) * st(E(P, S), g)
            + delta(R, N, g) * delta(P, S, g) * st(E(M, Q), g)
        )
        assert mismatch(fock, out, golden, extra=GENERAL) is None
        assert len(out) == 4
        assert [set(t.scalars) for t in out] == [
            {Delta(S, M), Delta(Q, R)},
            {Delta(S, M), Delta(P, N)},
            {Delta(R, N), Delta(Q, M)},
            {Delta(R, N), Delta(P, S)},
        ]

    def test_1S_2_4(self, fock):
        out = run(comm(cre(P) * cre(Q), ann(R) * ann(S)))
        st = fock.string
        golden = lambda g: (
            delta(Q, R, g) * st(cre(P) * ann(S), g)
            - delta(P, R, g) * st(cre(Q) * ann(S), g)
            + delta(Q, S, g) * st(ann(R) * cre(P), g)
            - delta(P, S, g) * st(ann(R) * cre(Q), g)
        )
        assert mismatch(fock, out, golden, extra=[P, Q, R, S]) is None
        assert len(out) == 4
        assert set(deltas(out)) == {Delta(Q, R), Delta(P, R), Delta(Q, S), Delta(P, S)}


class TestPipelineWithDummies:
    def test_commutator_with_general_number_operator_vanishes(self, fock):
        # [E_PQ, sum_p a†_p a_p] = 0: after stage 3 the two terms cancel numerically
        # (merging them is Simplify's job).
        out = run(comm(E(P, Q), T(E(p, p), summed=[p])))
        assert mismatch(fock, out, Expr()) is None
        assert all(not t.summed for t in out)

    def test_commutator_with_occupied_number_operator(self, fock):
        # [E_PQ, sum_i a†_i a_i]: P, Q are general, so sum_i delta_Qi cannot collapse.
        x = comm(E(P, Q), T(E(i, i), summed=[i]))
        out = run(x)
        assert mismatch(fock, out, x) is None
        assert all(t.summed == {i} for t in out)

    def test_occ_virt_excitation(self, fock):
        # [a†_a a_i, a†_j a_b]: deltas between occ and virt indices vanish.
        x = comm(E(a, i), E(j, b))
        out = run(x)
        assert mismatch(fock, out, x) is None
        assert all(d.p.space.overlaps(d.q.space) for d in deltas(out))

    def test_dangling_through_pipeline(self):
        # sum_k [a_k, a†_k a_P] = sum_k delta_kk a_P: the orbital count has no symbol.
        with pytest.raises(DanglingSum):
            run(T(comm(ann(k), E(k, P)), summed=[k]))
