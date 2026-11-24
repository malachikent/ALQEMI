"""Step 6 tests: pipeline stage 4, Simplify, and the whole pipeline.

These tests DEFINE the Step 6 API. Simplify is the first stage allowed to merge
terms and drop zeros, and the only one that renames dummies for display
(DESIGN_DECISIONS 12). It still never reorders operators (decision 8): two terms
are alike only if their operator strings are identical up to dummy names.

Values are checked against the oracle. Structure is checked where it is the
point of the stage: which terms merge, which do not, and the dummy names.

  alqemi/pipeline/simplify.py
    def simplify(x) -> Expr
        x: anything as_expr() accepts, with no Bracket and no ElementaryBracket
        (ValueError: run IdentityExpand and DeltaEvaluate first).
        1. Like terms merge. Two terms are alike when they are equal after
           (a) renaming their dummies, keeping each dummy's space, and
           (b) reading every delta as symmetric (delta_PQ == delta_QP).
           Operator order, free indices and coefficients are NOT normalised.
           The merged term takes the first occurrence's place, form (dummy
           names, delta orientation) and the sum of the coefficients.
        2. Terms whose coefficient is 0 are removed.
        3. relabel_dummies (below).
        Idempotent.
    def relabel_dummies(x) -> Expr
        Each term independently: its dummies, in order of first appearance
        (operators left to right, then deltas in stored order), get the first
        letters of their space not used as a FREE index name anywhere in x:
          OCC i j k l m n o ; VIRT a b c d e f g h ; GENERAL p q r s t u v w
        then i1, i2, ... (a1, ..., p1, ...) skipping names in use.
        Free indices never change. Nothing else changes.

  alqemi/pipeline/run.py
    def run(x, trace: list | None = None) -> Expr
        Linearize -> IdentityExpand -> DeltaEvaluate -> Simplify, with its own
        IndexPool. ``trace`` is passed to IdentityExpand. UnsupportedBracket
        and DanglingSum propagate.
"""
from fractions import Fraction

import pytest

from alqemi.core.expr import (
    Bracket,
    Delta,
    ElementaryBracket,
    Expr,
    Term,
    acomm,
    as_expr,
    comm,
)
from alqemi.core.indices import Index, Space
from alqemi.core.operators import OpString, ann, cre
from alqemi.pipeline.delta_evaluate import DanglingSum
from alqemi.pipeline.run import run
from alqemi.pipeline.simplify import relabel_dummies, simplify
from alqemi.rules.hjo import UnsupportedBracket
from alqemi.tests.evaluate import mismatch
from alqemi.tests.oracle import FockSpace, delta

P, Q, R, S, M, N = (Index(x) for x in "PQRSMN")
k, l = Index("k"), Index("l")
i, j = Index("i", Space.OCC), Index("j", Space.OCC)
a, b = Index("a", Space.VIRT), Index("b", Space.VIRT)
p1, p2, p3 = Index("p1"), Index("p2"), Index("p3")
i1, i2 = Index("i1", Space.OCC), Index("i2", Space.OCC)
a1 = Index("a1", Space.VIRT)
p, q, r = Index("p"), Index("q"), Index("r")
GENERAL = [P, Q, R, S, M, N]


def E(x, y):
    return cre(x) * ann(y)


def T(*factors, coeff=1, scalars=(), summed=()):
    return Term(coeff=coeff, scalars=tuple(scalars), factors=tuple(factors), summed=frozenset(summed))


def X(*terms):
    return Expr(tuple(terms))


@pytest.fixture(scope="module")
def fock():
    return FockSpace(n_occ=2, n_virt=2)


# =========================================================================
# Merging like terms
# =========================================================================
class TestMerge:
    def test_identical_terms_add(self):
        assert simplify(X(T(E(P, Q)), T(E(P, Q)))) == X(T(E(P, Q), coeff=2))

    def test_cancelling_terms_vanish(self):
        assert simplify(X(T(E(P, Q)), T(E(P, Q), coeff=-1))) == Expr()

    def test_exact_fractions(self):
        out = simplify(X(T(cre(P), coeff=Fraction(1, 3)), T(cre(P), coeff=Fraction(1, 6))))
        assert out == X(T(cre(P), coeff=Fraction(1, 2)))

    def test_merged_term_takes_first_place(self):
        out = simplify(X(T(cre(P)), T(cre(Q)), T(cre(P), coeff=2)))
        assert out == X(T(cre(P), coeff=3), T(cre(Q)))

    def test_symmetric_deltas_merge_first_orientation_kept(self):
        out = simplify(X(T(cre(R), scalars=[Delta(P, Q)]), T(cre(R), scalars=[Delta(Q, P)])))
        assert out == X(T(cre(R), coeff=2, scalars=[Delta(P, Q)]))

    def test_dummies_merge_up_to_renaming(self):
        out = simplify(X(T(E(k, P), summed=[k]), T(E(l, P), summed=[l])))
        assert out == X(T(E(p, P), coeff=2, summed=[p]))

    def test_two_dummies_merge_up_to_permutation(self):
        # sum_kl a†_k a†_l a_P  and  sum_kl a†_l a†_k a_P  are the same term
        x = X(T(cre(k), cre(l), ann(P), summed=[k, l]), T(cre(l), cre(k), ann(P), summed=[k, l]))
        out = simplify(x)
        assert len(out) == 1 and out.terms[0].coeff == 2

    def test_dummy_only_in_delta_merges(self):
        x = X(T(cre(Q), scalars=[Delta(i, P)], summed=[i]), T(cre(Q), scalars=[Delta(P, j)], summed=[j]))
        out = simplify(x)
        assert len(out) == 1 and out.terms[0].coeff == 2

    def test_dummy_space_matters(self):
        # sum over general k vs sum over occupied i: different terms
        x = X(T(cre(k), summed=[k]), T(cre(i), summed=[i]))
        assert len(simplify(x)) == 2

    def test_free_names_matter(self):
        assert len(simplify(X(T(cre(P)), T(cre(Q))))) == 2

    def test_free_vs_dummy_not_merged(self):
        assert len(simplify(X(T(cre(k), summed=[k]), T(cre(k))))) == 2

    def test_operator_order_matters(self):
        # a†_P a_Q and a_Q a†_P differ by delta_PQ; Simplify never reorders.
        x = X(T(cre(P), ann(Q)), T(ann(Q), cre(P), coeff=-1))
        assert len(simplify(x)) == 2

    def test_dummy_pattern_matters(self):
        # sum_i a†_i a_j  vs  sum_j a†_i a_j with the other index free: not alike
        x = X(T(E(i, j), summed=[i]), T(E(i, j), summed=[j]))
        assert len(simplify(x)) == 2

    def test_different_deltas_not_merged(self):
        x = X(T(cre(R), scalars=[Delta(P, Q)]), T(cre(R), scalars=[Delta(P, S)]))
        assert len(simplify(x)) == 2

    def test_zero_coefficient_dropped(self):
        assert simplify(X(T(cre(P), coeff=0), T(cre(Q)))) == X(T(cre(Q)))

    def test_empty(self):
        assert simplify(Expr()) == Expr()


class TestInputs:
    def test_rejects_brackets(self):
        with pytest.raises(ValueError):
            simplify(comm(cre(P), ann(Q)))

    def test_rejects_elementary_brackets(self):
        with pytest.raises(ValueError):
            simplify(T(cre(R), scalars=[ElementaryBracket(ann(P), cre(Q))]))

    def test_rejects_garbage(self):
        with pytest.raises(TypeError):
            simplify("a_P")

    def test_idempotent(self):
        x = X(T(E(p1, P), summed=[p1]), T(E(p2, P), summed=[p2]), T(cre(Q), scalars=[Delta(P, Q)]))
        once = simplify(x)
        assert simplify(once) == once


# =========================================================================
# Relabelling dummies (decision 12)
# =========================================================================
class TestRelabel:
    @pytest.mark.parametrize(
        "dummy, letter",
        [(p1, Index("p")), (i1, Index("i", Space.OCC)), (a1, Index("a", Space.VIRT))],
    )
    def test_first_letter_of_space(self, dummy, letter):
        out = relabel_dummies(T(cre(dummy), ann(P), summed=[dummy]))
        assert out == as_expr(T(cre(letter), ann(P), summed=[letter]))

    def test_order_of_first_appearance(self):
        out = relabel_dummies(T(cre(p2), ann(p1), summed=[p1, p2]))
        assert out == as_expr(T(cre(p), ann(q), summed=[p, q]))

    def test_operators_before_deltas(self):
        # p2 appears in the operators, p1 only in a delta: p2 -> p, p1 -> q
        out = relabel_dummies(T(cre(p2), scalars=[Delta(p1, P), Delta(p2, Q)], summed=[p1, p2]))
        assert out == as_expr(T(cre(p), scalars=[Delta(q, P), Delta(p, Q)], summed=[p, q]))

    def test_spaces_counted_separately(self):
        out = relabel_dummies(T(cre(a1), ann(i1), cre(p1), summed=[a1, i1, p1]))
        aa, ii = Index("a", Space.VIRT), Index("i", Space.OCC)
        assert out == as_expr(T(cre(aa), ann(ii), cre(p), summed=[aa, ii, p]))

    def test_skips_free_names_in_term(self):
        # free occupied i: the occupied dummy becomes j
        out = relabel_dummies(T(cre(i1), ann(i), summed=[i1]))
        assert out == as_expr(T(cre(j), ann(i), summed=[j]))

    def test_skips_free_names_anywhere_in_expr(self):
        x = X(T(cre(i1), ann(P), summed=[i1]), T(cre(i), ann(Q)))
        out = relabel_dummies(x)
        assert out.terms[0] == T(cre(j), ann(P), summed=[j])

    def test_skips_free_names_same_space(self):
        # a free general index named "p" blocks the letter p for general dummies
        out = relabel_dummies(T(cre(p1), ann(p), summed=[p1]))
        assert out == as_expr(T(cre(q), ann(p), summed=[q]))

    def test_skips_free_names_of_other_spaces(self):
        # names are blocked whatever the space: a free GENERAL index named "i"
        # blocks the letter i for OCCUPIED dummies too
        i_general = Index("i")
        out = relabel_dummies(T(cre(i1), ann(i_general), summed=[i1]))
        assert out == as_expr(T(cre(j), ann(i_general), summed=[j]))

    def test_terms_relabelled_independently(self):
        x = X(T(cre(p1), summed=[p1]), T(ann(p2), summed=[p2]))
        out = relabel_dummies(x)
        assert out == X(T(cre(p), summed=[p]), T(ann(p), summed=[p]))

    def test_user_dummy_names_relabelled_too(self):
        out = relabel_dummies(T(cre(k), ann(P), summed=[k]))
        assert out == as_expr(T(cre(p), ann(P), summed=[p]))

    def test_letters_run_out(self):
        dummies = [Index(f"p{n}") for n in range(1, 10)]
        term = Term(factors=(OpString(tuple(cre(d) for d in dummies)),), summed=frozenset(dummies))
        (out,) = relabel_dummies(term).terms
        names = [op.index.name for op in out.factors[0]]
        assert names == ["p", "q", "r", "s", "t", "u", "v", "w", "p1"]

    def test_fallback_skips_free_names(self):
        dummies = [Index(f"k{n}") for n in range(1, 10)]
        term = Term(factors=(OpString(tuple(cre(d) for d in dummies) + (ann(p1),)),), summed=frozenset(dummies))
        (out,) = relabel_dummies(term).terms
        assert [op.index.name for op in out.factors[0]][-2:] == ["p2", "p1"]

    def test_free_indices_untouched(self):
        x = T(E(P, p1), scalars=[Delta(Q, R)], summed=[p1])
        out = relabel_dummies(x)
        assert out.free_indices == {P, Q, R}

    def test_nothing_else_changes(self):
        x = X(T(E(P, Q), coeff=0), T(E(P, Q)))
        assert relabel_dummies(x) == x


@pytest.mark.parametrize(
    "x",
    [
        X(T(E(k, P), summed=[k]), T(E(l, P), summed=[l]), T(cre(Q), coeff=-1)),
        X(T(cre(R), scalars=[Delta(P, Q)]), T(cre(R), scalars=[Delta(Q, P)], coeff=-3)),
        X(T(cre(k), cre(l), ann(P), summed=[k, l]), T(cre(l), cre(k), ann(P), summed=[k, l])),
        X(T(cre(i1), ann(i), summed=[i1]), T(cre(i), ann(i1), summed=[i1], coeff=Fraction(1, 2))),
        X(T(cre(Q), scalars=[Delta(i, P)], summed=[i]), T(cre(Q), scalars=[Delta(P, j)], summed=[j])),
    ],
    ids=["dummies", "deltas", "permutation", "free occ", "delta-only dummy"],
)
def test_simplify_value_unchanged(fock, x):
    assert mismatch(fock, simplify(x), x) is None


# =========================================================================
# The whole pipeline
# =========================================================================
class TestRun:
    def test_golden_1_8_14(self, fock):
        out = run(comm(E(P, Q), E(R, S)))
        st = fock.string
        golden = lambda g: delta(Q, R, g) * st(E(P, S), g) - delta(P, S, g) * st(E(R, Q), g)
        assert mismatch(fock, out, golden, extra=[P, Q, R, S]) is None
        assert len(out) == 2

    def test_golden_1_8_15(self, fock):
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

    def test_golden_1_8_16(self, fock):
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

    def test_golden_1S_2_4(self, fock):
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

    def test_commutator_with_number_operator_is_zero(self):
        # [E_PQ, sum_p a†_p a_p] = 0, now as an empty Expr, not just numerically
        assert run(comm(E(P, Q), T(E(k, k), summed=[k]))) == Expr()

    def test_operator_commutes_with_itself(self):
        # [E_PQ, E_PQ] = delta_QP E_PQ - delta_PQ E_PQ = 0: symmetric deltas cancel
        assert run(comm(E(P, Q), E(P, Q))) == Expr()

    def test_jacobi_1_8_17(self, fock):
        A, B, C = E(P, Q), E(R, S), E(M, N)
        total = as_expr(comm(A, comm(B, C))) + as_expr(comm(C, comm(A, B))) + as_expr(comm(B, comm(C, A)))
        out = run(total)
        assert mismatch(fock, out, Expr()) is None
        assert out == Expr()  # every term cancels symbolically, not just numerically

    def test_dummies_relabelled_to_letters(self, fock):
        # [sum_k E_kP, sum_k E_kQ]: the pool's p1 is relabelled for display.
        # (Not [sum_k E_kP, sum_k E_Qk]: that contains sum_kl delta_kl, the
        # orbital count, and raises DanglingSum.)
        x = comm(T(E(k, P), summed=[k]), T(E(k, Q), summed=[k]))
        out = run(x)
        names = {ix.name for t in out for ix in t.summed}
        assert names <= {"p", "q"}
        assert mismatch(fock, out, x) is None

    def test_occupied_number_operator(self, fock):
        x = comm(E(P, Q), T(E(i, i), summed=[i]))
        out = run(x)
        assert mismatch(fock, out, x) is None
        assert all(t.summed == {i} for t in out)

    def test_cc_like_singles(self, fock):
        # [a†_P a_Q, sum_ia a†_a a_i]: general P, Q against an occ-virt excitation
        x = comm(E(P, Q), T(E(a, i), summed=[a, i]))
        out = run(x)
        assert mismatch(fock, out, x) is None

    def test_trace_passed_through(self):
        trace = []
        run(comm(cre(P), E(Q, R)), trace)
        assert [s.rule for s in trace] == ["1.8.6", "elementary", "elementary"]

    def test_unsupported_propagates(self):
        with pytest.raises(UnsupportedBracket):
            run(comm(ann(P), ann(Q)))

    def test_dangling_propagates(self):
        with pytest.raises(DanglingSum):
            run(T(comm(ann(k), E(k, P)), summed=[k]))

    def test_anticommutator_pipeline(self, fock):
        x = acomm(ann(P), cre(Q) * ann(R) * cre(S))
        out = run(x)
        assert mismatch(fock, out, x) is None
