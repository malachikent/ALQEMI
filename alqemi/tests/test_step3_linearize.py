"""Step 3 tests: pipeline stage 1, Linearize.

These tests DEFINE the Step 3 API. Linearize uses only the bilinearity of the
bracket definitions (1.8.1)/(1.8.2): for c-number coefficients and scalars,

    [sum_i c_i s_i A_i, sum_j d_j t_j B_j](+) = sum_ij c_i d_j s_i t_j [A_i, B_j](+)

and summations over dummies come out of a bracket the same way. Deltas and
elementary anticommutators are c-numbers, so they come out as scalars. No
bracket is expanded (that is IdentityExpand, stage 2), no delta or elementary
bracket is evaluated, no like terms are merged, nothing is reordered.

  alqemi/pipeline/__init__.py          (new package, may be empty)
  alqemi/pipeline/linearize.py
    def linearize(x, pool: IndexPool) -> Expr
        x: anything as_expr() accepts. The value is unchanged.
        Output is in LINEAR FORM (see is_linear):
          - every Bracket, at any depth, has left and right each equal to ONE
            "bare" Term: coeff 1, no scalars, no summed. Its factors are
            OpStrings and (bare, recursively) Brackets, in their original order
          - all coefficients, scalars and summed indices sit on the top-level
            terms
          - a bare argument with no factors (the identity) is kept, e.g.
            [A, delta_PQ] -> delta_PQ [A, 1]. Evaluating [A, 1] is stage 2's job
            ((1.8.5) with n = 0); Linearize does not do it
        Bracket kind (COMM / ACOMM) is preserved.
        Term order: each input term in turn; within a term, its factors'
        expansions combine left-major (like multiply); within a bracket, the
        left argument's terms are the outer loop and the right's the inner.
        An argument that is zero (Expr()) makes that product vanish (empty sum);
        a coefficient of 0 is kept, not dropped.
        Dummies: summed indices pulled out of an argument keep their names
        unless they clash with a name already pulled out of the same term. The
        first occurrence (earlier factor; left argument before right) keeps it,
        and the later one gets pool.fresh(same space, avoid=names in the term).
        The term's own summed indices and free indices never change.
        Idempotent: on input already in linear form, it returns the same terms.
    def is_linear(x) -> bool
        True iff x (an Expr) satisfies the linear-form condition above.

All value checks go through alqemi/tests/evaluate.py (the oracle).
"""
from fractions import Fraction

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
)
from alqemi.core.indices import Index, Space
from alqemi.core.operators import OpString, ann, cre
from alqemi.core.pool import IndexPool
from alqemi.pipeline.linearize import is_linear, linearize
from alqemi.tests.evaluate import mismatch
from alqemi.tests.oracle import FockSpace

P, Q, R, S, M, N = (Index(x) for x in "PQRSMN")
k, l = Index("k"), Index("l")
i, j = Index("i", Space.OCC), Index("j", Space.OCC)
a, b = Index("a", Space.VIRT), Index("b", Space.VIRT)
p1 = Index("p1")


def E(p, q):
    return cre(p) * ann(q)


def T(*factors, coeff=1, scalars=(), summed=()):
    return Term(coeff=coeff, scalars=tuple(scalars), factors=tuple(factors), summed=frozenset(summed))


def X(*terms):
    return Expr(tuple(terms))


def lin(x):
    return linearize(x, IndexPool())


def brackets(term):
    return [f for f in term.factors if isinstance(f, Bracket)]


@pytest.fixture(scope="module")
def fock():
    return FockSpace(n_occ=2, n_virt=2)


# =========================================================================
# What Linearize pulls out
# =========================================================================
class TestPullOut:
    def test_coefficients(self):
        out = lin(comm(T(cre(P), coeff=2), T(ann(Q), coeff=Fraction(-1, 2))))
        assert out.terms == (T(comm(cre(P), ann(Q)), coeff=-1),)

    def test_scalars(self):
        out = lin(comm(T(cre(R), scalars=[Delta(P, Q)]), ann(S)))
        assert out.terms == (T(comm(cre(R), ann(S)), scalars=[Delta(P, Q)]),)

    def test_elementary_bracket_comes_out_unevaluated(self):
        eb = ElementaryBracket(ann(P), cre(Q))
        out = lin(comm(T(ann(R), scalars=[eb]), cre(S)))
        assert out.terms == (T(comm(ann(R), cre(S)), scalars=[eb]),)

    def test_sums_distribute_left_major(self):
        left = X(T(cre(P)), T(cre(Q), coeff=3))
        right = X(T(ann(R)), T(ann(S), coeff=-1))
        out = lin(comm(left, right))
        assert out.terms == (
            T(comm(cre(P), ann(R))),
            T(comm(cre(P), ann(S)), coeff=-1),
            T(comm(cre(Q), ann(R)), coeff=3),
            T(comm(cre(Q), ann(S)), coeff=-3),
        )

    def test_summation_comes_out(self):
        out = lin(comm(T(E(k, P), summed=[k]), cre(Q)))
        assert out.terms == (T(comm(E(k, P), cre(Q)), summed=[k]),)

    def test_summation_space_preserved(self):
        out = lin(comm(T(E(i, P), summed=[i]), cre(Q)))
        assert out.terms[0].summed == {i}

    def test_outer_coeff_scalars_and_sum_combine(self):
        inner = T(E(l, Q), coeff=5, scalars=[Delta(l, R)], summed=[l])
        term = T(cre(k), comm(ann(P), inner), coeff=Fraction(1, 5), scalars=[Delta(k, S)], summed=[k])
        out = lin(term)
        expected = T(cre(k), comm(ann(P), E(l, Q)), coeff=1, scalars=[Delta(k, S), Delta(l, R)], summed=[k, l])
        assert out.terms == (expected,)

    def test_anticommutator_kind_preserved(self):
        out = lin(acomm(T(cre(P), coeff=2), ann(Q)))
        assert out.terms == (T(acomm(cre(P), ann(Q)), coeff=2),)
        assert brackets(out.terms[0])[0].kind is BracketKind.ACOMM


class TestDummies:
    def test_clashing_dummies_renamed_right_side(self):
        out = lin(comm(T(E(k, P), summed=[k]), T(E(Q, k), summed=[k])))
        assert out.terms == (T(comm(E(k, P), E(Q, p1)), summed=[k, p1]),)

    def test_renamed_dummy_keeps_space(self):
        out = lin(comm(T(E(i, P), summed=[i]), T(E(Q, i), summed=[i])))
        summed = out.terms[0].summed
        assert len(summed) == 2 and all(ix.space is Space.OCC for ix in summed)

    def test_sibling_brackets_clash_later_one_renamed(self):
        inner = T(E(P, k), summed=[k])
        out = lin(T(comm(cre(Q), inner), comm(cre(R), inner)))
        (term,) = out.terms
        assert term.summed == {k, p1}
        first, second = brackets(term)
        assert first == comm(cre(Q), E(P, k))
        assert second == comm(cre(R), E(P, p1))

    def test_fresh_name_avoids_names_in_term(self):
        # p1 is already a free index, so the renamed dummy must not be p1.
        out = lin(T(cre(p1), comm(T(E(k, P), summed=[k]), T(E(Q, k), summed=[k]))))
        (term,) = out.terms
        assert p1 in term.free_indices
        assert len(term.summed) == 2 and p1 not in term.summed

    def test_free_and_own_summed_untouched(self):
        term = T(E(k, M), comm(T(E(P, l), summed=[l]), cre(N)), summed=[k])
        out = lin(term)
        assert out.terms[0].free_indices == term.free_indices
        assert out.terms[0].summed == {k, l}

    def test_sum_in_each_term_of_a_sum(self):
        # Separate terms are separate scopes: both may keep the name k.
        arg = X(T(E(k, P), summed=[k]), T(E(Q, k), summed=[k]))
        out = lin(comm(arg, cre(R)))
        assert [t.summed for t in out.terms] == [{k}, {k}]

    def test_capture_is_impossible(self):
        # [sum_k A_k, delta_kS B]: k is free on the right. Pulling the sum out
        # would capture it, so Step 2's no-shadowing check rejects the input.
        with pytest.raises(ValueError):
            lin(comm(T(E(R, k), summed=[k]), T(E(M, N), scalars=[Delta(k, S)])))

    def test_pool_is_used_not_global(self):
        pool = IndexPool()
        pool.fresh(Space.GENERAL)  # p1 already issued by this pool
        out = linearize(comm(T(E(k, P), summed=[k]), T(E(Q, k), summed=[k])), pool)
        assert out.terms[0].summed == {k, Index("p2")}
        # and a new pool starts over
        assert lin(comm(T(E(k, P), summed=[k]), T(E(Q, k), summed=[k]))).terms[0].summed == {k, p1}


# =========================================================================
# Nesting and products
# =========================================================================
class TestNesting:
    def test_nested_inner_pulled_to_top(self):
        inner = comm(T(E(R, k), coeff=2, summed=[k]), T(E(M, N), scalars=[Delta(M, S)]))
        out = lin(comm(E(P, Q), inner))
        expected = T(comm(E(P, Q), comm(E(R, k), E(M, N))), coeff=2, scalars=[Delta(M, S)], summed=[k])
        assert out.terms == (expected,)

    def test_coefficient_on_nested_bracket(self):
        out = lin(comm(cre(P), T(comm(cre(Q), ann(R)), coeff=-3)))
        assert out.terms == (T(comm(cre(P), comm(cre(Q), ann(R))), coeff=-3),)

    def test_three_levels(self):
        deep = comm(T(cre(M), coeff=2), X(T(ann(N)), T(ann(k), summed=[k])))
        out = lin(comm(E(P, Q), comm(E(R, S), deep)))
        assert len(out) == 2
        assert [t.coeff for t in out.terms] == [2, 2]
        assert out.terms[1].summed == {k}
        assert is_linear(out)

    def test_product_inside_argument_stays_one_term(self):
        # [A, B [C, D]]: the argument is the single product B [C, D].
        arg = T(cre(R), comm(T(cre(S), coeff=4), ann(M)))
        out = lin(comm(ann(P), arg))
        assert out.terms == (T(comm(ann(P), T(cre(R), comm(cre(S), ann(M)))), coeff=4),)

    def test_product_of_brackets_combines_left_major(self):
        y = X(T(cre(Q)), T(cre(R), coeff=2))
        z = X(T(ann(S)), T(ann(M), coeff=3))
        out = lin(T(comm(cre(P), y), comm(z, ann(N))))
        assert [t.coeff for t in out.terms] == [1, 3, 2, 6]
        assert brackets(out.terms[1]) == [comm(cre(P), cre(Q)), comm(ann(M), ann(N))]
        assert brackets(out.terms[2]) == [comm(cre(P), cre(R)), comm(ann(S), ann(N))]

    def test_factor_order_kept_around_brackets(self):
        term = T(ann(R), comm(T(cre(P), coeff=2), ann(Q)), cre(S))
        (out,) = lin(term).terms
        assert out.factors == (OpString((ann(R),)), comm(cre(P), ann(Q)), OpString((cre(S),)))


# =========================================================================
# What Linearize must NOT do
# =========================================================================
class TestNoAlgebra:
    def test_does_not_expand_brackets(self):
        x = comm(E(P, Q), E(R, S))
        assert lin(x) == as_expr(x)

    def test_does_not_swap_arguments(self):
        out = lin(comm(T(E(R, S), coeff=2), E(P, Q)))
        assert brackets(out.terms[0])[0] == comm(E(R, S), E(P, Q))

    def test_does_not_evaluate_scalars(self):
        scalars = [Delta(P, P), Delta(i, a), ElementaryBracket(cre(P), cre(Q))]
        out = lin(comm(T(cre(R), scalars=scalars), ann(S)))
        assert len(out.terms[0].scalars) == 3

    def test_does_not_merge_like_terms(self):
        out = lin(comm(X(T(cre(P)), T(cre(P))), ann(Q)))
        assert len(out) == 2

    def test_zero_coefficient_kept(self):
        out = lin(comm(T(cre(P), coeff=0), ann(Q)))
        assert out.terms == (T(comm(cre(P), ann(Q)), coeff=0),)

    def test_zero_argument_vanishes(self):
        assert lin(comm(cre(P), Expr())) == Expr()
        assert lin(X(T(cre(Q)), T(comm(Expr(), ann(P))))) == X(T(cre(Q)))

    def test_identity_argument_kept(self):
        # [a_P, delta_QR] -> delta_QR [a_P, 1]; stage 2 evaluates [a_P, 1].
        out = lin(comm(ann(P), Delta(Q, R)))
        assert out.terms == (T(comm(ann(P), Term()), scalars=[Delta(Q, R)]),)

    def test_terms_without_brackets_pass_through(self):
        x = X(T(E(P, Q), coeff=2, scalars=[Delta(P, R)]), T(E(k, k), summed=[k]))
        assert lin(x) == x

    @pytest.mark.parametrize(
        "x",
        [
            comm(cre(P), E(Q, R)),
            comm(E(P, Q), E(R, S) * E(M, N)),
            comm(E(P, Q), comm(E(R, S), E(M, N))),
            comm(cre(P) * cre(Q), ann(R) * ann(S)),
        ],
        ids=["1.8.11", "1.8.15", "1.8.16", "1S.2.4"],
    )
    def test_golden_inputs_already_linear(self, x):
        assert is_linear(as_expr(x))
        assert lin(x) == as_expr(x)

    def test_idempotent(self):
        x = comm(X(T(E(k, P), coeff=2, summed=[k]), T(cre(Q))), T(E(R, k), summed=[k]))
        once = lin(x)
        assert linearize(once, IndexPool()) == once


class TestInputs:
    def test_accepts_term_bracket_and_expr(self):
        br = comm(cre(P), ann(Q))
        assert lin(br) == lin(T(br)) == lin(as_expr(br))

    def test_rejects_garbage(self):
        with pytest.raises(TypeError):
            lin("a_P")

    def test_returns_expr(self):
        assert isinstance(lin(cre(P)), Expr)


# =========================================================================
# is_linear
# =========================================================================
class TestIsLinear:
    @pytest.mark.parametrize(
        "x",
        [
            Expr(),
            as_expr(E(P, Q)),
            as_expr(T(comm(cre(P), ann(Q)), coeff=3, scalars=[Delta(P, Q)])),
            as_expr(T(comm(E(k, P), cre(Q)), summed=[k])),
            as_expr(comm(ann(P), Term())),
            as_expr(comm(E(P, Q), comm(E(R, S), E(M, N)))),
        ],
    )
    def test_linear(self, x):
        assert is_linear(x)

    @pytest.mark.parametrize(
        "x",
        [
            as_expr(comm(T(cre(P), coeff=2), ann(Q))),
            as_expr(comm(T(cre(P), scalars=[Delta(P, Q)]), ann(Q))),
            as_expr(comm(T(E(k, P), summed=[k]), cre(Q))),
            as_expr(comm(X(T(cre(P)), T(cre(Q))), ann(R))),
            as_expr(comm(Expr(), ann(R))),
            as_expr(comm(E(P, Q), comm(T(cre(R), coeff=2), ann(S)))),   # bad only deep inside
        ],
    )
    def test_not_linear(self, x):
        assert not is_linear(x)

    def test_output_always_linear(self):
        for case in ORACLE_CASES.values():
            assert is_linear(lin(case))


# =========================================================================
# Oracle: Linearize never changes the value
# =========================================================================
ORACLE_CASES = {
    "coefficients": comm(T(cre(P), coeff=2), T(ann(Q), coeff=Fraction(-1, 3))),
    "scalars": comm(T(cre(R), scalars=[Delta(P, Q)]), T(ann(S), scalars=[ElementaryBracket(ann(P), cre(M))])),
    "sums": comm(X(T(cre(P)), T(cre(Q), coeff=3)), X(T(ann(R)), T(ann(S), coeff=-1))),
    "dummy": comm(T(E(k, P), summed=[k]), cre(Q)),
    "clashing dummies": comm(T(E(k, P), summed=[k]), T(E(Q, k), summed=[k])),
    "occ and virt dummies": comm(T(E(i, P), summed=[i]), T(E(a, j), summed=[a])),
    "anticommutator": acomm(X(T(cre(P), coeff=2), T(cre(Q))), T(ann(k), summed=[k])),
    "nested": comm(E(P, Q), comm(T(E(R, k), coeff=2, summed=[k]), T(E(M, N), scalars=[Delta(M, S)]))),
    "sibling clash": T(comm(cre(Q), T(E(P, k), summed=[k])), comm(cre(R), T(E(P, k), summed=[k]))),
    "outer dummy": T(cre(k), comm(ann(P), T(E(l, Q), coeff=5, summed=[l])), scalars=[Delta(k, S)], summed=[k]),
    "product in argument": comm(ann(P), T(cre(R), comm(T(cre(S), coeff=4), ann(M)))),
    "identity argument": comm(ann(P), Delta(Q, R)),
    "anticommutator with identity": acomm(ann(P), T(scalars=[Delta(Q, R)], coeff=2)),
    "product of brackets": T(comm(cre(P), X(T(cre(Q)), T(cre(R), coeff=2))), comm(X(T(ann(S)), T(ann(M))), ann(N))),
}


@pytest.mark.parametrize("case", list(ORACLE_CASES))
def test_value_unchanged(fock, case):
    x = ORACLE_CASES[case]
    out = lin(x)
    assert out.free_indices == as_expr(x).free_indices
    assert mismatch(fock, out, x) is None
