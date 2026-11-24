"""Step 8 tests: tensors and the standard coupled-cluster operators.

These tests DEFINE Step 8. Values are always checked against the oracle, with
random tensor values that have exactly the declared symmetry
(tests/evaluate.py, TensorValues: one array per name, ranks and swap).
Structure is checked where it is the point:
what the builders produce, which terms Simplify merges or drops.

  alqemi/core/tensor.py
    class Swap(Enum): NONE | SYMMETRIC | ANTISYMMETRIC
        Symmetry under exchanging the whole upper group with the whole lower
        group: X^{u}_{l} = +X^{l}_{u} (SYMMETRIC) or -X^{l}_{u} (ANTISYMMETRIC).
        SYMMETRIC fits real orbitals and real Hermitian operators (f, <pq||rs>,
        h); ANTISYMMETRIC fits imaginary Hermitian ones; NONE fits amplitudes.
    @dataclass(frozen=True) class Tensor(name, upper=(), lower=(), swap=Swap.NONE)
        A commuting scalar X^{upper}_{lower}. Always antisymmetric within the
        upper indices and within the lower indices. Lists become tuples.
        ValueError: empty name; swap != NONE with len(upper) != len(lower).
        TypeError: a non-Index index, a non-Swap swap.
        Construction never reorders indices or detects zeros (Simplify does).
        .indices = upper + lower. str: "t^{ab}_{ij}", "s^{a}", "w"
        (an empty group is left out).

  Term: a Tensor is a scalar, like Delta and ElementaryBracket. Its indices
    count as occurrences (summed-index check, free_indices, rename, shadowing).
    Stored scalar order: Deltas, then ElementaryBrackets, then Tensors.

  Pipeline
    Linearize / IdentityExpand: tensors are c-numbers; they move out of bracket
      arguments like other scalars and are never changed.
    DeltaEvaluate: collapsing a summed index substitutes into tensors too
      (sum_q delta_qA f^{P}_{q} -> f^{P}_{A}); an index occurring only in a
      tensor still occurs (no DanglingSum).
    Simplify:
      - Like terms: as in Step 6, but also equal up to each tensor's
        symmetries; the sign that a symmetry brings is applied. The merged term
        keeps the first occurrence's form, with coefficient
        (that form's sign) * sum(coeff_k * sign_k).
      - Like terms are also equal up to reordering operators within each run
        of adjacent creators and each run of adjacent annihilators, with the
        permutation's sign (DESIGN_DECISIONS 14: {a+_p, a+_q} = {a_p, a_q} = 0
        is exact, never a delta). A creator and an annihilator are never
        swapped. The output keeps the first occurrence's operator order.
      - A term equal to minus itself is zero and is dropped (e.g. a repeated
        index within an upper or lower group, X^{P}_{P} with ANTISYMMETRIC swap,
        sum_ab t^{ab}_{IJ} s^{a} s^{b}, a repeated operator a+_P a+_P).
      - relabel_dummies: dummies in order of first appearance in operators,
        then deltas, then tensors (upper then lower), in stored order.

  alqemi/builders.py   (all return Expr; dummies come from ``pool``, a new
                        IndexPool when None, issued upper group first, left to right)
    tensor_operator(name, upper, lower, *, swap=Swap.NONE, coeff=None,
                    normal_ordered=False, pool=None)
        One term: coeff * sum X^{u1..un}_{l1..lm} a+_{u1}..a+_{un} a_{lm}..a_{l1},
        upper/lower are sequences of Space. Default coeff: 1 / prod over both
        groups and each space s of (count of s in the group)!, e.g. 1/4 for
        (V,V),(O,O); 1/2 for (O,V),(V,V); 1 for (O,V),(O,V).
        normal_ordered=True: the normal-ordered product {...} relative to the
        Fermi vacuum, written as a plain string: quasi-particle creators
        (a+ virtual, a occupied) moved left of quasi-annihilators (a virtual,
        a+ occupied), each group keeping its order, and coeff multiplied by
        the permutation's sign. Needs OCC/VIRT spaces only (ValueError for
        GENERAL). Its HF expectation value is 0. It is NOT the same operator as
        the plain string when that has an occupied a+ and an occupied a (they
        differ by contractions, e.g. {a+_i a_j} = a+_i a_j - delta_ij); the
        builders below add the constants that make their forms equal.
        ValueError: empty name, no indices. TypeError: a non-Space.
    fock_operator(*, normal_ordered=False, pool=None)
        F = sum_pq f^{p}_{q} a+_p a_q, f SYMMETRIC.
        normal_ordered: the blocks (O,O), (O,V), (V,O), (V,V) of F, each
        normal ordered, then the constant sum_i f^{i}_{i}. Same operator.
    fluctuation_potential(*, normal_ordered=False, pool=None)
        Phi = V - V_HF = 1/4 sum_pqrs v^{pq}_{rs} a+_p a+_q a_s a_r
                         - sum_pq sum_i v^{pi}_{qi} a+_p a_q,
        v = <pq||rs>, SYMMETRIC. Plain form: exactly these two terms.
        normal_ordered: the 9 blocks of V (upper, lower each in (O,O), (O,V),
        (V,V), upper-major), each normal ordered, then the constant
        -1/2 sum_ij v^{ij}_{ij}. Same operator.
    hamiltonian(*, form="fock", normal_ordered=False, pool=None)
        form="fock": fock_operator(...) + fluctuation_potential(...), one pool.
        form="core": sum_pq h^{p}_{q} a+_p a_q + 1/4 sum v^{pq}_{rs} a+_p a+_q a_s a_r,
        h SYMMETRIC. normal_ordered=True is only defined for form="fock"
        (ValueError otherwise); any other form is a ValueError.
        The two forms are the same operator when f_pq = h_pq + sum_i v^{pi}_{qi}.
    cluster(n, *, name="t", pool=None)
        T_n = (1/n!)^2 sum t^{a1..an}_{i1..in} a+_{a1}..a+_{an} a_{in}..a_{i1}.
        ValueError for n < 1.

  Public API: Tensor, Swap, tensor_operator, fock_operator,
  fluctuation_potential, hamiltonian, cluster join ``alqemi.__all__``.
  The Step 7 reference files core/cluster.py and core/tools.py are deleted;
  no file in the package imports sympy.
"""
import itertools
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import alqemi
from alqemi.core.expr import Bracket, Delta, ElementaryBracket, Expr, Term, comm
from alqemi.core.indices import Index, Space
from alqemi.core.operators import Kind, OpString, ann, cre
from alqemi.core.pool import IndexPool
from alqemi.core.tensor import Swap, Tensor
from alqemi.builders import (
    cluster,
    fluctuation_potential,
    fock_operator,
    hamiltonian,
    tensor_operator,
)
from alqemi.pipeline.delta_evaluate import DanglingSum, delta_evaluate
from alqemi.pipeline.identity_expand import identity_expand
from alqemi.pipeline.linearize import is_linear, linearize
from alqemi.pipeline.run import run
from alqemi.pipeline.simplify import simplify
from alqemi.tests.evaluate import TensorValues, evaluate, mismatch
from alqemi.tests.oracle import FockSpace, commutator

O, V, G = Space.OCC, Space.VIRT, Space.GENERAL
P, Q, R = Index("P"), Index("Q"), Index("R")
A, B = Index("A", V), Index("B", V)
I, J = Index("I", O), Index("J", O)
a, b, c = Index("a", V), Index("b", V), Index("c", V)
i, j = Index("i", O), Index("j", O)
p, q, k = Index("p"), Index("q"), Index("k")
a1, a2, a3 = (Index(f"a{n}", V) for n in (1, 2, 3))
i1, i2, i3, i4 = (Index(f"i{n}", O) for n in (1, 2, 3, 4))


def E(x, y):
    return cre(x) * ann(y)


def T(*factors, coeff=1, scalars=(), summed=()):
    return Term(coeff=coeff, scalars=tuple(scalars), factors=tuple(factors), summed=frozenset(summed))


def X(*terms):
    return Expr(tuple(terms))


def t2(u, l, name="t", swap=Swap.NONE):
    return Tensor(name, tuple(u), tuple(l), swap)


@pytest.fixture(scope="module")
def fock():
    return FockSpace(n_occ=2, n_virt=2)


@pytest.fixture(scope="module")
def values(fock):
    return TensorValues(fock, seed=8)


def matrix(fock, x, values):
    return evaluate(fock, x, {}, values)


def same_value(fock, values, lhs, rhs):
    return mismatch(fock, lhs, rhs, tensors=values) is None


# ---- direct (independent) matrices -----------------------------------------
def cd(fock, n):
    return fock._cre[n]


def an(fock, n):
    return fock._ann[n]


def occ(fock):
    return range(fock.n_occ)


def virt(fock):
    return range(fock.n_occ, fock.n_modes)


def direct_one_body(fock, arr):
    n = fock.n_modes
    return sum(arr[x, y] * cd(fock, x) @ an(fock, y) for x in range(n) for y in range(n))


def direct_two_body(fock, v):
    n = fock.n_modes
    out = np.zeros((fock.dim, fock.dim))
    for w in range(n):
        for x in range(n):
            for y in range(n):
                for z in range(n):
                    if v[w, x, y, z] != 0:
                        out += 0.25 * v[w, x, y, z] * cd(fock, w) @ cd(fock, x) @ an(fock, z) @ an(fock, y)
    return out


def direct_v_hf(fock, v):
    n = fock.n_modes
    return sum(
        v[x, m, y, m] * cd(fock, x) @ an(fock, y) for x in range(n) for y in range(n) for m in occ(fock)
    )


def hf_state(fock):
    state = np.zeros(fock.dim)
    state[0] = 1.0  # the empty Fock state
    for m in occ(fock):
        state = cd(fock, m) @ state
    return state


def hf_expectation(fock, mat):
    s = hf_state(fock)
    return float(s @ mat @ s)


# =========================================================================
# Tensor
class TestTensor:
    def test_fields_and_tuples(self):
        t = Tensor("t", [a, b], [i, j])
        assert t.upper == (a, b) and t.lower == (i, j) and t.swap is Swap.NONE
        assert t.indices == (a, b, i, j)

    def test_value_equality_and_hash(self):
        assert Tensor("t", (a,), (i,)) == Tensor("t", (a,), (i,))
        assert len({Tensor("t", (a,), (i,)), Tensor("t", (a,), (i,))}) == 1
        assert Tensor("t", (a,), (i,)) != Tensor("u", (a,), (i,))
        assert Tensor("f", (p,), (q,), Swap.SYMMETRIC) != Tensor("f", (p,), (q,))

    def test_construction_does_not_reorder(self):
        assert Tensor("t", (b, a), (i, j)).upper == (b, a)

    def test_repeated_index_allowed_at_construction(self):
        assert Tensor("t", (a, a), (i, j)).upper == (a, a)

    @pytest.mark.parametrize(
        "tensor, text",
        [
            (Tensor("t", (a, b), (i, j)), "t^{ab}_{ij}"),
            (Tensor("f", (p,), (q,), Swap.SYMMETRIC), "f^{p}_{q}"),
            (Tensor("s", (a,)), "s^{a}"),
            (Tensor("s", (), (i,)), "s_{i}"),
            (Tensor("w"), "w"),
        ],
    )
    def test_str(self, tensor, text):
        assert str(tensor) == text

    @pytest.mark.parametrize(
        "kwargs, error",
        [
            (dict(name="", upper=(a,)), ValueError),
            (dict(name="f", upper=(p, q), lower=(q,), swap=Swap.SYMMETRIC), ValueError),
            (dict(name="f", upper=(p,), lower=(), swap=Swap.ANTISYMMETRIC), ValueError),
            (dict(name="t", upper=("a",)), TypeError),
            (dict(name="t", upper=(a,), swap="symmetric"), TypeError),
        ],
    )
    def test_invalid(self, kwargs, error):
        with pytest.raises(error):
            Tensor(**kwargs)


# =========================================================================
# Term with tensors
class TestTermWithTensors:
    def test_summed_index_only_in_a_tensor(self):
        term = T(scalars=[Tensor("f", (i,), (i,), Swap.SYMMETRIC)], summed=[i])
        assert term.free_indices == frozenset()

    def test_summed_index_must_still_occur(self):
        with pytest.raises(ValueError):
            T(E(P, Q), scalars=[Tensor("t", (A,), (I,))], summed=[a])

    def test_free_indices_include_tensor_indices(self):
        term = T(E(P, Q), scalars=[Tensor("t", (A,), (I,))])
        assert term.free_indices == {P, Q, A, I}

    def test_rename_reaches_tensors(self):
        term = T(E(a, i), scalars=[Tensor("t", (a,), (i,))], summed=[a, i])
        out = term.rename({a: b, i: j})
        assert out.scalars == (Tensor("t", (b,), (j,)),)
        assert out.summed == {b, j}

    def test_scalar_order_is_canonical(self):
        t, f = Tensor("t", (A,), (I,)), Tensor("f", (P,), (Q,), Swap.SYMMETRIC)
        d, eb = Delta(P, Q), ElementaryBracket(ann(P), cre(Q))
        one = T(scalars=[t, eb, f, d])
        two = T(scalars=[f, d, t, eb])
        assert one == two
        assert one.scalars[:2] == (d, eb)
        assert set(one.scalars[2:]) == {t, f}

    def test_shadowing_through_a_tensor_index(self):
        inner = T(E(k, Q), summed=[k])
        with pytest.raises(ValueError):
            T(comm(inner, E(P, Q)), scalars=[Tensor("f", (k,), (P,))])

    def test_str(self):
        term = T(E(a, i), scalars=[Tensor("t", (a,), (i,))], summed=[a, i])
        assert str(term) == "Σ_{a,i} t^{a}_{i} a†_a a_i"


# =========================================================================
# Linearize and IdentityExpand
class TestEarlyStages:
    def test_linearize_moves_tensor_out(self, fock, values):
        t1 = T(E(a, i), scalars=[Tensor("t", (a,), (i,))], summed=[a, i])
        x = comm(t1, E(P, Q))
        out = linearize(x, IndexPool())
        assert is_linear(out) and len(out) == 1
        (term,) = out.terms
        assert [type(s) for s in term.scalars] == [Tensor]
        assert isinstance(term.factors[0], Bracket)
        assert same_value(fock, values, out, as_value(x))

    def test_linearize_freshens_dummies_inside_tensors(self, fock, values):
        x = comm(cluster(1), cluster(1))
        out = linearize(x, IndexPool())
        (term,) = out.terms
        assert len(term.summed) == 4  # both copies' dummies, renamed apart
        assert len({ix.name for ix in term.summed}) == 4
        assert same_value(fock, values, out, as_value(x))

    def test_identity_expand_keeps_tensors(self, fock, values):
        x = comm(cluster(1), E(P, Q))
        out = identity_expand(x, IndexPool())
        assert all(any(isinstance(s, Tensor) for s in t.scalars) for t in out)
        assert same_value(fock, values, out, as_value(x))


def as_value(x):
    """Wrap a bracket so mismatch() can read its free indices."""
    return x if isinstance(x, (Expr, Term)) else X(T(x))


# =========================================================================
# DeltaEvaluate
class TestDeltaEvaluate:
    def test_collapse_substitutes_into_tensor(self, fock, values):
        f = Tensor("f", (P,), (q,), Swap.SYMMETRIC)
        x = X(T(E(P, Q), scalars=[Delta(q, A), f], summed=[q]))
        out = delta_evaluate(x)
        (term,) = out.terms
        assert term.scalars == (Tensor("f", (P,), (A,), Swap.SYMMETRIC),)
        assert term.summed == frozenset()
        assert same_value(fock, values, out, x)

    def test_elementary_bracket_collapse_into_tensor(self, fock, values):
        f = Tensor("f", (P,), (q,), Swap.SYMMETRIC)
        x = X(T(ann(Q), scalars=[ElementaryBracket(ann(q), cre(A)), f], summed=[q]))
        out = delta_evaluate(x)
        (term,) = out.terms
        assert term.scalars == (Tensor("f", (P,), (A,), Swap.SYMMETRIC),)
        assert same_value(fock, values, out, x)

    def test_narrower_dummy_does_not_collapse(self, fock, values):
        x = X(T(ann(Q), scalars=[Delta(i, P), Tensor("t", (A,), (i,))], summed=[i]))
        out = delta_evaluate(x)
        (term,) = out.terms
        assert term.summed == {i}
        assert same_value(fock, values, out, x)

    def test_index_only_in_tensor_is_not_dangling(self, fock, values):
        f = Tensor("f", (k,), (k,), Swap.SYMMETRIC)
        x = X(T(E(P, Q), scalars=[ElementaryBracket(ann(k), cre(k)), f], summed=[k]))
        out = delta_evaluate(x)
        (term,) = out.terms
        assert term.scalars == (f,) and term.summed == {k}
        assert same_value(fock, values, out, x)

    def test_dangling_sum_still_raised(self):
        f = Tensor("f", (P,), (Q,), Swap.SYMMETRIC)
        x = X(T(E(P, Q), scalars=[ElementaryBracket(ann(k), cre(k)), f], summed=[k]))
        with pytest.raises(DanglingSum):
            delta_evaluate(x)


# =========================================================================
# Simplify
def tAB(u, l, coeff=1, name="t", swap=Swap.NONE, string=None):
    string = E(P, Q) if string is None else string
    return T(string, coeff=coeff, scalars=[Tensor(name, tuple(u), tuple(l), swap)])


SIMPLIFY_VALUE_CASES = {
    "antisym cancel": X(tAB((A, B), (I, J)), tAB((B, A), (I, J))),
    "antisym add": X(tAB((A, B), (I, J)), tAB((B, A), (I, J), coeff=-1)),
    "lower antisym": X(tAB((A, B), (I, J)), tAB((A, B), (J, I))),
    "both groups": X(tAB((A, B), (I, J)), tAB((B, A), (J, I), coeff=-1)),
    "first form kept": X(tAB((B, A), (I, J), coeff=3), tAB((A, B), (I, J))),
    "repeated upper": X(tAB((A, A), (I, J))),
    "repeated lower": X(tAB((A, B), (I, I))),
    "swap symmetric": X(tAB((P,), (Q,), name="f", swap=Swap.SYMMETRIC, string=ann(I)),
                        tAB((Q,), (P,), name="f", swap=Swap.SYMMETRIC, string=ann(I))),
    "swap antisymmetric": X(tAB((P,), (Q,), name="x", swap=Swap.ANTISYMMETRIC, string=ann(I)),
                            tAB((Q,), (P,), name="x", swap=Swap.ANTISYMMETRIC, string=ann(I))),
    "swap antisymmetric diagonal": X(tAB((P,), (P,), name="x", swap=Swap.ANTISYMMETRIC, string=ann(I))),
    "no swap": X(tAB((P,), (Q,), name="g", string=ann(I)), tAB((Q,), (P,), name="g", string=ann(I))),
    "different names": X(tAB((A, B), (I, J)), tAB((A, B), (I, J), name="u")),
    "dummies with sign": X(
        T(cre(a), cre(b), scalars=[Tensor("t", (a, b), (I, J))], summed=[a, b]),
        T(cre(b), cre(a), scalars=[Tensor("t", (a, b), (I, J))], summed=[a, b]),
    ),
    "dummies with sign add": X(
        T(cre(a), cre(b), scalars=[Tensor("t", (a, b), (I, J))], summed=[a, b]),
        T(cre(b), cre(a), coeff=-1, scalars=[Tensor("t", (a, b), (I, J))], summed=[a, b]),
    ),
    "minus itself": X(
        T(scalars=[Tensor("t", (a, b), (I, J)), Tensor("s", (a,)), Tensor("s", (b,))], summed=[a, b])
    ),
    "not minus itself": X(T(cre(a), cre(b), scalars=[Tensor("t", (a, b), (I, J))], summed=[a, b])),
    # same-kind swaps (DESIGN_DECISIONS 14)
    "creators swap cancel": X(T(cre(P), cre(Q)), T(cre(Q), cre(P))),
    "annihilators swap cancel": X(T(ann(P), ann(Q), cre(R)), T(ann(Q), ann(P), cre(R))),
    "swap with tensor": X(
        T(cre(A), cre(B), scalars=[Tensor("t", (A, B), (I, J))]),
        T(cre(B), cre(A), scalars=[Tensor("t", (B, A), (I, J))]),
    ),
    "swap keeps first order": X(T(cre(Q), cre(P), coeff=3), T(cre(P), cre(Q))),
    "three creators": X(T(cre(P), cre(Q), cre(R)), T(cre(R), cre(P), cre(Q))),
    "repeated operator": X(T(cre(P), cre(P), ann(Q))),
    "mixed kinds not swapped": X(T(cre(P), ann(Q)), T(ann(Q), cre(P), coeff=-1)),
    "separated runs not swapped": X(T(cre(P), ann(R), cre(Q)), T(cre(Q), ann(R), cre(P))),
}


class TestSimplify:
    @pytest.mark.parametrize("name", list(SIMPLIFY_VALUE_CASES))
    def test_value_unchanged(self, fock, values, name):
        x = SIMPLIFY_VALUE_CASES[name]
        assert mismatch(fock, simplify(x), x, extra=sorted(x.free_indices, key=str), tensors=values) is None

    @pytest.mark.parametrize(
        "name",
        ["antisym cancel", "lower antisym", "repeated upper", "repeated lower",
         "swap antisymmetric", "swap antisymmetric diagonal", "dummies with sign", "minus itself",
         "creators swap cancel", "annihilators swap cancel", "repeated operator"],
    )
    def test_vanishes(self, name):
        assert len(simplify(SIMPLIFY_VALUE_CASES[name])) == 0

    @pytest.mark.parametrize(
        "name, coeff, tensor",
        [
            ("antisym add", 2, Tensor("t", (A, B), (I, J))),
            ("both groups", 0, None),
            ("first form kept", 2, Tensor("t", (B, A), (I, J))),
            ("swap symmetric", 2, Tensor("f", (P,), (Q,), Swap.SYMMETRIC)),
        ],
    )
    def test_merges_with_sign(self, name, coeff, tensor):
        out = simplify(SIMPLIFY_VALUE_CASES[name])
        if coeff == 0:
            assert len(out) == 0
            return
        (term,) = out.terms
        assert term.coeff == coeff and term.scalars == (tensor,)

    @pytest.mark.parametrize(
        "name", ["no swap", "different names", "mixed kinds not swapped", "separated runs not swapped"]
    )
    def test_does_not_merge(self, name):
        assert len(simplify(SIMPLIFY_VALUE_CASES[name])) == 2

    @pytest.mark.parametrize(
        "name, coeff, factor",
        [
            ("swap with tensor", 2, cre(A) * cre(B)),
            ("swap keeps first order", 2, cre(Q) * cre(P)),
            ("three creators", 2, cre(P) * cre(Q) * cre(R)),  # a+R a+P a+Q = +a+P a+Q a+R
        ],
    )
    def test_same_kind_swap_merges(self, name, coeff, factor):
        (term,) = simplify(SIMPLIFY_VALUE_CASES[name]).terms
        assert term.coeff == coeff and term.factors == (factor,)

    def test_dummies_with_sign_add(self):
        (term,) = simplify(SIMPLIFY_VALUE_CASES["dummies with sign add"]).terms
        assert term.coeff == 2
        assert term.factors == (cre(a) * cre(b),)

    def test_not_minus_itself_kept(self):
        assert len(simplify(SIMPLIFY_VALUE_CASES["not minus itself"])) == 1

    def test_relabel_tensor_only_dummies(self):
        x = X(T(ann(Q), scalars=[Tensor("t", (a1,), (I,))], summed=[a1]))
        (term,) = simplify(x).terms
        assert term.scalars == (Tensor("t", (a,), (I,)),)

    def test_relabel_order_operators_then_tensors(self):
        x = X(T(cre(a1), scalars=[Tensor("s", (a2,)), Tensor("u", (a1, a2))], summed=[a1, a2]))
        (term,) = simplify(x).terms
        assert term.factors == (OpString((cre(a),)),)
        assert set(term.scalars) == {Tensor("s", (b,)), Tensor("u", (a, b))}

    def test_idempotent(self):
        for x in SIMPLIFY_VALUE_CASES.values():
            once = simplify(x)
            assert simplify(once) == once


# =========================================================================
# tensor_operator
def string_kinds(term):
    (string,) = term.factors
    return [(op.kind, op.index) for op in string]


def is_normal_ordered(string):
    quasi_creator = [
        (op.kind is Kind.CRE and op.index.space is V) or (op.kind is Kind.ANN and op.index.space is O)
        for op in string
    ]
    return quasi_creator == sorted(quasi_creator, reverse=True)


class TestTensorOperator:
    def test_t2_structure(self):
        (term,) = tensor_operator("t", [V, V], [O, O]).terms
        assert term.coeff == Fraction(1, 4)
        assert term.scalars == (Tensor("t", (a1, a2), (i1, i2)),)
        assert term.factors == (cre(a1) * cre(a2) * ann(i2) * ann(i1),)
        assert term.summed == {a1, a2, i1, i2}

    @pytest.mark.parametrize(
        "upper, lower, coeff",
        [
            ([G], [G], 1), ([V], [O], 1), ([V, V], [O, O], Fraction(1, 4)),
            ([O, V], [V, V], Fraction(1, 2)), ([O, V], [O, V], 1),
            ([V, V, V], [O, O, O], Fraction(1, 36)), ([V, O, V], [O], Fraction(1, 2)),
        ],
    )
    def test_default_coefficient(self, upper, lower, coeff):
        (term,) = tensor_operator("x", upper, lower).terms
        assert term.coeff == coeff

    def test_coeff_override(self):
        (term,) = tensor_operator("t", [V, V], [O, O], coeff=1).terms
        assert term.coeff == 1

    def test_swap_passed_to_tensor(self):
        (term,) = tensor_operator("f", [G], [G], swap=Swap.SYMMETRIC).terms
        assert term.scalars[0].swap is Swap.SYMMETRIC

    def test_normal_ordered_mixed(self):
        (term,) = tensor_operator("v", [O, V], [V, V], normal_ordered=True).terms
        # upper (O, V) -> i1, a1; lower (V, V) -> a2, a3; raw a+i1 a+a1 a_a3 a_a2
        assert term.scalars == (Tensor("v", (i1, a1), (a2, a3)),)
        assert term.factors == (cre(a1) * cre(i1) * ann(a3) * ann(a2),)
        assert term.coeff == Fraction(-1, 2)

    def test_normal_ordered_keeps_order_within_groups(self):
        (term,) = tensor_operator("v", [O, O], [O, O], normal_ordered=True).terms
        # raw a+i1 a+i2 a_i4 a_i3 -> a_i4 a_i3 a+i1 a+i2, each a_ passing two a+ (sign +)
        assert term.factors == (ann(i4) * ann(i3) * cre(i1) * cre(i2),)
        assert term.coeff == Fraction(1, 4)

    def test_normal_ordered_two_quasi_creators(self):
        (term,) = tensor_operator("v", [O, V], [O, V], normal_ordered=True).terms
        # raw a+i1 a+a1 a_a2 a_i2 -> a+a1 a_i2 a+i1 a_a2 (signs: -1, then +1)
        assert term.scalars == (Tensor("v", (i1, a1), (i2, a2)),)
        assert term.factors == (cre(a1) * ann(i2) * cre(i1) * ann(a2),)
        assert term.coeff == -1

    def test_normal_ordered_occupied_one_body(self):
        (term,) = tensor_operator("f", [O], [O], normal_ordered=True).terms
        assert term.factors == (ann(i2) * cre(i1),)
        assert term.coeff == -1

    def test_normal_ordered_needs_occ_virt(self):
        with pytest.raises(ValueError):
            tensor_operator("f", [G], [O], normal_ordered=True)

    BLOCKS = [([O, O], [O, O]), ([O, V], [V, O]), ([V, V], [O, O]), ([O, O], [V, V]),
              ([V, O], [O, V]), ([O], [V]), ([O], [O]), ([V], [V])]

    @pytest.mark.parametrize("upper, lower", BLOCKS)
    def test_normal_ordered_zero_hf_expectation(self, fock, values, upper, lower):
        normal = tensor_operator("w", upper, lower, normal_ordered=True)
        (term,) = normal.terms
        assert is_normal_ordered(term.factors[0])
        assert abs(hf_expectation(fock, matrix(fock, normal, values))) < 1e-12

    @pytest.mark.parametrize(
        "upper, lower", [([V, V], [O, O]), ([O, O], [V, V]), ([O], [V]), ([V], [O]), ([V], [V])]
    )
    def test_normal_ordered_same_value_without_contractions(self, fock, values, upper, lower):
        """No occupied a+ together with an occupied a: {X} is the plain product X."""
        plain = tensor_operator("w", upper, lower)
        normal = tensor_operator("w", upper, lower, normal_ordered=True)
        assert same_value(fock, values, plain, normal)

    @pytest.mark.parametrize("upper, lower", [([O], [O]), ([O, O], [O, O]), ([O, V], [V, O])])
    def test_normal_ordered_differs_with_contractions(self, fock, values, upper, lower):
        plain = tensor_operator("w", upper, lower)
        normal = tensor_operator("w", upper, lower, normal_ordered=True)
        assert not same_value(fock, values, plain, normal)

    @pytest.mark.parametrize(
        "args, kwargs, error",
        [
            (("", [V], [O]), {}, ValueError),
            (("t", [], []), {}, ValueError),
            (("t", ["virt"], [O]), {}, TypeError),
            (("f", [G, G], [G]), dict(swap=Swap.SYMMETRIC), ValueError),
        ],
    )
    def test_invalid(self, args, kwargs, error):
        with pytest.raises(error):
            tensor_operator(*args, **kwargs)

    def test_shared_pool_gives_distinct_names(self):
        pool = IndexPool()
        one = tensor_operator("t", [V], [O], pool=pool)
        two = tensor_operator("t", [V], [O], pool=pool)
        assert not ({ix.name for ix in one.terms[0].summed} & {ix.name for ix in two.terms[0].summed})

    def test_new_object_lambda_2(self, fock, values):
        """Extensibility: Lambda_2 = 1/4 sum l^{ij}_{ab} a+_i a+_j a_b a_a in one line."""
        lam = tensor_operator("l", [O, O], [V, V])
        arr = values.array("l", 2, 2)

        def direct(g):
            return sum(
                0.25 * arr[w, x, y, z] * cd(fock, w) @ cd(fock, x) @ an(fock, z) @ an(fock, y)
                for w in occ(fock) for x in occ(fock) for y in virt(fock) for z in virt(fock)
            )

        assert same_value(fock, values, lam, direct)


# =========================================================================
# Standard operators
class TestFockOperator:
    def test_plain_structure(self):
        (term,) = fock_operator().terms
        (f,) = term.scalars
        assert f.name == "f" and f.swap is Swap.SYMMETRIC
        assert term.coeff == 1 and len(term.summed) == 2
        assert all(ix.space is G for ix in term.summed)
        assert term.factors == (cre(f.upper[0]) * ann(f.lower[0]),)

    def test_plain_value(self, fock, values):
        farr = values.array("f", 1, 1, Swap.SYMMETRIC)
        assert same_value(fock, values, fock_operator(), lambda g: direct_one_body(fock, farr))

    def test_normal_ordered_structure(self):
        out = fock_operator(normal_ordered=True)
        assert len(out) == 5
        blocks = [tuple(ix.space for ix in t.scalars[0].indices) for t in out.terms[:4]]
        assert blocks == [(O, O), (O, V), (V, O), (V, V)]
        for term in out.terms[:4]:
            assert is_normal_ordered(term.factors[0])
        const = out.terms[4]
        assert const.factors == () and const.coeff == 1
        (f,) = const.scalars
        assert f.upper == f.lower and f.upper[0].space is O

    def test_normal_ordered_value(self, fock, values):
        assert same_value(fock, values, fock_operator(normal_ordered=True), fock_operator())


class TestFluctuationPotential:
    def test_plain_structure(self):
        two, one = fluctuation_potential().terms
        assert two.coeff == Fraction(1, 4) and len(two.factors[0]) == 4
        assert one.coeff == -1 and len(one.factors[0]) == 2
        for term in (two, one):
            (v,) = term.scalars
            assert v.name == "v" and v.swap is Swap.SYMMETRIC
        (v,) = one.scalars
        assert v.upper[1] == v.lower[1] and v.upper[1].space is O  # v^{p i}_{q i}

    def test_plain_value(self, fock, values):
        varr = values.array("v", 2, 2, Swap.SYMMETRIC)
        direct = direct_two_body(fock, varr) - direct_v_hf(fock, varr)
        assert same_value(fock, values, fluctuation_potential(), lambda g: direct)

    def test_normal_ordered_structure(self):
        out = fluctuation_potential(normal_ordered=True)
        assert len(out) == 10
        pairs = [(O, O), (O, V), (V, V)]
        expected = [(u, l) for u in pairs for l in pairs]
        got = [
            (tuple(ix.space for ix in t.scalars[0].upper), tuple(ix.space for ix in t.scalars[0].lower))
            for t in out.terms[:9]
        ]
        assert got == expected
        factor = {(O, O): Fraction(1, 2), (O, V): Fraction(1), (V, V): Fraction(1, 2)}
        for (u, l), term in zip(expected, out.terms[:9]):
            assert abs(term.coeff) == factor[u] * factor[l]
            assert is_normal_ordered(term.factors[0])
        const = out.terms[9]
        assert const.factors == () and const.coeff == Fraction(-1, 2)
        (v,) = const.scalars
        assert v.upper == v.lower and all(ix.space is O for ix in v.upper)

    def test_normal_ordered_value(self, fock, values):
        assert same_value(fock, values, fluctuation_potential(normal_ordered=True), fluctuation_potential())

    def test_operator_terms_have_zero_hf_expectation(self, fock, values):
        for term in fluctuation_potential(normal_ordered=True).terms[:9]:
            assert abs(hf_expectation(fock, matrix(fock, X(term), values))) < 1e-12
        for term in fock_operator(normal_ordered=True).terms[:4]:
            assert abs(hf_expectation(fock, matrix(fock, X(term), values))) < 1e-12


class TestHamiltonian:
    def test_fock_form_is_f_plus_phi(self, fock, values):
        assert same_value(fock, values, hamiltonian(), fock_operator() + fluctuation_potential())
        assert len(hamiltonian()) == 3

    def test_fock_form_value(self, fock, values):
        farr = values.array("f", 1, 1, Swap.SYMMETRIC)
        varr = values.array("v", 2, 2, Swap.SYMMETRIC)
        direct = direct_one_body(fock, farr) + direct_two_body(fock, varr) - direct_v_hf(fock, varr)
        assert same_value(fock, values, hamiltonian(), lambda g: direct)

    def test_one_pool_for_all_terms(self):
        names = [{ix.name for ix in t.summed} for t in hamiltonian().terms]
        assert sum(len(n) for n in names) == len(set().union(*names))

    def test_normal_ordered_same_value(self, fock, values):
        assert same_value(fock, values, hamiltonian(normal_ordered=True), hamiltonian())

    def test_hf_energy(self, fock, values):
        farr = values.array("f", 1, 1, Swap.SYMMETRIC)
        varr = values.array("v", 2, 2, Swap.SYMMETRIC)
        e_hf = sum(farr[m, m] for m in occ(fock)) - 0.5 * sum(
            varr[m, n, m, n] for m in occ(fock) for n in occ(fock)
        )
        for h in (hamiltonian(), hamiltonian(normal_ordered=True)):
            assert hf_expectation(fock, matrix(fock, h, values)) == pytest.approx(e_hf)

    def test_core_form_structure(self):
        one, two = hamiltonian(form="core").terms
        assert one.scalars[0].name == "h" and one.scalars[0].swap is Swap.SYMMETRIC
        assert two.scalars[0].name == "v" and two.coeff == Fraction(1, 4)

    def test_core_form_equals_fock_form(self, fock):
        vals = TensorValues(fock, seed=81)
        harr = vals.array("h", 1, 1, Swap.SYMMETRIC)
        varr = vals.array("v", 2, 2, Swap.SYMMETRIC)
        farr = harr + np.einsum("piqi->pq", varr[:, : fock.n_occ, :, : fock.n_occ])
        vals.set("f", 1, 1, Swap.SYMMETRIC, farr)
        assert mismatch(fock, hamiltonian(form="core"), hamiltonian(), tensors=vals) is None
        assert mismatch(fock, hamiltonian(form="core"), hamiltonian(normal_ordered=True), tensors=vals) is None

    @pytest.mark.parametrize("kwargs", [dict(form="core", normal_ordered=True), dict(form="bare")])
    def test_invalid(self, kwargs):
        with pytest.raises(ValueError):
            hamiltonian(**kwargs)


class TestCluster:
    def test_t2_structure(self):
        assert cluster(2) == tensor_operator("t", [V, V], [O, O])

    def test_name(self):
        (term,) = cluster(1, name="r").terms
        assert term.scalars[0].name == "r"

    @pytest.mark.parametrize("n, n_occ, n_virt", [(1, 2, 2), (2, 2, 2), (3, 3, 3)])
    def test_value(self, n, n_occ, n_virt):
        fk = FockSpace(n_occ=n_occ, n_virt=n_virt)
        vals = TensorValues(fk, seed=n)
        arr = vals.array("t", n, n)
        occs, virts = list(range(n_occ)), list(range(n_occ, fk.n_modes))
        direct = np.zeros((fk.dim, fk.dim))
        for us in itertools.product(virts, repeat=n):
            for ls in itertools.product(occs, repeat=n):
                mat = fk.identity
                for u in us:
                    mat = mat @ cd(fk, u)
                for l in reversed(ls):
                    mat = mat @ an(fk, l)
                direct += arr[us + ls] / math.factorial(n) ** 2 * mat
        assert mismatch(fk, cluster(n), lambda g: direct, tensors=vals) is None

    @pytest.mark.parametrize("n", [0, -1])
    def test_invalid(self, n):
        with pytest.raises(ValueError):
            cluster(n)


# =========================================================================
# The whole pipeline on CC commutators
def matrix_comm(fock, values, x, y):
    return commutator(matrix(fock, x, values), matrix(fock, y, values))


class TestCommutators:
    @pytest.mark.parametrize(
        "name, left, right",
        [
            ("[F, T1]", lambda: fock_operator(), lambda: cluster(1)),
            ("[F, T2]", lambda: fock_operator(), lambda: cluster(2)),
            ("[Phi, T1]", lambda: fluctuation_potential(), lambda: cluster(1)),
            ("[Phi, T2]", lambda: fluctuation_potential(), lambda: cluster(2)),
            ("[H, T1]", lambda: hamiltonian(), lambda: cluster(1)),
            ("[H, T2]", lambda: hamiltonian(), lambda: cluster(2)),
            ("[H_no, T1]", lambda: hamiltonian(normal_ordered=True), lambda: cluster(1)),
            ("[F_no, T2]", lambda: fock_operator(normal_ordered=True), lambda: cluster(2)),
        ],
    )
    def test_value(self, fock, values, name, left, right):
        l, r = left(), right()
        out = run(comm(l, r))
        expected = matrix_comm(fock, values, l, r)
        assert same_value(fock, values, out, lambda g: expected)
        assert simplify(out) == out

    def test_f_t2_terms_merge(self, fock, values):
        """[F, T2] without the same-kind merges has 4 terms in 2 equal pairs."""
        out = run(comm(fock_operator(), cluster(2)))
        assert len(out) == 2

    def test_double_commutator(self, fock, values):
        h, t = hamiltonian(), cluster(1)
        out = run(comm(comm(h, t), t))
        th = matrix(fock, t, values)
        expected = commutator(commutator(matrix(fock, h, values), th), th)
        assert same_value(fock, values, out, lambda g: expected)

    def test_constant_terms_drop_out(self, fock, values):
        plain = run(comm(fock_operator(), cluster(1)))
        normal = run(comm(fock_operator(normal_ordered=True), cluster(1)))
        assert same_value(fock, values, plain, normal)


# =========================================================================
# Public API and cleanup
STEP8_NAMES = {"Tensor", "Swap", "tensor_operator", "fock_operator", "fluctuation_potential", "hamiltonian", "cluster"}


@pytest.mark.parametrize("name", sorted(STEP8_NAMES))
def test_exported(name):
    assert name in alqemi.__all__ and hasattr(alqemi, name)


PACKAGE_DIR = Path(alqemi.__file__).resolve().parent


@pytest.mark.parametrize("path", ["core/cluster.py", "core/tools.py"])
def test_reference_files_deleted(path):
    assert not (PACKAGE_DIR / path).exists()


def test_no_sympy_anywhere():
    offenders = [
        str(path.relative_to(PACKAGE_DIR))
        for path in PACKAGE_DIR.rglob("*.py")
        if "__pycache__" not in path.parts
        and any(
            line.strip().startswith(("import sympy", "from sympy"))
            for line in path.read_text().splitlines()
        )
    ]
    assert offenders == []
