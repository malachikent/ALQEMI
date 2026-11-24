"""Step 9 tests: Fermi-vacuum projections and the BCH expansion.

These tests DEFINE Step 9. Every value is checked against the oracle: <HF| M |HF>
is computed directly from the Hartree-Fock state vector (occupied modes filled)
and the matrix M, with random tensor values (tests/evaluate.py, TensorValues).

Notation (DESIGN_DECISIONS, Step 9): |mu> = tau_mu |HF>, <mu| = <HF| tau_mu^dagger.
Excited determinants are never separate objects; Phi means only the fluctuation
potential.

  alqemi/builders.py
    excitation(n) -> Expr
        tau_mu of rank n: one term, coefficient 1, no tensor, no sum, FREE indices
        a, b, c, ... (VIRT) and i, j, k, ... (OCC):
        a+_a a+_b .. a_j a_i  (n = 2: a+_a a+_b a_j a_i).
        ValueError unless 1 <= n <= 7 (an int).
    deexcitation(n, *, name="l", pool=None) -> Expr
        Lambda_n = tensor_operator(name, [OCC] * n, [VIRT] * n): summed, with
        amplitudes. ValueError for n < 1.

  alqemi/pipeline/project.py
    vacuum_expectation(x) -> Expr
        <HF| x |HF> as operator-free terms, simplified (Step 6/8 Simplify).
        1. run(x): brackets are expanded first.
        2. Split: every summed GENERAL index becomes two copies of its term, one
           with an OCC index and one with a VIRT index. A free GENERAL index is a
           ValueError (its value depends on its space).
        3. Repeat on every term that still has operators. Each operator kills one
           side of the vacuum: a_a (VIRT) and a+_i (OCC) kill |HF>; a+_a and a_i
           kill <HF|. Take the rightmost |HF>-killer K:
             - none: every operator kills <HF|, the term is 0;
             - K is the last operator: the term is 0;
             - otherwise <HF| X K Y |HF> = <HF| X [K, Y](+) |HF> with Y the rest of
               the string: an anticommutator if len(Y) is odd, else a commutator
               (always rank-reducing); the result goes through IdentityExpand and
               DeltaEvaluate (not Simplify: its dummy-renaming search is costly,
               so it runs once, at the end).
        4. simplify. DanglingSum propagates (e.g. sum_p a+_p a_p = N_occ).
    projection(n, x) -> Expr
        n = 0: vacuum_expectation(x). n >= 1: <mu_n| x |HF>, i.e.
        vacuum_expectation(tau_mu^dagger x) with tau_mu = excitation(n); its free
        indices are excitation(n)'s. tau_mu^dagger = a+_i a+_j .. a_b a_a.
        ValueError if a free index of x has a name excitation(n) uses.

  alqemi/pipeline/bch.py
    bch(h, t, *, order=4) -> Expr
        sum_{k=0}^{order} (1/k!) C_k with C_0 = run(h), C_k = run([C_{k-1}, t]),
        simplified. Equals e^{-T} H e^{T} exactly once order reaches the point
        where the series ends (4 for a two-body H). ValueError for order < 0.

  Public API: excitation, deexcitation, vacuum_expectation, projection, bch join
  ``alqemi.__all__``.
"""
from fractions import Fraction

import numpy as np
import pytest

import alqemi
from alqemi.builders import (
    cluster,
    deexcitation,
    excitation,
    fluctuation_potential,
    fock_operator,
    hamiltonian,
    tensor_operator,
)
from alqemi.core.expr import Expr, Term, as_expr, comm
from alqemi.core.indices import Index, Space
from alqemi.core.operators import OpString, ann, cre
from alqemi.core.tensor import Swap, Tensor
from alqemi.pipeline.bch import bch
from alqemi.pipeline.delta_evaluate import DanglingSum
from alqemi.pipeline.project import projection, vacuum_expectation
from alqemi.pipeline.run import run
from alqemi.pipeline.simplify import simplify
from alqemi.tests.evaluate import TensorValues, evaluate, mismatch
from alqemi.tests.oracle import FockSpace, commutator

O, V, G = Space.OCC, Space.VIRT, Space.GENERAL
a, b, c = (Index(x, V) for x in "abc")
i, j, k = (Index(x, O) for x in "ijk")
A, B = Index("A", V), Index("B", V)
I, J = Index("I", O), Index("J", O)
P, Q = Index("P"), Index("Q")
p, q = Index("p"), Index("q")


def T(*factors, coeff=1, scalars=(), summed=()):
    return Term(coeff=coeff, scalars=tuple(scalars), factors=tuple(factors), summed=frozenset(summed))


def X(*terms):
    return Expr(tuple(terms))


@pytest.fixture(scope="module")
def fock():
    return FockSpace(n_occ=2, n_virt=2)


@pytest.fixture(scope="module")
def values(fock):
    return TensorValues(fock, seed=9)


# ---- oracle helpers --------------------------------------------------------------
def hf_state(fock):
    state = np.zeros(fock.dim)
    state[0] = 1.0  # the empty Fock state
    for m in range(fock.n_occ):
        state = fock._cre[m] @ state
    return state


def hf_expectation(fock, mat):
    s = hf_state(fock)
    return float(s @ mat @ s)


def matrix(fock, x, values, g=None):
    return evaluate(fock, x, {} if g is None else g, values)


def operator_free(result):
    return all(t.factors == () for t in result)


def check_vacuum(fock, values, result, x):
    """result (scalar terms) == <HF| x |HF> for every assignment of x's free indices."""
    assert operator_free(result)
    extra = sorted(as_expr(x).free_indices, key=str)

    def direct(g):
        return hf_expectation(fock, matrix(fock, x, values, g)) * fock.identity

    return mismatch(fock, result, direct, extra=extra, tensors=values) is None


def taudag(n):
    """<mu_n| = <HF| a+_i a+_j .. a_b a_a, written directly for the oracle."""
    occ, virt = [i, j, k][:n], [a, b, c][:n]
    ops = [cre(x) for x in occ] + [ann(x) for x in reversed(virt)]
    return OpString(tuple(ops)), occ + virt


def check_projection(fock, values, n, result, x):
    assert operator_free(result)
    string, free = taudag(n)

    def direct(g):
        return hf_expectation(fock, fock.string(string, g) @ matrix(fock, x, values)) * fock.identity

    return mismatch(fock, result, direct, extra=free, tensors=values) is None


def expm_nilpotent(mat):
    out, term, k = np.eye(len(mat)), np.eye(len(mat)), 1
    while True:
        term = term @ mat / k
        if not term.any():
            return out
        out, k = out + term, k + 1


# =========================================================================
# excitation and deexcitation
class TestExcitation:
    def test_rank_2(self):
        (term,) = excitation(2).terms
        assert term.coeff == 1 and term.scalars == () and term.summed == frozenset()
        assert term.factors == (cre(a) * cre(b) * ann(j) * ann(i),)

    def test_rank_1(self):
        (term,) = excitation(1).terms
        assert term.factors == (cre(a) * ann(i),)

    def test_rank_3_free_indices(self):
        assert excitation(3).free_indices == {a, b, c, i, j, k}

    @pytest.mark.parametrize("n", [0, -1, 8, 2.0, "2"])
    def test_invalid(self, n):
        with pytest.raises(ValueError):
            excitation(n)


class TestDeexcitation:
    def test_is_tensor_operator(self):
        assert deexcitation(2) == tensor_operator("l", [O, O], [V, V])

    def test_name(self):
        (term,) = deexcitation(1, name="z").terms
        assert term.scalars[0].name == "z"
        assert term.coeff == 1

    def test_invalid(self):
        with pytest.raises(ValueError):
            deexcitation(0)


# =========================================================================
# vacuum_expectation: elementary cases
ELEMENTARY = {
    "a_A a+_B": (X(T(ann(A), cre(B))), 1),
    "a+_I a_J": (X(T(cre(I), ann(J))), 1),
    "a+_A a_B": (X(T(cre(A), ann(B))), 0),
    "a_I a+_J": (X(T(ann(I), cre(J))), 0),
    "a_A": (X(T(ann(A))), 0),
    "a+_I": (X(T(cre(I))), 0),
    "a+_A": (X(T(cre(A))), 0),
    "a_I": (X(T(ann(I))), 0),
    "a+_I a_J a_A a+_B": (X(T(cre(I), ann(J), ann(A), cre(B))), 1),
    "a_A a+_B a+_I a_J": (X(T(ann(A), cre(B), cre(I), ann(J))), 1),
    "a+_I a+_J a_J a_I": (X(T(cre(I), cre(J), ann(J), ann(I))), 2),
    "a_A a_B a+_B a+_A": (X(T(ann(A), ann(B), cre(B), cre(A))), 2),
    "a+_I a_A a+_B a_J": (X(T(cre(I), ann(A), cre(B), ann(J))), 1),
    "constant": (X(T(scalars=[Tensor("w")], coeff=3)), 1),
}


class TestElementary:
    @pytest.mark.parametrize("name", list(ELEMENTARY))
    def test_value(self, fock, values, name):
        x, _ = ELEMENTARY[name]
        assert check_vacuum(fock, values, vacuum_expectation(x), x)

    @pytest.mark.parametrize("name", [n for n, (_, terms) in ELEMENTARY.items() if terms == 0])
    def test_zero(self, name):
        assert len(vacuum_expectation(ELEMENTARY[name][0])) == 0

    def test_single_contraction_is_a_delta(self):
        (term,) = vacuum_expectation(ELEMENTARY["a_A a+_B"][0]).terms
        assert term.coeff == 1 and term.factors == ()
        assert {ix for s in term.scalars for ix in s.indices} == {A, B}

    def test_constant_unchanged(self):
        (term,) = vacuum_expectation(ELEMENTARY["constant"][0]).terms
        assert term.coeff == 3 and term.scalars == (Tensor("w"),)


# =========================================================================
# vacuum_expectation: general indices, brackets, errors
class TestSplitting:
    def test_one_body_becomes_occupied_trace(self, fock, values):
        x = fock_operator()
        out = vacuum_expectation(x)
        (term,) = out.terms
        (f,) = term.scalars
        assert f.name == "f" and f.upper == f.lower and f.upper[0].space is O
        assert term.summed == {f.upper[0]} and term.coeff == 1
        assert check_vacuum(fock, values, out, x)

    def test_no_general_index_left(self, fock, values):
        out = vacuum_expectation(fluctuation_potential())
        assert all(ix.space is not G for t in out for ix in t.summed | t.free_indices)

    def test_free_general_index_rejected(self):
        with pytest.raises(ValueError):
            vacuum_expectation(X(T(cre(P), ann(Q))))

    def test_free_general_inside_bracket_rejected(self):
        with pytest.raises(ValueError):
            vacuum_expectation(comm(cre(P) * ann(Q), cluster(1)))

    def test_dangling_sum(self):
        with pytest.raises(DanglingSum):
            vacuum_expectation(X(T(cre(p), ann(p), summed=[p])))

    def test_mixed_free_and_general_summed(self, fock, values):
        x = X(T(cre(p), ann(A), scalars=[Tensor("g", (p,), (I,))], summed=[p]))
        out = vacuum_expectation(x)
        assert check_vacuum(fock, values, out, x)

    def test_brackets_are_expanded_first(self, fock, values):
        x = comm(fock_operator(), cluster(1))
        out = vacuum_expectation(x)
        assert check_vacuum(fock, values, out, X(T(x)))


# =========================================================================
# vacuum_expectation of the standard operators
class TestStandardOperators:
    def test_hf_energy(self, fock, values):
        """<HF| H |HF> = sum_i f_ii - 1/2 sum_ij <ij||ij>: exactly two terms."""
        out = vacuum_expectation(hamiltonian())
        assert sorted(t.coeff for t in out) == [Fraction(-1, 2), 1]
        names = sorted(t.scalars[0].name for t in out)
        assert names == ["f", "v"]
        assert all(ix.space is O for t in out for ix in t.summed)
        assert check_vacuum(fock, values, out, hamiltonian())

    def test_normal_ordered_forms_agree(self, fock, values):
        plain = vacuum_expectation(hamiltonian())
        normal = vacuum_expectation(hamiltonian(normal_ordered=True))
        assert len(normal) == 2
        assert mismatch(fock, plain, normal, tensors=values) is None

    def test_normal_ordered_operator_terms_vanish(self):
        for term in fluctuation_potential(normal_ordered=True).terms[:9]:
            assert len(vacuum_expectation(X(term))) == 0

    def test_cluster_operators_vanish(self):
        for n in (1, 2):
            assert len(vacuum_expectation(cluster(n))) == 0
            assert len(vacuum_expectation(deexcitation(n))) == 0

    def test_core_hamiltonian(self, fock, values):
        out = vacuum_expectation(hamiltonian(form="core"))
        assert check_vacuum(fock, values, out, hamiltonian(form="core"))


# =========================================================================
# projection
class TestProjection:
    def test_rank_0_is_vacuum_expectation(self):
        assert projection(0, hamiltonian()) == vacuum_expectation(hamiltonian())

    def test_singles_of_h_is_fock(self, fock, values):
        """<^a_i| H |HF> = f_ai (Brillouin)."""
        out = projection(1, hamiltonian())
        (term,) = out.terms
        assert term.coeff == 1 and term.summed == frozenset()
        (f,) = term.scalars
        assert f.name == "f" and set(f.indices) == {a, i}
        assert check_projection(fock, values, 1, out, hamiltonian())

    def test_doubles_of_h_is_integral(self, fock, values):
        """<^{ab}_{ij}| H |HF> = <ab||ij>."""
        out = projection(2, hamiltonian())
        (term,) = out.terms
        assert abs(term.coeff) == 1 and term.summed == frozenset()
        (v,) = term.scalars
        assert v.name == "v" and set(v.indices) == {a, b, i, j}
        assert check_projection(fock, values, 2, out, hamiltonian())

    def test_free_indices(self):
        assert projection(2, hamiltonian()).free_indices == {a, b, i, j}

    @pytest.mark.parametrize(
        "n, name, x",
        [
            (1, "[F, T1]", lambda: comm(fock_operator(), cluster(1))),
            (1, "[Phi, T2]", lambda: comm(fluctuation_potential(), cluster(2))),
            (2, "[F, T2]", lambda: comm(fock_operator(), cluster(2))),
            (2, "[H, T1]", lambda: comm(hamiltonian(), cluster(1))),
            (1, "H_no", lambda: hamiltonian(normal_ordered=True)),
            (2, "Lambda2", lambda: deexcitation(2)),
            (2, "T2", lambda: cluster(2)),
        ],
    )
    def test_value(self, fock, values, n, name, x):
        x = x()
        out = projection(n, x)
        assert check_projection(fock, values, n, out, x if isinstance(x, Expr) else X(T(x)))

    def test_name_clash_rejected(self):
        with pytest.raises(ValueError):
            projection(1, X(T(cre(a), ann(I))))


# =========================================================================
# bch
class TestBCH:
    def test_order_0(self, fock, values):
        out = bch(fock_operator(), cluster(1), order=0)
        assert mismatch(fock, out, fock_operator(), tensors=values) is None

    def test_order_1(self, fock, values):
        h, t = hamiltonian(), cluster(1)
        out = bch(h, t, order=1)
        hm, tm = matrix(fock, h, values), matrix(fock, t, values)
        assert mismatch(fock, out, lambda g: hm + commutator(hm, tm), tensors=values) is None

    def test_truncated_order_2_doubles(self, fock, values):
        h, t = hamiltonian(), cluster(2)
        out = bch(h, t, order=2)
        hm, tm = matrix(fock, h, values), matrix(fock, t, values)
        c1 = commutator(hm, tm)
        expected = hm + c1 + 0.5 * commutator(c1, tm)
        assert mismatch(fock, out, lambda g: expected, tensors=values) is None

    @pytest.mark.parametrize(
        "h, order", [(lambda: fock_operator(), 2), (lambda: hamiltonian(), 4)]
    )
    def test_exact_similarity_transform_t1(self, fock, values, h, order):
        h, t = h(), cluster(1)
        out = bch(h, t, order=order)
        hm, tm = matrix(fock, h, values), matrix(fock, t, values)
        exact = expm_nilpotent(-tm) @ hm @ expm_nilpotent(tm)
        assert mismatch(fock, out, lambda g: exact, tensors=values) is None

    def test_sum_of_cluster_operators(self, fock, values):
        h, t = fock_operator(), cluster(1) + cluster(2)
        out = bch(h, t, order=2)
        hm, tm = matrix(fock, h, values), matrix(fock, t, values)
        exact = expm_nilpotent(-tm) @ hm @ expm_nilpotent(tm)
        assert mismatch(fock, out, lambda g: exact, tensors=values) is None

    def test_result_is_simplified(self):
        out = bch(fock_operator(), cluster(1), order=2)
        assert simplify(out) == out

    def test_invalid_order(self):
        with pytest.raises(ValueError):
            bch(fock_operator(), cluster(1), order=-1)


# =========================================================================
# Coupled-cluster equations
class TestCoupledCluster:
    def test_ccsd_energy(self, fock, values):
        """E = E_HF + f_ia t^a_i + 1/4 <ij||ab> t^{ab}_{ij} + 1/2 <ij||ab> t^a_i t^b_j."""
        h, t = hamiltonian(), cluster(1) + cluster(2)
        out = projection(0, bch(h, t, order=2))
        assert len(out) == 5
        assert sorted(abs(term.coeff) for term in out) == sorted(
            [Fraction(1), Fraction(1, 2), Fraction(1), Fraction(1, 4), Fraction(1, 2)]
        )
        hm, tm = matrix(fock, h, values), matrix(fock, t, values)
        exact = hf_expectation(fock, expm_nilpotent(-tm) @ hm @ expm_nilpotent(tm))
        assert mismatch(fock, out, lambda g: exact * fock.identity, tensors=values) is None

    def test_ccd_doubles_residual(self, fock, values):
        h, t = hamiltonian(), cluster(2)
        x = bch(h, t, order=2)
        out = projection(2, x)
        assert out.free_indices == {a, b, i, j}
        hm, tm = matrix(fock, h, values), matrix(fock, t, values)
        c1 = commutator(hm, tm)
        hbar = hm + c1 + 0.5 * commutator(c1, tm)
        string, free = taudag(2)

        def direct(g):
            return hf_expectation(fock, fock.string(string, g) @ hbar) * fock.identity

        assert mismatch(fock, out, direct, extra=free, tensors=values) is None

    def test_ccsd_singles_residual_linear(self, fock, values):
        h, t = hamiltonian(), cluster(1) + cluster(2)
        x = bch(h, t, order=1)
        out = projection(1, x)
        hm, tm = matrix(fock, h, values), matrix(fock, t, values)
        hbar = hm + commutator(hm, tm)
        string, free = taudag(1)

        def direct(g):
            return hf_expectation(fock, fock.string(string, g) @ hbar) * fock.identity

        assert mismatch(fock, out, direct, extra=free, tensors=values) is None


# =========================================================================
# Public API
STEP9_NAMES = {"excitation", "deexcitation", "vacuum_expectation", "projection", "bch"}


@pytest.mark.parametrize("name", sorted(STEP9_NAMES))
def test_exported(name):
    assert name in alqemi.__all__ and hasattr(alqemi, name)
