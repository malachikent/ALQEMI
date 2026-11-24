"""Step 10 tests: two exact speed-ups, with results unchanged.

These tests DEFINE Step 10. Neither change may alter any result: the fast
Simplify must give EXACTLY the output of the brute-force search it replaces
(tests/reference_simplify.py), and projections must give the same values
(checked by the Step 9 oracle tests, which all still apply).

  1. Simplify (pipeline/simplify.py): fingerprint-restricted search.
     Each dummy gets a fingerprint that uses no dummy names: its space and the
     sorted list of where it occurs:
       - an operator: (kind, index of the run of same-kind operators it is in);
       - a delta: the partner's name if the partner is free, else "dummy";
       - a tensor: (name, swap, ranks, side), side = upper/lower, or "either"
         when the tensor has a swap symmetry.
     A renaming that maps a term onto a like term (or onto minus itself) keeps
     every fingerprint, so the search only permutes dummies with EQUAL
     fingerprints. Output is identical to the brute-force search.

  2. vacuum_expectation (pipeline/project.py): occupation-count pruning.
     Right after splitting general indices, a term whose operators change the
     number of electrons in the occupied orbitals, or in the virtual orbitals
     (created - annihilated != 0 in either space), is dropped: it maps |HF> to an
     orthogonal state, so its expectation value is exactly 0.

  Time budgets (about 10x slower than the implementation, so CI noise is safe;
  the pre-Step-10 code exceeds each of them):
    - simplify of the raw [[H, T2], T2] (32 terms): under 0.5 s (was 1.7 s);
    - CCSD energy, projection(0, bch(H, T1 + T2, order=2)): under 3 s (was ~9 s);
    - CCD doubles residual, projection(2, .) of a prebuilt bch(H, T2, order=2):
      under 4 s (about 7.6 s without the pruning, 3.5-7 s before Step 10).
"""
import time

import pytest

from alqemi.builders import cluster, deexcitation, fluctuation_potential, fock_operator, hamiltonian
from alqemi.core.expr import Delta, Expr, Term, comm
from alqemi.core.indices import Index, Space
from alqemi.core.operators import ann, cre
from alqemi.core.pool import IndexPool
from alqemi.core.tensor import Swap, Tensor
from alqemi.pipeline.bch import bch
from alqemi.pipeline.delta_evaluate import delta_evaluate
from alqemi.pipeline.identity_expand import identity_expand
from alqemi.pipeline.project import projection, vacuum_expectation
from alqemi.pipeline.run import run
from alqemi.pipeline.simplify import simplify
from alqemi.tests.reference_simplify import simplify as reference_simplify

O, V = Space.OCC, Space.VIRT


def raw(x):
    """Expanded and delta-evaluated, NOT simplified: Simplify's real input."""
    return delta_evaluate(identity_expand(x, IndexPool()))


def T(*factors, coeff=1, scalars=(), summed=()):
    return Term(coeff=coeff, scalars=tuple(scalars), factors=tuple(factors), summed=frozenset(summed))


# =========================================================================
# 1. Simplify gives exactly the brute-force output
def _corpus():
    h, hn = hamiltonian(), hamiltonian(normal_ordered=True)
    cases = {}
    for hname, hx in [("F", fock_operator()), ("Phi", fluctuation_potential()), ("H", h), ("H_no", hn)]:
        for tname, tx in [("T1", cluster(1)), ("T2", cluster(2)), ("T1+T2", cluster(1) + cluster(2)),
                          ("L2", deexcitation(2))]:
            cases[f"[{hname},{tname}]"] = lambda hx=hx, tx=tx: comm(hx, tx)
    cases["[[H,T1],T1]"] = lambda: comm(run(comm(hamiltonian(), cluster(1))), cluster(1))
    cases["[[H,T1],T2]"] = lambda: comm(run(comm(hamiltonian(), cluster(1))), cluster(2))
    cases["[[H,T2],T2]"] = lambda: comm(run(comm(hamiltonian(), cluster(2))), cluster(2))
    return cases


CORPUS = _corpus()


@pytest.mark.parametrize("name", list(CORPUS))
def test_identical_to_brute_force(name):
    x = raw(CORPUS[name]())
    assert simplify(x) == reference_simplify(x)


# Hand-made terms where fingerprints tie, merge with signs, or vanish.
a, b, c = (Index(n, V) for n in "abc")
i, j = (Index(n, O) for n in "ij")
I, J = Index("I", O), Index("J", O)
A, B = Index("A", V), Index("B", V)
p, q = Index("p"), Index("q")
P, Q = Index("P"), Index("Q")

HAND = {
    "tied dummies swap": Expr((
        T(cre(a), cre(b), scalars=[Tensor("t", (a, b), (I, J))], summed=[a, b]),
        T(cre(b), cre(a), scalars=[Tensor("t", (a, b), (I, J))], summed=[a, b]),
    )),
    "symmetric sum is zero": Expr((T(cre(p), cre(q), ann(P), summed=[p, q]),)),
    "two identical tensors": Expr((
        T(scalars=[Tensor("t", (a,), (i,)), Tensor("t", (b,), (j,)), Tensor("v", (i, j), (a, b), Swap.SYMMETRIC)],
          summed=[a, b, i, j]),
        T(scalars=[Tensor("t", (b,), (j,)), Tensor("t", (a,), (i,)), Tensor("v", (j, i), (b, a), Swap.SYMMETRIC)],
          summed=[a, b, i, j], coeff=2),
    )),
    "delta to free vs to dummy": Expr((
        T(cre(Q), scalars=[Tensor("f", (p,), (P,), Swap.SYMMETRIC)], summed=[p]),
        T(cre(Q), scalars=[Tensor("f", (P,), (p,), Swap.SYMMETRIC)], summed=[p]),
    )),
    "swap symmetric either side": Expr((
        T(ann(A), scalars=[Tensor("f", (a,), (A,), Swap.SYMMETRIC), Tensor("s", (a,))], summed=[a]),
        T(ann(A), scalars=[Tensor("f", (A,), (a,), Swap.SYMMETRIC), Tensor("s", (a,))], summed=[a]),
    )),
    "no swap keeps sides apart": Expr((
        T(ann(A), scalars=[Tensor("g", (a,), (A,)), Tensor("s", (a,))], summed=[a]),
        T(ann(A), scalars=[Tensor("g", (A,), (a,)), Tensor("s", (a,))], summed=[a]),
    )),
    "swap symmetric two dummies": Expr((
        T(ann(A), scalars=[Tensor("f", (a,), (b,), Swap.SYMMETRIC), Tensor("u", (a,)), Tensor("w", (b,))],
          summed=[a, b]),
        T(ann(A), scalars=[Tensor("f", (b,), (a,), Swap.SYMMETRIC), Tensor("u", (a,)), Tensor("w", (b,))],
          summed=[a, b]),
    )),
    "delta between dummies": Expr((
        T(cre(p), ann(q), scalars=[Delta(p, q)], summed=[p, q]),
        T(cre(q), ann(p), scalars=[Delta(p, q)], summed=[p, q]),
    )),
    "runs distinguish": Expr((
        T(cre(a), ann(I), cre(b), scalars=[Tensor("u", (a, b))], summed=[a, b]),
        T(cre(b), ann(I), cre(a), scalars=[Tensor("u", (a, b))], summed=[a, b]),
    )),
}


@pytest.mark.parametrize("name", list(HAND))
def test_hand_cases_identical_to_brute_force(name):
    x = HAND[name]
    assert simplify(x) == reference_simplify(x)


def test_hand_case_results():
    assert len(simplify(HAND["tied dummies swap"])) == 0
    assert len(simplify(HAND["symmetric sum is zero"])) == 0
    (term,) = simplify(HAND["two identical tensors"]).terms
    assert term.coeff == 3
    assert len(simplify(HAND["delta to free vs to dummy"])) == 1
    assert len(simplify(HAND["swap symmetric either side"])) == 1
    assert len(simplify(HAND["no swap keeps sides apart"])) == 2
    assert len(simplify(HAND["runs distinguish"])) == 0  # u^{ab} antisymmetric, a+_a, a+_b in separate runs
    (term,) = simplify(HAND["swap symmetric two dummies"]).terms
    assert term.coeff == 2
    (term,) = simplify(HAND["delta between dummies"]).terms
    assert term.coeff == 2  # u^{ab} antisymmetric, a+_a, a+_b in separate runs


# =========================================================================
# 2. Occupation-count pruning keeps every value (the Step 9 oracle tests cover
#    values); here: exact zeros for strings that change an occupation count.
@pytest.mark.parametrize(
    "string",
    [
        (cre(A), ann(I)),  # moves an electron occupied -> virtual
        (cre(I), ann(A)),
        (cre(A), cre(B), ann(J), ann(I)),
        (cre(A), ann(B), cre(I), ann(A)),
    ],
)
def test_occupation_changing_strings_vanish(string):
    assert len(vacuum_expectation(Expr((T(*string),)))) == 0


# =========================================================================
# Time budgets
def _best_of(f, repeats=2):
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        f()
        best = min(best, time.perf_counter() - t)
    return best


def test_budget_simplify_double_commutator():
    x = raw(CORPUS["[[H,T2],T2]"]())
    assert len(x) == 32
    assert _best_of(lambda: simplify(x)) < 0.5


def test_budget_ccsd_energy():
    h, t = hamiltonian(), cluster(1) + cluster(2)
    elapsed = _best_of(lambda: projection(0, bch(h, t, order=2)), repeats=1)
    assert elapsed < 3.0


def test_budget_ccd_residual_projection():
    hbar = bch(hamiltonian(), cluster(2), order=2)
    elapsed = _best_of(lambda: projection(2, hbar), repeats=1)
    assert elapsed < 4.0
