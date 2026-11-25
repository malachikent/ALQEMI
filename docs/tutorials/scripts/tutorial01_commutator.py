"""Tutorial 1: your first commutator.

Evaluate [a+_p a_q, a+_r a_s] with ALQEMI, watch the HJO identities it applies, and
check the answer yourself with explicit fermion matrices. The result is HJO (1.8.14):

    [a+_p a_q, a+_r a_s] = delta_qr a+_p a_s - delta_ps a+_r a_q

Run:  python tutorial01_commutator.py      (needs numpy for the matrix check only)
"""
import itertools

import numpy as np

from alqemi import Delta, Expr, Index, Kind, OpString, Term, ann, comm, cre, latex, run

CHECKS = []  # (label, ALQEMI value, reference value)

# ---- 1. Build and evaluate ------------------------------------------------------
p, q, r, s = (Index(name) for name in "pqrs")       # GENERAL space by default
x = comm(OpString((cre(p), ann(q))), OpString((cre(r), ann(s))))

trace = []
result = run(x, trace=trace)
print(result)
print(latex(result))
for step in trace:
    print(f"{step.rule:12s} {step.bracket}")

# ---- 2. Compare with the textbook form ------------------------------------------
expected = Expr((
    Term(scalars=(Delta(q, r),), factors=(OpString((cre(p), ann(s))),)),
    Term(-1, scalars=(Delta(p, s),), factors=(OpString((cre(r), ann(q))),)),
))
print("same as (1.8.14):", len(run(Expr(result.terms + tuple(t.scale(-1) for t in expected)))) == 0)

# ---- 3. Check with matrices -----------------------------------------------------
# Jordan-Wigner: n modes, a_k = Z x .. x Z x a x 1 x .. x 1 on 2^n states.
n = 3
a1, z, one = np.array([[0.0, 1.0], [0.0, 0.0]]), np.diag([1.0, -1.0]), np.eye(2)


def annihilator(k):
    m = np.array([[1.0]])
    for j in range(n):
        m = np.kron(m, z if j < k else a1 if j == k else one)
    return m


A = [annihilator(k) for k in range(n)]


def matrix(expr, values):
    """The matrix of an operator expression for one assignment index -> mode."""
    total = np.zeros((2 ** n, 2 ** n))
    for term in expr:
        c = float(term.coeff)
        for sc in term.scalars:
            c *= float(values[sc.p] == values[sc.q])     # only deltas appear here
        m = np.eye(2 ** n)
        for factor in term.factors:
            for op in factor:
                a = A[values[op.index]]
                m = m @ (a.T if op.kind is Kind.CRE else a)
        total += c * m
    return total


worst = 0.0
for modes in itertools.product(range(n), repeat=4):
    values = dict(zip((p, q, r, s), modes))
    epq = A[values[p]].T @ A[values[q]]
    ers = A[values[r]].T @ A[values[s]]
    worst = max(worst, np.abs(matrix(result, values) - (epq @ ers - ers @ epq)).max())
print(f"largest matrix difference over all {n ** 4} index assignments: {worst}")
CHECKS.append(("matrix check", worst, 0.0))
