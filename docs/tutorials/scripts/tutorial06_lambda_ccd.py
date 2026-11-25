"""Tutorial 6: Left-hand (Lambda) CCD equations and a property.

Derive the Lambda-CCD equations  <HF|(1 + Lambda2) [Hbar, tau_ij^ab]|HF> = 0  and the
first-order property  <HF|(1 + Lambda2) e^{-T} U e^{T}|HF>  for a one-electron
operator U (here the z dipole), then check the property against a finite-field
derivative of the CCD energy, dE/d(eps) for H + eps U with the orbitals held fixed.
That check needs no CC program at all: it holds only if both the Lambda equations
and the property expression are right.

Two practical points this tutorial shows:

* ALQEMI has no product of two expressions yet; ``times`` below builds one with
  ``Term``. Dummy names must not clash, so the operators share one IndexPool.
* ``bch`` relabels dummies to letters (i, j, a, b, ...). Before Hbar goes into a
  commutator with tau_ij^ab, whose free indices are i, j, a, b, the clashing
  dummies are renamed (``move_dummies``).

Run:  python tutorial06_lambda_ccd.py
"""
import numpy as np

from alqemi import Expr, IndexPool, Space, Term, bch, cluster, comm, deexcitation
from alqemi import einsum_code, excitation, hamiltonian, latex, projection, simplify
from alqemi import tensor_operator, vacuum_expectation

from integrals import SpinOrbitalIntegrals
from scf import run_scf

CHECKS = []  # (label, ALQEMI value, reference value)


def times(left, right):
    """left * right for expressions whose dummy names do not clash."""
    return Expr(tuple(
        Term(l.coeff * r.coeff, l.scalars + r.scalars, l.factors + r.factors, l.summed | r.summed)
        for l in left for r in right
    ))


def move_dummies(expr, names, pool):
    """Rename every dummy of ``expr`` called one of ``names`` to a fresh pool name."""
    out = []
    for term in expr:
        avoid = set(names) | {ix.name for ix in term.summed}
        mapping = {}
        for ix in term.summed:
            if ix.name in names:
                mapping[ix] = pool.fresh(ix.space, avoid)
                avoid.add(mapping[ix].name)
        out.append(term.rename(mapping))
    return Expr(tuple(out))


# ---- 1. Derive ------------------------------------------------------------------
pool = IndexPool()
H, T2 = hamiltonian(), cluster(2)
Lambda2 = deexcitation(2, pool=pool)          # l^{ij}_{ab}, numbered dummies
tau = excitation(2)                           # a+_a a+_b a_j a_i, free a, b, i, j

Hbar = bch(H, T2, order=2)                    # enough for CCD
energy = vacuum_expectation(Hbar)
t_residual = projection(2, Hbar)

Hbar = move_dummies(Hbar, {ix.name for ix in tau.free_indices}, pool)
H_tau = Expr((Term(factors=(comm(Hbar, tau),)),))
l_residual = simplify(vacuum_expectation(H_tau) + vacuum_expectation(times(Lambda2, H_tau)))
print(latex(l_residual, lhs=r"R^{ij}_{ab}", terms_per_line=3, group=True))

# Property: U = sum_pq u^p_q a+_p a_q, and e^{-T} U e^{T} = U + [U, T2] (one-body U).
U = tensor_operator("u", [Space.GENERAL], [Space.GENERAL], pool=pool)
Ubar = bch(U, T2, order=1)
Ubar = move_dummies(Ubar, {ix.name for ix in Lambda2.terms[0].summed}, pool)
prop = simplify(vacuum_expectation(Ubar) + vacuum_expectation(times(Lambda2, Ubar)))
print(latex(prop, lhs=r"\langle U \rangle"))

# ---- 2. Generate code -----------------------------------------------------------
integrals = {"f": "moeris.f_{block}", "v": "moeris.g_{block}", "u": "moeris.u_{block}"}
namespace = {}
for expr, name in [(energy, "ccd_energy"), (t_residual, "ccd_t_residual"),
                   (l_residual, "ccd_l_residual"), (prop, "ccd_property")]:
    exec(einsum_code(expr, name, integrals=integrals, group=True), namespace)
ccd_energy, ccd_t_residual = namespace["ccd_energy"], namespace["ccd_t_residual"]
ccd_l_residual, ccd_property = namespace["ccd_l_residual"], namespace["ccd_property"]

# ---- 3. Numbers -----------------------------------------------------------------
scf = run_scf()
dipole_z = scf.dipole_ao[2]
moeris = SpinOrbitalIntegrals(scf.h_ao, scf.eri_ao, scf.ca, scf.cb, scf.nocc_a, scf.nocc_b,
                              one_body={"u": dipole_z})


class Perturbed:
    """moeris for H + eps U: every Fock block gains eps * u (orbitals unchanged)."""

    def __init__(self, base, eps):
        self.base, self.eps = base, eps

    def __getattr__(self, name):
        if name.startswith("f_"):
            return getattr(self.base, name) + self.eps * getattr(self.base, "u_" + name[2:])
        return getattr(self.base, name)


def denominator(m):
    fo, fv = np.diag(m.f_OO), np.diag(m.f_VV)
    return fo[:, None, None, None] + fo[None, :, None, None] - fv[None, None, :, None] - fv[None, None, None, :]


def solve_t(m):
    d2, t2 = denominator(m), np.zeros_like(m.g_OOVV)
    for _ in range(500):
        r2 = ccd_t_residual(m, t2)
        t2 = t2 + r2 / d2
        if np.abs(r2).max() < 1e-11:
            return t2
    raise RuntimeError("T2 did not converge")


def solve_l(m, t2):
    d2, l2 = denominator(m), np.zeros_like(m.g_OOVV)
    for _ in range(500):
        r2 = ccd_l_residual(m, l2, t2)
        l2 = l2 + r2 / d2
        if np.abs(r2).max() < 1e-11:
            return l2
    raise RuntimeError("Lambda2 did not converge")


t2 = solve_t(moeris)
l2 = solve_l(moeris, t2)
analytic = ccd_property(moeris, l2, t2)

eps = 1e-4
e_plus = ccd_energy(Perturbed(moeris, eps), solve_t(Perturbed(moeris, eps)))
e_minus = ccd_energy(Perturbed(moeris, -eps), solve_t(Perturbed(moeris, -eps)))
finite_field = (e_plus - e_minus) / (2 * eps)
CHECKS.append(("dE/d eps (z dipole)", analytic, finite_field))

for label, ours, theirs in CHECKS:
    print(f"{label:20s} Lambda {ours:.10f}  finite field {theirs:.10f}  diff {ours - theirs:.1e}")
