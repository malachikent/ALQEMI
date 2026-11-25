"""Tutorial 3: CCD from scratch.

Derive the CCD energy and doubles residual from the BCH expansion, look at them in
LaTeX with and without P(ij) grouping, then solve them for water/STO-3G.

Psi4 has no conventional CCD to compare with, so the numerical check is one that
needs no CCD program: the first Jacobi step from T2 = 0 gives the first-order
amplitudes, so its energy must equal the program's MP2 energy exactly.

Run:  python tutorial03_ccd.py
"""
import numpy as np

from alqemi import bch, cluster, einsum_code, expand_permutations, group_permutations
from alqemi import hamiltonian, latex, projection, simplify, vacuum_expectation

from integrals import SpinOrbitalIntegrals
from scf import reference_energy, run_scf

CHECKS = []  # (label, ALQEMI value, reference value)

# ---- 1. Derive ------------------------------------------------------------------
H, T2 = hamiltonian(), cluster(2)
Hbar = bch(H, T2, order=2)          # [[[H,T2],T2],T2] and beyond vanish in these projections
energy = vacuum_expectation(Hbar)
residual = projection(2, Hbar)

print(latex(energy, lhs="E"))
print(latex(residual, lhs=r"R^{ab}_{ij}", terms_per_line=3))

# ---- 2. Group under (1 - P_ij), (1 - P_ab) --------------------------------------
groups = group_permutations(residual)
print(f"{len(residual)} terms -> {len(groups)} groups")
print(latex(residual, lhs=r"R^{ab}_{ij}", terms_per_line=3, group=True))
assert len(simplify(expand_permutations(groups) - residual)) == 0  # grouping is exact
CHECKS.append(("residual terms", len(residual), 18))
CHECKS.append(("groups", len(groups), 10))

# ---- 3. Generate code and solve -------------------------------------------------
namespace = {}
exec(einsum_code(residual, "ccd_residual", group=True), namespace)
exec(einsum_code(energy, "ccd_energy"), namespace)
ccd_residual, ccd_energy = namespace["ccd_residual"], namespace["ccd_energy"]

scf = run_scf()
moeris = SpinOrbitalIntegrals(scf.h_ao, scf.eri_ao, scf.ca, scf.cb, scf.nocc_a, scf.nocc_b)
fo, fv = np.diag(moeris.f_OO), np.diag(moeris.f_VV)
d2 = fo[:, None, None, None] + fo[None, :, None, None] - fv[None, None, :, None] - fv[None, None, None, :]

t2 = np.zeros_like(moeris.g_OOVV)
for iteration in range(200):
    r2 = ccd_residual(moeris, t2)
    t2 = t2 + r2 / d2
    if iteration == 0:
        CHECKS.append(("MP2 energy (1st step)", ccd_energy(moeris, t2) + scf.e_nuc, reference_energy(scf, "mp2")))
    if np.abs(r2).max() < 1e-10:
        break
e_ccd = ccd_energy(moeris, t2) + scf.e_nuc
print(f"CCD energy {e_ccd:.10f} after {iteration + 1} iterations")

for label, ours, theirs in CHECKS:
    print(f"{label:22s} ALQEMI {ours:.10f}  reference {theirs:.10f}")
