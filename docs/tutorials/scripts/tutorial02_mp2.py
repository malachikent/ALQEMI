"""Tutorial 2: MP2 from first principles.

Derive the first-order doubles equation and the second-order energy with ALQEMI,
generate numpy code for both, solve the amplitude equation for water/STO-3G, and
compare the energy with the SCF program's own MP2.

Run:  python tutorial02_mp2.py          (Psi4, the default)
      ALQEMI_PROGRAM=pyscf python tutorial02_mp2.py
"""
import numpy as np

from alqemi import bch, cluster, einsum_code, fluctuation_potential, fock_operator
from alqemi import hamiltonian, latex, projection, vacuum_expectation

from integrals import SpinOrbitalIntegrals
from scf import reference_energy, run_scf

CHECKS = []  # (label, ALQEMI value, reference value)

# ---- 1. Derive ------------------------------------------------------------------
# First-order amplitudes:  <ab_ij| [F, T2] + V |HF> = 0
# Energy through 2nd order: <HF| H + [H, T2] |HF> = E_HF + 1/4 sum t <ij||ab>
T2 = cluster(2)
residual = projection(2, bch(fock_operator(), T2, order=1)) + projection(2, fluctuation_potential())
energy = vacuum_expectation(bch(hamiltonian(), T2, order=1))

print(latex(residual, lhs=r"R^{ab}_{ij}", group=True))
print(latex(energy, lhs="E"))

# ---- 2. Generate code -----------------------------------------------------------
code = einsum_code(residual, "mp2_residual", group=True) + "\n\n" + einsum_code(energy, "mp2_energy")
print(code)
namespace = {}
exec(code, namespace)
mp2_residual, mp2_energy = namespace["mp2_residual"], namespace["mp2_energy"]

# ---- 3. Numbers -----------------------------------------------------------------
scf = run_scf()
moeris = SpinOrbitalIntegrals(scf.h_ao, scf.eri_ao, scf.ca, scf.cb, scf.nocc_a, scf.nocc_b)

fo, fv = np.diag(moeris.f_OO), np.diag(moeris.f_VV)
d2 = fo[:, None, None, None] + fo[None, :, None, None] - fv[None, None, :, None] - fv[None, None, None, :]

t2 = np.zeros_like(moeris.g_OOVV)
for iteration in range(100):
    r2 = mp2_residual(moeris, t2)
    t2 = t2 + r2 / d2
    if np.abs(r2).max() < 1e-12:
        break

e_hf = mp2_energy(moeris, np.zeros_like(t2)) + scf.e_nuc
e_mp2 = mp2_energy(moeris, t2) + scf.e_nuc
CHECKS.append(("HF energy", e_hf, scf.e_scf))
CHECKS.append(("MP2 energy", e_mp2, reference_energy(scf, "mp2")))

for label, ours, theirs in CHECKS:
    print(f"{label:12s} ALQEMI {ours:.10f}  {scf.program} {theirs:.10f}  diff {ours - theirs:.1e}")
