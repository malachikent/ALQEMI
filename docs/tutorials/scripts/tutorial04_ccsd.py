"""Tutorial 4: CCSD, from derivation to a working module.

Derive the CCSD energy and singles/doubles residuals, write them to a Python module
``ccsd_residuals.py``, import it, and solve CCSD for water/STO-3G. The energy is
compared with the SCF program's own CCSD.

Run:  python tutorial04_ccsd.py        (writes ccsd_residuals.py to ALQEMI_OUTPUT or .)
"""
import importlib.util
import os
from fractions import Fraction

import numpy as np

from alqemi import bch, cluster, comm, einsum_code, hamiltonian, projection, run, simplify
from alqemi import vacuum_expectation

from integrals import SpinOrbitalIntegrals
from scf import reference_energy, run_scf

CHECKS = []  # (label, ALQEMI value, reference value)

# ---- 1. Derive ------------------------------------------------------------------
# For a two-body H the BCH series ends at order 4. Order 4 is expensive with
# T = T1 + T2, but only one piece of it survives the projections: H must contract
# with all four T's, which leaves excitation rank a + 2b - 2 for T1^a T2^b (a + b = 4).
# That is 2 (doubles) only for T1^4, and never 1 or 0. So: BCH to order 3 with
# T1 + T2, plus the fourth nested commutator with T1 alone.
H, T1, T2 = hamiltonian(), cluster(1), cluster(2)
Hbar = bch(H, T1 + T2, order=3)

c4 = H
for k in range(4):
    c4 = run(comm(c4, T1))          # [[[[H, T1], T1], T1], T1]

energy = vacuum_expectation(bch(H, T1 + T2, order=2))   # higher orders vanish here
r1 = projection(1, Hbar)
r2 = simplify(projection(2, Hbar) + projection(2, c4).scale(Fraction(1, 24)))
print(f"energy {len(energy)} terms, singles {len(r1)} terms, doubles {len(r2)} terms")

# ---- 2. Write a module ----------------------------------------------------------
source = "\n\n".join([
    einsum_code(energy, "ccsd_energy"),
    einsum_code(r1, "ccsd_r1", group=True),
    einsum_code(r2, "ccsd_r2", group=True),
])
path = os.path.join(os.environ.get("ALQEMI_OUTPUT", "."), "ccsd_residuals.py")
with open(path, "w", encoding="utf-8") as handle:
    handle.write(source + "\n")
spec = importlib.util.spec_from_file_location("ccsd_residuals", path)
ccsd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ccsd)

# ---- 3. Solve -------------------------------------------------------------------
scf = run_scf()
moeris = SpinOrbitalIntegrals(scf.h_ao, scf.eri_ao, scf.ca, scf.cb, scf.nocc_a, scf.nocc_b)
fo, fv = np.diag(moeris.f_OO), np.diag(moeris.f_VV)
d1 = fo[:, None] - fv[None, :]
d2 = fo[:, None, None, None] + fo[None, :, None, None] - fv[None, None, :, None] - fv[None, None, None, :]

t1 = np.zeros_like(moeris.f_OV)
t2 = np.zeros_like(moeris.g_OOVV)
for iteration in range(200):
    res1 = ccsd.ccsd_r1(moeris, t1, t2)
    res2 = ccsd.ccsd_r2(moeris, t1, t2)
    t1, t2 = t1 + res1 / d1, t2 + res2 / d2
    if max(np.abs(res1).max(), np.abs(res2).max()) < 1e-10:
        break
e_ccsd = ccsd.ccsd_energy(moeris, t1, t2) + scf.e_nuc
CHECKS.append(("CCSD energy", e_ccsd, reference_energy(scf, "ccsd")))

for label, ours, theirs in CHECKS:
    print(f"{label:12s} ALQEMI {ours:.10f}  {scf.program} {theirs:.10f}  diff {ours - theirs:.1e}")
