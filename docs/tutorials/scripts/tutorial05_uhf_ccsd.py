"""Tutorial 5: UHF-CCSD with spin integration.

Spin-integrate the spin-orbital CCSD equations into alpha/beta blocks, generate one
function per block, and solve UHF-CCSD for the water cation (doublet, STO-3G).
The energy is compared with the SCF program's own UHF-CCSD.

Blocks follow ALQEMI's convention: upper case alpha, lower case beta, occupied
first; only the alpha-beta mixed blocks are stored (OoVv, never oOvV).

Run:  python tutorial05_uhf_ccsd.py
"""
import inspect
from fractions import Fraction

import numpy as np

from alqemi import bch, cluster, comm, einsum_code, hamiltonian, projection, run, simplify
from alqemi import spin_integrate, vacuum_expectation

from integrals import UHFIntegrals
from scf import reference_energy, run_scf

CHECKS = []  # (label, ALQEMI value, reference value)

# ---- 1. Spin-orbital CCSD (as in tutorial 4) --------------------------------------
H, T1, T2 = hamiltonian(), cluster(1), cluster(2)
Hbar = bch(H, T1 + T2, order=3)
c4 = H
for k in range(4):
    c4 = run(comm(c4, T1))
energy = vacuum_expectation(bch(H, T1 + T2, order=2))
r1 = projection(1, Hbar)
r2 = simplify(projection(2, Hbar) + projection(2, c4).scale(Fraction(1, 24)))

# ---- 2. Spin-integrate: one Expr per spin block ----------------------------------
blocks = {"energy": spin_integrate(energy), "r1": spin_integrate(r1), "r2": spin_integrate(r2)}
for kind, parts in blocks.items():
    print(kind, {block: len(expr) for block, expr in parts.items()})

functions = {}
for kind, parts in blocks.items():
    for block, expr in parts.items():
        name = f"{kind}_{block}" if block else kind
        namespace = {}
        exec(einsum_code(expr, name, group=True), namespace)
        functions[name] = namespace[name]


def call(name, moeris, amplitudes):
    """Call a generated function with the amplitude blocks it asks for."""
    f = functions[name]
    wanted = list(inspect.signature(f).parameters)[1:]
    return f(moeris, *(amplitudes[a] for a in wanted))


# ---- 3. Solve ---------------------------------------------------------------------
scf = run_scf(charge=1, multiplicity=2, reference="uhf")
moeris = UHFIntegrals(scf.h_ao, scf.eri_ao, scf.ca, scf.cb, scf.nocc_a, scf.nocc_b)

e = {s: np.diag(getattr(moeris, f"f_{s}{s}")) for s in "OoVv"}


def denominator(block):
    """e_occ + ... - e_vir - ... broadcast over the block's axes (OOVV order)."""
    half, d = len(block) // 2, 0.0
    for axis, space in enumerate(block):
        shape = [1] * len(block)
        shape[axis] = -1
        d = d + (1 if axis < half else -1) * e[space].reshape(shape)
    return d


amplitudes = {
    "t1_OV": np.zeros_like(moeris.f_OV), "t1_ov": np.zeros_like(moeris.f_ov),
    "t2_OOVV": np.zeros_like(moeris.g_OOVV), "t2_OoVv": np.zeros_like(moeris.g_OoVv),
    "t2_oovv": np.zeros_like(moeris.g_oovv),
}
for iteration in range(200):
    residuals = {name: call(name, moeris, amplitudes) for name in functions if name != "energy"}
    for name, r in residuals.items():
        kind, block = name.split("_")
        target = ("t1_" if kind == "r1" else "t2_") + block
        amplitudes[target] = amplitudes[target] + r / denominator(block)
    if max(np.abs(r).max() for r in residuals.values()) < 1e-10:
        break

e_ccsd = call("energy", moeris, amplitudes) + scf.e_nuc
CHECKS.append(("UHF energy", call("energy", moeris, {k: 0 * v for k, v in amplitudes.items()}) + scf.e_nuc, scf.e_scf))
CHECKS.append(("UHF-CCSD energy", e_ccsd, reference_energy(scf, "ccsd")))

for label, ours, theirs in CHECKS:
    print(f"{label:16s} ALQEMI {ours:.10f}  {scf.program} {theirs:.10f}  diff {ours - theirs:.1e}")
