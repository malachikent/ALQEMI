"""Run an SCF with Psi4 or PySCF and hand back what the tutorials need.

``run_scf`` returns an ``SCF`` record of plain numpy arrays (AO integrals, MO
coefficients, occupations), so everything after it is program-independent.
``reference_energy`` asks the same program for a correlated energy to compare with.

Pick the program with the ``ALQEMI_PROGRAM`` environment variable ("psi4", the
default, or "pyscf") or the ``program`` argument.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# Water at a fixed geometry (Angstrom) and its doublet cation, small enough for STO-3G
# coupled cluster in seconds.
WATER = """
O  0.000000  0.000000  0.117790
H  0.000000  0.755453 -0.471161
H  0.000000 -0.755453 -0.471161
"""


@dataclass
class SCF:
    program: str
    h_ao: np.ndarray      # core Hamiltonian, AO basis
    eri_ao: np.ndarray    # (mu nu|la si), chemist's notation
    dipole_ao: np.ndarray # (x, y, z) electric dipole integrals <mu|r|nu>, shape (3, n, n)
    ca: np.ndarray        # alpha MO coefficients (AO x MO)
    cb: np.ndarray        # beta MO coefficients
    nocc_a: int
    nocc_b: int
    e_nuc: float
    e_scf: float
    handle: object        # the program's own objects, for reference_energy


def _program(program):
    program = program or os.environ.get("ALQEMI_PROGRAM", "psi4")
    if program not in ("psi4", "pyscf"):
        raise ValueError(f"unknown program {program!r}; use 'psi4' or 'pyscf'")
    return program


def run_scf(geometry=WATER, *, basis="sto-3g", charge=0, multiplicity=1,
            reference="rhf", program=None) -> SCF:
    """Converged RHF/UHF for ``geometry`` (no symmetry, all electrons correlated later)."""
    program = _program(program)
    if program == "psi4":
        return _psi4_scf(geometry, basis, charge, multiplicity, reference)
    return _pyscf_scf(geometry, basis, charge, multiplicity, reference)


def reference_energy(scf: SCF, method: str) -> float:
    """The program's own total energy for ``method`` ("mp2" or "ccsd")."""
    if scf.program == "psi4":
        return _psi4_reference(scf, method)
    return _pyscf_reference(scf, method)


# ---- Psi4 -----------------------------------------------------------------------
def _psi4_options(reference):
    return {
        "basis": None,  # set by the caller
        "reference": reference,
        "scf_type": "pk",
        "mp2_type": "conv",
        "cc_type": "conv",
        "freeze_core": False,
        "e_convergence": 1e-12,
        "d_convergence": 1e-10,
        "r_convergence": 1e-10,
    }


def _psi4_scf(geometry, basis, charge, multiplicity, reference):
    import psi4

    psi4.core.be_quiet()
    psi4.core.clean()
    psi4.core.clean_options()
    mol = psi4.geometry(f"{charge} {multiplicity}\n{geometry}\nsymmetry c1\nno_reorient\nno_com")
    options = _psi4_options(reference)
    options["basis"] = basis
    psi4.set_options(options)
    e_scf, wfn = psi4.energy("scf", molecule=mol, return_wfn=True)
    mints = psi4.core.MintsHelper(wfn.basisset())
    h_ao = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())
    return SCF(
        program="psi4",
        h_ao=h_ao,
        eri_ao=np.asarray(mints.ao_eri()),
        dipole_ao=np.array([np.asarray(m) for m in mints.ao_dipole()]),
        ca=np.asarray(wfn.Ca()),
        cb=np.asarray(wfn.Cb()),
        nocc_a=wfn.nalpha(),
        nocc_b=wfn.nbeta(),
        e_nuc=mol.nuclear_repulsion_energy(),
        e_scf=e_scf,
        handle=(mol, options),
    )


def _psi4_reference(scf, method):
    import psi4

    mol, options = scf.handle
    psi4.core.clean()
    psi4.set_options(options)
    return psi4.energy(method, molecule=mol)


# ---- PySCF ----------------------------------------------------------------------
def _pyscf_scf(geometry, basis, charge, multiplicity, reference):
    from pyscf import gto, scf as pscf

    mol = gto.M(atom=geometry, basis=basis, charge=charge, spin=multiplicity - 1,
                unit="angstrom", symmetry=False, verbose=0)
    mf = (pscf.RHF if reference == "rhf" else pscf.UHF)(mol)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    e_scf = mf.kernel()
    if reference == "rhf":
        ca = cb = mf.mo_coeff
        nocc_a = nocc_b = mol.nelectron // 2
    else:
        ca, cb = mf.mo_coeff
        nocc_a, nocc_b = mol.nelec
    return SCF(
        program="pyscf",
        h_ao=mol.intor("int1e_kin") + mol.intor("int1e_nuc"),
        eri_ao=mol.intor("int2e"),
        dipole_ao=-mol.intor("int1e_r"),  # Psi4's ao_dipole() is -<mu|r|nu> (electron charge)
        ca=np.asarray(ca),
        cb=np.asarray(cb),
        nocc_a=nocc_a,
        nocc_b=nocc_b,
        e_nuc=mol.energy_nuc(),
        e_scf=e_scf,
        handle=mf,
    )


def _pyscf_reference(scf, method):
    from pyscf import cc, mp

    mf = scf.handle
    if method == "mp2":
        return mp.MP2(mf).run(verbose=0).e_tot
    if method == "ccsd":
        solver = cc.CCSD(mf)
        solver.conv_tol = 1e-12
        solver.conv_tol_normt = 1e-10
        return solver.run(verbose=0).e_tot
    raise ValueError(f"no PySCF reference for {method!r}")
