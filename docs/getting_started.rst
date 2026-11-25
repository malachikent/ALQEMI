Getting Started
===============

Installation
------------

ALQEMI is pure Python (3.11 or newer) with no run-time dependencies.

.. code-block:: bash

   git clone https://github.com/malachikent/ALQEMI.git
   cd ALQEMI
   pip install -e .              # ALQEMI itself
   pip install -e ".[test]"      # plus pytest, numpy and opt_einsum for the tests

Running the generated code needs numpy (and optionally opt_einsum). The
numerical tutorials also need an SCF program, either `Psi4 <https://psicode.org>`_
(``conda install -c conda-forge psi4``) or `PySCF <https://pyscf.org>`_
(``pip install pyscf``).

Your first commutator
---------------------

Indices carry their orbital space. Operators are built with ``cre`` (creator) and
``ann`` (annihilator), and ``run`` evaluates a commutator:

>>> from alqemi import Index, Space, OpString, cre, ann, comm, run
>>> p, q = Index("p"), Index("q")
>>> i, a = Index("i", Space.OCC), Index("a", Space.VIRT)
>>> x = comm(OpString((cre(p), ann(q))), OpString((cre(a), ann(i))))
>>> print(run(x))
δ_qa a†_p a_i - δ_pi a†_a a_q

This is HJO (1.8.14). The deltas are left for you to use: ``p`` and ``q`` are general
indices, so ALQEMI cannot decide them.

Your first coupled-cluster equations
------------------------------------

``hamiltonian`` and ``cluster`` build operators, and ``bch`` evaluates
e\ :sup:`-T` H e\ :sup:`T` term by term:

>>> from alqemi import bch, cluster, hamiltonian, vacuum_expectation, projection
>>> Hbar = bch(hamiltonian(), cluster(2), order=2)
>>> energy = vacuum_expectation(Hbar)
>>> print(energy)
Σ_{i} f^{i}_{i} - 1/2 Σ_{i,j} v^{ij}_{ij} - 1/4 Σ_{a,b,i,j} t^{ab}_{ij} v^{ij}_{ba}

``v`` is the antisymmetrized integral ⟨pq‖rs⟩ in physicist's notation. The first two
terms are the Hartree–Fock energy. The CCD doubles residual is the projection onto
⟨Φ\ :sub:`ij`\ :sup:`ab`\ \|:

>>> residual = projection(2, Hbar)
>>> len(residual)
18

Print it, or turn it into code
------------------------------

>>> from alqemi import latex, einsum_code, group_permutations
>>> print(latex(energy, lhs="E"))
E = \sum_{i} f^{i}_{i} - \frac{1}{2} \sum_{ij} \langle ij \| ij \rangle - \frac{1}{4} \sum_{abij} t^{ab}_{ij}\, \langle ij \| ba \rangle
>>> len(group_permutations(residual))       # grouped under (1 - P_ij), (1 - P_ab)
10
>>> print(einsum_code(energy, "ccd_energy"))
import numpy as np
<BLANKLINE>
<BLANKLINE>
def ccd_energy(moeris, t2):
    r = 0.0
    # Σ_{i} f^{i}_{i}
    r += np.einsum("ii->", moeris.f_OO, optimize=True)
    # -1/2 Σ_{i,j} v^{ij}_{ij}
    r -= 0.5 * np.einsum("ijij->", moeris.g_OOOO, optimize=True)
    # -1/4 Σ_{a,b,i,j} t^{ab}_{ij} v^{ij}_{ba}
    r -= 0.25 * np.einsum("ijab,ijba->", t2, moeris.g_OOVV, optimize=True)
    return r

The generated function reads integral blocks from one object, ``moeris``
(``moeris.f_OO``, ``moeris.g_OOVV``, ...), with occupied axes first. See
:doc:`user_guide/codegen` for the layout, and :doc:`tutorials/index` for complete
programs checked against Psi4 and PySCF.

Where next
----------

* :doc:`cheatsheet`: every function on one page.
* :doc:`user_guide/index`: how ALQEMI works and why.
* :doc:`tutorials/index`: MP2, CCD, CCSD, UHF-CCSD and Λ-CCD from derivation to numbers.
