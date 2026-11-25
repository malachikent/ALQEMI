Tutorials
=========

Each tutorial is a complete Python script in ``docs/tutorials/scripts``. The page
shows the whole script and explains it. Every script ends by recording ``CHECKS``,
pairs of numbers that must agree, such as ALQEMI's CCSD energy and Psi4's. The test
suite runs every script and compares each pair itself, so the numbers on these pages
are tested.

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Tutorial
     - You will learn
     - Checked against
   * - :doc:`01_commutator`
     - brackets, the trace, checking with matrices
     - fermion matrices
   * - :doc:`02_mp2`
     - first-order amplitudes, generated code, an SCF interface
     - Psi4 / PySCF MP2
   * - :doc:`03_ccd`
     - BCH, projections, P(ij) grouping
     - MP2 (first iteration)
   * - :doc:`04_ccsd`
     - truncating BCH, writing a module
     - Psi4 / PySCF CCSD
   * - :doc:`05_uhf_ccsd`
     - spin integration, one function per block
     - Psi4 / PySCF UHF-CCSD
   * - :doc:`06_lambda_ccd`
     - left-hand equations, properties, operator products
     - finite field
   * - :doc:`07_latex`
     - publication output
     - pdflatex

Setup
-----

Tutorials 2–6 need numpy and one SCF program:

.. code-block:: bash

   conda install -c conda-forge psi4       # Psi4 (the default), or
   pip install pyscf                       # PySCF

Run a tutorial from inside the scripts folder:

.. code-block:: bash

   cd docs/tutorials/scripts
   python tutorial02_mp2.py                          # Psi4
   ALQEMI_PROGRAM=pyscf python tutorial02_mp2.py     # PySCF

The molecule is water in a minimal basis (STO-3G, 14 spin orbitals), so every
tutorial runs in seconds. For UHF, it is the water cation.

The two helpers
---------------

Generated code needs an object ``moeris`` with integral blocks ``f_OV``, ``g_OOVV``,
and so on (:doc:`../user_guide/codegen`). ``integrals.py`` builds them with numpy
from AO integrals and MO coefficients, so it works with any SCF program:

* ``SpinOrbitalIntegrals``: spin-orbital blocks (spaces O and V).
* ``UHFIntegrals``: α/β blocks (O, V for α and o, v for β), opposite-spin blocks
  without exchange.

Both build a block the first time it is read, so you never list the blocks you need.

.. literalinclude:: scripts/integrals.py
   :language: python

``scf.py`` runs the SCF with Psi4 or PySCF and returns plain numpy arrays, and it asks
the same program for reference MP2 and CCSD energies:

.. literalinclude:: scripts/scf.py
   :language: python

.. toctree::
   :maxdepth: 1
   :hidden:

   01_commutator
   02_mp2
   03_ccd
   04_ccsd
   05_uhf_ccsd
   06_lambda_ccd
   07_latex
