5. UHF-CCSD
===========

**Goal:** turn the spin-orbital CCSD equations into α/β block equations for an
unrestricted reference, generate one function per block, and reproduce UHF-CCSD for
an open-shell molecule.

Spin integration
----------------

``spin_integrate`` splits each equation into blocks of its free-index spins:

.. code-block:: text

   energy {'': 13}
   r1 {'OV': 26, 'ov': 26}
   r2 {'OOVV': 97, 'OoVv': 86, 'oovv': 97}

Upper case is α and lower case is β. Only the αβ mixed block ``OoVv`` is produced,
never βα. Every block becomes one generated function, ``r2_OoVv(moeris, t1_OV, ...)``,
whose arguments are the amplitude blocks it needs. The small ``call`` helper reads
those names from the function signature, so the solver loop never lists them.

Integrals
---------

``UHFIntegrals`` builds blocks from separate α and β orbitals. A same-spin block such
as ``g_OOVV`` holds ⟨IJ\|AB⟩ − ⟨IJ\|BA⟩, and an opposite-spin block such as ``g_OoVv``
holds only ⟨Ij\|Ab⟩. Its exchange part vanishes by spin. All 21 two-electron blocks
ALQEMI can ask for are built on demand.

Solving
-------

Each block is updated with its own orbital-energy denominator, built from the spins
in the block name (``denominator("OoVv")`` = ε\ :sub:`I` + ε\ :sub:`j` −
ε\ :sub:`A` − ε\ :sub:`b`).

Results (H\ :sub:`2`\ O\ :sup:`+`, doublet, STO-3G)
------------------------------------------------------

.. code-block:: text

   UHF energy       ALQEMI -74.6557767324  psi4 -74.6557767324  diff 1.4e-14
   UHF-CCSD energy  ALQEMI -74.6947739978  psi4 -74.6947739978  diff 2.8e-12

PySCF agrees to 4×10\ :sup:`-11` E\ :sub:`h`.

The script
----------

.. literalinclude:: scripts/tutorial05_uhf_ccsd.py
   :language: python
