Spin integration (UHF)
======================

Spin-orbital equations hold for any reference. For an unrestricted (UHF) reference
each spin orbital is either α or β, and integrals and amplitudes split into spin
blocks. ``spin_integrate(x)`` does that split:

>>> from alqemi import bch, cluster, hamiltonian, projection, spin_integrate
>>> r1 = projection(1, bch(hamiltonian(), cluster(2), order=1))
>>> print(r1)
f^{a}_{i} - Σ_{b,j} f^{j}_{b} t^{ba}_{ij} + 1/2 Σ_{b,j,k} t^{ba}_{jk} v^{jk}_{ib} + 1/2 Σ_{b,c,j} t^{bc}_{ij} v^{ja}_{cb}
>>> blocks = spin_integrate(r1)
>>> sorted(blocks)
['OV', 'ov']
>>> print(blocks["OV"])
f^{A}_{I} - Σ_{B,J} f^{J}_{B} t^{BA}_{IJ} - Σ_{a,i} f^{i}_{a} t^{aA}_{Ii} + 1/2 Σ_{B,J,K} t^{BA}_{JK} v^{JK}_{IB} + Σ_{J,a,i} t^{aA}_{Ji} v^{Ji}_{Ia} + 1/2 Σ_{B,C,J} t^{BC}_{IJ} v^{JA}_{CB} + Σ_{B,a,i} t^{Ba}_{Ii} v^{iA}_{aB}

Conventions
-----------

* **Names.** α indices are upper case and β indices are lower case.
* **Blocks.** Keys name the spins of the free indices, occupied first: ``""`` for an
  energy, ``"OV"``/``"ov"`` for singles, and ``"OOVV"``/``"OoVv"``/``"oovv"`` for
  doubles. Only the αβ mixed block is produced. βα (``oOvV``) is the same
  information, so it is never stored.
* **How.** Each summed index is copied once as α and once as β. Copies in which a
  tensor violates spin conservation (an integral ⟨pq‖rs⟩ needs spin(p) + spin(q) =
  spin(r) + spin(s), and the same for amplitudes) are dropped, and the rest are
  simplified.
* **Integrals.** In an opposite-spin integral only the direct part survives:
  ⟨Ij‖Ab⟩ = ⟨Ij\|Ab⟩. Generated code reads these from blocks such as ``g_OoVv``. The
  same-spin blocks keep both parts.

Energies split the same way:

>>> e = spin_integrate(projection(0, bch(hamiltonian(), cluster(2), order=1)))
>>> print(e[""])
Σ_{I} f^{I}_{I} + Σ_{i} f^{i}_{i} - 1/2 Σ_{I,J} v^{IJ}_{IJ} - Σ_{I,i} v^{Ii}_{Ii} - 1/2 Σ_{i,j} v^{ij}_{ij} - 1/4 Σ_{A,B,I,J} t^{AB}_{IJ} v^{IJ}_{BA} - Σ_{A,I,a,i} t^{Aa}_{Ii} v^{Ii}_{aA} - 1/4 Σ_{a,b,i,j} t^{ab}_{ij} v^{ij}_{ba}

From blocks to code
-------------------

Generate one function per block, then solve the coupled equations. Amplitudes are
named after their block: ``t1_OV``, ``t2_OoVv``. :doc:`../tutorials/05_uhf_ccsd`
does this for UHF-CCSD and matches Psi4 and PySCF to 10\ :sup:`-10` E\ :sub:`h`.
