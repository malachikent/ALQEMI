4. CCSD to working code
=======================

**Goal:** derive the full CCSD equations, write them to a module you can keep, and
reproduce the CCSD energy of Psi4 and PySCF.

How much BCH is needed
----------------------

``bch(H, T1 + T2, order=4)`` is exact, but the fourth nested commutator of the whole
operator is large and takes many minutes. Only one piece of it survives the
projections. H must contract with all four T's, and T1\ :sup:`a` T2\ :sup:`b`
(a + b = 4) then has excitation rank at least a + 2b − 2. That is 2 only for
T1\ :sup:`4`, and never 1 or 0. So the script uses

.. code-block:: python

   Hbar = bch(H, T1 + T2, order=3)
   c4 = H
   for k in range(4):
       c4 = run(comm(c4, T1))                    # [[[[H, T1], T1], T1], T1]
   r2 = simplify(projection(2, Hbar) + projection(2, c4).scale(Fraction(1, 24)))

The single extra term is the familiar t\ :sub:`i`\ :sup:`c` t\ :sub:`j`\ :sup:`d`
t\ :sub:`k`\ :sup:`a` t\ :sub:`l`\ :sup:`b` ⟨kl‖cd⟩ term of the CCSD doubles. With it
the derivation takes about 6 seconds. The result is identical to the full order-4
expansion, which was checked once and took 19 minutes.

The energy needs only order 2 and the singles only order 3. The result is 5 energy
terms, 14 singles terms and 63 doubles terms.

A module of your own
--------------------

The script joins three ``einsum_code`` functions and writes them to
``ccsd_residuals.py`` (in ``ALQEMI_OUTPUT`` or the current folder). It then imports
that file like any module. Keep it, commit it, edit it: it is plain numpy with one
comment per term.

Results (water, STO-3G)
-----------------------

.. code-block:: text

   energy 5 terms, singles 14 terms, doubles 63 terms
   CCSD energy  ALQEMI -75.0126602528  psi4 -75.0126602528  diff 6.2e-12

PySCF: −75.0126602527, difference 6×10\ :sup:`-12` E\ :sub:`h`. The difference is
the convergence threshold, not the equations.

The script
----------

.. literalinclude:: scripts/tutorial04_ccsd.py
   :language: python
