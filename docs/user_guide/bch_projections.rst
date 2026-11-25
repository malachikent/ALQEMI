BCH and projections
===================

The similarity-transformed Hamiltonian
--------------------------------------

``bch(H, T, order)`` evaluates the Baker–Campbell–Hausdorff series

.. math::

   e^{-T} H e^{T} = H + [H, T] + \frac{1}{2!}[[H, T], T] + \frac{1}{3!}[[[H, T], T], T]
                    + \frac{1}{4!}[[[[H, T], T], T], T] + \cdots

up to ``order`` nested commutators. Each nested commutator is evaluated and simplified
before the next one is formed. For a two-body H the series ends after order 4, where
it is exact. The factors 1/k! are always written out: no factor is folded into a
coefficient you cannot trace back.

Projections
-----------

``vacuum_expectation(x)`` is ⟨HF\|x\|HF⟩. ``projection(n, x)`` is ⟨μ\ :sub:`n`\|x\|HF⟩,
the projection onto the n-fold excited determinant, with free indices a, b, ... and
i, j, .... Together they give energies and residuals:

>>> from alqemi import bch, cluster, hamiltonian, projection, vacuum_expectation
>>> H, T1 = hamiltonian(), cluster(1)
>>> print(vacuum_expectation(bch(H, T1, order=2)))
Σ_{i} f^{i}_{i} - 1/2 Σ_{i,j} v^{ij}_{ij} + Σ_{a,i} f^{i}_{a} t^{a}_{i} + 1/2 Σ_{a,b,i,j} t^{a}_{i} t^{b}_{j} v^{ji}_{ba}
>>> print(projection(1, bch(H, T1, order=1)))
f^{a}_{i} + Σ_{b} f^{a}_{b} t^{b}_{i} - Σ_{j} f^{j}_{i} t^{a}_{j} - Σ_{b,j} t^{b}_{j} v^{ja}_{ib}

The free indices of a projection are always named a, b, ..., i, j, .... A free index of
``x`` with one of those names is an error, so a projection never silently renames.

How far to expand
-----------------

The projections need only a few orders. H is at most two-body, so H can contract with
at most four T's. The excitation rank of H · T1\ :sup:`a` T2\ :sup:`b` with every T
contracted is at least a + 2b − 2:

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Quantity
     - order
     - why
   * - energy, any CC
     - 2
     - rank 0 needs a + 2b ≤ 2
   * - CCD doubles
     - 2
     - T2² is the most H can reach with rank 2
   * - CCSD singles
     - 3
     - a + 2b ≤ 3
   * - CCSD doubles
     - 4, but only T1\ :sup:`4`
     - a + b = 4 and a + 2b ≤ 4 forces b = 0

``bch(H, T1 + T2, order=4)`` is exact but slow, because the full fourth-order operator
is large. :doc:`../tutorials/04_ccsd` uses order 3 for T1 + T2 and adds the single
fourth-order T1 term. The result is identical and takes seconds.

Two facts worth knowing
-----------------------

* For a one-body operator U, [[U, T], T] = 0 when T contains only excitations. So
  e\ :sup:`-T` U e\ :sup:`T` = U + [U, T] (``bch(U, T, order=1)``).
* After the brackets are evaluated, ``vacuum_expectation`` drops every term whose
  operators cannot take \|HF⟩ back to itself (unbalanced occupations), before it
  reduces the rest. Terms that cannot reach rank n cost little.
