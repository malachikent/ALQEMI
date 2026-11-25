3. CCD from scratch
===================

**Goal:** derive coupled-cluster doubles from the BCH expansion, read it in LaTeX,
group it with permutation operators, and solve it.

Derivation
----------

.. code-block:: python

   Hbar = bch(hamiltonian(), cluster(2), order=2)
   energy = vacuum_expectation(Hbar)
   residual = projection(2, Hbar)

Order 2 is exact here. Every T\ :sub:`2` raises the excitation rank by 2 and H lowers
it by at most 2, so a third T\ :sub:`2` cannot fit in a doubles projection (see
:doc:`../user_guide/bch_projections`). The residual has 18 terms.

Grouping
--------

``latex(residual, group=True)`` folds the 18 terms into 10, using
(1 − P\ :sub:`ij`), (1 − P\ :sub:`ab`) and their product:

.. math::

   \begin{aligned}
   R^{ab}_{ij} &= \langle ab \| ij \rangle + (1 - P_{ab}) \sum_{c} f^{a}_{c} t^{cb}_{ij}
     - (1 - P_{ij}) \sum_{k} f^{k}_{j} t^{ab}_{ik}
     - (1 - P_{ij})(1 - P_{ab}) \sum_{ck} t^{cb}_{ik} \langle ka \| jc \rangle \\
   &\quad - \frac{1}{2} \sum_{cd} t^{cd}_{ij} \langle ab \| dc \rangle
     - \frac{1}{2} \sum_{kl} t^{ab}_{kl} \langle lk \| ij \rangle
     + \frac{1}{2} (1 - P_{ab}) \sum_{cdkl} t^{cb}_{ij} t^{da}_{kl} \langle kl \| cd \rangle \\
   &\quad - (1 - P_{ij}) \sum_{cdkl} t^{cb}_{ik} t^{da}_{jl} \langle kl \| cd \rangle
     + \frac{1}{2} (1 - P_{ij}) \sum_{cdkl} t^{ab}_{ik} t^{cd}_{jl} \langle kl \| dc \rangle
     + \frac{1}{4} \sum_{cdkl} t^{ab}_{kl} t^{cd}_{ij} \langle lk \| dc \rangle
   \end{aligned}

The script asserts that ``expand_permutations`` turns the groups back into exactly
the 18 terms. The generated code uses the same grouping (``group=True``), so it makes
10 contractions instead of 18.

Checking without a CCD program
------------------------------

Psi4 has no conventional CCD, so the check is one that needs no CCD program. The
first Jacobi step from T\ :sub:`2` = 0 gives t = ⟨ab‖ij⟩ / D, the MP2 amplitudes, so
the energy after that step must equal the program's MP2 energy. This tests the
linear part of the residual and the energy. The full CCD equations are tested in the
next tutorial, as part of CCSD.

.. code-block:: text

   CCD energy -75.0124134206 after 28 iterations
   residual terms         ALQEMI 18.0000000000  reference 18.0000000000
   groups                 ALQEMI 10.0000000000  reference 10.0000000000
   MP2 energy (1st step)  ALQEMI -74.9987553080  reference -74.9987553080

The script
----------

.. literalinclude:: scripts/tutorial03_ccd.py
   :language: python
