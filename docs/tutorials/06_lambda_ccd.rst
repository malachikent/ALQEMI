6. Λ-CCD and a property
=======================

**Goal:** derive the left-hand (Λ) equations of CCD and the expectation value of a
one-electron operator, and prove both right with a finite-field derivative.

The equations
-------------

The CC Lagrangian L = ⟨HF\|(1 + Λ\ :sub:`2`) e\ :sup:`-T` H e\ :sup:`T`\|HF⟩ is
stationary in the amplitudes t\ :sub:`μ` when

.. math::

   \langle \mathrm{HF} | (1 + \Lambda_2) [\bar H, \tau_{\mu}] | \mathrm{HF} \rangle = 0,
   \qquad \bar H = e^{-T} H e^{T}.

The derivative of the energy with respect to a perturbation εU is then
⟨HF\|(1 + Λ\ :sub:`2`) e\ :sup:`-T` U e\ :sup:`T`\|HF⟩, with no derivative of T
needed. For a one-body U, e\ :sup:`-T` U e\ :sup:`T` = U + [U, T\ :sub:`2`].

Two practical steps
-------------------

**Operator products.** ALQEMI has no ``Expr * Expr`` yet. The ``times`` helper multiplies
term by term with ``Term(...)``. That is only safe if the dummy names of the two
factors differ, so Λ\ :sub:`2` and U take their dummies from one shared
``IndexPool``.

**Name clashes.** ``bch`` returns dummies relabelled to i, j, a, b, ..., the same
names as the free indices of τ\ :sub:`μ` = ``excitation(2)``. ``move_dummies``
renames the clashing dummies of H̄ to fresh pool names before the commutator is
formed. ALQEMI never renames anything silently (:doc:`../user_guide/limits`).

The Λ equations come out in the familiar form, 25 terms that group into 13:

.. math::

   \begin{aligned}
   R^{ij}_{ab} &= \langle ij \| ab \rangle + (1 - P_{ab}) \sum_{c} f^{c}_{a} l^{ij}_{cb}
     - (1 - P_{ij}) \sum_{k} f^{j}_{k} l^{ik}_{ab}
     - (1 - P_{ij})(1 - P_{ab}) \sum_{ck} l^{ik}_{cb} \langle jc \| ka \rangle \\
   &\quad - \frac{1}{2} \sum_{cd} l^{ij}_{cd} \langle cd \| ba \rangle
     - \frac{1}{2} \sum_{kl} l^{kl}_{ab} \langle ji \| kl \rangle + \ldots
   \end{aligned}

and the property is

.. math::

   \langle U \rangle = \sum_{i} u^{i}_{i}
     + \frac{1}{2} \sum_{abcij} l^{ij}_{ab} t^{cb}_{ij} u^{a}_{c}
     - \frac{1}{2} \sum_{abijk} l^{ij}_{ab} t^{ab}_{ik} u^{k}_{j}.

The check
---------

With the orbitals held fixed, adding εU to H adds εu to every Fock block. The script
solves CCD at ε = ±10\ :sup:`-4` and takes the central difference
(E(+ε) − E(−ε)) / 2ε. That difference must equal ⟨U⟩ from Λ. It can only agree if
the T equations, the Λ equations and the property expression are all right. No
other program is involved.

.. code-block:: text

   dE/d eps (z dipole)  Lambda -0.6644341883  finite field -0.6644341881  diff -2.1e-10

The 2×10\ :sup:`-10` difference is the O(ε\ :sup:`2`) error of the finite difference.

The script
----------

.. literalinclude:: scripts/tutorial06_lambda_ccd.py
   :language: python
