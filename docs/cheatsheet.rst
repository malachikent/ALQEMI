Cheatsheet
==========

Everything is imported from the top-level package: ``from alqemi import ...``.

Symbols
-------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Call
     - Meaning
   * - ``Index("p")``
     - general orbital index (``Space.GENERAL``)
   * - ``Index("i", Space.OCC)``, ``Index("a", Space.VIRT)``
     - occupied / virtual index; the space is never read from the name
   * - ``Index("I", Space.OCC, Spin.ALPHA)``
     - spin-labelled index (``Spin.ALPHA`` / ``Spin.BETA``), made by ``spin_integrate``
   * - ``cre(p)``, ``ann(q)``
     - a†\ :sub:`p`, a\ :sub:`q`: an ``Op`` with ``Kind.CRE`` / ``Kind.ANN``
   * - ``OpString((cre(p), ann(q)))``
     - the product a†\ :sub:`p` a\ :sub:`q`
   * - ``Delta(p, q)``
     - Kronecker delta δ\ :sub:`pq`
   * - ``ElementaryBracket(ann(p), cre(q))``
     - unevaluated [a\ :sub:`p`, a†\ :sub:`q`]\ :sub:`+` (stages 1–2 only)
   * - ``Tensor("t", upper, lower, Swap.NONE)``
     - t\ :sup:`upper`\ :sub:`lower`, antisymmetric within each group; ``Swap`` is the
       symmetry under exchanging the groups (``SYMMETRIC`` for f, h, v; ``NONE`` for t, l)

Expressions
-----------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Call
     - Meaning
   * - ``comm(A, B)``, ``acomm(A, B)``
     - [A, B] and [A, B]\ :sub:`+`: a ``Bracket`` of kind ``BracketKind.COMM`` / ``ACOMM``
   * - ``Term(coeff, scalars, factors, summed)``
     - coeff × scalars × operator factors, summed over ``summed`` (a frozenset)
   * - ``Expr(terms)``
     - a sum of terms; ``+``, ``-``, ``len``, iteration
   * - ``IndexPool()``
     - source of fresh dummy names (p1, i1, a1, ...); share one to avoid clashes

Pipeline
--------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Call
     - Meaning
   * - ``run(x, trace=None)``
     - all four stages; pass a list as ``trace`` to collect ``TraceStep(rule, bracket)``
   * - ``linearize(x, pool)``
     - stage 1: flatten into a sum of terms
   * - ``identity_expand(x, pool, trace=None)``
     - stage 2: apply HJO identities until no bracket is left
   * - ``delta_evaluate(x)``
     - stage 3: elementary anticommutators → deltas, collapse sums
   * - ``simplify(x)``
     - stage 4: combine like terms (exact, sign-aware)
   * - ``relabel_dummies(x)``
     - rename dummies to letters (i, j, a, b, ...)

Errors: ``UnsupportedBracket`` (no rank-reducing expansion: use ``acomm`` for two
odd strings), ``DanglingSum`` (a summed index vanished in delta evaluation: its sum would be the
size of a space, which has no symbol).

Operators
---------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Call
     - Meaning
   * - ``hamiltonian(form="fock")``
     - H = F + Φ (``fock_operator()`` + ``fluctuation_potential()``); ``form="core"``: h + V
   * - ``tensor_operator("u", [Space.GENERAL], [Space.GENERAL])``
     - Σ u\ :sup:`p`\ :sub:`q` a†\ :sub:`p` a\ :sub:`q` (any rank, any spaces)
   * - ``cluster(n)``
     - T\ :sub:`n` with amplitudes ``t``
   * - ``excitation(n)``
     - τ\ :sub:`μ` = a†\ :sub:`a` a†\ :sub:`b` a\ :sub:`j` a\ :sub:`i` (free indices)
   * - ``deexcitation(n)``
     - Λ\ :sub:`n` with amplitudes ``l``
   * - ``normal_ordered=True``, ``pool=``
     - options of the builders above

Equations
---------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Call
     - Meaning
   * - ``bch(H, T, order=4)``
     - Σ\ :sub:`k` (1/k!) [...[H, T], ... T]; exact at order 4 for a two-body H
   * - ``vacuum_expectation(x)``
     - ⟨HF\| x \|HF⟩
   * - ``projection(n, x)``
     - ⟨μ\ :sub:`n`\ \| x \|HF⟩; free indices a, b, ..., i, j, ...
   * - ``spin_integrate(x)``
     - dict of spin blocks: ``""``, ``"OV"``/``"ov"``, ``"OOVV"``/``"OoVv"``/``"oovv"``

Output
------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Call
     - Meaning
   * - ``latex(x, lhs=None, terms_per_line=None, group=False)``
     - LaTeX; ``terms_per_line`` gives an ``align*`` block; ``group`` adds (1 - P\ :sub:`ij`)
   * - ``einsum_code(x, name, group=False)``
     - Python source of ``def name(moeris, t1, t2, ...)``; ``backend="opt_einsum"``
   * - ``group_permutations(x)``
     - tuple of ``PermutedTerm(term, swaps)``
   * - ``expand_permutations(groups)``
     - the exact inverse of ``group_permutations``

Conventions
-----------

* ``v`` is ⟨pq‖rs⟩, antisymmetrized, physicist's notation; it prints as
  ``\langle pq \| rs \rangle``.
* Generated code reads ``moeris.f_OV``, ``moeris.g_OOVV``, ...: occupied axes first.
  With spin, upper case is α and lower case is β (``g_OoVv``); only αβ mixed blocks
  exist, and opposite-spin blocks have no exchange part. These are the 21 two-electron
  blocks OOOO OOOV OOVV OVOV OVVV VVVV (and lower case), OoOo OoOv oOoV OoVv OvOv oVoV
  OvVv oVvV VvVv, and Fock blocks f_OO f_OV f_VV (and lower case).
* Amplitude arguments are ``t1``, ``t2`` (spin-orbital) or ``t2_OoVv`` (spin blocks).
* Coefficients are exact ``Fraction``\ s; BCH factors are always written out.

Quick examples
--------------

>>> from alqemi import *
>>> i, a = Index("i", Space.OCC), Index("a", Space.VIRT)
>>> print(run(acomm(OpString((ann(i),)), OpString((cre(a),)))))
0
>>> print(projection(1, hamiltonian()))
f^{a}_{i}
>>> blocks = spin_integrate(projection(1, hamiltonian()))
>>> sorted(blocks)
['OV', 'ov']
>>> print(latex(blocks["OV"]))
f^{A}_{I}
>>> print(cluster(2))
1/4 Σ_{a1,a2,i1,i2} t^{a1a2}_{i1i2} a†_a1 a†_a2 a_i2 a_i1
