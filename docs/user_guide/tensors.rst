Tensors and symmetry
====================

A ``Tensor`` is a named coefficient with upper and lower indices, X\ :sup:`upper`\
:sub:`lower`. Tensors are scalars. They appear in a ``Term``'s ``scalars``, next to
deltas.

>>> from alqemi import Index, Space, Swap, Tensor
>>> i, j = Index("i", Space.OCC), Index("j", Space.OCC)
>>> a, b = Index("a", Space.VIRT), Index("b", Space.VIRT)
>>> t = Tensor("t", (a, b), (i, j))
>>> print(t)
t^{ab}_{ij}

Two kinds of symmetry
---------------------

* **Within a group.** Every tensor is antisymmetric under exchanging two upper or two
  lower indices: t\ :sup:`ba`\ :sub:`ij` = −t\ :sup:`ab`\ :sub:`ij`. This matches the
  antisymmetrized integrals and amplitudes of spin-orbital theory.
* **Between the groups.** ``swap`` says what happens when the whole upper group is
  exchanged with the whole lower group:

  .. list-table::
     :header-rows: 1
     :widths: 25 45 30

     * - ``Swap``
       - meaning
       - used for
     * - ``Swap.NONE``
       - no relation
       - t, l amplitudes
     * - ``Swap.SYMMETRIC``
       - X\ :sup:`pq`\ :sub:`rs` = X\ :sup:`rs`\ :sub:`pq`
       - f, h, v (real orbitals)
     * - ``Swap.ANTISYMMETRIC``
       - X\ :sup:`pq`\ :sub:`rs` = −X\ :sup:`rs`\ :sub:`pq`
       - anti-Hermitian operators

A ``Tensor`` stores its indices exactly as given. It does not reorder them or detect
zeros. ``simplify`` uses the symmetries when it compares terms:

>>> from alqemi import Expr, Term, simplify
>>> v_ab = Tensor("v", (a, b), (i, j), Swap.SYMMETRIC)
>>> v_ba = Tensor("v", (b, a), (i, j), Swap.SYMMETRIC)
>>> x = Expr((Term(scalars=(v_ab,)), Term(scalars=(v_ba,))))
>>> print(x)
v^{ab}_{ij} + v^{ba}_{ij}
>>> print(simplify(x))
0

The integral v
--------------

``v`` is always the antisymmetrized two-electron integral in physicist's notation:

.. math::

   v^{pq}_{rs} = \langle pq \| rs \rangle = \langle pq | rs \rangle - \langle pq | sr \rangle,
   \qquad \langle pq | rs \rangle = (pr|qs).

``latex`` prints it as ⟨pq‖rs⟩, and generated code reads it from the ``moeris.g_*``
blocks.

Operators from tensors
----------------------

``tensor_operator`` builds Σ X a†...a with a tensor of your choice. You give the space
of each creator (``upper``) and annihilator (``lower``) index. The default coefficient
counts each distinct operator once:

>>> from alqemi import tensor_operator
>>> print(tensor_operator("u", [Space.GENERAL], [Space.GENERAL]))
Σ_{p1,p2} u^{p1}_{p2} a†_p1 a_p2
>>> print(tensor_operator("x", [Space.VIRT, Space.VIRT], [Space.OCC, Space.OCC]))
1/4 Σ_{a1,a2,i1,i2} x^{a1a2}_{i1i2} a†_a1 a†_a2 a_i2 a_i1

The annihilators are written in reverse order, a\ :sub:`i2` a\ :sub:`i1`, so that
X\ :sup:`ab`\ :sub:`ij` a†\ :sub:`a` a†\ :sub:`b` a\ :sub:`j` a\ :sub:`i` is the usual
excitation.
