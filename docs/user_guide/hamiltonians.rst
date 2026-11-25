Hamiltonians and cluster operators
==================================

The Hamiltonian
---------------

``hamiltonian()`` writes H in one of two equivalent forms:

* ``form="fock"`` (default): H = F + Φ, the Fock operator plus the fluctuation
  potential. This is the natural form for Hartree–Fock-based theories, and generated
  code reads Fock blocks ``f_OO``, ``f_OV``, ``f_VV``.
* ``form="core"``: H = h + V, the core Hamiltonian plus the two-electron operator.

>>> from alqemi import fock_operator, fluctuation_potential, hamiltonian
>>> print(fock_operator())
Σ_{p1,p2} f^{p1}_{p2} a†_p1 a_p2
>>> print(fluctuation_potential())
1/4 Σ_{p1,p2,p3,p4} v^{p1p2}_{p3p4} a†_p1 a†_p2 a_p4 a_p3 - Σ_{i1,p5,p6} v^{p5i1}_{p6i1} a†_p5 a_p6
>>> print(hamiltonian(form="core"))
Σ_{p1,p2} h^{p1}_{p2} a†_p1 a_p2 + 1/4 Σ_{p3,p4,p5,p6} v^{p3p4}_{p5p6} a†_p3 a†_p4 a_p6 a_p5

Φ = V − Σ\ :sub:`pqi` ⟨pi‖qi⟩ a†\ :sub:`p` a\ :sub:`q` removes the mean-field part that
F already contains. Both forms give the same equations. Only the integrals they are
written with differ.

``normal_ordered=True`` writes each operator in normal order relative to \|HF⟩. It
splits the general indices into occupied and virtual blocks and adds the
fully-contracted constants. You rarely need it: the bracket evaluation does not
depend on normal order.

Excitation operators
--------------------

>>> from alqemi import cluster, deexcitation, excitation
>>> print(cluster(1))
Σ_{a1,i1} t^{a1}_{i1} a†_a1 a_i1
>>> print(cluster(2))
1/4 Σ_{a1,a2,i1,i2} t^{a1a2}_{i1i2} a†_a1 a†_a2 a_i2 a_i1
>>> print(deexcitation(2))
1/4 Σ_{a1,a2,i1,i2} l^{i1i2}_{a1a2} a†_i1 a†_i2 a_a2 a_a1
>>> print(excitation(2))
a†_a a†_b a_j a_i

* ``cluster(n)`` is T\ :sub:`n` = (1/n!)\ :sup:`2` Σ t\ :sup:`ab...`\ :sub:`ij...`
  a†\ :sub:`a` a†\ :sub:`b` ... a\ :sub:`j` a\ :sub:`i`. ``name=`` changes the
  amplitude name.
* ``deexcitation(n)`` is Λ\ :sub:`n`, with amplitudes l\ :sup:`ij...`\ :sub:`ab...`.
* ``excitation(n)`` is a single excitation string τ\ :sub:`μ` with *free* indices
  a, b, ..., i, j, .... ``projection`` uses it, and you can use it to build the
  left-hand equations (:doc:`../tutorials/06_lambda_ccd`).

Operators can be added: ``cluster(1) + cluster(2)`` is T = T\ :sub:`1` + T\ :sub:`2`.
Each builder makes its own fresh dummies. Pass a shared ``IndexPool`` (``pool=``) when
you combine their terms by hand.
