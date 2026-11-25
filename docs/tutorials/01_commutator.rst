1. Your first commutator
========================

**Goal:** evaluate [a†\ :sub:`p` a\ :sub:`q`, a†\ :sub:`r` a\ :sub:`s`], see which
identities produce the answer, and check it yourself with matrices.

The script, step by step
------------------------

1. **Build.** ``Index("p")`` is a general index. ``comm`` makes the bracket and
   ``run`` evaluates it.
2. **Trace.** ``trace`` collects one ``TraceStep`` per identity: the derivation rule
   (1.8.4) first, then the elementary anticommutators that become deltas.
3. **Compare.** The result is HJO (1.8.14), δ\ :sub:`qr` a†\ :sub:`p` a\ :sub:`s` −
   δ\ :sub:`ps` a†\ :sub:`r` a\ :sub:`q`. Subtracting the textbook form and running
   again gives nothing.
4. **Check with matrices.** With Jordan–Wigner matrices for 3 modes, the commutator
   of the matrices must equal ALQEMI's result for every one of the 3\ :sup:`4` index
   assignments. This is how ALQEMI's own test suite checks its algebra: never by
   comparing expression trees, always against an independent calculation.

Output
------

.. code-block:: text

   δ_qr a†_p a_s - δ_ps a†_r a_q
   \delta_{qr}\, a^{\dagger}_{p} a_{s} - \delta_{ps}\, a^{\dagger}_{r} a_{q}
   1.8.4        [a†_p a_q, a†_r a_s]
   1.8.6L       [a†_p a_q, a†_r]
   elementary   [a†_p, a†_r]_+
   elementary   [a_q, a†_r]_+
   1.8.6L       [a†_p a_q, a_s]
   elementary   [a†_p, a_s]_+
   elementary   [a_q, a_s]_+
   same as (1.8.14): True
   largest matrix difference over all 81 index assignments: 0.0

Try next
--------

* Replace ``Index("r")`` by ``Index("a", Space.VIRT)`` and ``Index("s")`` by
  ``Index("i", Space.OCC)``. The term δ\ :sub:`ps` stays, but δ\ :sub:`qa` now
  requires q to be virtual.
* Evaluate [a†\ :sub:`p` a\ :sub:`q`, a†\ :sub:`r` a†\ :sub:`s` a\ :sub:`u`
  a\ :sub:`t`] and compare with HJO (1.8.15).

The script
----------

.. literalinclude:: scripts/tutorial01_commutator.py
   :language: python
