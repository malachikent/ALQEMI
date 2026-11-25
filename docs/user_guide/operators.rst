Operators and brackets
======================

Operator strings
----------------

``cre(p)`` is a†\ :sub:`p` and ``ann(q)`` is a\ :sub:`q`. An ``OpString`` is their
product, in the order given:

>>> from alqemi import Index, Space, OpString, cre, ann, comm, acomm, run
>>> p, q, r, s = (Index(n) for n in "pqrs")
>>> E_pq = OpString((cre(p), ann(q)))
>>> print(E_pq)
a†_p a_q

Brackets
--------

``comm(A, B)`` is [A, B] and ``acomm(A, B)`` is [A, B]\ :sub:`+`. A and B can be
operator strings, terms, expressions or other brackets. Nothing is evaluated until
you call ``run``:

>>> x = comm(E_pq, OpString((cre(r), ann(s))))
>>> print(x)
[a†_p a_q, a†_r a_s]
>>> print(run(x))
δ_qr a†_p a_s - δ_ps a†_r a_q

How a bracket is evaluated
--------------------------

ALQEMI does not use Wick's theorem. It applies the bracket identities of HJO
section 1.8, such as [A, BC] = [A, B]C + B[A, C], until only elementary anticommutators
[a\ :sub:`p`, a†\ :sub:`q`]\ :sub:`+` = δ\ :sub:`pq` remain. Two principles choose which
identity to apply:

* **Rank reduction.** A rule is used only if every bracket it creates is smaller.
  Two strings of odd length are anticommuted; every other pair is commuted.
* **Right first.** The right string is split first, and the left only when the right
  cannot be split. This reproduces HJO's printed results, (1.8.14)–(1.8.16).

The result keeps the operator order the identities produce. ALQEMI never reorders
toward normal order. The ``trace`` argument records each identity used:

>>> trace = []
>>> _ = run(x, trace=trace)
>>> [step.rule for step in trace]
['1.8.4', '1.8.6L', 'elementary', 'elementary', '1.8.6L', 'elementary', 'elementary']

Anticommutators
---------------

Two single operators anticommute to a delta:

>>> i, a = Index("i", Space.OCC), Index("a", Space.VIRT)
>>> print(run(acomm(OpString((ann(p),)), OpString((cre(q),)))))
δ_pq
>>> print(run(acomm(OpString((ann(i),)), OpString((cre(a),)))))
0

The second is zero because an occupied and a virtual orbital are never the same.

Commuting two single operators has no rank-reducing expansion, so it raises
``UnsupportedBracket`` rather than returning something wrong. See :doc:`limits`.
