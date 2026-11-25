The pipeline
============

``run(x)`` passes an expression through four stages. You can also call them one by
one:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Stage
     - What it does
   * - ``linearize(x, pool)``
     - Distribute products over sums, so the result is a flat sum of terms.
   * - ``identity_expand(x, pool)``
     - Apply HJO identities until every bracket is an elementary anticommutator
       [a\ :sub:`p`, a†\ :sub:`q`]\ :sub:`+` (kept unevaluated).
   * - ``delta_evaluate(x)``
     - Turn elementary anticommutators into deltas, use δ\ :sub:`pp` = 1 and
       δ\ :sub:`ia` = 0, and collapse sums over deltas.
   * - ``simplify(x)``
     - Combine like terms exactly, then relabel dummies to letters.

Deltas exist only after stage 3. Before that an elementary anticommutator is an
``ElementaryBracket``, a scalar that stage 3 evaluates:

>>> from alqemi import Index, IndexPool, OpString, comm, cre, ann
>>> from alqemi import linearize, identity_expand, delta_evaluate, simplify
>>> p, q, r, s = (Index(n) for n in "pqrs")
>>> x = comm(OpString((cre(p), ann(q))), OpString((cre(r), ann(s))))
>>> pool = IndexPool()
>>> stage2 = identity_expand(x, pool)
>>> print(stage2)
-[a†_p, a†_r]_+ a_q a_s + [a_q, a†_r]_+ a†_p a_s - [a†_p, a_s]_+ a†_r a_q + [a_q, a_s]_+ a†_r a†_p
>>> print(delta_evaluate(stage2))
δ_qr a†_p a_s - δ_ps a†_r a_q

Stage 2 produces four elementary anticommutators. Two of them, [a†, a†]\ :sub:`+` and
[a, a]\ :sub:`+`, are zero, and stage 3 drops them. The other two become deltas.

Like terms
----------

``simplify`` recognizes terms that differ only by dummy names, by tensor symmetry, or
by swapping adjacent operators of the same kind, and tracks the sign of each:

>>> from fractions import Fraction
>>> from alqemi import Expr, Space, Tensor, Term
>>> i, j = Index("i", Space.OCC), Index("j", Space.OCC)
>>> a, b = Index("a", Space.VIRT), Index("b", Space.VIRT)
>>> half = Fraction(1, 2)
>>> x = Expr((Term(half, (Tensor("t", (a,), (i,)),), (), frozenset({a, i})),
...           Term(half, (Tensor("t", (b,), (j,)),), (), frozenset({b, j}))))
>>> print(x)
1/2 Σ_{a,i} t^{a}_{i} + 1/2 Σ_{b,j} t^{b}_{j}
>>> print(simplify(x))
Σ_{a,i} t^{a}_{i}

It never swaps a creator with an annihilator (that would need a delta) and never
changes the order of the operators it outputs.

Tracing
-------

``run(x, trace=[])`` records every identity applied, as ``TraceStep(rule, bracket)``.
Rule names are HJO equation numbers, such as ``1.8.4`` and ``1.8.6L``. This is the
quickest way to see why a result looks the way it does.
