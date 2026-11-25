Limits and errors
=================

ALQEMI raises an error rather than return something it cannot justify. These are the
cases you are most likely to meet.

UnsupportedBracket
------------------

Every expansion must reduce rank (see :doc:`operators`). Commuting two strings of odd
length has no rank-reducing expansion:

>>> from alqemi import Index, Space, OpString, acomm, comm, cre, ann, run
>>> p, i = Index("p"), Index("i", Space.OCC)
>>> run(comm(OpString((cre(p),)), OpString((ann(i),))))
Traceback (most recent call last):
    ...
alqemi.rules.hjo.UnsupportedBracket: no rank-reducing expansion: [a†_p, a_i]
>>> print(run(acomm(OpString((cre(p),)), OpString((ann(i),)))))
δ_pi

Use the anticommutator, or rewrite the bracket so that one side has even length.
Partial expansions that do not reduce rank are deliberately not offered.

Free general indices
--------------------

The value of ⟨HF\|a†\ :sub:`p` a\ :sub:`p`\|HF⟩ depends on whether p is occupied, so a
free ``GENERAL`` index is an error in ``vacuum_expectation`` and ``projection``:

>>> from alqemi import Expr, Term, vacuum_expectation
>>> vacuum_expectation(Expr((Term(factors=(OpString((cre(p), ann(p))),)),)))
Traceback (most recent call last):
    ...
ValueError: free general index p: give it the OCC or VIRT space

Name clashes
------------

Free indices are never renamed. A projection names its own free indices a, b, ...,
i, j, ..., so an expression that already has a free ``a`` cannot be projected:

>>> from alqemi import projection
>>> a = Index("a", Space.VIRT)
>>> projection(2, Expr((Term(factors=(OpString((cre(a),)),)),)))
Traceback (most recent call last):
    ...
ValueError: free indices ['a'] of x clash with the projection's own

Simplified output names its dummies i, j, a, b too. Before such an expression goes
into a bracket with ``excitation(n)``, rename its clashing dummies with
``Term.rename`` and an ``IndexPool``. :doc:`../tutorials/06_lambda_ccd` has a
six-line helper for this.

Not yet supported
-----------------

* **Products of two expressions.** There is no ``Expr * Expr``. Build the product
  term by term with ``Term(...)`` (:doc:`../tutorials/06_lambda_ccd`).
* **Deltas in generated code.** ``einsum_code`` raises ``ValueError`` for an
  expression with a free delta.
* **Triples permutation operators** P(i/jk) in ``group_permutations``.
* **Full fourth-order BCH with T1 + T2 is slow** (minutes). For CCSD, use order 3 plus
  the T1\ :sup:`4` term, as in :doc:`../tutorials/04_ccsd`.

DanglingSum
-----------

After deltas are evaluated, a summed index can disappear from its term. For example
Σ\ :sub:`k` δ\ :sub:`kk` a\ :sub:`p` is (number of orbitals) × a\ :sub:`p`. ALQEMI has no
symbol for the size of a space, so ``delta_evaluate`` raises ``DanglingSum`` instead of
guessing. ``.term`` and ``.index`` say where it happened.
