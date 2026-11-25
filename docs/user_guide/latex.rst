LaTeX and P(ij) grouping
========================

``latex(x)`` prints any ALQEMI object as LaTeX, directly from ALQEMI's own objects
(no SymPy):

>>> from alqemi import Index, Space, OpString, bch, cluster, cre, ann, hamiltonian
>>> from alqemi import latex, projection
>>> i, a = Index("i", Space.OCC), Index("a", Space.VIRT)
>>> print(latex(OpString((cre(a), ann(i)))))
a^{\dagger}_{a} a_{i}
>>> print(latex(Index("a12", Space.VIRT)))
a_{12}

Conventions:

* ``v`` prints as the double-bar integral ``\langle pq \| rs \rangle``. Pass
  ``double_bar=("g",)`` if your integral tensor has another name.
* Sums are explicit (``\sum_{ck}``), and every coefficient, including BCH factors, is
  shown as ``\frac{1}{2}``.
* α and β are told apart by case, as in the rest of ALQEMI.
* In Jupyter, an ``Expr`` renders as math by itself (``_repr_latex_``).

Long equations
--------------

``lhs`` adds a left-hand side. ``terms_per_line`` (which needs ``lhs``) gives an
``align*`` block:

>>> r = projection(2, bch(hamiltonian(), cluster(2), order=1))
>>> print(latex(r, lhs="R", terms_per_line=4))
\begin{align*}
R &= \langle ab \| ij \rangle + \sum_{c} f^{a}_{c}\, t^{cb}_{ij} - \sum_{c} f^{b}_{c}\, t^{ca}_{ij} - \sum_{k} f^{k}_{j}\, t^{ab}_{ik} \\
&\quad + \sum_{k} f^{k}_{i}\, t^{ab}_{jk} - \sum_{ck} t^{cb}_{ik}\, \langle ka \| jc \rangle + \sum_{ck} t^{cb}_{jk}\, \langle ka \| ic \rangle + \sum_{ck} t^{ca}_{ik}\, \langle kb \| jc \rangle \\
&\quad - \sum_{ck} t^{ca}_{jk}\, \langle kb \| ic \rangle - \frac{1}{2} \sum_{cd} t^{cd}_{ij}\, \langle ab \| dc \rangle - \frac{1}{2} \sum_{kl} t^{ab}_{kl}\, \langle lk \| ij \rangle
\end{align*}

Permutation operators
---------------------

Many terms of a doubles residual come in sets related by swapping i ↔ j or a ↔ b.
``group=True`` writes each set once, with P\ :sub:`pq` exchanging p and q:

>>> print(latex(r, group=True))
\langle ab \| ij \rangle + (1 - P_{ab}) \sum_{c} f^{a}_{c}\, t^{cb}_{ij} - (1 - P_{ij}) \sum_{k} f^{k}_{j}\, t^{ab}_{ik} - (1 - P_{ij})(1 - P_{ab}) \sum_{ck} t^{cb}_{ik}\, \langle ka \| jc \rangle - \frac{1}{2} \sum_{cd} t^{cd}_{ij}\, \langle ab \| dc \rangle - \frac{1}{2} \sum_{kl} t^{ab}_{kl}\, \langle lk \| ij \rangle

The 11 terms become 6. The grouping itself is available as data:

>>> from alqemi import expand_permutations, group_permutations, simplify
>>> groups = group_permutations(r)
>>> len(r), len(groups)
(11, 6)
>>> print(latex(groups[1]))
(1 - P_{ab}) \sum_{c} f^{a}_{c}\, t^{cb}_{ij}
>>> len(simplify(expand_permutations(groups) - r))      # exact: expands back to r
0

What grouping guarantees:

* **Exact.** Partners are found with the canonical keys ``simplify`` uses, so dummy
  renaming and tensor symmetry, signs included, are accounted for.
  ``expand_permutations`` gives back the original expression.
* **Greedy and in order.** For each term the product (1 − P\ :sub:`ij`)(1 −
  P\ :sub:`ab`) is tried first, then (1 − P\ :sub:`ij`), then (1 − P\ :sub:`ab`).
* **Same spin and space only.** Only two free indices of the same space and spin can
  be swapped. A UHF ``OoVv`` block therefore has no groups, while ``OOVV`` and
  ``oovv`` do.
* **Never a term with itself.** A swap that maps a term to ± itself is never used for
  that term.
* **Doubles form only.** Triples operators such as P(i/jk) are planned for later.
