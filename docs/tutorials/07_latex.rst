7. Publication-ready equations
==============================

**Goal:** turn derived equations into a LaTeX document you can paste into a paper or
thesis.

What the script does
--------------------

* ``latex(energy, lhs="E", terms_per_line=3)`` makes an ``align*`` block with three
  terms per line.
* ``latex(residual, ..., group=True)`` writes the CCD residual with
  (1 − P\ :sub:`ij`)(1 − P\ :sub:`ab`) factors, so 10 groups instead of 18 terms.
* ``spin_integrate(residual)["OoVv"]`` is the αβ block. Its free indices have
  different spins, so it has nothing to group, and it is printed term by term.
* The pieces go into a minimal ``article`` with ``amsmath``. If ``pdflatex`` is
  installed, the script compiles it.

The output, ``ccd_equations.tex``, needs only ``amsmath``. Every command ALQEMI emits
(``\langle``, ``\|``, ``\sum``, ``\frac``, ``a^{\dagger}``) is standard LaTeX.

Tips
----

* Change ``terms_per_line`` until the lines fit your page width. Two or three terms
  per line suit a doubles residual.
* ``lhs`` is copied as given. Write ``0 = R^{ab}_{ij}`` to present the equations as
  equations.
* In a Jupyter notebook, just display an ``Expr``. It renders through
  ``_repr_latex_`` with the same printer.

The script
----------

.. literalinclude:: scripts/tutorial07_latex.py
   :language: python
