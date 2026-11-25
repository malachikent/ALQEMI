Developer Guide
===============

Ground rules
------------

ALQEMI's design is fixed in three files at the repository root:

* ``MATHEMATICAL_CONTRACT.md``: the mathematics and every convention. It is the
  source of truth. If code or tests disagree with it, the disagreement is resolved
  before anything else changes.
* ``DESIGN_DECISIONS.md``: why each choice was made, what each step delivered, and
  what is planned.
* ``CLAUDE.md``: the short list of non-negotiables (no Wick's theorem, no SymPy,
  rank reduction, never reorder toward normal order, ...).

Running the tests
-----------------

.. code-block:: bash

   pip install -e ".[test]"
   pytest alqemi/tests                 # about 45 s

Tests never compare expression trees. They compare results with
``alqemi/tests/oracle.py``, which builds fermion operators as matrices (Jordan–Wigner)
and checks every assignment of indices to modes. Generated code is checked by
running it on random tensors and on a small UHF model. The oracle and golden tests
are never edited to make code pass.

The documentation tests (``alqemi/tests/test_step15_docs.py``) run every ``>>>``
example in the docs and every tutorial script. Tutorials 2–6 are skipped unless Psi4
or PySCF is installed. The ``docs`` CI job installs both, so they always run there.

Building the documentation
--------------------------

.. code-block:: bash

   conda env create -f devtools/conda-envs/docs_env.yaml    # Sphinx + Psi4 + PySCF
   conda activate alqemi-docs
   pip install -e . --no-deps
   sphinx-build -W --keep-going -b html docs docs/_build/html
   open docs/_build/html/index.html                          # macOS

``-W`` turns warnings into errors, as on ReadTheDocs. For a live preview while
writing, use ``pip install sphinx-autobuild`` and ``sphinx-autobuild docs
docs/_build/html``.

ReadTheDocs builds every push to ``main`` and every tag from ``.readthedocs.yaml``,
using the lighter ``docs/requirements.yaml`` environment. The pages only *include*
the tutorial scripts, so that build needs no SCF program.

Writing documentation
---------------------

* Pages are reStructuredText. Examples in the cheatsheet and the user guide are
  doctests: write ``>>>`` and the exact output, and the test suite checks it.
* A new public name must appear in ``docs/api.rst`` and in ``docs/cheatsheet.rst``.
  Tests enforce both.
* A tutorial is a script in ``docs/tutorials/scripts`` that records
  ``CHECKS = [(label, ALQEMI value, reference value), ...]``, plus a page that
  ``literalinclude``\ s it. Add its labels and tolerances to ``CHECK_SPEC`` in
  ``test_step15_docs.py``.

How a step is added
-------------------

ALQEMI grows in numbered steps. Each step goes in this order:

1. **Discuss** the scope and decide the open questions.
2. **Spec and tests.** A new ``alqemi/tests/test_stepN_*.py`` whose docstring is the
   specification. A scratch prototype must pass it, and *planted bugs* (deliberate
   mistakes in the prototype) must make it fail. A test that misses a planted bug is
   strengthened, or the miss is shown to change no result.
3. **Commit** the spec and tests.
4. **Implement** until the whole suite passes, and update ``docs/api.rst``,
   ``CHANGELOG.md``, ``DESIGN_DECISIONS.md`` and ``docs/progress.md``.
5. **Release** once CI is green. Rename "Unreleased" in the CHANGELOG, tag
   ``vX.Y.Z`` on ``main``, and publish a GitHub release. versioningit reads the
   version from the tag.
