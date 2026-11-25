ALQEMI
======

**A**\ utomated **L**\ ogic for **Q**\ uantum **E**\ xpression, **M**\ anipulation,
and **I**\ mplementation.

ALQEMI derives second-quantized equations, such as coupled-cluster amplitude
equations, by evaluating commutators with the identities of Helgaker, Jørgensen
and Olsen (HJO), *Molecular Electronic-Structure Theory*, chapter 1. It does not use
Wick's theorem or SymPy. Coefficients are exact fractions. The results can be printed
as LaTeX or turned into numpy ``einsum`` code, in spin-orbital or UHF spin-integrated
form.

.. code-block:: python

   from alqemi import bch, cluster, hamiltonian, latex, projection

   Hbar = bch(hamiltonian(), cluster(2), order=2)
   print(latex(projection(2, Hbar), lhs=r"R^{ab}_{ij}", group=True))   # the CCD equations

.. grid:: 1 1 2 3

    .. grid-item-card:: Getting Started
      :link: getting_started
      :link-type: doc

      Install ALQEMI and derive your first equations.

    .. grid-item-card:: Cheatsheet
      :link: cheatsheet
      :link-type: doc

      Every public function on one page.

    .. grid-item-card:: User Guide
      :link: user_guide/index
      :link-type: doc

      The concepts, one page per topic.

    .. grid-item-card:: Tutorials
      :link: tutorials/index
      :link-type: doc

      From a single commutator to UHF-CCSD and Λ-CCD, checked against Psi4 and PySCF.

    .. grid-item-card:: API Reference
      :link: api
      :link-type: doc

      Signatures and docstrings.

    .. grid-item-card:: Developer Guide
      :link: developer_guide
      :link-type: doc

      Tests, docs builds and how a step is added.

.. toctree::
   :maxdepth: 2
   :hidden:
   :titlesonly:

   getting_started
   cheatsheet
   user_guide/index
   tutorials/index
   api
   developer_guide
