Code generation
===============

``einsum_code(expr, name)`` returns the source of a Python function that evaluates an
operator-free expression, such as an energy or a residual, with ``numpy.einsum``:

>>> from alqemi import bch, cluster, einsum_code, hamiltonian, vacuum_expectation
>>> energy = vacuum_expectation(bch(hamiltonian(), cluster(2), order=1))
>>> print(einsum_code(energy, "ccd_energy"))
import numpy as np
<BLANKLINE>
<BLANKLINE>
def ccd_energy(moeris, t2):
    r = 0.0
    # Σ_{i} f^{i}_{i}
    r += np.einsum("ii->", moeris.f_OO, optimize=True)
    # -1/2 Σ_{i,j} v^{ij}_{ij}
    r -= 0.5 * np.einsum("ijij->", moeris.g_OOOO, optimize=True)
    # -1/4 Σ_{a,b,i,j} t^{ab}_{ij} v^{ij}_{ba}
    r -= 0.25 * np.einsum("ijab,ijba->", t2, moeris.g_OOVV, optimize=True)
    return r

Use it with ``exec``, or write it to a ``.py`` file and import it
(:doc:`../tutorials/04_ccsd`).

The integral object
-------------------

The function reads every integral from one argument, ``moeris``, by attribute:

* ``moeris.f_XY``: Fock block, ``f[p, q]`` = f\ :sup:`p`\ :sub:`q` (``h_XY`` for the
  core Hamiltonian).
* ``moeris.g_WXYZ``: antisymmetrized integral, ``g[p, q, r, s]`` = ⟨pq‖rs⟩.

Block letters give the space of each axis, **occupied axes first** (OOVV order). Only
canonical blocks are read. The code uses the symmetries of f and ⟨pq‖rs⟩ to turn any
other block into a canonical one, so you only need these:

* spin-orbital: ``f_OO f_OV f_VV``, ``g_OOOO g_OOOV g_OOVV g_OVOV g_OVVV g_VVVV``;
* UHF: the same with upper case for α and lower case for β, plus the αβ blocks
  ``g_OoOo g_OoOv g_oOoV g_OoVv g_OvOv g_oVoV g_OvVv g_oVvV g_VvVv``, which is
  21 two-electron blocks in all.

Amplitudes are positional arguments in OOVV order: ``t2[i, j, a, b]`` =
t\ :sup:`ab`\ :sub:`ij`. They are named ``t1``, ``t2``, or ``t2_OoVv`` and so on
with spin.

Options
-------

``integrals=`` maps tensor names to access templates, and ``amplitudes=`` sets the
argument names:

>>> print(einsum_code(energy, "e", integrals={"f": "F.{block}", "v": "eri.{block}"}, amplitudes="T{rank}"))
import numpy as np
<BLANKLINE>
<BLANKLINE>
def e(F, eri, T2):
    r = 0.0
    # Σ_{i} f^{i}_{i}
    r += np.einsum("ii->", F.OO, optimize=True)
    # -1/2 Σ_{i,j} v^{ij}_{ij}
    r -= 0.5 * np.einsum("ijij->", eri.OOOO, optimize=True)
    # -1/4 Σ_{a,b,i,j} t^{ab}_{ij} v^{ij}_{ba}
    r -= 0.25 * np.einsum("ijab,ijba->", T2, eri.OOVV, optimize=True)
    return r

* ``backend="opt_einsum"`` uses ``opt_einsum.contract`` instead of ``np.einsum``.
* ``group=True`` evaluates each (1 − P\ :sub:`ij`)(1 − P\ :sub:`ab`) group once into
  ``tmp`` and adds its permutations with ``swapaxes``. The numbers are identical to
  term-by-term code, and there are fewer contractions:

.. code-block:: python

   # (1 - P_ij)(1 - P_ab) -Σ_{c,k} t^{cb}_{ik} v^{ka}_{jc}
   tmp = np.einsum("ikcb,kajc->ijab", t2, moeris.g_OVOV, optimize=True)
   r -= tmp - tmp.swapaxes(0, 1) - tmp.swapaxes(2, 3) + tmp.swapaxes(0, 1).swapaxes(2, 3)

Other operators
---------------

Any tensor can be an integral. To read a property operator ``u`` from ``moeris.u_OV``
and so on, add ``"u": "moeris.u_{block}"`` to ``integrals``. The helper classes in
:doc:`../tutorials/index` build such blocks from AO matrices.

Deltas are not supported yet: generate code only from expressions without free deltas.
