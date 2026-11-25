Indices, spaces and spin
========================

An ``Index`` has a name, an orbital space and, optionally, a spin:

>>> from alqemi import Index, Space, Spin
>>> p = Index("p")                      # Space.GENERAL: any orbital
>>> i = Index("i", Space.OCC)           # occupied in the reference
>>> a = Index("a", Space.VIRT)          # virtual
>>> p.space, i.space, a.space
(<Space.GENERAL: 'general'>, <Space.OCC: 'occ'>, <Space.VIRT: 'virt'>)

The space is part of the index, never read from its name. ``Index("a", Space.OCC)`` is
a legal occupied index called *a*, and two indices with the same name but different
spaces are different indices:

>>> Index("k", Space.OCC) == Index("k", Space.VIRT)
False

Why it matters
--------------

The space decides what an operator does to the Hartree–Fock reference \|HF⟩:

* a\ :sub:`i` and a†\ :sub:`a` excite (create a hole or a particle);
* a†\ :sub:`i` \|HF⟩ = 0 and a\ :sub:`a` \|HF⟩ = 0;
* δ\ :sub:`ia` = 0 because the spaces are disjoint.

A ``GENERAL`` index covers both spaces. ``vacuum_expectation`` and ``projection`` split
each summed general index into an occupied and a virtual copy. A *free* general index
must be given a space first, because its value depends on that space.

Spin
----

Spin-orbital equations have no spin labels. ``spin_integrate`` (see :doc:`spin`)
produces indices with ``Spin.ALPHA`` or ``Spin.BETA``. By convention α names are upper
case and β names are lower case, so the spin can be read from printed output:

>>> I = Index("I", Space.OCC, Spin.ALPHA)
>>> j = Index("j", Space.OCC, Spin.BETA)
>>> print(I, j)
I j
>>> I.spin
<Spin.ALPHA: 'alpha'>

Dummy names
-----------

Builders take fresh dummy names from an ``IndexPool``: numbered names such as ``p1``,
``i1``, ``a1``. Products that need fresh names take them from the pool, so nothing
clashes. Only the final, simplified output is relabelled to plain letters
(``relabel_dummies``):

>>> from alqemi import IndexPool
>>> pool = IndexPool()
>>> pool.fresh(Space.OCC, set()), pool.fresh(Space.VIRT, set())
(Index(name='i1', space=<Space.OCC: 'occ'>, spin=None), Index(name='a1', space=<Space.VIRT: 'virt'>, spin=None))

If you combine expressions yourself, share one pool between the builders
(``cluster(2, pool=pool)``, ...) so their dummies cannot collide.
:doc:`../tutorials/06_lambda_ccd` shows this.
