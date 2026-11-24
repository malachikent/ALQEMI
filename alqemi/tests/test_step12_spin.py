"""Step 12 tests: spin integration for UHF.

These tests DEFINE Step 12. Values are checked on a small UHF-like model with
3 alpha + 2 beta occupied and 2 alpha + 3 beta virtual spin-orbitals (unequal, so
an alpha/beta mix-up cannot go unnoticed, and at least two of each, so no block is
zero by antisymmetry alone). Integrals are built from random REAL
chemist's (pq|rs) with 8-fold symmetry, <pq||rs> = <pq|rs> - <pq|sr>, zero unless
spins match electron by electron; f and the amplitudes are random and conserve
spin. A spin block's value must equal the spin-orbital value (Step 11 code, already
checked against the oracle) at the corresponding spin-orbital modes.

  alqemi/core/indices.py
    Spin: ALPHA, BETA.
    Index(name, space=Space.GENERAL, spin=None): spin is a Spin or None (TypeError
      otherwise); it takes part in equality and hashing and is never parsed from
      the name. Spin-orbital indices have spin None. str(Index) is still its name.
    Term.rename must keep the spin too (ValueError), like the space.

  Simplify (pipeline/simplify.py) with spin
    Like terms only exchange dummies of the same space AND spin; spin is part of
    every key, so Σ_C x^C y_C (alpha) and Σ_c x^c y_c (beta) never merge.
    relabel_dummies: ALPHA dummies get upper-case letters (I, J, A, B, ...); BETA
    and spinless dummies lower-case. Output unchanged for spin-orbital input.

  alqemi/pipeline/spin.py
    spin_integrate(expr) -> dict[str, Expr]
      Input: operator-free terms (e.g. a projection result) whose scalars are
      Tensors with equal upper and lower ranks; every index OCC or VIRT, without
      spin; as many free OCC as free VIRT indices. Every tensor is assumed to
      conserve spin (f, v, t, l, r, ... of a spin-conserving theory): it vanishes
      unless its upper indices carry the same spins as its lower ones.
      Blocks. The n free OCC indices sorted by name, then the n free VIRT indices
      sorted by name; for k = n, n-1, ..., 0 the first k of each get ALPHA, the
      rest BETA (alpha before beta; Sz conserved). The key has one letter per free
      index in that order: O / o for alpha / beta OCC, V / v for alpha / beta VIRT.
      projection(2, .) -> "OOVV", "OoVv", "oovv"; projection(1, .) -> "OV", "ov";
      no free indices -> one block "" (e.g. the energy). Keys in that order.
      Each block: free indices renamed to name.upper() (ALPHA) or name.lower()
      (BETA); each summed index becomes an ALPHA and a BETA copy of its term; a
      copy is dropped if any tensor's upper spins differ from its lower spins
      (exact, by spin conservation); then Simplify.
      Exact: a block's value equals the spin-orbital value at those spins. Other
      spin cases of a projection follow from its antisymmetry (e.g. beta-alpha is
      alpha-beta transposed) or vanish (Sz).
      ValueError: operators; Delta or ElementaryBracket; a GENERAL index; an index
      that already has a spin; a tensor with unequal ranks; unequal numbers of free
      OCC and VIRT indices; free names that clash after the case change.

  alqemi/codegen/einsum.py (Step 11 unchanged for spin-orbital input)
    - All indices have a spin, or none does (ValueError if mixed).
    - Block letters: O / V for alpha (or no spin), o / v for beta.
    - Output axes: OCC first, then ALPHA before BETA, then by name: r[I,j,A,b].
    - Amplitudes: the default template is "{name}{rank}_{block}" for spin input
      ("{name}{rank}" without spin); {block} = the array's axis letters (t2_OoVv,
      t1_ov). Within each group a stable sort puts OCC first, then ALPHA first,
      with the permutation's sign; axes occupied first. So t^{Ab}_{Ij} ->
      t2_OoVv[I,j,A,b] and the beta-alpha t^{bA}_{jI} -> +t2_OoVv[I,j,A,b].
    - Integrals: candidates are the permutations within the upper and within the
      lower group (with sign), the group exchange under a swap symmetry, and, for a
      two-electron integral (2 upper, 2 lower) whose electrons have opposite spins
      (there <pq||rs> = <pq|rs>), the real-orbital images <rq|ps>, <ps|rq>, <rs|pq>
      (no sign; REAL orbitals, as in the user's eris). A candidate must pair spins
      electron by electron: upper[k] and lower[k] have the same spin. The smallest
      key wins (space letters, O before V, position by position; then spins, alpha
      before beta); the first candidate in enumeration order (no exchange first,
      permutations in lexicographic order) breaks ties, which gives Step 11's
      result for spin-orbital input. Every spin-conserving <pq||rs> then uses one
      of the user's 21 eris blocks:
        OOOO OOOV OOVV OVOV OVVV VVVV, the same in lower case (beta-beta), and
        OoOo OoOv oOoV OoVv OvOv oVoV OvVv oVvV VvVv;
      f uses f_OO, f_oo, f_OV, f_ov, f_VV, f_vv. ValueError if an integral
      cannot pair its spins (it does not conserve spin).

  Public API: Spin and spin_integrate join ``alqemi.__all__``.
"""
import itertools
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

import alqemi
from alqemi.builders import cluster, fluctuation_potential, hamiltonian
from alqemi.codegen.einsum import einsum_code
from alqemi.core.expr import Delta, Expr, Term, comm
from alqemi.core.indices import Index, Space, Spin
from alqemi.core.operators import ann, cre
from alqemi.core.tensor import Swap, Tensor
from alqemi.pipeline.bch import bch
from alqemi.pipeline.project import projection
from alqemi.pipeline.run import run
from alqemi.pipeline.simplify import relabel_dummies, simplify
from alqemi.pipeline.spin import spin_integrate
from alqemi.tests.evaluate import TensorValues

O, V, G = Space.OCC, Space.VIRT, Space.GENERAL
AL, BE = Spin.ALPHA, Spin.BETA
SYM, NONE = Swap.SYMMETRIC, Swap.NONE
a, b, c, d = (Index(x, V) for x in "abcd")
i, j, k, l = (Index(x, O) for x in "ijkl")

USER_BLOCKS = set(
    "OOOO OOOV OOVV OVOV OVVV VVVV oooo ooov oovv ovov ovvv vvvv "
    "OoOo OoOv oOoV OoVv OvOv oVoV OvVv oVvV VvVv".split()
)
FOCK_BLOCKS = ["OO", "oo", "OV", "ov", "VV", "vv"]


def T(coeff=1, scalars=(), summed=(), factors=()):
    return Term(coeff=coeff, scalars=tuple(scalars), factors=tuple(factors), summed=frozenset(summed))


def X(*terms):
    return Expr(tuple(terms))


def sp(name, space, spin):
    return Index(name, space, spin)


# ---- the UHF-like model -------------------------------------------------------------
NOA, NOB, NVA, NVB = 3, 2, 2, 3
SPIN = [AL] * NOA + [BE] * NOB + [AL] * NVA + [BE] * NVB
N_OCC, N = NOA + NOB, len(SPIN)
SL = {  # spin-orbital modes of each spin block letter
    "O": slice(0, NOA), "o": slice(NOA, N_OCC),
    "V": slice(N_OCC, N_OCC + NVA), "v": slice(N_OCC + NVA, N),
}
SO = {O: slice(0, N_OCC), V: slice(N_OCC, N)}  # spin-orbital blocks


def _spin_mask(ndim, n_upper):
    mask = np.ones((N,) * ndim)
    for idx in itertools.product(range(N), repeat=ndim):
        if Counter(SPIN[m] for m in idx[:n_upper]) != Counter(SPIN[m] for m in idx[n_upper:]):
            mask[idx] = 0.0
    return mask


@pytest.fixture(scope="module")
def values():
    vals = TensorValues(SimpleNamespace(n_modes=N), seed=12)  # only n_modes is read
    rng = np.random.default_rng(12)
    same = np.array([[float(SPIN[p] is SPIN[q]) for q in range(N)] for p in range(N)])
    g = rng.normal(size=(N,) * 4)  # chemist's (pq|rs), real: 8-fold symmetric
    g = g + g.transpose(1, 0, 2, 3)
    g = g + g.transpose(0, 1, 3, 2)
    g = g + g.transpose(2, 3, 0, 1)
    chem = g * same[:, :, None, None] * same[None, None, :, :]
    plain = chem.transpose(0, 2, 1, 3)  # <pq|rs> = (pr|qs)
    vals.set("v", 2, 2, SYM, plain - plain.transpose(0, 1, 3, 2))
    f = rng.normal(size=(N, N))
    vals.set("f", 1, 1, SYM, (f + f.T) * same)
    for name in "trl":
        for n in (1, 2):
            vals.set(name, n, n, NONE, vals.array(name, n, n, NONE) * _spin_mask(2 * n, n))
    return vals


class UHFMoeris:
    """Exactly the user's 21 ERI blocks and 6 Fock blocks; nothing else."""

    def __init__(self, values):
        v, f = values.array("v", 2, 2, SYM), values.array("f", 1, 1, SYM)
        for blk in USER_BLOCKS:
            setattr(self, "g_" + blk, v[tuple(SL[ch] for ch in blk)].copy())
        for blk in FOCK_BLOCKS:
            setattr(self, "f_" + blk, f[tuple(SL[ch] for ch in blk)].copy())


class SOMoeris:
    """Spin-orbital blocks (Step 11 code) over all modes."""

    def __init__(self, values):
        v, f = values.array("v", 2, 2, SYM), values.array("f", 1, 1, SYM)
        for blk in ["OOOO", "OOOV", "OOVV", "OVOV", "OVVV", "VVVV"]:
            setattr(self, "g_" + blk, v[tuple(SO[O if ch == "O" else V] for ch in blk)].copy())
        for blk in ["OO", "OV", "VV"]:
            setattr(self, "f_" + blk, f[tuple(SO[O if ch == "O" else V] for ch in blk)].copy())


def uhf_amplitude(values, name, n, blk):
    """blk in axis order, occupied first: OoVv holds t^{Ab}_{Ij} at [I,j,A,b]."""
    full = values.array(name, n, n, NONE)
    sub = full[tuple(SL[ch] for ch in blk[n:] + blk[:n])]  # upper (virt) then lower (occ)
    return sub.transpose(list(range(n, 2 * n)) + list(range(n))).copy()


def so_amplitude(values, name, n):
    full = values.array(name, n, n, NONE)
    sub = full[tuple([SO[V]] * n + [SO[O]] * n)]
    return sub.transpose(list(range(n, 2 * n)) + list(range(n))).copy()


def uhf_arrays(values):
    out = {"moeris": UHFMoeris(values)}
    for name in "trl":
        for blk in ["OV", "ov"]:
            out[f"{name}1_{blk}"] = uhf_amplitude(values, name, 1, blk)
        for blk in ["OOVV", "OoVv", "oovv"]:
            out[f"{name}2_{blk}"] = uhf_amplitude(values, name, 2, blk)
    return out


def so_arrays(values):
    out = {"moeris": SOMoeris(values)}
    for name in "trl":
        out[f"{name}1"] = so_amplitude(values, name, 1)
        out[f"{name}2"] = so_amplitude(values, name, 2)
    return out


def call(src, arrays):
    namespace = {}
    exec(compile(src, "<generated>", "exec"), namespace)
    fn = next(v for key, v in namespace.items() if callable(v) and key not in ("np", "oe"))
    params = fn.__code__.co_varnames[: fn.__code__.co_argcount]
    return np.asarray(fn(**{p: arrays[p] for p in params}))


def spin_orbital_reference(expr, values):
    """The spin-orbital result over all modes, from Step 11 code."""
    return call(einsum_code(expr, "so"), so_arrays(values))


def block_slice(key):
    """Spin-orbital positions (within the OCC and VIRT ranges) of a block's axes."""
    out = []
    for ch in key:
        s = SL[ch]
        base = 0 if ch in "Oo" else N_OCC
        out.append(slice(s.start - base, s.stop - base))
    return tuple(out)


def brute_block(expr, values, key):
    """Loops over a block expression: dummies over the modes of their space AND spin."""
    modes = {
        (O, AL): range(0, NOA), (O, BE): range(NOA, N_OCC),
        (V, AL): range(N_OCC, N_OCC + NVA), (V, BE): range(N_OCC + NVA, N),
    }
    free = sorted(expr.free_indices, key=lambda ix: (ix.space is not O, ix.spin is not AL, ix.name))
    lists = [list(modes[(ix.space, ix.spin)]) for ix in free]
    out = np.zeros(tuple(len(m) for m in lists))
    for pos in itertools.product(*(range(len(m)) for m in lists)):
        g = {ix: m[p] for ix, m, p in zip(free, lists, pos)}
        total = 0.0
        for term in expr:
            dummies = sorted(term.summed, key=lambda ix: ix.name)
            for dv in itertools.product(*(modes[(ix.space, ix.spin)] for ix in dummies)):
                gg = dict(g)
                gg.update(zip(dummies, dv))
                val = float(term.coeff)
                for s in term.scalars:
                    val *= values.value(s, gg)
                total += val
        out[pos] = total
    return out


# ---- derived equations ------------------------------------------------------------------
PHI, H = fluctuation_potential(), hamiltonian()
T1, T2 = cluster(1), cluster(2)
R1 = cluster(1, name="r")

DERIVED = {
    "<ab|[Phi,T2]>": lambda: projection(2, comm(PHI, T2)),
    "<a|[Phi,T2]>": lambda: projection(1, comm(PHI, T2)),
    "<ab|[[Phi,T1],R1]>": lambda: projection(2, comm(run(comm(PHI, T1)), R1)),
    "<a|[[Phi,T1],T2]>": lambda: projection(1, comm(run(comm(PHI, T1)), T2)),
    "CCSD energy": lambda: projection(0, bch(H, T1 + T2, order=2)),
    "CCD doubles": lambda: projection(2, bch(H, T2, order=2)),
}


@pytest.fixture(scope="module")
def derived():
    out = {}
    for name, make in DERIVED.items():
        x = make()
        out[name] = (x, spin_integrate(x))
    return out


# =========================================================================
# 1. Spin on Index
class TestSpinIndex:
    def test_default_none(self):
        assert Index("i", O).spin is None

    def test_equality_and_hash(self):
        assert sp("i", O, AL) != sp("i", O, BE) != Index("i", O)
        assert len({sp("i", O, AL), sp("i", O, BE), Index("i", O)}) == 3

    def test_spin_type(self):
        with pytest.raises(TypeError):
            Index("i", O, "alpha")

    def test_str_is_name(self):
        assert str(sp("I", O, AL)) == "I"

    def test_rename_keeps_spin(self):
        x = sp("I", O, AL)
        term = T(scalars=[Tensor("s", (), (x,))], summed=[x])
        with pytest.raises(ValueError):
            term.rename({x: sp("J", O, BE)})
        assert term.rename({x: sp("J", O, AL)}).summed == {sp("J", O, AL)}


# =========================================================================
# 2. Simplify with spin
class TestSimplifySpin:
    def test_alpha_and_beta_copies_do_not_merge(self):
        ca, cb = sp("C", V, AL), sp("c", V, BE)
        x = X(T(scalars=[Tensor("x", (ca,)), Tensor("y", (), (ca,))], summed=[ca]),
              T(scalars=[Tensor("x", (cb,)), Tensor("y", (), (cb,))], summed=[cb]))
        assert len(simplify(x)) == 2

    def test_same_spin_dummies_merge(self):
        ca, da = sp("C", V, AL), sp("D", V, AL)
        x = X(T(scalars=[Tensor("x", (ca,)), Tensor("y", (), (ca,))], summed=[ca]),
              T(scalars=[Tensor("x", (da,)), Tensor("y", (), (da,))], summed=[da]))
        (term,) = simplify(x).terms
        assert term.coeff == 2

    def test_relabel_alpha_upper_beta_lower(self):
        p1, p2 = sp("a7", V, AL), sp("a8", V, BE)
        x = X(T(scalars=[Tensor("x", (p1, p2))], summed=[p1, p2]))
        (term,) = relabel_dummies(x).terms
        assert term.summed == {sp("A", V, AL), sp("a", V, BE)}

    def test_relabel_avoids_free_names(self):
        free, dummy = sp("A", V, AL), sp("a7", V, AL)
        x = X(T(scalars=[Tensor("x", (free, dummy))], summed=[dummy]))
        (term,) = relabel_dummies(x).terms
        assert term.summed == {sp("B", V, AL)}


# =========================================================================
# 3. spin_integrate: structure
class TestStructure:
    def test_doubles_keys(self, derived):
        assert list(derived["<ab|[Phi,T2]>"][1]) == ["OOVV", "OoVv", "oovv"]

    def test_singles_keys(self, derived):
        assert list(derived["<a|[Phi,T2]>"][1]) == ["OV", "ov"]

    def test_energy_one_block(self, derived):
        assert list(derived["CCSD energy"][1]) == [""]

    def test_triples_keys(self):
        x = X(T(scalars=[Tensor("t", (a, b, c), (i, j, k))]))
        assert list(spin_integrate(x)) == ["OOOVVV", "OOoVVv", "OooVvv", "ooovvv"]

    def test_free_names_and_spins(self, derived):
        block = derived["<ab|[Phi,T2]>"][1]["OoVv"]
        assert block.free_indices == {sp("I", O, AL), sp("j", O, BE), sp("A", V, AL), sp("b", V, BE)}

    @pytest.mark.parametrize("name", list(DERIVED))
    def test_every_index_has_spin_and_tensors_conserve_it(self, derived, name):
        for expr in derived[name][1].values():
            for term in expr:
                for t in term.scalars:
                    assert all(ix.spin is not None for ix in t.indices)
                    assert Counter(ix.spin for ix in t.upper) == Counter(ix.spin for ix in t.lower)

    def test_term_counts_phi_t2(self, derived):
        blocks = derived["<ab|[Phi,T2]>"][1]
        assert {k: len(e) for k, e in blocks.items()} == {"OOVV": 10, "OoVv": 8, "oovv": 10}
        singles = derived["<a|[Phi,T2]>"][1]
        assert {k: len(e) for k, e in singles.items()} == {"OV": 4, "ov": 4}

    def test_simple_case_by_hand(self):
        # sum_c f^a_c t^c_i -> OV: sum_C f^A_C t^C_I ; ov: sum_c f^a_c t^c_i
        x = X(T(scalars=[Tensor("f", (a,), (c,), SYM), Tensor("t", (c,), (i,))], summed=[c]))
        blocks = spin_integrate(x)
        A_, C_, I_ = sp("A", V, AL), sp("B", V, AL), sp("I", O, AL)
        (term,) = blocks["OV"].terms
        (dummy,) = term.summed
        assert dummy.spin is AL and dummy.space is V
        assert {ix for s in term.scalars for ix in s.indices} - {dummy} == {A_, I_}
        assert len(blocks["ov"]) == 1


# =========================================================================
# 4. spin_integrate: values (exact against the spin-orbital result)
@pytest.mark.parametrize("name", list(DERIVED))
def test_block_values_equal_spin_orbital_values(values, derived, name):
    x, blocks = derived[name]
    ref = spin_orbital_reference(x, values)
    for key, expr in blocks.items():
        got = brute_block(expr, values, key)
        want = ref[block_slice(key)] if key else ref
        assert np.allclose(got, want), key


def test_other_spin_cases_follow_or_vanish(values, derived):
    x, _ = derived["<ab|[Phi,T2]>"]
    ref = spin_orbital_reference(x, values)  # r[i,j,a,b] over all modes
    oa, ob = block_slice("O")[0], block_slice("o")[0]
    va, vb = block_slice("V")[0], block_slice("v")[0]
    # beta-alpha = alpha-beta transposed (i<->j and a<->b), no sign
    assert np.allclose(ref[ob, oa, vb, va], ref[oa, ob, va, vb].transpose(1, 0, 3, 2))
    # alpha-beta with the virtual spins swapped: minus the a<->b transpose
    assert np.allclose(ref[oa, ob, vb, va], -ref[oa, ob, va, vb].transpose(0, 1, 3, 2))
    # Sz-changing blocks vanish
    assert np.allclose(ref[oa, oa, vb, vb], 0.0)
    assert np.allclose(ref[oa, oa, va, vb], 0.0)


# =========================================================================
# 5. Code generation with spin
def line(term, **kw):
    return einsum_code(X(term), "x", **kw).splitlines()[6].strip()


class TestCodegenSpin:
    def test_default_amplitude_names(self):
        I_, j_, A_, b_ = sp("I", O, AL), sp("j", O, BE), sp("A", V, AL), sp("b", V, BE)
        assert line(T(scalars=[Tensor("t", (A_, b_), (I_, j_))])) == \
            'r += np.einsum("IjAb->IjAb", t2_OoVv, optimize=True)'

    def test_beta_alpha_amplitude_is_transposed_without_sign(self):
        I_, j_, A_, b_ = sp("I", O, AL), sp("j", O, BE), sp("A", V, AL), sp("b", V, BE)
        assert line(T(scalars=[Tensor("t", (b_, A_), (j_, I_))])) == \
            'r += np.einsum("IjAb->IjAb", t2_OoVv, optimize=True)'

    def test_amplitude_group_sort_sign(self):
        I_, j_, A_, b_ = sp("I", O, AL), sp("j", O, BE), sp("A", V, AL), sp("b", V, BE)
        assert line(T(scalars=[Tensor("t", (b_, A_), (I_, j_))])) == \
            'r -= np.einsum("IjAb->IjAb", t2_OoVv, optimize=True)'

    def test_singles_names(self):
        i_, a_ = sp("i", O, BE), sp("a", V, BE)
        assert line(T(scalars=[Tensor("t", (a_,), (i_,))])) == 'r += np.einsum("ia->ia", t1_ov, optimize=True)'

    def test_output_alpha_first(self):
        I_, j_, A_, b_ = sp("I", O, AL), sp("j", O, BE), sp("A", V, AL), sp("b", V, BE)
        src = einsum_code(X(T(scalars=[Tensor("t", (A_, b_), (I_, j_))])), "x")
        assert "->IjAb" in src

    def test_output_spin_before_name(self):
        # alpha before beta even when the beta index's name sorts first
        jA, iB, aA, bB = sp("j", O, AL), sp("i", O, BE), sp("a", V, AL), sp("b", V, BE)
        src = einsum_code(X(T(scalars=[Tensor("t", (aA, bB), (jA, iB))])), "x")
        assert "->jiab" in src

    @pytest.mark.parametrize("blk", FOCK_BLOCKS)
    def test_fock_blocks(self, blk):
        names = {"O": ("P", O, AL), "o": ("p", O, BE), "V": ("Q", V, AL), "v": ("q", V, BE)}
        up, lo = (sp(*names[ch]) for ch in blk)
        lo = Index("r" if lo.spin is BE else "R", lo.space, lo.spin) if lo.name == up.name else lo
        assert f"moeris.f_{blk}," in line(T(scalars=[Tensor("f", (lo,), (up,), SYM)]))

    def test_mixed_spin_and_spinless(self):
        with pytest.raises(ValueError):
            einsum_code(X(T(scalars=[Tensor("t", (sp("A", V, AL),), (i,))])), "x")

    def test_integral_that_breaks_spin(self):
        x = X(T(scalars=[Tensor("v", (sp("I", O, AL), sp("J", O, AL)), (sp("A", V, AL), sp("b", V, BE)), SYM)]))
        with pytest.raises(ValueError):
            einsum_code(x, "x")

    def test_spinless_unchanged(self):
        assert '"ijab->ijab", moeris.g_OOVV' in line(T(scalars=[Tensor("v", (a, b), (i, j), SYM)]))


KINDS = [(O, AL), (O, BE), (V, AL), (V, BE)]
NAMES = {(O, AL): "IJKL", (O, BE): "ijkl", (V, AL): "ABCD", (V, BE): "abcd"}


def _all_spin_conserving_v():
    out = []
    for combo in itertools.product(KINDS, repeat=4):
        used = Counter()
        ixs = []
        for kind in combo:
            ixs.append(sp(NAMES[kind][used[kind]], *kind))
            used[kind] += 1
        up, lo = ixs[:2], ixs[2:]
        if Counter(x.spin for x in up) == Counter(x.spin for x in lo):
            out.append(Tensor("v", tuple(up), tuple(lo), SYM))
    return out


SPIN_V = _all_spin_conserving_v()


def test_all_spin_conserving_integrals_land_on_user_blocks():
    used = set()
    for t in SPIN_V:
        src = einsum_code(X(T(scalars=[t])), "x")
        (blk,) = [w[len("moeris.g_"):] for w in src.replace(",", " ").split() if w.startswith("moeris.g_")]
        assert blk in USER_BLOCKS, (t, blk)
        used.add(blk)
    assert used == USER_BLOCKS


def test_every_integral_block_has_the_right_value(values):
    arrays = uhf_arrays(values)
    v = values.array("v", 2, 2, SYM)
    for t in SPIN_V:
        got = call(einsum_code(X(T(scalars=[t])), "x"), arrays)
        free = sorted(t.indices, key=lambda ix: (ix.space is not O, ix.spin is not AL, ix.name))
        sl = [SL[{(O, AL): "O", (O, BE): "o", (V, AL): "V", (V, BE): "v"}[(ix.space, ix.spin)]] for ix in free]
        want = v[tuple(sl[free.index(ix)] for ix in t.indices)]
        perm = [t.indices.index(ix) for ix in free]
        assert np.allclose(got, want.transpose(perm)), t


@pytest.mark.parametrize("name", list(DERIVED))
def test_generated_block_code_equals_spin_orbital_values(values, derived, name):
    x, blocks = derived[name]
    ref = spin_orbital_reference(x, values)
    arrays = uhf_arrays(values)
    for key, expr in blocks.items():
        got = call(einsum_code(expr, "block"), arrays)
        want = ref[block_slice(key)] if key else ref
        assert np.allclose(got, want), key


def textbook_uhf_ccsd_energy(m, t1_OV, t1_ov, t2_OOVV, t2_OoVv, t2_oovv):
    """UHF CCSD energy from H (reference part included), written by hand."""
    e = np.einsum("ii->", m.f_OO) + np.einsum("ii->", m.f_oo)
    e -= 0.5 * np.einsum("ijij->", m.g_OOOO) + np.einsum("ijij->", m.g_OoOo) + 0.5 * np.einsum("ijij->", m.g_oooo)
    e += np.einsum("ia,ia->", m.f_OV, t1_OV) + np.einsum("ia,ia->", m.f_ov, t1_ov)
    e += 0.25 * np.einsum("ijab,ijab->", m.g_OOVV, t2_OOVV)
    e += np.einsum("ijab,ijab->", m.g_OoVv, t2_OoVv)
    e += 0.25 * np.einsum("ijab,ijab->", m.g_oovv, t2_oovv)
    e += 0.5 * np.einsum("ijab,ia,jb->", m.g_OOVV, t1_OV, t1_OV)
    e += np.einsum("ijab,ia,jb->", m.g_OoVv, t1_OV, t1_ov)
    e += 0.5 * np.einsum("ijab,ia,jb->", m.g_oovv, t1_ov, t1_ov)
    return e


def test_textbook_uhf_ccsd_energy(values, derived):
    _, blocks = derived["CCSD energy"]
    arrays = uhf_arrays(values)
    got = call(einsum_code(blocks[""], "energy"), arrays)
    want = textbook_uhf_ccsd_energy(
        arrays["moeris"], arrays["t1_OV"], arrays["t1_ov"], arrays["t2_OOVV"], arrays["t2_OoVv"], arrays["t2_oovv"]
    )
    assert np.isclose(got, want)


# =========================================================================
# 6. Errors
class TestErrors:
    def test_operators(self):
        with pytest.raises(ValueError):
            spin_integrate(X(T(factors=[cre(a) * ann(i)])))

    def test_delta(self):
        with pytest.raises(ValueError):
            spin_integrate(X(T(scalars=[Delta(i, j), Tensor("f", (a,), (b,), SYM)])))

    def test_general(self):
        with pytest.raises(ValueError):
            spin_integrate(X(T(scalars=[Tensor("f", (Index("p"),), (i,), SYM)])))

    def test_already_spin(self):
        with pytest.raises(ValueError):
            spin_integrate(X(T(scalars=[Tensor("t", (sp("A", V, AL),), (sp("I", O, AL),))])))

    def test_unequal_ranks(self):
        with pytest.raises(ValueError):
            spin_integrate(X(T(scalars=[Tensor("s", (a,), ()), Tensor("u", (), (i,))])))

    def test_unequal_free_counts(self):
        with pytest.raises(ValueError):
            spin_integrate(X(T(scalars=[Tensor("f", (a,), (b,), SYM)])))

    def test_free_name_clash(self):
        big, small = Index("I", O), Index("i", O)
        x = X(T(scalars=[Tensor("t", (a, b), (big, small))]))
        with pytest.raises(ValueError):
            spin_integrate(x)


# =========================================================================
# 7. Packaging
def test_public_api():
    assert {"Spin", "spin_integrate"} <= set(alqemi.__all__)
    assert alqemi.Spin is Spin and alqemi.spin_integrate is spin_integrate
