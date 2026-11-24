"""Step 11 tests: einsum code generation.

These tests DEFINE Step 11. Generated code is checked numerically: it is executed
on random tensor values (tests/evaluate.py, TensorValues) and compared with a
direct loop evaluation of the same ALQEMI expression, and, for the CCSD energy
and the CCD doubles residual, with the textbook spin-orbital formulas written
by hand. Exact source text is fixed only where the format is the point.

  alqemi/codegen/einsum.py
    einsum_code(expr, name, *, integrals=None, amplitudes="{name}{rank}",
                backend="numpy") -> str
        Python source of one function ``name`` that evaluates ``expr``, an
        operator-free Expr (or Term), e.g. a projection(...) result:

            import numpy as np


            def ccd_doubles(moeris, t2):
                r = 0.0
                # Σ_{c} f^{a}_{c} t^{cb}_{ij}
                r += np.einsum("ac,ijcb->ijab", moeris.f_VV, t2, optimize=True)
                ...
                return r

        Layout
          - First line: "import numpy as np" (backend "numpy") or
            "import opt_einsum as oe" (backend "opt_einsum"); two blank lines;
            the def; the body is indented 4 spaces; the source ends in "\\n".
          - Parameters: the integral objects (each integral template's leading
            name, e.g. ``moeris``), sorted, then the amplitude arguments, sorted.
          - Body: "r = 0.0", then for each term, in the expression's order, a
            comment "# " + str(term) and one line
                r += <c> * np.einsum("<spec>", <operands>, optimize=True)
            (backend "opt_einsum": oe.contract("<spec>", <operands>)), then
            "return r". "r = 0.0" first means r never aliases an input: a
            single-tensor term's einsum may return a view, and 0.0 + view is a
            new array. A scalar expression returns a 0-d result (a number).
          - A term without tensors (a pure number, no free indices) is
            "r += <c>".
          - Sign and coefficient: c = term.coeff times the sign of the
            canonical forms below. "r -=" when c < 0, with |c| written as:
            omitted when 1 ("r -= np.einsum(...)"); an integer "2 * "; a
            fraction exact in binary as a float "0.5 * ", "0.75 * "; any other
            fraction "(1 / 3) * ".

        Index orientation: OCCUPIED FIRST everywhere (the OOVV convention).
          - Output axes: the free OCC indices sorted by name, then the free VIRT
            indices sorted by name. projection(2, .) gives r[i, j, a, b].
          - Einsum letters: an index whose name is one ASCII letter keeps it;
            any other index gets an unused letter.

        Integrals: tensors whose name is a key of ``integrals`` (default
          {"f": "moeris.f_{block}", "h": "moeris.h_{block}",
           "v": "moeris.g_{block}"}, the antisymmetrized <pq||rs> in physicist's
          order). Each is rewritten exactly to an occupied-first block:
            1. Within the upper and within the lower group, a stable sort puts
               OCC before VIRT; sign = parity of the permutation.
            2. If the tensor has a swap symmetry and the lower group's block
               ("O"/"V" letters) sorts before the upper's, exchange the groups;
               sign -1 more if Swap.ANTISYMMETRIC.
          The operand is the template with {block} = upper block + lower block
          (e.g. "OVOV"), axes upper then lower. So v^{ak}_{ic} -> -g_OVOV[k,a,i,c]
          and v^{ab}_{ij} -> g_OOVV[i,j,a,b]; ⟨pq||rs⟩ terms only ever use the
          six unique blocks OOOO, OOOV, OOVV, OVOV, OVVV, VVVV, and f, h only
          OO, OV, VV.

        Amplitudes: every other tensor. The argument name is
          amplitudes.format(name=tensor name, rank=n) for n upper and n lower
          indices, rank "<n_upper>_<n_lower>" otherwise (t1, t2, l2, s1_0).
          Step 1 above applies (with its sign); the array's axes are then all
          indices, OCC first, otherwise in upper-then-lower order (storage,
          no sign): t^{ab}_{ij} -> t2[i,j,a,b], l^{ij}_{ab} -> l2[i,j,a,b],
          t^{a}_{i} -> t1[i,a]. Every occurrence of one argument must have the
          same (upper block, lower block).

        ValueError for: a term with operators; a Delta or ElementaryBracket
        (deltas between free indices are not supported yet); a GENERAL index; a
        term whose free indices differ from the expression's; ``name`` or an
        amplitude argument that is not an identifier; an integral template
        without a leading name; an argument name equal to another, to ``name``,
        or to r, np, oe; one amplitude argument with two block patterns; an
        unknown backend.

  Spin-orbital code only. Spin integration for UHF (αα/αβ/ββ blocks) is a later
  derivation step. The module imports no numpy (it only writes text).
  Public API: einsum_code joins ``alqemi.__all__``.
"""
import itertools
import subprocess
import sys
from fractions import Fraction

import numpy as np
import pytest

import alqemi
from alqemi.builders import cluster, deexcitation, hamiltonian
from alqemi.codegen.einsum import einsum_code
from alqemi.core.expr import Delta, Expr, Term, as_expr, comm, multiply
from alqemi.core.indices import Index, Space
from alqemi.core.operators import ann, cre
from alqemi.core.pool import IndexPool
from alqemi.core.tensor import Swap, Tensor
from alqemi.pipeline.bch import bch
from alqemi.pipeline.project import projection, vacuum_expectation
from alqemi.tests.evaluate import TensorValues
from alqemi.tests.oracle import FockSpace

O, V, G = Space.OCC, Space.VIRT, Space.GENERAL
a, b, c, d = (Index(x, V) for x in "abcd")
i, j, k, l = (Index(x, O) for x in "ijkl")
p = Index("p")
SYM, ANTI, NONE = Swap.SYMMETRIC, Swap.ANTISYMMETRIC, Swap.NONE

N_OCC, N_VIRT = 2, 3  # unequal, so a swapped axis cannot go unnoticed
OCC, VIR = slice(0, N_OCC), slice(N_OCC, N_OCC + N_VIRT)
SL = {O: OCC, V: VIR}


def T(coeff=1, scalars=(), summed=(), factors=()):
    return Term(coeff=coeff, scalars=tuple(scalars), factors=tuple(factors), summed=frozenset(summed))


def X(*terms):
    return Expr(tuple(terms))


def f_(u, w):
    return Tensor("f", (u,), (w,), SYM)


def v_(u1, u2, w1, w2):
    return Tensor("v", (u1, u2), (w1, w2), SYM)


def t_(upper, lower, name="t"):
    return Tensor(name, tuple(upper), tuple(lower))


# ---- numerical harness -----------------------------------------------------------
@pytest.fixture(scope="module")
def values():
    return TensorValues(FockSpace(n_occ=N_OCC, n_virt=N_VIRT), seed=11)


class Moeris:
    """Only the unique occupied-first blocks; any other block is an AttributeError."""

    def __init__(self, values):
        for name, attr in [("f", "f"), ("h", "h")]:
            full = values.array(name, 1, 1, SYM)
            for blk in ["OO", "OV", "VV"]:
                setattr(self, f"{attr}_{blk}", self._block(full, blk))
        full = values.array("v", 2, 2, SYM)
        for blk in ["OOOO", "OOOV", "OOVV", "OVOV", "OVVV", "VVVV"]:
            setattr(self, f"g_{blk}", self._block(full, blk))

    @staticmethod
    def _block(full, blk):
        return full[tuple(SL[O if ch == "O" else V] for ch in blk)].copy()


def amplitude(values, name, upper, lower):
    """The block array of tensor ``name`` for spaces (upper, lower), axes occupied first."""
    full = values.array(name, len(upper), len(lower), NONE)
    spaces = list(upper) + list(lower)
    blk = full[tuple(SL[s] for s in spaces)]
    order = sorted(range(len(spaces)), key=lambda n: spaces[n] is not O)
    return blk.transpose(order).copy()


def call(src, name, **arrays):
    namespace = {}
    exec(compile(src, f"<{name}>", "exec"), namespace)
    return namespace[name](**arrays)


def brute(expr, values):
    """Direct loops over every index value; output axes occupied first, by name."""
    expr = as_expr(expr)
    free = sorted(expr.free_indices, key=lambda ix: (ix.space is not O, ix.name))
    modes = {O: range(0, N_OCC), V: range(N_OCC, N_OCC + N_VIRT)}
    shape = tuple(N_OCC if ix.space is O else N_VIRT for ix in free)
    out = np.zeros(shape)
    for fv in itertools.product(*(modes[ix.space] for ix in free)):
        pos = tuple(m if ix.space is O else m - N_OCC for ix, m in zip(free, fv))
        total = 0.0
        for term in expr:
            dummies = sorted(term.summed, key=lambda ix: ix.name)
            for dv in itertools.product(*(modes[ix.space] for ix in dummies)):
                g = dict(zip(free, fv))
                g.update(zip(dummies, dv))
                val = float(term.coeff)
                for s in term.scalars:
                    val *= values.value(s, g)
                total += val
        out[pos] = total
    return out


# ---- derived equations (module scope: derived once) --------------------------------
@pytest.fixture(scope="module")
def ccsd_energy():
    return projection(0, bch(hamiltonian(), cluster(1) + cluster(2), order=2))


@pytest.fixture(scope="module")
def ccd_doubles():
    return projection(2, bch(hamiltonian(), cluster(2), order=2))


@pytest.fixture(scope="module")
def singles_order2():
    return projection(1, bch(hamiltonian(), cluster(1) + cluster(2), order=2))


@pytest.fixture(scope="module")
def lambda_energy():
    """<HF| Lambda2 [H, T2] |HF>: uses l2 (OOVV) and t2 together."""
    pool = IndexPool()
    return vacuum_expectation(multiply(deexcitation(2), as_expr(comm(hamiltonian(), cluster(2))), pool))


def arrays_for(values, *, t1=False, t2=False, l2=False):
    out = {"moeris": Moeris(values)}
    if t1:
        out["t1"] = amplitude(values, "t", [V], [O])
    if t2:
        out["t2"] = amplitude(values, "t", [V, V], [O, O])
    if l2:
        out["l2"] = amplitude(values, "l", [O, O], [V, V])
    return out


# =========================================================================
# 1. Exact source format
class TestFormat:
    def test_single_term_source(self):
        src = einsum_code(X(T(scalars=[v_(a, b, i, j)])), "r2")
        assert src == (
            "import numpy as np\n"
            "\n"
            "\n"
            "def r2(moeris):\n"
            "    r = 0.0\n"
            "    # v^{ab}_{ij}\n"
            '    r += np.einsum("ijab->ijab", moeris.g_OOVV, optimize=True)\n'
            "    return r\n"
        )

    def test_opt_einsum_backend_source(self):
        src = einsum_code(X(T(scalars=[f_(a, c), t_([c], [i])], summed=[c])), "r1", backend="opt_einsum")
        assert src.splitlines()[0] == "import opt_einsum as oe"
        assert '    r += oe.contract("ac,ic->ia", moeris.f_VV, t1)' in src.splitlines()
        assert "np." not in src

    def test_empty_expression(self):
        src = einsum_code(Expr(()), "zero")
        assert src == "import numpy as np\n\n\ndef zero():\n    r = 0.0\n    return r\n"
        assert call(src, "zero") == 0.0

    def test_comment_is_the_term(self):
        term = T(-1, [f_(k, j), t_([a, b], [i, k])], [k])
        lines = einsum_code(X(term), "x").splitlines()
        assert lines[5] == f"    # {term}"

    def test_parameters_sorted_integrals_then_amplitudes(self):
        x = X(T(scalars=[t_([a], [i]), t_([a, b], [i, j]), t_([a, b], [i, j], "l"), f_(i, a)], summed=[a, b, i, j]))
        line = einsum_code(x, "e").splitlines()[3]
        assert line == "def e(moeris, l2, t1, t2):"

    @pytest.mark.parametrize(
        "coeff, text",
        [
            (1, "r += np.einsum("),
            (-1, "r -= np.einsum("),
            (2, "r += 2 * np.einsum("),
            (Fraction(1, 2), "r += 0.5 * np.einsum("),
            (Fraction(-3, 4), "r -= 0.75 * np.einsum("),
            (Fraction(1, 3), "r += (1 / 3) * np.einsum("),
            (Fraction(-2, 3), "r -= (2 / 3) * np.einsum("),
        ],
    )
    def test_coefficients(self, coeff, text):
        src = einsum_code(X(T(coeff, [f_(i, a)], [i, a])), "e")
        assert src.splitlines()[6] == "    " + text + '"ia->", moeris.f_OV, optimize=True)'

    def test_pure_number_term(self):
        src = einsum_code(X(T(Fraction(-1, 2))), "c")
        assert "    r -= 0.5" in src.splitlines()
        assert call(src, "c") == -0.5

    def test_ends_with_newline_and_compiles(self, ccd_doubles):
        src = einsum_code(ccd_doubles, "ccd_doubles")
        assert src.endswith("return r\n")
        compile(src, "<ccd>", "exec")

    def test_one_line_per_term(self, ccd_doubles):
        src = einsum_code(ccd_doubles, "ccd_doubles")
        assert sum(1 for line in src.splitlines() if "np.einsum(" in line) == len(ccd_doubles)


# =========================================================================
# 2. Occupied-first orientation and canonical blocks
def operand_line(term, **kw):
    return einsum_code(X(term), "x", **kw).splitlines()[6].strip()


class TestOrientation:
    def test_output_axes_occupied_first(self):
        assert '"ijab->ijab"' in operand_line(T(scalars=[v_(a, b, i, j)]))

    def test_output_sorted_by_name_within_space(self):
        line = operand_line(T(scalars=[t_([b, a], [j, i])]))
        assert line == 'r += np.einsum("jiba->ijab", t2, optimize=True)'

    def test_t2_axes(self):
        assert '"ijab->ijab", t2' in operand_line(T(scalars=[t_([a, b], [i, j])]))

    def test_t1_axes(self):
        assert '"ia->ia", t1' in operand_line(T(scalars=[t_([a], [i])]))

    def test_l2_axes(self):
        assert '"ijab->ijab", l2' in operand_line(T(scalars=[t_([i, j], [a, b], "l")]))

    def test_v_swapped_to_oovv(self):
        assert '"ijab->ijab", moeris.g_OOVV' in operand_line(T(scalars=[v_(a, b, i, j)]))

    def test_v_within_group_sort_sign(self):
        line = operand_line(T(scalars=[v_(a, k, i, c)], summed=[]))
        assert line == 'r -= np.einsum("kaic->ikac", moeris.g_OVOV, optimize=True)'

    def test_v_both_groups_sorted_sign_cancels(self):
        line = operand_line(T(scalars=[v_(a, k, c, i)]))
        assert line == 'r += np.einsum("kaic->ikac", moeris.g_OVOV, optimize=True)'

    @pytest.mark.parametrize(
        "tensor, block",
        [
            (v_(i, j, k, l), "OOOO"), (v_(i, a, j, k), "OOOV"), (v_(a, i, j, k), "OOOV"),
            (v_(i, j, a, b), "OOVV"), (v_(a, b, i, j), "OOVV"), (v_(i, a, j, b), "OVOV"),
            (v_(a, i, b, j), "OVOV"), (v_(i, a, b, c), "OVVV"), (v_(a, b, i, c), "OVVV"),
            (v_(a, b, c, d), "VVVV"), (f_(i, a), "OV"), (f_(a, i), "OV"), (f_(i, j), "OO"),
            (f_(a, b), "VV"),
        ],
    )
    def test_only_unique_blocks(self, tensor, block):
        assert f"moeris.{tensor.name if tensor.name != 'v' else 'g'}_{block}," in operand_line(T(scalars=[tensor]))

    def test_antisymmetric_swap_sign(self):
        w = Tensor("w", (a,), (i,), ANTI)
        line = operand_line(T(scalars=[w]), integrals={"w": "w_{block}"})
        assert line == 'r -= np.einsum("ia->ia", w_OV, optimize=True)'

    def test_no_swap_keeps_block(self):
        x = Tensor("x", (a,), (i,))
        line = operand_line(T(scalars=[x]), integrals={"x": "ints.x_{block}"})
        assert line == 'r += np.einsum("ai->ia", ints.x_VO, optimize=True)'

    def test_mixed_amplitude_sign_and_axes(self):
        s1 = operand_line(T(scalars=[Tensor("s", (i, a), (j, b))]))
        s2 = operand_line(T(scalars=[Tensor("s", (a, i), (j, b))]))
        assert s1 == 'r += np.einsum("ijab->ijab", s2, optimize=True)'
        assert s2 == 'r -= np.einsum("ijab->ijab", s2, optimize=True)'

    def test_unequal_ranks_name(self):
        assert '"a->a", s1_0' in operand_line(T(scalars=[Tensor("s", (a,), ())]))

    def test_amplitude_template(self):
        line = operand_line(T(scalars=[t_([a, b], [i, j])]), amplitudes="{name}{rank}_amp")
        assert line.endswith("t2_amp, optimize=True)")

    def test_custom_integrals_root_is_parameter(self):
        src = einsum_code(X(T(scalars=[v_(a, b, i, j)])), "x", integrals={"v": "eri.v_{block}"})
        assert "def x(eri):" in src and "eri.v_OOVV" in src

    def test_long_names_get_letters(self):
        a1, i1 = Index("a1", V), Index("i1", O)
        x = X(T(scalars=[f_(i1, a1), t_([a1], [i1])], summed=[a1, i1]))
        line = einsum_code(x, "e").splitlines()[6]
        spec = line.split('"')[1]
        lhs = spec.split("->")[0].split(",")
        assert len(lhs) == 2 and all(len(s) == 2 for s in lhs) and lhs[0] == lhs[1]


# =========================================================================
# 3. Numbers: generated code == direct loops over the same expression
class TestValues:
    def test_ccsd_energy(self, values, ccsd_energy):
        src = einsum_code(ccsd_energy, "ccsd_energy")
        got = call(src, "ccsd_energy", **arrays_for(values, t1=True, t2=True))
        assert np.ndim(got) == 0
        assert np.isclose(got, brute(ccsd_energy, values))

    def test_ccd_doubles(self, values, ccd_doubles):
        src = einsum_code(ccd_doubles, "ccd_doubles")
        got = call(src, "ccd_doubles", **arrays_for(values, t2=True))
        assert got.shape == (N_OCC, N_OCC, N_VIRT, N_VIRT)
        assert np.allclose(got, brute(ccd_doubles, values))

    def test_singles_order2(self, values, singles_order2):
        src = einsum_code(singles_order2, "r1")
        got = call(src, "r1", **arrays_for(values, t1=True, t2=True))
        assert got.shape == (N_OCC, N_VIRT)
        assert np.allclose(got, brute(singles_order2, values))

    def test_lambda_with_l2(self, values, lambda_energy):
        src = einsum_code(lambda_energy, "lam")
        got = call(src, "lam", **arrays_for(values, t2=True, l2=True))
        assert np.isclose(got, brute(lambda_energy, values))

    def test_normal_ordered_hamiltonian(self, values):
        x = projection(2, bch(hamiltonian(normal_ordered=True), cluster(2), order=1))
        got = call(einsum_code(x, "r2"), "r2", **arrays_for(values, t2=True))
        assert np.allclose(got, brute(x, values))

    def test_signed_blocks_value(self, values):
        x = X(T(scalars=[v_(a, k, i, c), t_([c], [k])], summed=[c, k]),
              T(-1, [v_(a, b, c, i), t_([b], [k]), t_([c], [k])], [b, c, k]))
        got = call(einsum_code(x, "e"), "e", **arrays_for(values, t1=True))
        assert np.allclose(got, brute(x, values))

    def test_inputs_not_modified(self, values):
        x = X(T(scalars=[v_(a, b, i, j)]), T(2, [v_(a, b, i, j)]))
        arrays = arrays_for(values)
        before = arrays["moeris"].g_OOVV.copy()
        got = call(einsum_code(x, "e"), "e", **arrays)
        assert np.allclose(arrays["moeris"].g_OOVV, before)
        assert np.allclose(got, 3 * before)

    def test_opt_einsum_values(self, values, ccd_doubles):
        pytest.importorskip("opt_einsum")
        src = einsum_code(ccd_doubles, "ccd_doubles", backend="opt_einsum")
        got = call(src, "ccd_doubles", **arrays_for(values, t2=True))
        assert np.allclose(got, brute(ccd_doubles, values))


# =========================================================================
# 4. Textbook spin-orbital formulas (occupied first), written by hand
def textbook_energy(m, t1, t2):
    e = np.einsum("ii->", m.f_OO) - 0.5 * np.einsum("ijij->", m.g_OOOO)
    e += np.einsum("ia,ia->", m.f_OV, t1)
    e += 0.25 * np.einsum("ijab,ijab->", m.g_OOVV, t2)
    e += 0.5 * np.einsum("ijab,ia,jb->", m.g_OOVV, t1, t1)
    return e


def textbook_ccd(m, t2):
    """Spin-orbital CCD residual, no HF assumption (f off-diagonal kept)."""
    g_OVVO = -m.g_OVOV.transpose(0, 1, 3, 2)  # <kb||cj> = -<kb||jc>
    r = m.g_OOVV.copy()  # <ab||ij> = <ij||ab> (real), as [i,j,a,b]
    pab = np.einsum("bc,ijac->ijab", m.f_VV, t2)
    r += pab - pab.transpose(0, 1, 3, 2)
    pij = -np.einsum("kj,ikab->ijab", m.f_OO, t2)
    r += pij - pij.transpose(1, 0, 2, 3)
    r += 0.5 * np.einsum("klij,klab->ijab", m.g_OOOO, t2)
    r += 0.5 * np.einsum("abcd,ijcd->ijab", m.g_VVVV, t2)
    x = np.einsum("kbcj,ikac->ijab", g_OVVO, t2)
    r += x - x.transpose(1, 0, 2, 3) - x.transpose(0, 1, 3, 2) + x.transpose(1, 0, 3, 2)
    r += 0.25 * np.einsum("klcd,ijcd,klab->ijab", m.g_OOVV, t2, t2)
    y = np.einsum("klcd,ikac,jlbd->ijab", m.g_OOVV, t2, t2)
    r += y - y.transpose(1, 0, 2, 3)
    y = -0.5 * np.einsum("klcd,ikab,jlcd->ijab", m.g_OOVV, t2, t2)
    r += y - y.transpose(1, 0, 2, 3)
    y = -0.5 * np.einsum("klcd,ijac,klbd->ijab", m.g_OOVV, t2, t2)
    r += y - y.transpose(0, 1, 3, 2)
    return r


class TestTextbook:
    def test_ccsd_energy(self, values, ccsd_energy):
        arrays = arrays_for(values, t1=True, t2=True)
        got = call(einsum_code(ccsd_energy, "ccsd_energy"), "ccsd_energy", **arrays)
        assert np.isclose(got, textbook_energy(arrays["moeris"], arrays["t1"], arrays["t2"]))

    def test_ccd_doubles(self, values, ccd_doubles):
        arrays = arrays_for(values, t2=True)
        got = call(einsum_code(ccd_doubles, "ccd_doubles"), "ccd_doubles", **arrays)
        assert np.allclose(got, textbook_ccd(arrays["moeris"], arrays["t2"]))


# =========================================================================
# 5. Errors
class TestErrors:
    def test_operators(self):
        with pytest.raises(ValueError):
            einsum_code(X(T(factors=[cre(a) * ann(i)])), "x")

    def test_delta(self):
        with pytest.raises(ValueError):
            einsum_code(X(T(scalars=[Delta(i, j), f_(a, b)])), "x")

    def test_general_free(self):
        with pytest.raises(ValueError):
            einsum_code(X(T(scalars=[f_(p, i)])), "x")

    def test_general_summed(self):
        with pytest.raises(ValueError):
            einsum_code(X(T(scalars=[f_(p, i), t_([a], [p])], summed=[p])), "x")

    def test_free_indices_differ(self):
        with pytest.raises(ValueError):
            einsum_code(X(T(scalars=[f_(i, a)]), T(scalars=[f_(i, b)])), "x")

    @pytest.mark.parametrize("name", ["", "2x", "a-b", "def x", None])
    def test_bad_name(self, name):
        with pytest.raises(ValueError):
            einsum_code(X(T(scalars=[f_(i, a)])), name)

    def test_bad_amplitude_name(self):
        with pytest.raises(ValueError):
            einsum_code(X(T(scalars=[t_([a], [i])])), "x", amplitudes="{name}-{rank}")

    def test_template_without_leading_name(self):
        with pytest.raises(ValueError):
            einsum_code(X(T(scalars=[f_(i, a)])), "x", integrals={"f": "{block}"})

    @pytest.mark.parametrize("amp", ["moeris", "r", "np", "oe", "x"])
    def test_argument_clash(self, amp):
        x = X(T(scalars=[t_([a], [i]), f_(i, a)], summed=[a, i]))
        with pytest.raises(ValueError):
            einsum_code(x, "x", amplitudes=amp)

    def test_two_block_patterns(self):
        x = X(T(scalars=[Tensor("s", (i, a), (j, b))]), T(scalars=[Tensor("s", (i, j), (a, b))]))
        with pytest.raises(ValueError):
            einsum_code(x, "x")

    def test_unknown_backend(self):
        with pytest.raises(ValueError):
            einsum_code(X(T(scalars=[f_(i, a)])), "x", backend="torch")


# =========================================================================
# 6. Packaging
def test_public_api():
    assert "einsum_code" in alqemi.__all__
    assert alqemi.einsum_code is einsum_code


def test_no_numpy_import():
    code = "import sys, alqemi.codegen.einsum; print('numpy' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_term_accepted():
    src = einsum_code(T(scalars=[f_(i, a)]), "x")
    assert "moeris.f_OV" in src
