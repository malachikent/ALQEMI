"""Step 14 tests: grouping terms with permutation operators (1 - P_ij), (1 - P_ab).

These tests DEFINE Step 14. Settled with the user (2026-11-24): the doubles forms
(1 - P_{ij}), (1 - P_{ab}) and (1 - P_{ij})(1 - P_{ab}) only (triples forms such as
P(i/jk) later); printed as (1 - P_{ij}), the style of the user's hand-written code;
used by both the LaTeX printer and code generation; opt-in (group=True), so
default output stays term by term.

  alqemi/pipeline/permute.py
    PermutedTerm(term, swaps)
      Frozen. Value: (1 - P_s1)(1 - P_s2) term, where P_s exchanges the two
      indices of swap s everywhere in the term. ``swaps`` has 0, 1 or 2 pairs;
      each pair is two FREE indices of ``term`` with the same space and spin,
      sorted by name; with two pairs the OCC pair comes first and the pairs are
      disjoint. ValueError otherwise (TypeError for a non-Term).
    expand_permutations(groups) -> Expr
      Each group written out, in order: the term, then for each non-empty subset
      of the swaps (in the order P_s1, P_s2, P_s1 P_s2) the term with those
      exchanges applied and sign (-1)^(subset size). The expansion is exact.
    group_permutations(expr) -> tuple[PermutedTerm, ...]
      The swaps an expression allows: for each (space, spin), the free indices of
      that kind if there are exactly two of them (a doubles-type pair).
      Each term's value is c K with K a canonical form (Simplify's like-term key,
      so dummy renaming and tensor symmetries are taken into account, with their
      sign) and c its coefficient times that sign. Greedy, in expression order,
      over terms not yet used; for the current term t:
        1. if two swaps exist: group t with three other unused terms whose values
           are exactly -P_occ t, -P_virt t and +P_occ P_virt t;
        2. else (or if not all three exist): with an unused term equal to
           -P_occ t; then with one equal to -P_virt t;
        3. else t stands alone (swaps = ()).
      Partners are distinct terms, never t itself. A swap under which t maps to
      plus or minus itself (same canonical form, e.g. <ab||ij> under P_ij) is
      never used for t, so such a term stands alone, even when the expression
      (unsimplified) holds a copy of it. The
      representative is t as stored. Always: expand_permutations(result) equals
      expr exactly (as like terms; checked with Simplify).

  alqemi/printing/latex.py
    latex(x, ..., group=False): with group=True an Expr is grouped first. A
    PermutedTerm prints as [coefficient] (1 - P_{ij})(1 - P_{ab}) [\\sum ...]
    factors, i.e. the coefficient as in Step 13, then "(1 - P_{pq})" for each
    swap (pq the pair's index list) with no space between them and one space
    after the last, then the term without its coefficient. Sums
    and align* (lhs, terms_per_line) count groups as terms.

  alqemi/codegen/einsum.py
    einsum_code(..., group=False): with group=True an Expr is grouped first; a
    group without swaps is written as in Step 11; a group with swaps as
        # (1 - P_ij)(1 - P_ab) <str(term)>
        tmp = np.einsum("<spec>", <operands>, optimize=True)
        r -= 0.5 * (tmp - tmp.swapaxes(0, 1) - tmp.swapaxes(2, 3) + tmp.swapaxes(0, 1).swapaxes(2, 3))
    (one swap: (tmp - tmp.swapaxes(p, q))), axes = the positions of the swapped
    indices in the output, coefficient and sign as in Step 11. With group=True,
    "tmp" is also a reserved argument name. Numbers are identical to group=False.

  Public API: PermutedTerm, group_permutations, expand_permutations join
  ``alqemi.__all__``.
"""
import itertools
from collections import Counter
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

import alqemi
from alqemi.builders import cluster, fluctuation_potential, hamiltonian
from alqemi.codegen.einsum import einsum_code
from alqemi.core.expr import Expr, Term, comm
from alqemi.core.indices import Index, Space, Spin
from alqemi.core.tensor import Swap, Tensor
from alqemi.pipeline.bch import bch
from alqemi.pipeline.permute import PermutedTerm, expand_permutations, group_permutations
from alqemi.pipeline.project import projection
from alqemi.pipeline.run import run
from alqemi.pipeline.simplify import simplify
from alqemi.pipeline.spin import spin_integrate
from alqemi.printing.latex import latex
from alqemi.tests.evaluate import TensorValues

O, V = Space.OCC, Space.VIRT
AL, BE = Spin.ALPHA, Spin.BETA
SYM, NONE = Swap.SYMMETRIC, Swap.NONE
a, b, c, d = (Index(x, V) for x in "abcd")
i, j, k, l = (Index(x, O) for x in "ijkl")


def T(coeff=1, scalars=(), summed=()):
    return Term(coeff=coeff, scalars=tuple(scalars), summed=frozenset(summed))


def X(*terms):
    return Expr(tuple(terms))


def v_(u1, u2, w1, w2):
    return Tensor("v", (u1, u2), (w1, w2), SYM)


def same(x, y):
    """x and y are equal sums (like terms merged by Simplify)."""
    return len(simplify(Expr(tuple(x.terms) + tuple(t.scale(-1) for t in y.terms)))) == 0


# ---- derived equations ------------------------------------------------------------------
PHI, H = fluctuation_potential(), hamiltonian()
T1, T2, R1 = cluster(1), cluster(2), cluster(1, name="r")
DERIVED = {
    "CCD doubles": lambda: projection(2, bch(H, T2, order=2)),
    "<ab|[Phi,T2]>": lambda: projection(2, comm(PHI, T2)),
    "<ab|[[Phi,T1],R1]>": lambda: projection(2, comm(run(comm(PHI, T1)), R1)),
    "<ab|[[Phi,T1],T2]>": lambda: projection(2, comm(run(comm(PHI, T1)), T2)),
    "<a|[Phi,T2]>": lambda: projection(1, comm(PHI, T2)),
    "CCSD energy": lambda: projection(0, bch(H, T1 + T2, order=2)),
}


@pytest.fixture(scope="module")
def derived():
    out = {name: make() for name, make in DERIVED.items()}
    for key, block in spin_integrate(out["<ab|[Phi,T2]>"]).items():
        out[f"<ab|[Phi,T2]> {key}"] = block
    return out


# =========================================================================
# 1. PermutedTerm and expansion
class TestPermutedTerm:
    t = T(scalars=[Tensor("s", (a,), (i,)), Tensor("u", (b,), (j,))])

    def test_no_swaps(self):
        assert PermutedTerm(self.t, ()).swaps == ()

    def test_valid_pairs(self):
        g = PermutedTerm(self.t, ((i, j), (a, b)))
        assert g.swaps == ((i, j), (a, b))

    @pytest.mark.parametrize("swaps", [
        ((j, i),),                       # not sorted by name
        ((a, b), (i, j)),                # virtual pair before occupied
        ((i, a),),                       # different spaces
        ((i, k),),                       # k is not free
        ((i, j), (i, j)),                # repeated
        ((i, j), (a, b), (i, j)),        # three pairs
    ])
    def test_invalid(self, swaps):
        with pytest.raises(ValueError):
            PermutedTerm(self.t, swaps)

    def test_spin_must_match(self):
        I_, j_ = Index("I", O, AL), Index("j", O, BE)
        t = T(scalars=[Tensor("s", (), (I_, j_))])
        with pytest.raises(ValueError):
            PermutedTerm(t, ((I_, j_),))

    def test_type(self):
        with pytest.raises(TypeError):
            PermutedTerm("t", ())

    def test_expand_order_and_signs(self):
        g = PermutedTerm(self.t, ((i, j), (a, b)))
        terms = expand_permutations((g,)).terms
        s, u = (lambda up, lo: Tensor("s", (up,), (lo,))), (lambda up, lo: Tensor("u", (up,), (lo,)))
        assert [t.coeff for t in terms] == [1, -1, -1, 1]
        assert set(terms[1].scalars) == {s(a, j), u(b, i)}  # P_ij
        assert set(terms[2].scalars) == {s(b, i), u(a, j)}  # P_ab
        assert set(terms[3].scalars) == {s(b, j), u(a, i)}  # P_ij P_ab


# =========================================================================
# 2. Grouping
class TestGrouping:
    def test_pair(self):
        x = X(T(scalars=[Tensor("s", (a,), (i,)), Tensor("u", (b,), (j,))]),
              T(-1, [Tensor("s", (a,), (j,)), Tensor("u", (b,), (i,))]))
        (g,) = group_permutations(x)
        assert g.swaps == ((i, j),) and g.term == x.terms[0]

    def test_product_before_pairs(self):
        base = T(scalars=[Tensor("s", (a,), (i,)), Tensor("u", (b,), (j,))])
        (g,) = group_permutations(expand_permutations((PermutedTerm(base, ((i, j), (a, b))),)))
        assert g.swaps == ((i, j), (a, b))

    def test_occupied_pair_tried_before_virtual(self):
        # t, -P_ij t and -P_ab t, but no P_ij P_ab partner: t pairs with -P_ij t
        base = T(scalars=[Tensor("s", (a,), (i,)), Tensor("u", (b,), (j,))])
        swap = lambda m: base._rename(m)
        x = X(base, swap({i: j, j: i}).scale(-1), swap({a: b, b: a}).scale(-1))
        groups = group_permutations(x)
        assert [g.swaps for g in groups] == [((i, j),), ()]
        assert groups[1].term == x.terms[2]

    def test_partner_up_to_dummies_and_symmetry(self):
        # -P_ab of sum_c f^a_c t^{cb}_{ij} is sum_d f^b_d t^{ad}_{ij}... written differently
        x = X(T(scalars=[Tensor("f", (a,), (c,), SYM), Tensor("t", (c, b), (i, j))], summed=[c]),
              T(scalars=[Tensor("f", (d,), (b,), SYM), Tensor("t", (a, d), (i, j))], summed=[d]))
        (g,) = group_permutations(x)
        assert g.swaps == ((a, b),)

    def test_wrong_coefficient_not_grouped(self):
        x = X(T(scalars=[Tensor("s", (a,), (i,)), Tensor("u", (b,), (j,))]),
              T(Fraction(-1, 2), [Tensor("s", (a,), (j,)), Tensor("u", (b,), (i,))]))
        assert [g.swaps for g in group_permutations(x)] == [(), ()]

    def test_same_sign_not_grouped(self):
        x = X(T(scalars=[Tensor("s", (a,), (i,)), Tensor("u", (b,), (j,))]),
              T(scalars=[Tensor("s", (a,), (j,)), Tensor("u", (b,), (i,))]))
        assert [g.swaps for g in group_permutations(x)] == [(), ()]

    def test_antisymmetric_term_stands_alone(self):
        x = X(T(scalars=[v_(a, b, i, j)]))
        (g,) = group_permutations(x)
        assert g.swaps == ()

    def test_duplicate_of_self_symmetric_term_not_paired(self):
        # unsimplified input: t^{ab}_{ij} twice. P_ij maps it to minus itself, so
        # the swap is never used for it (the copy is not a P_ij partner).
        t = T(scalars=[Tensor("t", (a, b), (i, j))])
        assert [g.swaps for g in group_permutations(X(t, t))] == [(), ()]

    def test_duplicate_mixed_spin_not_paired(self):
        I_, j_, A_, b_ = Index("I", O, AL), Index("j", O, BE), Index("A", V, AL), Index("b", V, BE)
        t = T(scalars=[Tensor("t", (A_, b_), (I_, j_))])
        assert [g.swaps for g in group_permutations(X(t, t))] == [(), ()]

    def test_no_pair_for_three_occupied(self):
        x = X(T(scalars=[Tensor("t", (a, b, c), (i, j, k))]))
        assert group_permutations(x)[0].swaps == ()

    def test_mixed_spin_block_has_no_swaps(self, derived):
        assert all(g.swaps == () for g in group_permutations(derived["<ab|[Phi,T2]> OoVv"]))

    def test_same_spin_block_groups(self, derived):
        groups = group_permutations(derived["<ab|[Phi,T2]> OOVV"])
        assert any(len(g.swaps) == 2 for g in groups)
        I_, J_, A_, B_ = (Index(n, s, AL) for n, s in [("I", O), ("J", O), ("A", V), ("B", V)])
        assert {g.swaps for g in groups} <= {(), ((I_, J_),), ((A_, B_),), ((I_, J_), (A_, B_))}

    def test_ccd_grouped(self, derived):
        groups = group_permutations(derived["CCD doubles"])
        assert len(groups) == 10
        assert Counter(len(g.swaps) for g in groups) == Counter({0: 4, 1: 5, 2: 1})

    @pytest.mark.parametrize("name", list(DERIVED) + ["<ab|[Phi,T2]> OOVV", "<ab|[Phi,T2]> OoVv",
                                                      "<ab|[Phi,T2]> oovv"])
    def test_expansion_is_exact(self, derived, name):
        x = derived[name]
        groups = group_permutations(x)
        assert same(expand_permutations(groups), x)
        assert sum(2 ** len(g.swaps) for g in groups) == len(x)

    def test_representative_is_stored_term(self, derived):
        x = derived["CCD doubles"]
        assert all(g.term in x.terms for g in group_permutations(x))

    def test_no_free_indices(self, derived):
        x = derived["CCSD energy"]
        assert [g.swaps for g in group_permutations(x)] == [()] * len(x)


# =========================================================================
# 3. LaTeX
class TestLatex:
    def test_default_unchanged(self, derived):
        x = derived["CCD doubles"]
        assert latex(x) == latex(x, group=False)
        assert "P_" not in latex(x)

    def test_ring_term(self, derived):
        text = latex(derived["CCD doubles"], group=True)
        assert r" - (1 - P_{ij})(1 - P_{ab}) \sum_{ck} t^{cb}_{ik}\, \langle ka \| jc \rangle" in text

    def test_pairs(self, derived):
        text = latex(derived["CCD doubles"], group=True)
        assert r" + (1 - P_{ab}) \sum_{c} f^{a}_{c}\, t^{cb}_{ij}" in text
        assert r" - (1 - P_{ij}) \sum_{k} f^{k}_{j}\, t^{ab}_{ik}" in text

    def test_starts_with_lone_term(self, derived):
        assert latex(derived["CCD doubles"], group=True).startswith(r"\langle ab \| ij \rangle + (1 - P_{ab})")

    def test_permuted_term_alone(self):
        g = PermutedTerm(T(Fraction(-1, 2), [Tensor("s", (a,), (i,)), Tensor("u", (b,), (j,))]), ((i, j),))
        assert latex(g) == r"-\frac{1}{2} (1 - P_{ij}) s^{a}_{i}\, u^{b}_{j}"

    def test_align_counts_groups(self, derived):
        text = latex(derived["CCD doubles"], group=True, lhs="R", terms_per_line=5)
        assert text.count(r" \\") == 1  # 10 groups, 5 per line: two lines

    def test_spin_block(self, derived):
        text = latex(derived["<ab|[Phi,T2]> OOVV"], group=True)
        assert "(1 - P_{IJ})" in text and "(1 - P_{AB})" in text


# =========================================================================
# 4. Code generation
N_OCC, N_VIRT = 3, 4
N = N_OCC + N_VIRT
SO = {O: slice(0, N_OCC), V: slice(N_OCC, N)}


@pytest.fixture(scope="module")
def values():
    return TensorValues(SimpleNamespace(n_modes=N), seed=14)


def so_arrays(values):
    v, f = values.array("v", 2, 2, SYM), values.array("f", 1, 1, SYM)
    m = SimpleNamespace()
    for blk in ["OOOO", "OOOV", "OOVV", "OVOV", "OVVV", "VVVV"]:
        setattr(m, "g_" + blk, v[tuple(SO[O if ch == "O" else V] for ch in blk)].copy())
    for blk in ["OO", "OV", "VV"]:
        setattr(m, "f_" + blk, f[tuple(SO[O if ch == "O" else V] for ch in blk)].copy())
    out = {"moeris": m}
    for name in "tr":
        for n in (1, 2):
            full = values.array(name, n, n, NONE)
            sub = full[tuple([SO[V]] * n + [SO[O]] * n)]
            out[f"{name}{n}"] = sub.transpose(list(range(n, 2 * n)) + list(range(n))).copy()
    return out


def call(src, arrays):
    namespace = {}
    exec(compile(src, "<generated>", "exec"), namespace)
    fn = next(v for key, v in namespace.items() if callable(v) and key not in ("np", "oe"))
    params = fn.__code__.co_varnames[: fn.__code__.co_argcount]
    return np.asarray(fn(**{p: arrays[p] for p in params}))


class TestCodegen:
    def test_pair_text(self):
        x = X(T(scalars=[Tensor("s", (a,), (i,)), Tensor("u", (b,), (j,))]),
              T(-1, [Tensor("s", (a,), (j,)), Tensor("u", (b,), (i,))]))
        lines = einsum_code(x, "x", group=True).splitlines()
        assert lines[3] == "def x(s1, u1):"
        assert lines[5] == "    # (1 - P_ij) s^{a}_{i} u^{b}_{j}"
        assert lines[6] == '    tmp = np.einsum("ia,jb->ijab", s1, u1, optimize=True)'
        assert lines[7] == "    r += tmp - tmp.swapaxes(0, 1)"

    def test_product_text_with_coefficient(self):
        base = T(Fraction(-1, 2), [Tensor("s", (a,), (i,)), Tensor("u", (b,), (j,))])
        x = expand_permutations((PermutedTerm(base, ((i, j), (a, b))),))
        lines = einsum_code(x, "x", group=True).splitlines()
        assert lines[7] == ("    r -= 0.5 * (tmp - tmp.swapaxes(0, 1) - tmp.swapaxes(2, 3)"
                            " + tmp.swapaxes(0, 1).swapaxes(2, 3))")

    def test_virtual_pair_axes(self):
        x = X(T(scalars=[Tensor("s", (a,), (i,)), Tensor("u", (b,), (j,))]),
              T(-1, [Tensor("s", (b,), (i,)), Tensor("u", (a,), (j,))]))
        assert "    r += tmp - tmp.swapaxes(2, 3)" in einsum_code(x, "x", group=True).splitlines()

    def test_default_unchanged(self, derived):
        x = derived["CCD doubles"]
        assert einsum_code(x, "x") == einsum_code(x, "x", group=False)
        assert "tmp" not in einsum_code(x, "x")

    def test_fewer_contractions(self, derived):
        src = einsum_code(derived["CCD doubles"], "x", group=True)
        assert src.count("np.einsum(") == 10

    def test_tmp_reserved_when_grouping(self):
        x = X(T(scalars=[Tensor("s", (a,), (i,))]))
        assert "def x(tmp):" in einsum_code(x, "x", amplitudes="tmp")
        with pytest.raises(ValueError):
            einsum_code(x, "x", amplitudes="tmp", group=True)

    @pytest.mark.parametrize("name", ["CCD doubles", "<ab|[Phi,T2]>", "<ab|[[Phi,T1],R1]>", "<ab|[[Phi,T1],T2]>",
                                      "<a|[Phi,T2]>", "CCSD energy"])
    def test_same_numbers(self, values, derived, name):
        x = derived[name]
        arrays = so_arrays(values)
        grouped = call(einsum_code(x, "f", group=True), arrays)
        plain = call(einsum_code(x, "f"), arrays)
        assert np.allclose(grouped, plain)


# =========================================================================
# 5. Packaging
def test_public_api():
    assert {"PermutedTerm", "group_permutations", "expand_permutations"} <= set(alqemi.__all__)
    assert alqemi.group_permutations is group_permutations
