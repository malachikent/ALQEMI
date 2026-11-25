"""Step 13 tests: direct LaTeX printing (decision 13: no SymPy).

These tests DEFINE Step 13. The output format is the point, so most checks compare
exact strings. Settled with the user (2026-11-24): <pq||rs> double-bar integrals,
explicit sums, alpha/beta as upper/lower case (the names ALQEMI already gives),
term by term (P(ij)/P(ab) grouping is a later step).

  alqemi/printing/latex.py
    latex(x, *, double_bar=("v",), lhs=None, terms_per_line=None) -> str
      x: Index, Op, OpString, Delta, ElementaryBracket, Tensor, Bracket, Term or
      Expr; TypeError otherwise. Output is ASCII only (no Σ, †, δ characters).

      Index        its name; a name of letters then digits prints with a subscript:
                   a12 -> a_{12} (IndexPool names). Spin is not printed: alpha and
                   beta indices already differ by case (Step 12).
      index list   the indices' LaTeX run together (ij, Ab) if every one is a single
                   character, otherwise separated by commas (a_{1},i).
      Op           a^{\\dagger}_{p} (creator), a_{p} (annihilator).
      OpString     its operators separated by one space.
      Delta        \\delta_{pq} (index list of p, q).
      ElementaryBracket  [a_{p}, a^{\\dagger}_{q}]_{+}.
      Tensor       a name in ``double_bar`` with 2 upper and 2 lower indices prints
                   as the double-bar integral \\langle pq \\| rs \\rangle (upper
                   indices left, lower right; \\| is the double bar ‖). Any other
                   tensor: name, then ^{upper} if it has upper indices, then
                   _{lower} if it has lower ones: t^{ab}_{ij}, f^{a}_{c}, s^{a}, s_{i}.
      Bracket      [L, R] (commutator) or [L, R]_{+} (anticommutator), L and R
                   printed as expressions.
      Term         [coefficient] [\\sum_{dummies} ] factors. Factors are the
                   scalars in stored order, then the operator factors, separated by
                   "\\, ". Dummies sorted by name, as an index list. Coefficient: 1
                   nothing, -1 a leading "-", an integer n "n ", a fraction p/q
                   "\\frac{p}{q} " (negative: "-\\frac{p}{q} "). A term without
                   factors prints its number alone: 1, -2, \\frac{1}{2}.
      Expr         terms joined by " + ", or " - " with the term's own minus sign
                   removed; the empty sum prints 0.

      lhs: "lhs = rhs". With terms_per_line=n (n >= 1) the result is an align*
      environment, n terms per line:
          \\begin{align*}
          lhs &= t1 + t2 \\\\
          &\\quad + t3 - t4 \\\\
          &\\quad + t5
          \\end{align*}
      (lines joined by " \\\\" and a newline; a line that starts with a negative
      term starts "&\\quad - "). terms_per_line needs lhs and n >= 1 (ValueError).

    Expr._repr_latex_() returns "$" + latex(expr) + "$" (Jupyter display).

  The BCH factors are shown whenever an expression holds them: they are its
  coefficients (bch includes 1/k!; a bare commutator has none).
  The module imports no numpy. Public API: latex joins ``alqemi.__all__``.
"""
import re
import subprocess
import sys
from fractions import Fraction

import pytest

import alqemi
from alqemi.builders import cluster, fluctuation_potential, hamiltonian
from alqemi.core.expr import Delta, ElementaryBracket, Expr, Term, acomm, comm
from alqemi.core.indices import Index, Space, Spin
from alqemi.core.operators import OpString, ann, cre
from alqemi.core.tensor import Swap, Tensor
from alqemi.pipeline.bch import bch
from alqemi.pipeline.project import projection
from alqemi.pipeline.spin import spin_integrate
from alqemi.printing.latex import latex

O, V, G = Space.OCC, Space.VIRT, Space.GENERAL
SYM = Swap.SYMMETRIC
a, b, c, d = (Index(x, V) for x in "abcd")
i, j, k, l = (Index(x, O) for x in "ijkl")
p, q = Index("p"), Index("q")


def T(coeff=1, scalars=(), summed=(), factors=()):
    return Term(coeff=coeff, scalars=tuple(scalars), factors=tuple(factors), summed=frozenset(summed))


def X(*terms):
    return Expr(tuple(terms))


def v_(u1, u2, w1, w2):
    return Tensor("v", (u1, u2), (w1, w2), SYM)


# =========================================================================
# 1. Pieces
class TestPieces:
    @pytest.mark.parametrize("ix, text", [(a, "a"), (Index("a12", V), "a_{12}"), (Index("P"), "P"),
                                          (Index("I", O, Spin.ALPHA), "I"), (Index("ab"), "ab")])
    def test_index(self, ix, text):
        assert latex(ix) == text

    def test_ops(self):
        assert latex(cre(p)) == r"a^{\dagger}_{p}"
        assert latex(ann(q)) == "a_{q}"

    def test_opstring(self):
        assert latex(cre(p) * ann(q)) == r"a^{\dagger}_{p} a_{q}"

    def test_delta(self):
        assert latex(Delta(p, q)) == r"\delta_{pq}"
        assert latex(Delta(Index("a1", V), b)) == r"\delta_{a_{1},b}"

    def test_elementary_bracket(self):
        assert latex(ElementaryBracket(ann(p), cre(q))) == r"[a_{p}, a^{\dagger}_{q}]_{+}"

    def test_tensors(self):
        assert latex(Tensor("t", (a, b), (i, j))) == "t^{ab}_{ij}"
        assert latex(Tensor("f", (a,), (c,), SYM)) == "f^{a}_{c}"
        assert latex(Tensor("s", (a,))) == "s^{a}"
        assert latex(Tensor("s", (), (i,))) == "s_{i}"
        assert latex(Tensor("w")) == "w"

    def test_double_bar(self):
        assert latex(v_(k, a, j, c)) == r"\langle ka \| jc \rangle"

    def test_double_bar_only_for_two_by_two(self):
        assert latex(Tensor("v", (a,), (i,))) == "v^{a}_{i}"

    def test_double_bar_option(self):
        assert latex(v_(k, a, j, c), double_bar=()) == "v^{ka}_{jc}"
        assert latex(Tensor("g", (k, a), (j, c)), double_bar=("g",)) == r"\langle ka \| jc \rangle"

    def test_tensor_index_list_with_long_names(self):
        assert latex(Tensor("t", (Index("a1", V),), (i,))) == "t^{a_{1}}_{i}"
        assert latex(Tensor("t", (Index("a1", V), b), (i, j))) == "t^{a_{1},b}_{ij}"

    def test_commutators(self):
        assert latex(comm(cre(p), ann(q))) == r"[a^{\dagger}_{p}, a_{q}]"
        assert latex(acomm(ann(p), cre(q))) == r"[a_{p}, a^{\dagger}_{q}]_{+}"

    def test_nested_bracket(self):
        inner = comm(cre(p), ann(q))
        assert latex(comm(cre(Index("r")), T(factors=[inner]))) == r"[a^{\dagger}_{r}, [a^{\dagger}_{p}, a_{q}]]"

    def test_type_error(self):
        with pytest.raises(TypeError):
            latex(3)


# =========================================================================
# 2. Terms and sums
class TestTerms:
    def test_sum_and_factors(self):
        term = T(-1, [Tensor("t", (c, b), (i, k)), v_(k, a, j, c)], [c, k])
        assert latex(term) == r"-\sum_{ck} t^{cb}_{ik}\, \langle ka \| jc \rangle"

    @pytest.mark.parametrize("coeff, prefix", [
        (1, ""), (-1, "-"), (2, "2 "), (-3, "-3 "),
        (Fraction(1, 2), r"\frac{1}{2} "), (Fraction(-1, 4), r"-\frac{1}{4} "), (Fraction(2, 3), r"\frac{2}{3} "),
    ])
    def test_coefficients(self, coeff, prefix):
        assert latex(T(coeff, [Tensor("t", (a,), (i,))])) == prefix + "t^{a}_{i}"

    @pytest.mark.parametrize("coeff, text", [(1, "1"), (-2, "-2"), (Fraction(1, 2), r"\frac{1}{2}"),
                                             (Fraction(-1, 6), r"-\frac{1}{6}")])
    def test_pure_numbers(self, coeff, text):
        assert latex(T(coeff)) == text

    def test_operators_after_scalars(self):
        term = T(scalars=[Delta(p, q), Tensor("h", (p,), (q,), SYM)], factors=[cre(p) * ann(q)], summed=[p, q])
        assert latex(term) == r"\sum_{pq} \delta_{pq}\, h^{p}_{q}\, a^{\dagger}_{p} a_{q}"

    def test_dummies_sorted_by_name(self):
        term = T(scalars=[Tensor("x", (k, c))], summed=[k, c])
        assert latex(term).startswith(r"\sum_{ck} ")

    def test_expression_signs(self):
        x = X(T(scalars=[Tensor("x", (a,))]), T(-1, [Tensor("y", (a,))]), T(Fraction(-1, 2), [Tensor("z", (a,))]),
              T(2, [Tensor("w", (a,))]))
        assert latex(x) == r"x^{a} - y^{a} - \frac{1}{2} z^{a} + 2 w^{a}"

    def test_leading_negative_term(self):
        assert latex(X(T(-1, [Tensor("y", (a,))]))) == "-y^{a}"

    def test_empty_is_zero(self):
        assert latex(Expr(())) == "0"

    def test_ascii_only(self, ccd):
        text = latex(ccd)
        assert text.isascii()

    def test_every_bar_is_a_double_bar(self, ccd):
        text = latex(ccd)
        assert "|" in text
        assert all(text[n - 1] == "\\" for n, ch in enumerate(text) if ch == "|")
        assert text.count(r"\|") == text.count(r"\langle") == text.count(r"\rangle")

    def test_braces_balance(self, ccd):
        depth = 0
        for ch in latex(ccd, lhs="R", terms_per_line=3):
            depth += {"{": 1, "}": -1}.get(ch, 0)
            assert depth >= 0
        assert depth == 0


# =========================================================================
# 3. Derived equations (golden)
@pytest.fixture(scope="module")
def ccsd_energy():
    return projection(0, bch(hamiltonian(), cluster(1) + cluster(2), order=2))


@pytest.fixture(scope="module")
def ccd():
    return projection(2, bch(hamiltonian(), cluster(2), order=2))


def test_ccsd_energy(ccsd_energy):
    assert latex(ccsd_energy) == (
        r"\sum_{i} f^{i}_{i}"
        r" - \frac{1}{2} \sum_{ij} \langle ij \| ij \rangle"
        r" + \sum_{ai} f^{i}_{a}\, t^{a}_{i}"
        r" + \frac{1}{2} \sum_{abij} t^{a}_{i}\, t^{b}_{j}\, \langle ji \| ba \rangle"
        r" - \frac{1}{4} \sum_{abij} t^{ab}_{ij}\, \langle ij \| ba \rangle"
    )


def test_ccd_term_by_term(ccd):
    text = latex(ccd)
    assert text.startswith(r"\langle ab \| ij \rangle + \sum_{c} f^{a}_{c}\, t^{cb}_{ij} - \sum_{c} f^{b}_{c}\, t^{ca}_{ij}")
    assert text.endswith(r" + \frac{1}{4} \sum_{cdkl} t^{ab}_{kl}\, t^{cd}_{ij}\, \langle lk \| dc \rangle")
    assert text.count(r"\sum") == len(ccd) - 1  # every term but <ab||ij> is summed


def test_bch_factors_are_shown():
    # bch carries 1/2 on its double commutator; projection keeps it in the coefficients
    x = projection(0, bch(fluctuation_potential(), cluster(1), order=2))
    assert r"\frac{1}{2}" in latex(x)


def test_spin_integrated_block():
    blocks = spin_integrate(X(T(scalars=[Tensor("f", (a,), (c,), SYM), Tensor("t", (c, b), (i, j))], summed=[c])))
    # the alpha dummy gets the first free upper-case virtual letter (A is taken)
    assert latex(blocks["OoVv"]) == r"\sum_{B} f^{A}_{B}\, t^{Bb}_{Ij}"


# =========================================================================
# 4. Equations over several lines
class TestEquation:
    x = X(*(T(sign, [Tensor(n, (a,))]) for sign, n in [(1, "p"), (1, "q"), (-1, "r"), (1, "s"), (-1, "u")]))

    def test_lhs(self):
        assert latex(X(T(scalars=[Tensor("x", (a,))])), lhs="R^{a}") == "R^{a} = x^{a}"

    def test_align(self):
        assert latex(self.x, lhs="R", terms_per_line=2) == (
            "\\begin{align*}\n"
            "R &= p^{a} + q^{a} \\\\\n"
            "&\\quad - r^{a} + s^{a} \\\\\n"
            "&\\quad - u^{a}\n"
            "\\end{align*}"
        )

    def test_align_one_line(self):
        assert latex(self.x, lhs="R", terms_per_line=10) == (
            "\\begin{align*}\nR &= p^{a} + q^{a} - r^{a} + s^{a} - u^{a}\n\\end{align*}"
        )

    def test_align_empty(self):
        assert latex(Expr(()), lhs="R", terms_per_line=2) == "\\begin{align*}\nR &= 0\n\\end{align*}"

    @pytest.mark.parametrize("kw", [dict(terms_per_line=2), dict(lhs="R", terms_per_line=0)])
    def test_align_errors(self, kw):
        with pytest.raises(ValueError):
            latex(self.x, **kw)


# =========================================================================
# 5. Jupyter, packaging
def test_repr_latex():
    x = X(T(scalars=[Tensor("t", (a,), (i,))]))
    assert x._repr_latex_() == "$t^{a}_{i}$"


def test_public_api():
    assert "latex" in alqemi.__all__
    assert alqemi.latex is latex


def test_no_numpy_import():
    code = "import sys, alqemi.printing.latex; print('numpy' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_no_unicode_in_index_lists():
    assert re.fullmatch(r"[ -~]+", latex(Tensor("t", (a, b), (i, j))))
