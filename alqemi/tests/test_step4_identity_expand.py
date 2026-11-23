"""Step 4 tests: the HJO rule library and pipeline stage 2, IdentityExpand.

These tests DEFINE the Step 4 API. Results are checked ONLY against the oracle
(alqemi/tests/evaluate.py), never by expression-tree equality. Structure is
tested where it is the point: which rule fires (the route), what stays
unevaluated, and the invariants of the output.

  alqemi/rules/__init__.py                 (new package, may be empty)
  alqemi/rules/hjo.py
    class UnsupportedBracket(Exception)     .bracket -> the offending Bracket
    class Side(Enum): LEFT, RIGHT           which argument a rule splits
    A STRING BRACKET is a Bracket whose two arguments are each one bare Term
    whose factors are a single OpString or nothing (the identity, length 0).
    def is_elementary(br) -> bool
        string bracket, ACOMM, both strings of length 1: [x, y]_+ with x, y
        single operators. It becomes an ElementaryBracket scalar (decision 10);
        it is never evaluated here.
    def is_rank_reducing(br) -> bool        HJO Sec. 1.8 principle of rank reduction
        string bracket whose kind is ACOMM iff BOTH strings have odd length
    @dataclass(frozen=True) class Rule:
        name: str   "1.8.4", "1.8.5", "1.8.6", "1.8.7", "1.8.8a", "1.8.8b",
                    "1.8.9", each also with suffix "L" (14 rules)
        side: Side
        .matches(br) -> bool
            string bracket of the rule's kind whose split string (left for L,
            right otherwise) has length n meeting the equation's condition.
            The ONLY condition besides the kind is n:
              1.8.4, 1.8.6, 1.8.8a, 1.8.8b : n == 2
              1.8.5                        : n == 0 or n >= 2
              1.8.7                        : n even (0 included)
              1.8.9                        : n odd, n >= 3
        .creates(br) -> tuple[Bracket, ...]  brackets apply() creates, in order
        .apply(br) -> Expr                   ValueError if not matches(br)
      The split string is split into its single operators. RIGHT forms are the
      contract's (1.8.4)-(1.8.9) with A = left string, B_k = right operators.
      (1.8.8) has two forms: a = [A,B1]B2 + B1[A,B2]_+, b = [A,B1]_+B2 - B1[A,B2].
      LEFT forms ("L") split the left string A = A_1..A_n against B. They are
      derived via (1.8.1)/(1.8.2): swap, apply the right form, swap each created
      bracket back. The created brackets read [A_k, B], as in HJO's (1.8.13):
        1.8.4L  [A1,B]A2 + A1[A2,B]
        1.8.5L  sum_k A_<k [A_k,B] A_>k
        1.8.6L  -[A1,B]_+A2 + A1[A2,B]_+
        1.8.7L  sum_k (-1)^k A_<k [A_k,B]_+ A_>k
        1.8.8aL -[A1,B]A2 + A1[A2,B]_+
        1.8.8bL [A1,B]_+A2 + A1[A2,B]
        1.8.9L  sum_k (-1)^(k-1) A_<k [A_k,B]_+ A_>k
      Every rule is a valid identity even when it is not rank-reducing.
    RULES: tuple[Rule, ...]    priority order: 1.8.4, 1.8.6, 1.8.8a, 1.8.8b,
                               1.8.5, 1.8.7, 1.8.9 (binary forms first)
    def select_rule(br) -> Rule
        for a non-elementary string bracket:
          1. if either string is empty: 1.8.5 (right empty) or 1.8.5L (left
             empty), if it matches
          2. else the first RIGHT rule in RULES order that matches AND whose
             created brackets are all rank-reducing (decision 11: right first,
             which reproduces HJO's printed (1.8.14)-(1.8.16) and (1S.2.4))
          3. else the first L rule, same test
          4. else raise UnsupportedBracket (strict, decision 6)
        ValueError for an elementary bracket.

  alqemi/pipeline/identity_expand.py
    @dataclass(frozen=True) class TraceStep: rule: str ; bracket: Bracket
        rule is a Rule.name, or "elementary" for the leaf conversion
    def identity_expand(x, pool, trace: list | None = None) -> Expr
        1. linearize(x, pool)
        2. repeat until no Bracket remains anywhere: take the first bracket of
           the first term that has one, innermost first (a bracket's arguments
           are expanded before the bracket itself). If it is elementary, it
           becomes an ElementaryBracket scalar; otherwise select_rule(br).apply(br)
           replaces it in place (prefix * result * suffix). Re-linearize. Each
           step appends a TraceStep to ``trace`` when one is given.
        Output: no Bracket anywhere; scalars are the input's plus
        ElementaryBrackets; no Delta is created; nothing is merged or
        reordered toward normal order. UnsupportedBracket propagates.
"""
from fractions import Fraction

import pytest

from alqemi.core.expr import (
    Bracket,
    BracketKind,
    Delta,
    ElementaryBracket,
    Expr,
    Term,
    acomm,
    as_expr,
    comm,
)
from alqemi.core.indices import Index, Space
from alqemi.core.operators import Kind, OpString, ann, cre
from alqemi.core.pool import IndexPool
from alqemi.pipeline.identity_expand import TraceStep, identity_expand
from alqemi.pipeline.linearize import is_linear
from alqemi.rules.hjo import (
    RULES,
    Rule,
    Side,
    UnsupportedBracket,
    is_elementary,
    is_rank_reducing,
    select_rule,
)
from alqemi.tests.evaluate import mismatch
from alqemi.tests.oracle import FockSpace, delta

P, Q, R, S, M, N = (Index(x) for x in "PQRSMN")
k = Index("k")
i, j = Index("i", Space.OCC), Index("j", Space.OCC)
a, b = Index("a", Space.VIRT), Index("b", Space.VIRT)

GENERAL = [P, Q, R, S, M, N]


def E(p, q):
    return cre(p) * ann(q)


def T(*factors, coeff=1, scalars=(), summed=()):
    return Term(coeff=coeff, scalars=tuple(scalars), factors=tuple(factors), summed=frozenset(summed))


def string(n, start=0):
    """Alternating a†/a string of length n over distinct general indices."""
    ops = [(cre if t % 2 == 0 else ann)(GENERAL[(start + t) % 6]) for t in range(n)]
    return OpString(tuple(ops))


def B(kind, nl, nr):
    """String bracket of the given kind with strings of length nl and nr."""
    left, right = string(nl), string(nr, start=3)
    return Bracket(kind, left, right) if nl or nr else Bracket(kind, Term(), Term())


def by_name(name):
    (rule,) = [r for r in RULES if r.name == name]
    return rule


def expand(x, trace=None):
    return identity_expand(x, IndexPool(), trace)


def has_bracket(expr):
    return any(isinstance(f, Bracket) for t in expr for f in t.factors)


def ops_of(term):
    """Multiset of operators in a term: in its strings and its elementary brackets."""
    out = []
    for f in term.factors:
        out.extend(f.ops)
    for s in term.scalars:
        if isinstance(s, ElementaryBracket):
            out.extend([s.left, s.right])
    return sorted(out, key=str)


@pytest.fixture(scope="module")
def fock():
    return FockSpace(n_occ=2, n_virt=2)


@pytest.fixture(scope="module")
def small_fock():
    # Rule identities hold for ANY operators; 3 modes keep the sweep fast.
    return FockSpace(n_occ=1, n_virt=2)


# =========================================================================
# The rule library
# =========================================================================
EQUATIONS = ["1.8.4", "1.8.5", "1.8.6", "1.8.7", "1.8.8a", "1.8.8b", "1.8.9"]


class TestLibrary:
    def test_exactly_the_hjo_rules(self):
        expected = set(EQUATIONS) | {e + "L" for e in EQUATIONS}
        assert {r.name for r in RULES} == expected
        assert len(RULES) == 14

    def test_sides(self):
        for r in RULES:
            assert r.side is (Side.LEFT if r.name.endswith("L") else Side.RIGHT)

    def test_priority_order_binary_first(self):
        right = [r.name for r in RULES if r.side is Side.RIGHT]
        assert right == ["1.8.4", "1.8.6", "1.8.8a", "1.8.8b", "1.8.5", "1.8.7", "1.8.9"]

    def test_jacobi_is_not_a_rule(self):
        # (1.8.17) relates expressions of the same rank; it is only a test.
        assert not any("1.8.17" in r.name for r in RULES)

    def test_rules_are_immutable(self):
        assert isinstance(RULES, tuple)
        assert all(isinstance(r, Rule) for r in RULES)


# Which rules match which (kind, len(left), len(right)): only kind and the
# split string's length n matter (CLAUDE.md: parity is the only condition).
MATCHES = {
    "1.8.4": (BracketKind.COMM, lambda n: n == 2),
    "1.8.5": (BracketKind.COMM, lambda n: n == 0 or n >= 2),
    "1.8.6": (BracketKind.COMM, lambda n: n == 2),
    "1.8.7": (BracketKind.COMM, lambda n: n % 2 == 0),
    "1.8.8a": (BracketKind.ACOMM, lambda n: n == 2),
    "1.8.8b": (BracketKind.ACOMM, lambda n: n == 2),
    "1.8.9": (BracketKind.ACOMM, lambda n: n % 2 == 1 and n >= 3),
}
SHAPES = [(kind, nl, nr) for kind in BracketKind for nl in range(0, 5) for nr in range(0, 5)]


class TestMatches:
    @pytest.mark.parametrize("name", EQUATIONS)
    def test_matches_depends_only_on_kind_and_n(self, name):
        kind, cond = MATCHES[name]
        for side_name, n_of in ((name, lambda nl, nr: nr), (name + "L", lambda nl, nr: nl)):
            rule = by_name(side_name)
            for k_, nl, nr in SHAPES:
                expected = k_ is kind and cond(n_of(nl, nr))
                assert rule.matches(B(k_, nl, nr)) is expected, (side_name, k_, nl, nr)

    def test_space_and_operator_kind_do_not_matter(self):
        # [a_i, a†_a a_j] has the same shape as [a†_P, a_Q a†_R]: same rule.
        assert select_rule(comm(ann(i), cre(a) * ann(j))) == select_rule(comm(cre(P), ann(Q) * cre(R)))

    def test_non_string_bracket_never_matches(self):
        nested = comm(cre(P), comm(cre(Q), ann(R)))
        assert not any(r.matches(nested) for r in RULES)

    def test_apply_rejects_non_matching(self):
        with pytest.raises(ValueError):
            by_name("1.8.4").apply(B(BracketKind.COMM, 1, 3))


# Every rule is an identity: apply(br) has the same value as br, rank-reducing or not.
APPLY_CASES = [
    (name + side, kind, nl, nr)
    for name in EQUATIONS
    for side in ("", "L")
    for kind, nl, nr in SHAPES
    if nl <= 3 and nr <= 3 and by_name(name + side).matches(B(kind, nl, nr))
]


class TestApply:
    @pytest.mark.parametrize("name, kind, nl, nr", APPLY_CASES)
    def test_is_an_identity(self, small_fock, name, kind, nl, nr):
        rule = by_name(name)
        br = B(kind, nl, nr)
        assert mismatch(small_fock, rule.apply(br), br) is None

    @pytest.mark.parametrize("name", [e + s for e in EQUATIONS for s in ("", "L")])
    def test_creates_lists_the_brackets_apply_makes(self, name):
        rule = by_name(name)
        shapes = [s for s in SHAPES if rule.matches(B(*s))]
        for s in shapes:
            br = B(*s)
            made = tuple(f for t in rule.apply(br) for f in t.factors if isinstance(f, Bracket))
            assert rule.creates(br) == made

    def test_created_brackets_are_string_brackets_of_single_ops(self):
        br = B(BracketKind.COMM, 2, 4)
        for rule in RULES:
            if rule.matches(br):
                for c in rule.creates(br):
                    sides = [c.left.terms[0].factors, c.right.terms[0].factors]
                    lengths = sorted(len(fs[0]) if fs else 0 for fs in sides)
                    assert lengths[0] == 1

    def test_left_form_keeps_split_piece_on_the_left(self):
        # HJO (1.8.13): [a†_P a_Q, E_RS] = [a†_P, E_RS] a_Q + a†_P [a_Q, E_RS]
        made = by_name("1.8.4L").creates(comm(E(P, Q), E(R, S)))
        assert made == (comm(cre(P), E(R, S)), comm(ann(Q), E(R, S)))

    def test_contract_1_8_8_both_forms(self, fock):
        br = acomm(ann(P), cre(Q) * cre(R))
        assert mismatch(fock, by_name("1.8.8a").apply(br), br) is None
        assert mismatch(fock, by_name("1.8.8b").apply(br), br) is None


# =========================================================================
# Rank reduction and rule selection
# =========================================================================
class TestRankReduction:
    @pytest.mark.parametrize(
        "kind, nl, nr, expected",
        [
            (BracketKind.COMM, 2, 2, True),
            (BracketKind.COMM, 1, 2, True),
            (BracketKind.COMM, 2, 1, True),
            (BracketKind.COMM, 1, 1, False),   # odd, odd commutator
            (BracketKind.COMM, 1, 3, False),
            (BracketKind.ACOMM, 1, 1, True),   # odd, odd anticommutator
            (BracketKind.ACOMM, 1, 3, True),
            (BracketKind.ACOMM, 1, 2, False),  # anticommutator with an even string
            (BracketKind.ACOMM, 2, 2, False),
            (BracketKind.ACOMM, 1, 0, False),  # the identity is even
            (BracketKind.COMM, 1, 0, True),
        ],
    )
    def test_is_rank_reducing(self, kind, nl, nr, expected):
        assert is_rank_reducing(B(kind, nl, nr)) is expected

    def test_is_elementary(self):
        assert is_elementary(acomm(ann(P), cre(Q)))
        assert is_elementary(acomm(cre(P), cre(Q)))
        assert not is_elementary(comm(ann(P), cre(Q)))
        assert not is_elementary(acomm(ann(P), cre(Q) * ann(R)))


# The parity table is NOT hand-coded in select_rule; it follows from
# "first rule whose created brackets are all rank-reducing, right side first".
# L rules appear only where the right string cannot be split (length 1).
SELECTION = [
    (BracketKind.COMM, 2, 2, "1.8.4"),     # HJO (1.8.14)
    (BracketKind.COMM, 1, 2, "1.8.6"),     # HJO (1.8.10)-(1.8.12)
    (BracketKind.COMM, 2, 1, "1.8.6L"),    # [E_PQ, a†_R]: right cannot split
    (BracketKind.COMM, 2, 3, "1.8.5"),
    (BracketKind.COMM, 3, 2, "1.8.6"),
    (BracketKind.COMM, 3, 4, "1.8.7"),
    (BracketKind.COMM, 4, 4, "1.8.5"),     # HJO (1.8.15) shape
    (BracketKind.COMM, 1, 4, "1.8.7"),
    (BracketKind.COMM, 4, 1, "1.8.7L"),
    (BracketKind.COMM, 4, 3, "1.8.5"),
    (BracketKind.ACOMM, 1, 3, "1.8.9"),
    (BracketKind.ACOMM, 3, 1, "1.8.9L"),
    (BracketKind.ACOMM, 3, 3, "1.8.9"),
    (BracketKind.COMM, 2, 0, "1.8.5"),     # [A, 1] = 0: (1.8.5) with n = 0
    (BracketKind.COMM, 0, 2, "1.8.5L"),
    (BracketKind.COMM, 1, 0, "1.8.5"),
    (BracketKind.COMM, 0, 0, "1.8.5"),
]
UNSUPPORTED = [
    (BracketKind.COMM, 1, 1),
    (BracketKind.COMM, 1, 3),
    (BracketKind.COMM, 3, 3),
    (BracketKind.ACOMM, 1, 2),
    (BracketKind.ACOMM, 2, 1),
    (BracketKind.ACOMM, 2, 2),
    (BracketKind.ACOMM, 2, 3),
    (BracketKind.ACOMM, 1, 0),
    (BracketKind.ACOMM, 0, 0),
]


class TestSelection:
    @pytest.mark.parametrize("kind, nl, nr, name", SELECTION)
    def test_selected_rule(self, kind, nl, nr, name):
        assert select_rule(B(kind, nl, nr)).name == name

    @pytest.mark.parametrize("kind, nl, nr", UNSUPPORTED)
    def test_unsupported(self, kind, nl, nr):
        br = B(kind, nl, nr)
        with pytest.raises(UnsupportedBracket) as err:
            select_rule(br)
        assert err.value.bracket == br

    @pytest.mark.parametrize("kind, nl, nr, name", SELECTION)
    def test_selected_rule_creates_only_rank_reducing_brackets(self, kind, nl, nr, name):
        br = B(kind, nl, nr)
        assert all(is_rank_reducing(c) for c in select_rule(br).creates(br))

    def test_right_preferred_when_both_sides_work(self):
        # [E_PQ, E_RS]: 1.8.4L (HJO's (1.8.13) step) would also be rank-reducing,
        # but right-first reproduces the book's printed results (decision 11).
        br = comm(E(P, Q), E(R, S))
        assert by_name("1.8.4L").matches(br)
        assert all(is_rank_reducing(c) for c in by_name("1.8.4L").creates(br))
        assert select_rule(br).name == "1.8.4"

    def test_unsupported_is_never_quit(self):
        # It is an ordinary exception: catchable, the process keeps running.
        assert issubclass(UnsupportedBracket, Exception)
        assert not issubclass(UnsupportedBracket, SystemExit)

    def test_elementary_is_not_selected(self):
        with pytest.raises(ValueError):
            select_rule(acomm(ann(P), cre(Q)))


# =========================================================================
# IdentityExpand: structure of the output
# =========================================================================
class TestExpandStructure:
    def test_no_brackets_remain(self):
        out = expand(comm(E(P, Q), comm(E(R, S), E(M, N))))
        assert not has_bracket(out)
        assert is_linear(out)

    def test_elementary_brackets_not_evaluated(self):
        out = expand(comm(cre(P), E(Q, R)))
        scalars = [s for t in out for s in t.scalars]
        assert scalars and all(isinstance(s, ElementaryBracket) for s in scalars)
        # (1.2.27): [a†_P, a†_Q]_+ = 0 is kept, not dropped
        assert ElementaryBracket(cre(P), cre(Q)) in scalars

    def test_no_delta_created(self):
        x = T(comm(E(P, Q), E(R, S)), scalars=[Delta(M, N)])
        out = expand(x)
        assert all([s for s in t.scalars if isinstance(s, Delta)] == [Delta(M, N)] for t in out)

    @pytest.mark.parametrize("nl, nr", [(2, 2), (2, 4), (4, 2), (4, 4), (1, 2), (2, 1), (1, 4), (3, 2)])
    def test_one_contraction_per_bracket(self, nl, nr):
        # HJO Exercise 1.2: [n-string, m-string] gives strings of n + m - 2 operators;
        # here each term carries exactly one elementary bracket.
        out = expand(B(BracketKind.COMM, nl, nr))
        for t in out:
            assert len(t.scalars) == 1
            assert sum(len(f) for f in t.factors) == nl + nr - 2

    def test_nested_gives_two_contractions(self):
        out = expand(comm(E(P, Q), comm(E(R, S), E(M, N))))
        assert all(len(t.scalars) == 2 for t in out)

    @pytest.mark.parametrize("nl, nr", [(2, 2), (1, 2), (3, 1), (2, 3)])
    def test_operators_are_conserved(self, nl, nr):
        # No operator is created or lost: each one ends up in a string or an
        # elementary bracket. (DeltaEvaluate, stage 3, is what removes them.)
        br = B(BracketKind.ACOMM if nl % 2 and nr % 2 else BracketKind.COMM, nl, nr)
        expected = sorted(list(string(nl)) + list(string(nr, start=3)), key=str)
        for t in expand(br):
            assert ops_of(t) == expected

    def test_identity_argument_vanishes(self):
        assert len(expand(comm(ann(P), Delta(Q, R)))) == 0

    def test_terms_without_brackets_pass_through(self):
        x = Expr((T(E(P, Q), coeff=2, scalars=[Delta(P, R)]), T(E(k, k), summed=[k])))
        assert expand(x) == x

    def test_unsupported_propagates(self):
        with pytest.raises(UnsupportedBracket):
            expand(T(cre(M), comm(ann(P), ann(Q))))

    def test_unsupported_inside_nested(self):
        with pytest.raises(UnsupportedBracket):
            expand(comm(E(P, Q), acomm(E(R, S), E(M, N))))

    def test_not_normal_ordered(self):
        # (1S.2.4): the route produces a_R a†_Q-type strings; nothing reorders them.
        out = expand(comm(cre(P) * cre(Q), ann(R) * ann(S)))
        strings = [f for t in out for f in t.factors]
        assert any(s[0].kind is Kind.ANN and s[1].kind is Kind.CRE for s in strings if len(s) == 2)


# =========================================================================
# IdentityExpand: the route (trace)
# =========================================================================
class TestTrace:
    def test_trace_steps(self):
        trace = []
        expand(comm(cre(P), E(Q, R)), trace)
        assert all(isinstance(s, TraceStep) for s in trace)
        assert trace[0] == TraceStep("1.8.6", comm(cre(P), E(Q, R)))

    def test_route_1_8_11(self):
        trace = []
        expand(comm(cre(P), E(Q, R)), trace)
        assert [s.rule for s in trace] == ["1.8.6", "elementary", "elementary"]

    # Routes (decision 11, right first). These pin which rule fires; they are what
    # makes the final operator order match HJO's printed results.
    def test_route_1_8_14(self):
        trace = []
        expand(comm(E(P, Q), E(R, S)), trace)
        assert [s.rule for s in trace] == [
            "1.8.4",
            "1.8.6L", "elementary", "elementary",
            "1.8.6L", "elementary", "elementary",
        ]
        assert trace[1].bracket == comm(E(P, Q), cre(R))
        assert trace[4].bracket == comm(E(P, Q), ann(S))

    def test_route_1_8_15(self):
        trace = []
        expand(comm(E(P, Q), E(R, S) * E(M, N)), trace)
        assert [s.rule for s in trace] == ["1.8.5"] + ["1.8.6L", "elementary", "elementary"] * 4
        assert [s.bracket for s in trace if s.rule == "1.8.6L"] == [
            comm(E(P, Q), cre(R)), comm(E(P, Q), ann(S)), comm(E(P, Q), cre(M)), comm(E(P, Q), ann(N))
        ]

    def test_route_1S_2_4(self):
        trace = []
        expand(comm(cre(P) * cre(Q), ann(R) * ann(S)), trace)
        assert [s.rule for s in trace] == [
            "1.8.4",
            "1.8.6L", "elementary", "elementary",
            "1.8.6L", "elementary", "elementary",
        ]

    def test_innermost_first(self):
        trace = []
        inner = comm(E(R, S), E(M, N))
        expand(comm(E(P, Q), inner), trace)
        assert trace[0].bracket == inner

    def test_trace_optional(self):
        assert len(expand(comm(cre(P), E(Q, R)))) == 2


# =========================================================================
# IdentityExpand: values (oracle only)
# =========================================================================
class TestGoldenAgainstOracle:
    def test_1_8_11(self, fock):
        lhs = comm(cre(P), E(Q, R))
        golden = lambda g: -delta(P, R, g) * fock.op(cre(Q), g)
        assert mismatch(fock, expand(lhs), golden, extra=[P, Q, R]) is None

    def test_1_8_12(self, fock):
        lhs = comm(ann(P), E(Q, R))
        golden = lambda g: delta(P, Q, g) * fock.op(ann(R), g)
        assert mismatch(fock, expand(lhs), golden, extra=[P, Q, R]) is None

    def test_1_8_14(self, fock):
        st = fock.string
        golden = lambda g: delta(Q, R, g) * st(E(P, S), g) - delta(P, S, g) * st(E(R, Q), g)
        assert mismatch(fock, expand(comm(E(P, Q), E(R, S))), golden, extra=[P, Q, R, S]) is None

    def test_1_8_15(self, fock):
        st = fock.string
        golden = lambda g: (
            delta(Q, R, g) * st(E(P, S) * E(M, N), g)
            - delta(P, S, g) * st(E(R, Q) * E(M, N), g)
            + delta(Q, M, g) * st(E(R, S) * E(P, N), g)
            - delta(P, N, g) * st(E(R, S) * E(M, Q), g)
        )
        out = expand(comm(E(P, Q), E(R, S) * E(M, N)))
        assert mismatch(fock, out, golden, extra=GENERAL) is None

    def test_1_8_16(self, fock):
        st = fock.string
        golden = lambda g: (
            delta(S, M, g) * delta(Q, R, g) * st(E(P, N), g)
            - delta(S, M, g) * delta(P, N, g) * st(E(R, Q), g)
            - delta(R, N, g) * delta(Q, M, g) * st(E(P, S), g)
            + delta(R, N, g) * delta(P, S, g) * st(E(M, Q), g)
        )
        out = expand(comm(E(P, Q), comm(E(R, S), E(M, N))))
        assert mismatch(fock, out, golden, extra=GENERAL) is None

    def test_1S_2_4(self, fock):
        st = fock.string
        golden = lambda g: (
            delta(Q, R, g) * st(cre(P) * ann(S), g)
            - delta(P, R, g) * st(cre(Q) * ann(S), g)
            + delta(Q, S, g) * st(ann(R) * cre(P), g)
            - delta(P, S, g) * st(ann(R) * cre(Q), g)
        )
        out = expand(comm(cre(P) * cre(Q), ann(R) * ann(S)))
        assert mismatch(fock, out, golden, extra=[P, Q, R, S]) is None

    def test_jacobi_1_8_17(self, small_fock):
        # Consistency check through the whole stage, never a rule.
        fock = small_fock
        A, Bs, C = E(P, Q), E(R, S), E(M, N)
        total = as_expr(comm(A, comm(Bs, C))) + as_expr(comm(C, comm(A, Bs))) + as_expr(comm(Bs, comm(C, A)))
        assert mismatch(fock, expand(total), Expr()) is None


ORACLE_CASES = {
    "anticommutator 1,3": acomm(ann(P), cre(Q) * ann(R) * cre(S)),
    "anticommutator 3,1": acomm(cre(P) * ann(Q) * cre(R), ann(S)),
    "anticommutator 3,3": acomm(cre(P) * ann(Q) * cre(R), ann(S) * cre(M) * ann(N)),
    "commutator 2,1": comm(E(P, Q), cre(R)),
    "commutator 2,3": comm(E(P, Q), cre(R) * ann(S) * cre(M)),
    "commutator 3,2": comm(cre(P) * ann(Q) * cre(R), E(S, M)),
    "commutator 1,4": comm(ann(P), E(Q, R) * E(S, M)),
    "commutator 4,4": comm(E(P, Q) * E(R, S), E(M, N) * E(P, Q)),
    "repeated index": comm(E(P, P), E(P, Q)),
    "occ/virt": comm(E(a, i), E(j, b)),
    "products around": T(cre(M), comm(E(P, Q), E(R, S)), ann(N), coeff=Fraction(-3, 2)),
    "two brackets": T(comm(cre(P), E(Q, R)), comm(E(S, M), ann(N))),
    "with scalars": T(comm(E(P, Q), E(R, S)), scalars=[Delta(M, N), ElementaryBracket(ann(M), cre(N))]),
    "summed": T(comm(E(k, P), E(Q, k)), summed=[k]),
    "sum inside argument": comm(E(P, Q), T(E(R, k), summed=[k], coeff=2)),
    "nested anticommutator": comm(E(P, Q), acomm(ann(R), cre(S) * ann(M) * cre(N))),
    "elementary inside": comm(ann(P), acomm(ann(Q), cre(R))),
    "triple nested": comm(E(P, Q), comm(E(R, S), comm(E(M, N), E(P, R)))),
}


SLOW_CASES = {"triple nested"}  # 6 free indices: checked on the 3-mode space


@pytest.mark.parametrize("case", list(ORACLE_CASES))
def test_value_unchanged(fock, small_fock, case):
    x = ORACLE_CASES[case]
    out = expand(x)
    assert not has_bracket(out)
    assert mismatch(small_fock if case in SLOW_CASES else fock, out, x) is None
