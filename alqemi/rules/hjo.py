"""The HJO rule library: expansions (1.8.4)-(1.8.9) and rule selection by rank reduction.

A *string bracket* is a Bracket whose two arguments are each one bare Term holding
a single OpString or nothing (the identity). Rules apply only to string brackets;
IdentityExpand makes sure every bracket is one before a rule is chosen.

Right forms are the contract's equations with A = left string and the right string
split into its single operators B_1..B_n. Left forms ("L") split the left string
instead; they follow from the right forms via (1.8.1)/(1.8.2) (swap the bracket,
expand, swap every created bracket back), so created brackets read [A_k, B].

Rule selection follows HJO's principle of rank reduction (Sec. 1.8): rank
reduction follows upon anticommutation of two strings of half-integral rank (odd
length) and upon commutation of all other strings. A rule fires only if every
bracket it creates is rank-reducing; otherwise the bracket is unsupported
(DESIGN_DECISIONS 6). Right forms are tried first (decision 11).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

from alqemi.core.expr import Bracket, BracketKind, Expr, Term
from alqemi.core.operators import OpString

COMM, ACOMM = BracketKind.COMM, BracketKind.ACOMM


class UnsupportedBracket(Exception):
    """A bracket with no rank-reducing expansion in the rule library."""

    def __init__(self, bracket: Bracket, reason: str = "no rank-reducing expansion") -> None:
        super().__init__(f"{reason}: {bracket}")
        self.bracket = bracket


class Side(Enum):
    LEFT = "left"
    RIGHT = "right"


# ---- string brackets ------------------------------------------------------------
def _as_string(arg: Expr) -> Optional[OpString]:
    if len(arg) != 1:
        return None
    (t,) = arg.terms
    if t.coeff != 1 or t.scalars or t.summed:
        return None
    if not t.factors:
        return OpString()
    if len(t.factors) == 1 and isinstance(t.factors[0], OpString):
        return t.factors[0]
    return None


def string_args(br) -> Optional[Tuple[OpString, OpString]]:
    """(left, right) strings of a string bracket, or None for any other object."""
    if not isinstance(br, Bracket):
        return None
    left, right = _as_string(br.left), _as_string(br.right)
    if left is None or right is None:
        return None
    return left, right


def is_elementary(br) -> bool:
    """[x, y]_+ of two single operators: the leaf that becomes an ElementaryBracket."""
    strings = string_args(br)
    return (
        strings is not None
        and br.kind is ACOMM
        and len(strings[0]) == 1
        and len(strings[1]) == 1
    )


def is_rank_reducing(br) -> bool:
    """A string bracket whose kind is ACOMM exactly when both strings have odd length."""
    strings = string_args(br)
    if strings is None:
        return False
    both_odd = len(strings[0]) % 2 == 1 and len(strings[1]) % 2 == 1
    return (br.kind is ACOMM) == both_odd


# ---- equations ------------------------------------------------------------------
# For each equation: the bracket kind it rewrites, the condition on n (length of
# the split string), and the right-form expansion as (sign, k, created kind) for
# B_1..B_{k-1} [A, B_k](+) B_{k+1}..B_n, with k counted from 0.
Piece = Tuple[int, int, BracketKind]


@dataclass(frozen=True)
class _Equation:
    kind: BracketKind
    condition: Callable[[int], bool]
    pieces: Callable[[int], List[Piece]]


_EQUATIONS: Dict[str, _Equation] = {
    "1.8.4": _Equation(COMM, lambda n: n == 2, lambda n: [(1, 0, COMM), (1, 1, COMM)]),
    "1.8.5": _Equation(
        COMM, lambda n: n == 0 or n >= 2, lambda n: [(1, k, COMM) for k in range(n)]
    ),
    "1.8.6": _Equation(COMM, lambda n: n == 2, lambda n: [(1, 0, ACOMM), (-1, 1, ACOMM)]),
    "1.8.7": _Equation(
        COMM, lambda n: n % 2 == 0, lambda n: [((-1) ** k, k, ACOMM) for k in range(n)]
    ),
    "1.8.8a": _Equation(ACOMM, lambda n: n == 2, lambda n: [(1, 0, COMM), (1, 1, ACOMM)]),
    "1.8.8b": _Equation(ACOMM, lambda n: n == 2, lambda n: [(1, 0, ACOMM), (-1, 1, COMM)]),
    "1.8.9": _Equation(
        ACOMM,
        lambda n: n % 2 == 1 and n >= 3,
        lambda n: [((-1) ** k, k, ACOMM) for k in range(n)],
    ),
}

# Binary forms first so HJO's equation names appear for n = 2.
_PRIORITY = ("1.8.4", "1.8.6", "1.8.8a", "1.8.8b", "1.8.5", "1.8.7", "1.8.9")


@dataclass(frozen=True)
class Rule:
    """One HJO expansion, applied on one side of a string bracket."""

    name: str
    side: Side

    @property
    def _equation(self) -> _Equation:
        return _EQUATIONS[self.name.rstrip("L")]

    def matches(self, br) -> bool:
        strings = string_args(br)
        if strings is None:
            return False
        eq = self._equation
        n = len(strings[0] if self.side is Side.LEFT else strings[1])
        return br.kind is eq.kind and eq.condition(n)

    def _expansion(self, br: Bracket) -> List[Tuple[int, OpString, Bracket, OpString]]:
        if not self.matches(br):
            raise ValueError(f"rule {self.name} does not apply to {br}")
        left, right = string_args(br)
        eq = self._equation
        split = left if self.side is Side.LEFT else right
        out = []
        for sign, k, created_kind in eq.pieces(len(split)):
            single = OpString((split[k],))
            if self.side is Side.RIGHT:
                created = Bracket(created_kind, left, single)
            else:
                # (1.8.1)/(1.8.2): swapping the original bracket gives -1 for a
                # commutator; swapping each created bracket back gives -1 again
                # for a commutator, +1 for an anticommutator.
                created = Bracket(created_kind, single, right)
                sign *= (-1 if eq.kind is COMM else 1) * (-1 if created_kind is COMM else 1)
            out.append((sign, split[:k], created, split[k + 1 :]))
        return out

    def creates(self, br) -> Tuple[Bracket, ...]:
        """The brackets apply() would create, in order."""
        return tuple(created for _, _, created, _ in self._expansion(br))

    def apply(self, br) -> Expr:
        return Expr(
            tuple(
                Term(coeff=sign, factors=(before, created, after))
                for sign, before, created, after in self._expansion(br)
            )
        )


RULES: Tuple[Rule, ...] = tuple(Rule(name, Side.RIGHT) for name in _PRIORITY) + tuple(
    Rule(name + "L", Side.LEFT) for name in _PRIORITY
)
_BY_NAME = {rule.name: rule for rule in RULES}


def select_rule(br) -> Rule:
    """The rule IdentityExpand applies to a non-elementary string bracket."""
    strings = string_args(br)
    if strings is None:
        raise ValueError(f"not a string bracket: {br}")
    if is_elementary(br):
        raise ValueError(f"elementary bracket needs no rule: {br}")
    left, right = strings
    if len(left) == 0 or len(right) == 0:
        # [A, 1] = 0 by (1.8.5) with n = 0; an anticommutator with 1 is unsupported.
        rule = _BY_NAME["1.8.5" if len(right) == 0 else "1.8.5L"]
        if rule.matches(br):
            return rule
        raise UnsupportedBracket(br, "anticommutator with the identity has no rank-reducing expansion")
    for side in (Side.RIGHT, Side.LEFT):
        for rule in RULES:
            if rule.side is side and rule.matches(br):
                if all(is_rank_reducing(c) for c in rule.creates(br)):
                    return rule
    raise UnsupportedBracket(br)
