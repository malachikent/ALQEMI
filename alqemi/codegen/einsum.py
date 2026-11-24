"""einsum code generation: derived expressions as Python functions for prototype codes.

``einsum_code`` writes one function per expression, one ``np.einsum`` (or
``opt_einsum.contract``) line per term. Conventions (DESIGN_DECISIONS, Step 11):

- Occupied first everywhere: output r[i,j,a,b], amplitudes t2[i,j,a,b], t1[i,a].
- Integrals are blocks of one integrals object (moeris.g_OOVV, moeris.f_OV, ...),
  each rewritten exactly to an occupied-first block with the sign this costs:
  OCC before VIRT within the upper and within the lower group (antisymmetry),
  then the groups exchanged if the lower block sorts first (swap symmetry). So
  <pq||rs> only uses OOOO, OOOV, OOVV, OVOV, OVVV, VVVV.
- Amplitudes are plain arguments (t1, t2, l2), already in their block.

Spin-orbital code only; the module writes text and imports no numpy.
"""
from __future__ import annotations

import re
import string
from fractions import Fraction
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

from alqemi.core.expr import Delta, ElementaryBracket, as_expr
from alqemi.core.indices import Index, Space
from alqemi.core.tensor import Swap, Tensor

DEFAULT_INTEGRALS: Mapping[str, str] = {
    "f": "moeris.f_{block}",
    "h": "moeris.h_{block}",
    "v": "moeris.g_{block}",  # <pq||rs>, physicist's order
}

_BACKENDS = {
    "numpy": ("import numpy as np", "np.einsum", ", optimize=True"),
    "opt_einsum": ("import opt_einsum as oe", "oe.contract", ""),
}
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_LEADING_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_RESERVED = {"r", "np", "oe"}
_BLOCK_LETTER = {Space.OCC: "O", Space.VIRT: "V"}
_LETTERS = string.ascii_letters


def einsum_code(
    expr,
    name: str,
    *,
    integrals: Optional[Mapping[str, str]] = None,
    amplitudes: str = "{name}{rank}",
    backend: str = "numpy",
) -> str:
    """Python source of a function ``name`` that evaluates the operator-free ``expr``.

    ``integrals`` maps tensor names to access templates with a ``{block}`` field
    (default DEFAULT_INTEGRALS); every other tensor is an amplitude argument named
    ``amplitudes.format(name=..., rank=...)``. ``backend`` is "numpy" or "opt_einsum".
    Raises ValueError for input it cannot write (see the Step 11 spec).
    """
    if not isinstance(name, str) or not _IDENTIFIER.match(name):
        raise ValueError(f"function name {name!r} is not an identifier")
    if backend not in _BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; use 'numpy' or 'opt_einsum'")
    head, call, tail = _BACKENDS[backend]
    integrals = dict(DEFAULT_INTEGRALS if integrals is None else integrals)

    expr = as_expr(expr)
    free = expr.free_indices
    _check_spaces(free)
    output = sorted(free, key=lambda ix: (ix.space is not Space.OCC, ix.name))

    objects: Set[str] = set()
    arguments: Set[str] = set()
    patterns: Dict[str, Tuple[str, str]] = {}
    body: List[str] = []
    for term in expr:
        if term.factors:
            raise ValueError(f"term {term} has operators: project it first")
        if term.free_indices != free:
            raise ValueError(f"term {term} does not have the expression's free indices")
        sign = 1
        operands: List[Tuple[Tuple[Index, ...], str]] = []
        for s in term.scalars:
            if isinstance(s, (Delta, ElementaryBracket)):
                raise ValueError(f"{type(s).__name__} {s} is not supported by code generation yet")
            _check_spaces(s.indices)
            upper, s_up = _occupied_first(s.upper)
            lower, s_lo = _occupied_first(s.lower)
            sign *= s_up * s_lo
            if s.name in integrals:
                template = integrals[s.name]
                root = _LEADING_NAME.match(template)
                if root is None:
                    raise ValueError(f"integral template {template!r} must start with a name")
                objects.add(root.group(0))
                if s.swap is not Swap.NONE and _block(lower) < _block(upper):
                    upper, lower = lower, upper
                    if s.swap is Swap.ANTISYMMETRIC:
                        sign = -sign
                operands.append((upper + lower, template.format(block=_block(upper) + _block(lower))))
            else:
                arg = _amplitude_name(amplitudes, s)
                pattern = (_block(upper), _block(lower))
                if patterns.setdefault(arg, pattern) != pattern:
                    raise ValueError(f"amplitude {arg} occurs as blocks {patterns[arg]} and {pattern}")
                arguments.add(arg)
                axes, _ = _occupied_first(upper + lower)  # storage order: no sign
                operands.append((axes, arg))
        body.append(f"    # {term}")
        body.append(_line(term.coeff * sign, operands, output, call, tail))

    params = sorted(objects) + sorted(arguments)
    clash = (objects & arguments) | (set(params) & (_RESERVED | {name}))
    if clash:
        raise ValueError(f"argument name(s) {sorted(clash)} clash with each other or with r, np, oe, {name}")
    lines = [f"def {name}({', '.join(params)}):", "    r = 0.0", *body, "    return r"]
    return head + "\n\n\n" + "\n".join(lines) + "\n"


# ---- internals --------------------------------------------------------------------
def _check_spaces(indices) -> None:
    for ix in indices:
        if ix.space is Space.GENERAL:
            raise ValueError(f"general index {ix.name}: code generation needs OCC or VIRT indices")


def _occupied_first(indices: Sequence[Index]) -> Tuple[Tuple[Index, ...], int]:
    """Stable sort OCC before VIRT, with the parity of the permutation."""
    order = sorted(range(len(indices)), key=lambda n: indices[n].space is not Space.OCC)
    inversions = sum(1 for x in range(len(order)) for y in range(x) if order[y] > order[x])
    return tuple(indices[n] for n in order), (-1) ** inversions


def _block(indices: Sequence[Index]) -> str:
    return "".join(_BLOCK_LETTER[ix.space] for ix in indices)


def _amplitude_name(template: str, t: Tensor) -> str:
    n_up, n_lo = len(t.upper), len(t.lower)
    rank = str(n_up) if n_up == n_lo else f"{n_up}_{n_lo}"
    arg = template.format(name=t.name, rank=rank)
    if not _IDENTIFIER.match(arg):
        raise ValueError(f"amplitude argument {arg!r} is not an identifier")
    return arg


def _letters(output: Sequence[Index], operands) -> Dict[Index, str]:
    """One einsum letter per index: its own name if that is one ASCII letter."""
    indices = list(dict.fromkeys(list(output) + [ix for axes, _ in operands for ix in axes]))
    own = {ix.name for ix in indices if len(ix.name) == 1 and ix.name in _LETTERS}
    spare = (ch for ch in _LETTERS if ch not in own)
    letters: Dict[Index, str] = {}
    taken: Set[str] = set()
    for ix in indices:
        ch = ix.name if ix.name in own and ix.name not in taken else next(spare)
        taken.add(ch)
        letters[ix] = ch
    return letters


def _number(c: Fraction) -> str:
    """|c| as Python text: 2, 0.5, 0.75 (exact floats), or (1 / 3)."""
    if c.denominator == 1:
        return str(c.numerator)
    if Fraction(float(c)) == c:
        return repr(float(c))
    return f"({c.numerator} / {c.denominator})"


def _line(c: Fraction, operands, output: Sequence[Index], call: str, tail: str) -> str:
    op = "-=" if c < 0 else "+="
    size = abs(c)
    if not operands:
        return f"    r {op} {_number(size)}"
    letters = _letters(output, operands)
    spec = ",".join("".join(letters[ix] for ix in axes) for axes, _ in operands)
    spec += "->" + "".join(letters[ix] for ix in output)
    args = ", ".join(src for _, src in operands)
    factor = "" if size == 1 else f"{_number(size)} * "
    return f'    r {op} {factor}{call}("{spec}", {args}{tail})'
