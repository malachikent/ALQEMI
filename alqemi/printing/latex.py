"""LaTeX printing, directly from ALQEMI's own objects (decision 13: no SymPy).

Conventions (DESIGN_DECISIONS, Step 13):
- Double-bar integrals <pq||rs> as \\langle pq \\| rs \\rangle (``double_bar`` names,
  2 upper and 2 lower indices; \\| is the double bar); other tensors as
  name^{upper}_{lower}, e.g. t^{ab}_{ij}, f^{a}_{c}.
- Explicit sums \\sum_{ck}; coefficients as \\frac{p}{q}, so BCH factors show
  whenever an expression holds them.
- Index names as they are: alpha/beta indices already differ by case (Step 12);
  IndexPool names get a subscript (a12 -> a_{12}).
- Terms are printed one by one; P(ij)/P(ab) grouping is planned.
"""
from __future__ import annotations

import re
from fractions import Fraction

from alqemi.core.expr import Bracket, BracketKind, Delta, ElementaryBracket, Expr, Term
from alqemi.core.indices import Index
from alqemi.core.operators import Kind, Op, OpString
from alqemi.core.tensor import Tensor

_NUMBERED = re.compile(r"([A-Za-z]+)(\d+)\Z")


def latex(x, *, double_bar=("v",), lhs=None, terms_per_line=None) -> str:
    """LaTeX for an Index, Op, OpString, Delta, ElementaryBracket, Tensor, Bracket,
    Term or Expr (TypeError otherwise).

    ``lhs`` gives "lhs = rhs"; with ``terms_per_line`` (needs ``lhs``, >= 1) the
    result is an align* block with that many terms per line.
    """
    if terms_per_line is not None:
        if lhs is None or not isinstance(terms_per_line, int) or terms_per_line < 1:
            raise ValueError("terms_per_line needs lhs and an integer >= 1")
        return _align(x, lhs, terms_per_line, tuple(double_bar))
    body = _latex(x, tuple(double_bar))
    return body if lhs is None else f"{lhs} = {body}"


def _latex(x, db) -> str:
    if isinstance(x, Index):
        return _index(x)
    if isinstance(x, Op):
        return (r"a^{\dagger}_{" if x.kind is Kind.CRE else "a_{") + _index(x.index) + "}"
    if isinstance(x, OpString):
        return " ".join(_latex(op, db) for op in x)
    if isinstance(x, Delta):
        return r"\delta_{" + _list((x.p, x.q)) + "}"
    if isinstance(x, ElementaryBracket):
        return f"[{_latex(x.left, db)}, {_latex(x.right, db)}]_{{+}}"
    if isinstance(x, Tensor):
        if x.name in db and len(x.upper) == 2 and len(x.lower) == 2:
            return r"\langle " + _list(x.upper) + r" \| " + _list(x.lower) + r" \rangle"
        out = x.name
        if x.upper:
            out += "^{" + _list(x.upper) + "}"
        if x.lower:
            out += "_{" + _list(x.lower) + "}"
        return out
    if isinstance(x, Bracket):
        suffix = "_{+}" if x.kind is BracketKind.ACOMM else ""
        return f"[{_expr(x.left, db)}, {_expr(x.right, db)}]{suffix}"
    if isinstance(x, Term):
        return _term(x, db)
    if isinstance(x, Expr):
        return _expr(x, db)
    raise TypeError(f"cannot print {type(x).__name__} as LaTeX")


def _index(ix: Index) -> str:
    """The name; letters then digits get a subscript (a12 -> a_{12})."""
    m = _NUMBERED.match(ix.name)
    return f"{m.group(1)}_{{{m.group(2)}}}" if m else ix.name


def _list(indices) -> str:
    """Indices run together if all are one character, else comma-separated."""
    parts = [_index(ix) for ix in indices]
    return "".join(parts) if all(len(s) == 1 for s in parts) else ",".join(parts)


def _number(c: Fraction) -> str:
    if c.denominator == 1:
        return str(c.numerator)
    return rf"\frac{{{c.numerator}}}{{{c.denominator}}}"


def _term(t: Term, db) -> str:
    factors = [_latex(s, db) for s in t.scalars] + [_latex(f, db) for f in t.factors]
    if not factors:
        return ("-" if t.coeff < 0 else "") + _number(abs(t.coeff))
    body = r"\, ".join(factors)
    if t.summed:
        body = r"\sum_{" + _list(sorted(t.summed, key=lambda ix: ix.name)) + "} " + body
    c = t.coeff
    sign = "-" if c < 0 else ""
    mag = abs(c)
    return sign + ("" if mag == 1 else _number(mag) + " ") + body


def _signed(terms, db):
    """(sign, text without its sign) for each term."""
    out = []
    for t in terms:
        text = _term(t, db)
        out.append(("-", text[1:]) if text.startswith("-") else ("+", text))
    return out


def _join(parts) -> str:
    text = ""
    for n, (sign, body) in enumerate(parts):
        if n == 0:
            text = ("-" if sign == "-" else "") + body
        else:
            text += f" {sign} {body}"
    return text


def _expr(e: Expr, db) -> str:
    return _join(_signed(e.terms, db)) if len(e) else "0"


def _align(x, lhs, n, db) -> str:
    if isinstance(x, Term):
        x = Expr((x,))
    if not isinstance(x, Expr):
        raise TypeError("terms_per_line needs a Term or Expr")
    parts = _signed(x.terms, db)
    if not parts:
        lines = [f"{lhs} &= 0"]
    else:
        lines = []
        for start in range(0, len(parts), n):
            chunk = parts[start:start + n]
            if start == 0:
                lines.append(f"{lhs} &= " + _join(chunk))
            else:
                lines.append(r"&\quad " + " ".join(f"{s} {b}" for s, b in chunk))
    return "\\begin{align*}\n" + " \\\\\n".join(lines) + "\n\\end{align*}"
