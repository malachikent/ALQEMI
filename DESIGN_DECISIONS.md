# ALQEMI v3 — Design Decisions

Status key: SETTLED = agreed, copy into CLAUDE.md. OPEN = needs a decision.
Source: Helgaker, Jørgensen & Olsen (HJO), Ch. 1 (Helgaker_excerpt.pdf in this project).

## SETTLED (2026-10-22)

1. **No Wick's theorem, anywhere.** Do not use `sympy.physics.secondquant.Commutator`
   (its constructor reorders arguments and calls `wicks()` for two single fermion
   operators) or `wicks`. SymPy may be used only for final printing.
2. **Own immutable core.** Frozen dataclasses: every rewrite returns new objects, so
   nothing simplifies behind our back and every step can be logged. Exact
   `fractions.Fraction` coefficients.
3. **Orbital space lives on the Index object**, never parsed from the name.
   `Index(name, space)` with `Space.GENERAL | OCC | VIRT`; equality is by value
   (name AND space).
4. **Dummies are a property of a Term** (`Term.summed: frozenset[Index]`), not of an
   Index. Products of terms always freshen dummies through an explicit `IndexPool`.
   No global state.
5. **Identities are Rule objects** with `name`, `matches()`, `apply()`. Rule names cite
   HJO equation numbers. The rule library is exactly:
   - elementary relations (1.2.27)–(1.2.29);
   - expansions (1.8.4)–(1.8.9); parity conditions: (1.8.7) n even, (1.8.9) n odd;
   - antisymmetry/symmetry of the definitions (1.8.1)/(1.8.2), needed to move a product
     from the left slot to the right slot.
   The Jacobi identity (1.8.17) is NOT a rule (HJO: it relates expressions of the same
   rank); it is used only as a test.
6. **Rule selection follows HJO's principle of rank reduction (Sec. 1.8).** Rank of a
   string = (number of elementary operators)/2. Anticommute two strings of
   half-integral rank (odd length); commute all other pairs. A rule may be applied only
   if every bracket it produces is of a rank-reducing kind. Consequence: a commutator
   of two odd strings, or an anticommutator involving an even string, has no
   rank-reducing expansion and raises `UnsupportedBracket` (never `quit()`).
7. **Pipeline:** Linearize → IdentityExpand (until only ElementaryBrackets remain) →
   DeltaEvaluate → Simplify. Kronecker deltas exist only from stage 3 on.
8. **Results are never reordered toward normal order.** E.g. (1S.2.4) contains a_R a†_P;
   ALQEMI keeps operator order exactly as the identities produce it.
9. **Testing against a numerical oracle** (`tests/oracle.py`: Jordan–Wigner matrices,
   all index assignments respecting spaces), not against expression-tree equality.
   The oracle never uses Wick or ALQEMI's algebra.

## SETTLED (2026-11-23)

10. **ElementaryBracket is an unevaluated scalar.** HJO evaluates nested commutators by
    substituting the inner result, deltas included (1.8.16 "evaluated using (1.8.14)").
    ALQEMI is stricter: elementary anticommutators [a_p, a†_q]_+ stay as unevaluated
    leaves through stages 1–2. Linearity may move them as scalars, but no δ, δ_pp = 1,
    δ_ia = 0, or summation collapse happens before stage 3 (DeltaEvaluate).
11. **Default expansion route follows HJO.** When both are valid and rank-reducing, split
    the LEFT string with (1.8.4), as HJO does for (1.8.13), rather than the right string
    with (1.8.5). That way ALQEMI's trace reads like the book.

## Golden results (all verified by the oracle)
(1.8.11), (1.8.12), (1.8.14), (1.8.15), (1.8.16), (1S.2.4). Property for later steps
(HJO Exercise 1.2): [even string n, even string m] expands into strings of at most
n + m − 2 elementary operators.

## Step 1 (done 2026-10-22): symbol layer
- `alqemi/core/indices.py`: `Space` (with `overlaps`), `Index`.
- `alqemi/core/operators.py`: `Kind`, `Op`, `cre`, `ann`, `OpString`.
- Spec + tests: `alqemi/tests/test_step1_core.py` (70 checks; validated against a
  throwaway reference implementation, and confirmed to catch planted bugs).

## Step 2 (done 2026-11-23): expression layer
Still no algebra: no bracket is expanded, no delta evaluated, no like terms merged.
- `alqemi/core/pool.py`: `IndexPool.fresh(space, avoid)`; names p1.., i1.., a1.. per space.
- `alqemi/core/expr.py`: `Delta`, `ElementaryBracket` (unevaluated, decision 10),
  `BracketKind`, `Bracket`/`comm`/`acomm`, `Term`, `Expr`, `as_expr`,
  `multiply_terms`, `multiply`.
- Representation choices made in the spec:
  - Term = coeff · scalars · ordered factors, summed. Scalars commute, so they are
    stored in a canonical order. Factors never reorder: adjacent OpStrings concatenate
    and empty ones drop.
  - `Delta` and `ElementaryBracket` use structural equality (δ_PQ ≠ δ_QP);
    symmetry belongs to DeltaEvaluate/Simplify.
  - Expr keeps terms verbatim: no merging, and zero coefficients are not dropped.
  - Term/Expr have no `*`; products go through `multiply(…, pool)`, so dummies are
    always freshened. Right-term dummies are renamed first, then left-term dummies
    that clash with a free index on the right.
  - No shadowing: a name bound in a nested bracket term may not appear in the
    enclosing term. That makes `Term.rename` a plain substitution.
- Spec + tests: `alqemi/tests/test_step2_expr.py` (110 checks). Numerical referee:
  `alqemi/tests/evaluate.py` (reads Step 2 objects, builds oracle matrices, no ALQEMI
  algebra). Validated against a throwaway reference implementation; 21 planted bugs
  checked: 20 caught, and the remaining mutation changes no behaviour.

## OPEN
- None.
