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
12. **Dummy names: numbered inside, letters only in the final output.** `IndexPool` keeps
    issuing numbered names (p1, i1, a1, ...), which never run out and never clash.
    Dummies are relabelled to textbook letters (occ i, j, k, l, ...; virt a, b, c, d, ...;
    general p, q, r, s, ...) only once, at the end (Simplify or final printing). Free
    indices are never relabelled.

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

## Step 3 (spec + tests written 2026-11-23): pipeline stage 1, Linearize
Only the bilinearity of (1.8.1)/(1.8.2): coefficients, scalars (deltas and elementary
anticommutators are c-numbers) and summations come out of bracket arguments.
- `alqemi/pipeline/linearize.py`: `linearize(x, pool) -> Expr` and `is_linear(x)`.
- Linear form: every bracket, at any depth, has exactly one bare argument term on each
  side (coeff 1, no scalars, no summed); everything else sits on the top-level terms.
- Choices made in the spec:
  - An identity argument is kept ([A, δ] → δ [A, 1]). Stage 2 evaluates it: (1.8.5)
    with n = 0 gives 0 for a commutator, and an anticommutator with an even string
    has no rank-reducing expansion. No new rule is needed.
  - A zero argument makes the product vanish (empty sum). A zero coefficient is kept.
  - Term order is deterministic and left-major, as in `multiply`.
  - Clashing pulled-out dummies: the first occurrence keeps its name, the later one
    gets a fresh pool name.
  - Capture ([Σ_k A_k, δ_kS B] with k free on the right) cannot happen: Step 2's
    no-shadowing check rejects such input.
- Spec + tests: `alqemi/tests/test_step3_linearize.py` (65 checks, 14 oracle cases).
  Validated against a throwaway reference implementation; all 17 planted bugs caught.

## OPEN
- None.
