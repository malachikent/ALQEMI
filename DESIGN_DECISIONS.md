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
   Reconfirmed 2026-11-23 after checking HJO: the book says to reduce rank "to the
   greatest extent possible" and does not say what to do when no route reduces fully.
   ALQEMI keeps the strict reading. See "Deferred: partial expansion" below.
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
11. **Default expansion route: split the RIGHT string first** (revised 2026-11-23; it
    was "left first"). Use the left string only when the right one cannot be split
    rank-reducingly (e.g. [E_PQ, a†_R]). Reason, checked against the HJO text: ALQEMI
    never reorders, so the route fixes the final operator order. Right-first reproduces
    the book's printed (1.8.14), (1.8.15) and (1.8.16) term for term, and gives the same
    operator strings as (1S.2.4). Left-first gives different operator strings for
    (1.8.15) (e.g. a†_R a†_M a_N a_Q) and (1S.2.4) (e.g. a_S a†_Q). The trade-off is
    that the intermediate step for (1.8.13) is [E_PQ, a†_R] a_S + a†_R [E_PQ, a_S],
    not the book's [a†_P, E_RS] a_Q + a†_P [a_Q, E_RS]; the final (1.8.14) is the same.
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

## Step 3 (done 2026-11-23): pipeline stage 1, Linearize
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
  - A fresh name avoids every name in the whole input term, including factors after
    the bracket and deeper brackets. Otherwise a later free a†_p1 could be captured
    by a new Σ_p1. The reference implementation missed this; 2 tests were added.
- Spec + tests: `alqemi/tests/test_step3_linearize.py` (67 checks, 14 oracle cases).
  Validated against a throwaway reference implementation; all 17 planted bugs caught.
- `core/expr.py` gained `Term.all_names` and `freshen_dummies(target, clash, avoid,
  pool)`, now shared by `multiply_terms` and Linearize.

## Step 4 (done 2026-11-23): rule library and stage 2, IdentityExpand
- `alqemi/rules/hjo.py`: `Rule`, `Side`, `RULES` (14), `is_elementary`,
  `is_rank_reducing`, `select_rule`, `UnsupportedBracket`. Kept apart from the legacy
  SymPy-based `alqemi/algebra/`, which is left untouched.
- `alqemi/pipeline/identity_expand.py`: `identity_expand(x, pool, trace)`, `TraceStep`.
- Choices made in the spec:
  - Rules split a string into its single operators (n = string length). The binary
    equations (1.8.4)/(1.8.6)/(1.8.8) are the n = 2 cases and take priority, so
    HJO's names show up in the trace. HJO's own derivations sometimes split into
    larger pieces ((1S.2.3)); with right-first, single-operator splits still give the
    book's printed results.
  - Left forms ("L") are derived from the right forms via (1.8.1)/(1.8.2) and keep
    the split piece on the left: [A_k, B], as in HJO (1.8.13).
  - Selection is not a parity table. It picks the first rule (right side first per
    decision 11, binary first) whose created brackets are all rank-reducing; the
    parity table is a test. Left forms fire only when the right string cannot be split
    (e.g. [E_PQ, a†_R] uses 1.8.6L).
    A consequence: (1.8.8) never fires on single-operator splits, because one created
    bracket is always non-rank-reducing. It stays in the library as a tested identity.
  - [x, y]_+ of two single operators is the leaf. It becomes an ElementaryBracket
    scalar (trace step "elementary"), not a rule application.
  - An empty side is handled first: [A, 1] = 0 by (1.8.5) with n = 0; [A, 1]_+ raises.
  - Traversal is innermost-first, then left to right. A nested bracket's result is
    re-linearized, so its ElementaryBrackets move out as scalars (decision 10).
  - Odd/odd commutators raise anywhere, including after nesting: e.g.
    [a†_P, [E_QR, [E_SM, a_N]]] reaches [a†_P, single operator] and raises.
- Results are tested only against the oracle. Structure is tested for the route
  (trace), invariants (no brackets left, no deltas created, one elementary bracket per
  bracket level, operators conserved, n + m − 2 string length) and unsupported cases.
- Checked against the HJO text (Sec. 1.2.3, 1.8, Exercises/Solutions 1.2-1.3) on
  2026-11-23: rank-reduction rule, (1.8.1)-(1.8.17) and (1.2.27)-(1.2.29) all agree with
  the spec and with MATHEMATICAL_CONTRACT.md. The route was switched to right-first.
- Spec + tests: `alqemi/tests/test_step4_identity_expand.py` (223 checks; route tests
  for (1.8.14), (1.8.15), (1S.2.4)). Validated against a throwaway reference
  implementation; all 21 planted bugs caught, and left-first fails 11 tests.

## Step 5 (spec + tests written 2026-11-23): pipeline stage 3, DeltaEvaluate
- `alqemi/pipeline/delta_evaluate.py`: `delta_evaluate(x)`, `DanglingSum`.
  `alqemi/core/indices.py` gains `Space.contains`.
- Choices made in the spec:
  - [x, y]_+ becomes Delta(x.index, y.index): the delta keeps the bracket's order.
    That is how HJO writes every printed result ([a_Q, a†_R]_+ → δ_QR in (1.8.14)),
    so the golden deltas come out exactly as printed. Creator-first (the form of the
    definition (1.2.29)) would flip about half of them. Same-kind brackets give 0.
    Note for Simplify (Step 6): δ_PQ X and δ_QP X are like terms, so deltas must be
    compared as symmetric when merging.
  - δ_pp → 1; δ_ia (non-overlapping spaces) → 0; terms with a zero factor are removed.
    Terms with a 0 coefficient are kept for Simplify.
  - A summed index collapses only onto an index whose space it contains. Σ_p δ_pi
    becomes i, but Σ_i δ_iP stays (P may be virtual). When both indices are summed,
    the wider one goes; on a tie, the right one goes.
  - Free deltas never substitute into operators (δ_QR a†_P a_S stays), matching HJO.
  - Equal-valued deltas (δ_PQ δ_QP) are kept once.
  - A sum that no longer occurs (Σ_k δ_kk a_P = M a_P, with M the number of
    orbitals) raises `DanglingSum`. See "Deferred: space dimensions".
- Spec + tests: `alqemi/tests/test_step5_delta_evaluate.py` (66 checks). Validated
  against a throwaway reference implementation; all 18 planted bugs caught.

## Deferred: space dimensions (not planned; revisit only if needed)
Σ_k δ_kk = number of orbitals in k's space, which ALQEMI has no symbol for, so
DeltaEvaluate raises `DanglingSum`. It arises from inputs like Σ_k [a_k, a†_k a_P]. If
needed, add a scalar for a space's dimension (e.g. N_occ, N_virt, N_general) and
produce it instead of raising.

## Deferred: partial expansion (not planned; revisit only if needed)
Some brackets have no fully rank-reducing expansion. Strict mode (decision 6) raises
`UnsupportedBracket` for them. HJO's "to the greatest extent possible" would allow
expanding them anyway, keeping some terms at full rank:
- [odd, odd] commutator, e.g. [a_P, a_Q]: no product to split. Only the definitions
  help: [A, B] = [A, B]_+ − 2BA, so [a_P, a_Q] = −2 a_Q a_P. Nothing is reduced.
- Anticommutator with an even string, e.g. [a_P, a†_Q a_R]_+: by (1.8.8b) it becomes
  δ_PQ a_R − 2 a†_Q a_P a_R. One term reduces, one keeps full rank.
- [even, even] anticommutator, e.g. [E_PQ, E_RS]_+ = [E_PQ, E_RS] + 2 E_RS E_PQ.
These rarely arise in practice: commutators of excitation operators (BCH, coupled
cluster) are always even/even and reduce fully. If we ever need them, add a fallback
"expand by definition" rule from (1.8.1)/(1.8.2), tried only after every library rule
fails and marked as non-reducing in the trace. Also decide a tie-break between partial
routes (e.g. 1.8.8a vs 1.8.8b). It would be a new rule beyond the current library, so
CLAUDE.md would need updating too.

## OPEN
- None.
