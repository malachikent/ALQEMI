# ALQEMI v3 — Design Decisions

Status key: SETTLED = agreed, copy into CLAUDE.md. OPEN = needs a decision.
Source: Helgaker, Jørgensen & Olsen (HJO), Ch. 1 (Helgaker_excerpt.pdf in this project).

## SETTLED (2026-10-22)

1. **No Wick's theorem, anywhere.** Do not use `sympy.physics.secondquant.Commutator`
   (its constructor reorders arguments and calls `wicks()` for two single fermion
   operators) or `wicks`. (Originally SymPy was allowed for final printing; superseded
   by decision 13.)
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

## SETTLED (2026-11-24)

13. **No SymPy in v3, including printing.** SymPy's `Add`/`Mul` sort their arguments,
    which would undo decision 8 and Simplify's first-occurrence term order, and its
    `KroneckerDelta` evaluates itself, so the printer would be doing math. ALQEMI prints
    text and LaTeX directly from its own objects. An optional `to_sympy()` converter for
    downstream algebra may be added later; it would never be part of the pipeline.

14. **Simplify may swap adjacent same-kind operators when matching like terms.**
    {a†_p, a†_q} = {a_p, a_q} = 0 makes such a swap exact, a sign only, never a delta,
    so it is not a move toward normal order. Like-term matching tries every order
    within each run of adjacent creators and each run of adjacent annihilators (with
    the permutation's sign); a repeated operator in a run makes the term zero. A
    creator and an annihilator are never swapped, and the merged term keeps its first
    occurrence's operator order. Reason: without it, [F, T2] has 4 terms that are 2
    equal pairs (f^{p}_{b} t^{ba}_{ji} a†_p a†_a a_i a_j = f^{p}_{b} t^{ab}_{ji} a†_a a†_p a_i a_j).

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

## Step 5 (done 2026-11-23): pipeline stage 3, DeltaEvaluate
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

## Step 6 (done 2026-11-23): pipeline stage 4, Simplify, and `run`
- `alqemi/pipeline/simplify.py`: `simplify(x)`, `relabel_dummies(x)`.
  `alqemi/pipeline/run.py`: `run(x, trace=None)` runs all four stages with its own pool.
- Choices made in the spec:
  - Like terms: equal after renaming dummies (keeping spaces) and reading deltas as
    symmetric. Operator order is never normalised (decision 8), so a†_P a_Q and
    a_Q a†_P stay separate terms.
  - A merged term keeps the first occurrence's position, dummy names and delta
    orientation. Terms whose coefficients sum to 0 are removed.
  - Dummies are relabelled per term, in order of first appearance (operators left to
    right, then deltas), to the first free letters of their space: occ i-o, virt a-h,
    general p-w, then i1/a1/p1... Names used by any free index in the expression are
    skipped, whatever its space (decision 12).
  - Matching dummies tries every renaming within each space, which costs n! per
    space. That is fine for the handful of dummies in a term; revisit if it isn't.
- Results: [E_PQ, Σ_p E_pp], [E_PQ, E_PQ] and the Jacobi identity (1.8.17) all
  simplify to exactly 0.
- Spec + tests: `alqemi/tests/test_step6_simplify.py` (55 checks). Validated against a
  throwaway reference implementation; all 15 planted bugs caught.

## Step 7 (done 2026-11-24): public API, legacy cleanup, packaging
- No new mathematics. Spec + tests: `alqemi/tests/test_step7_package.py` (53 checks).
  Validated against a throwaway reference implementation; all 21 planted bugs caught.
- Public API: `alqemi.__all__` is exactly the Step 1-6 user-facing names (symbols,
  expressions, `IndexPool`, the pipeline stages, `run`, the two errors). Internals
  (`as_expr`, `multiply`, `RULES`, `select_rule`, ...) stay in their submodules.
  `__version__` falls back to "0+unknown" without `_version.py`. `import alqemi` loads
  neither sympy nor numpy.
- Removed: `alqemi.py` (cookiecutter `canvas`), `sample.py`, `algebra/`, `data/`,
  `tests/legacy/`. Kept as reference until Step 8 rebuilds them: `core/cluster.py`
  (legacy FluctuationPotential) and `core/tools.py` (DoubleBar); nothing imports them.
- Kept: `codegen/`, the future home of einsum code generation (see Step 8 notes).
- Packaging: no runtime dependencies (numpy moves to the `test` extra, sympy goes);
  `requires-python >=3.11`; package discovery `where=["."], include=["alqemi*"]`
  (the cookiecutter's `where=["alqemi"]` installed `core`, `pipeline`, ... as top-level
  packages). `docs/api.rst` lists the public API; README badges point at
  malachikent/ALQEMI. CI runs on pushes to every branch and on PRs to main.
- The rest of the cookiecutter (versioningit, CI, docs, conda test env, LICENSE,
  py.typed) stays: it is what ships the package. PyPI upload on a version tag and a
  conda-forge recipe come later.
- Checked: a wheel built from the repo contains only `alqemi.*`, declares no run-time
  dependencies, and works from a clean virtualenv outside the repo.

## Step 8 (done 2026-11-24): tensors and coupled-cluster operators
- Spec + tests: `alqemi/tests/test_step8_tensors.py` (163 checks, spec in its docstring).
  Validated against a throwaway reference implementation: 815 tests pass, and 34 of 35
  planted bugs are caught (the miss is equivalent: `cluster(0)` still raises).
  `tests/evaluate.py` gained `TensorValues`: random arrays per (name, ranks, swap)
  with exactly the declared symmetry, and a `tensors=` argument.
- `Tensor(name, upper, lower, swap)` in `core/tensor.py`: a commuting scalar, always
  antisymmetric within the upper and within the lower indices. `Swap` (NONE |
  SYMMETRIC | ANTISYMMETRIC) is the per-tensor upper<->lower exchange symmetry:
  SYMMETRIC for f, <pq||rs> (stored as tensor "v") and h (real orbitals); NONE for
  amplitudes; ANTISYMMETRIC for imaginary Hermitian operators. Construction never
  reorders or zeroes anything.
- Pipeline: Linearize and IdentityExpand treat tensors as scalars. DeltaEvaluate
  substitutes into them and counts their indices as occurrences. Simplify merges
  terms equal up to tensor symmetries (with the sign), drops terms equal to minus
  themselves, and relabels tensor-only dummies (after operators and deltas). It also
  matches terms up to same-kind operator swaps (decision 14), so [F, T2] has 2 terms.
- `alqemi/builders.py`: `tensor_operator` (the general builder; default coefficient
  1/prod(count of each space in each group)!), `fock_operator`, `fluctuation_potential`
  (renamed from Phi, lowercase like the others), `hamiltonian(form="fock"|"core")`,
  `cluster(n)`. All return Expr with numbered pool dummies.
  - Plain forms by default; `normal_ordered=True` gives the legacy block form
    (Fermi-vacuum normal-ordered blocks plus constants), the same operator.
  - A single `tensor_operator(..., normal_ordered=True)` is the normal-ordered
    product {X}, which is NOT the plain product when it holds an occupied a+ and an
    occupied a ({a+_i a_j} = a+_i a_j - delta_ij). Only the builders' complete forms
    are equal, because they add the constants (F = F_N + sum_i f_ii,
    Phi = V_N - 1/2 sum_ij <ij||ij>). Writing a definition in normal order is not
    reordering a result (decision 8 still holds for the pipeline).
  - hamiltonian(form="fock") = F + Phi; form="core" = h + V. Equal when
    f_pq = h_pq + sum_i <pi||qi>; no HF assumption (f need not be diagonal).
- Deleted `core/cluster.py` and `core/tools.py`; no sympy anywhere.
- Implementation: Simplify's canonical forms sort each anticommuting group (operator
  run or tensor index group) and use the sort's parity as the sign; a repeat means 0.
  Timings: [H, T2] 0.02 s (8 terms), [H_no, T2] 0.3 s (28 terms), [[H, T2], T2]
  3.6 s (14 terms).
- The Step 7 public-API list grows by Tensor, Swap and the five builders.
- Step 6's `test_two_dummies_merge_up_to_permutation` expected sum_kl a†_k a†_l a_P +
  sum_kl a†_l a†_k a_P to merge into one term; both are 0 (decision 14), so it now uses
  sum_kl a†_k a_P a†_l, and a new test expects the zero.
- Out of scope: projections / Fermi-vacuum expectation values (needed for CC equations
  and response functions). That needs its own no-Wick design, in a later step.
- End goal (later step): generate Python code from derived expressions, as
  `np.einsum` / `opt_einsum.contract` calls, for the prototype codes (`alqemi/codegen/`).

## Step 9 (done 2026-11-24): Fermi-vacuum projections
- Spec + tests: `alqemi/tests/test_step9_projection.py` (73 checks, spec in its docstring).
  Validated against a throwaway reference implementation: 893 tests pass, all 19
  planted bugs caught. The reference reproduces the CCSD energy (5 terms) and the
  standard 18-term CCD doubles residual.
- Goal: CC energy and amplitude equations, and later response-function terms, as
  vacuum expectation values. Excited determinants are never separate objects:
  |mu> = tau_mu |HF> and <mu| = <HF| tau_mu^dagger (HJO's CC notation), so every
  projection is a vacuum expectation value.
- Names (ASCII in code; Greek only in docs and printed output):
  - `excitation(n)`: tau_mu for rank n, the bare excitation string with FREE indices
    and coefficient 1, e.g. a+_a a+_b a_j a_i for n = 2.
  - `vacuum_expectation(X)`: <HF| X |HF>.
  - `projection(n, X)`: <mu_n| X |HF> = vacuum_expectation(tau_mu^dagger X); n = 0 gives
    <HF| X |HF> (the energy).
  - `deexcitation(n, name="l")`: Lambda_n = (1/n!)^2 sum l^{i..}_{a..} a+_i.. a_a..,
    summed with amplitudes, like `cluster(n)`. Not a projector.
- Notation: excited determinants are written |mu_n> or |^{ab}_{ij}> in docs and
  printed output, never Phi_ij^ab; Phi means only the fluctuation potential.
- Method (settled 2026-11-24): absorb a killer into a bracket. Because K|HF> = 0
  for a ket-side killer K (a_a virtual, a+_i occupied),
      <HF| X K Y |HF> = <HF| X [K, Y]_(+) |HF>,
  with Y the whole rest of the string: an anticommutator when Y has odd length, a
  commutator when Y has even length (always rank-reducing, decision 6). The bracket
  is then evaluated by the existing pipeline (IdentityExpand, DeltaEvaluate,
  Simplify). Repeat until only scalars remain. No Wick's theorem and no reordering
  as a step of its own.
  Spec details: once every index is OCC or VIRT, each operator kills exactly one
  side (a_a, a+_i kill |HF>; a+_a, a_i kill <HF|), so no bra-side rule is needed:
  take the RIGHTMOST |HF>-killer; if there is none, or it is the last operator, the
  term is 0. Inside the loop only IdentityExpand and DeltaEvaluate run; Simplify
  runs once at the end (per-step Simplify made the CCD residual 6.5x slower).
- `bch(H, T, order=4)` is part of Step 9: H + [H,T] + 1/2 [[H,T],T] + ..., each
  nested commutator evaluated and simplified before projection.
- General indices (settled 2026-11-24): split. Each summed GENERAL index becomes an
  OCC copy and a VIRT copy of its term before projection; results name their blocks
  (f_ov, v_oovv, ...), as code generation needs. A free GENERAL index is a ValueError.
- Implemented as specified; the real code also catches all 19 planted bugs. Timings:
  CCSD energy from bch(H, T1+T2, order=2) about 9 s in total; CCD doubles residual
  (18 terms) 3.5 s after bch.
- Modules: `excitation`, `deexcitation` in `builders.py`; `vacuum_expectation`,
  `projection` in `pipeline/project.py`; `bch` in `pipeline/bch.py`. The Step 7
  public-API list grows by these five names.

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

## Deferred: free general indices in projections (not planned; revisit for EPT/CPT)
Propagator methods (electron-propagator theory, cluster perturbation theory) project
expressions with FREE general indices labelling the ionized/attached orbital, e.g.
<HF| a_p [H, a+_q] |HF>. `vacuum_expectation` raises ValueError for these, because
the value depends on the index's space. Names are not the problem (decision 3;
dummies are renamed around free names). If needed: a `blocks(x)` helper returning
{space choice of the free general indices: specialized copy of x}, then project each
block (results are block-structured, as EPT writes them and codegen needs). Also
possible: `excitation(n, virt="cd", occ="kl")` to choose the projection's free names.

## Step 10 (done 2026-11-24): two exact speed-ups
- Spec + tests: `alqemi/tests/test_step10_performance.py` (36 checks, spec in its
  docstring). Referee: `alqemi/tests/reference_simplify.py`, the brute-force Simplify
  of Steps 6/8 kept verbatim; the fast Simplify must give EXACTLY its output.
  Validated against a prototype: 929 tests pass; 8 of 9 planted bugs caught. The real
  code gives the same 8 of 9. The miss, pruning only by the occupied count, is
  equivalent in practice: H, T, Lambda and the projectors conserve the total electron
  number, so an unchanged occupied count implies an unchanged virtual count; it
  changes neither results nor timing (CCD projection 0.43 s either way). Full suite
  38 s -> 29 s.
- Profiling (2026-11-24): 96% of [[H, T2], T2] was Simplify's like-term search
  (41,472 renamings for 32 terms); projections spent their time building Terms for
  split copies that are zero.
- 1. Simplify: only exchange dummies with equal name-free fingerprints (space; and
  where it occurs: operator kind + run index, delta partner (free name or "dummy"),
  tensor name/swap/ranks/side, side "either" under a swap). Any renaming that maps a
  term onto a like term keeps fingerprints, so the output is unchanged.
  [[H, T2], T2]: 3.6 s -> 0.13 s.
- 2. vacuum_expectation: after splitting, drop terms whose operators change the
  occupied or the virtual electron count; they map |HF> to an orthogonal state, so
  they are exactly 0 (particle-number conservation, not Wick).
  CCD residual projection: 3.5 s -> 0.44 s; CCSD energy derivation: ~9 s -> 0.3 s.
- Time budgets in the tests are several times the measured times (the CCD-residual
  budget was tightened from 4 s to 2 s at implementation: unpruned it takes 3.5 s on
  an idle machine, so 4 s could not detect a missing pruning).
- CCD-residual budget made relative (2026-11-24, before 0.2.0): the fixed 2 s failed
  on slow CI runners (macOS 3.11: 2.4 s; Windows 3.11: 3.0 s) with correct results.
  The test now times the projection with and without the pruning on the same
  machine and requires at least 3x (about 8x locally); it also checks both give
  the same result. Verified to fail when the pruning is removed.
- The other two budgets made relative too (2026-11-24, after 0.2.0): simplify of
  [[H, T2], T2] at least 5x faster than the brute-force reference (about 33x
  locally); the CCSD energy derivation at least 4x faster than with the
  brute-force Simplify in run, bch and project (about 11x). Both check the results
  agree, and both fail when the fingerprint restriction is removed. No fixed time
  limits remain.

## Step 11 (done 2026-11-24): einsum code generation
- Spec + tests: `alqemi/tests/test_step11_codegen.py` (76 checks, spec in its
  docstring). Generated code is executed on random tensor values and compared with
  direct loops over the same expression; the CCSD energy and the CCD doubles
  residual are also compared with the textbook spin-orbital formulas written by hand.
  Validated against a prototype: 1006 tests pass; 13 of 13 planted bugs caught.
  The real code (`alqemi/codegen/einsum.py`) gives the same: 1006 pass, 13 of 13.
- `einsum_code(expr, name, *, integrals=None, amplitudes="{name}{rank}",
  backend="numpy")` returns the source of one Python function, one einsum line
  per term, each under a comment with the ALQEMI term. Backends: `np.einsum(...,
  optimize=True)` or `opt_einsum.contract`.
- Index orientation (settled 2026-11-24, the user's convention): OCCUPIED FIRST
  everywhere. The algebra and printed terms keep t^{ab}_{ij}; only array axes
  change: t2[i,j,a,b], l2[i,j,a,b], t1[i,a], and the residual r[i,j,a,b].
- Integrals (settled 2026-11-24): block attributes of one integrals object, as
  the user's prototype codes store them (moeris.g_OOVV, ...). `g` is ALQEMI's `v`,
  <pq||rs> in physicist's order. Each integral is rewritten exactly to an
  occupied-first block (antisymmetry within each group, then the swap symmetry),
  so <pq||rs> uses only OOOO, OOOV, OOVV, OVOV, OVVV, VVVV and f, h only OO, OV,
  VV. The access pattern is a template (`integrals={"v": "moeris.g_{block}"}`).
  Rejected: full arrays plus o/v slices (keeps every redundant block, about 4x the
  memory, and views with gaps can slow einsum).
- Fock blocks (settled 2026-11-24): the same form as the ERI blocks, attributes
  f_OO, f_OV, f_VV of the integrals object (moeris.f_VV[a,c] = f^{a}_{c}),
  occupied first; under UHF later, f_oo, f_ov, f_vv for beta. The derivations
  never assume Hartree-Fock, so f_OV terms (e.g. f_ia t_ia in the CCSD energy) and
  off-diagonal f_OO/f_VV are kept; with an HF reference f_OV = 0 and those terms
  contribute nothing. Possible later: an option that drops f_OV terms at
  generation time, for speed only.
- Amplitudes are plain arguments (t1, t2, l2), already in their block.
- Not supported yet: deltas between free indices (ValueError; response-theory
  matrix elements such as delta_ij f_ab will need them).
- `opt_einsum` is in the test extra; its numeric test is skipped when it is not
  installed. The module itself imports no numpy.

## Step 12 (done 2026-11-24): spin integration for UHF
- Spec + tests: `alqemi/tests/test_step12_spin.py` (61 checks, spec in its
  docstring). Validated against a prototype: 1069 tests pass; 15 of 15 planted bugs
  caught; the real code gives the same (1069 pass, 15 of 15). Values on a UHF-like model (3a + 2b occupied, 2a + 3b virtual; integrals
  from random real (pq|rs)): every spin block equals the spin-orbital result at its
  spins, and generated code reads only the user's 21 ERI + 6 Fock blocks.
- Design: `Index` gains `spin` (Spin.ALPHA / BETA / None), part of equality, never
  parsed from the name; Simplify puts spin in every key and fingerprint and names
  alpha dummies in upper case. `spin_integrate(expr)` works on projected,
  tensor-only terms: free indices get the canonical spin blocks (alpha before beta,
  Sz conserved: OOVV, OoVv, oovv; OV, ov; "" for scalars), every summed index an
  alpha and a beta copy, copies with a spin-changing tensor are dropped (exact by
  spin conservation), then Simplify. No Wick, no new algebra.
- Code generation: block letters O/o, V/v; amplitudes named "{name}{rank}_{block}"
  (t2_OoVv[i,J,a,B]); integrals canonicalized over antisymmetry, the swap, and, for
  opposite-spin two-electron integrals only (there <pq||rs> = <pq|rs>), the
  REAL-orbital symmetry <pq|rs> = <rq|ps>, with spins paired electron by electron.
  This lands every spin-conserving <pq||rs> on exactly the user's 21 blocks; without
  the real-orbital images it produced blocks such as g_OvVo. Assumes real orbitals,
  as the user's eris does.
- Private cross-check (2026-11-24, scratchpad only, never committed) against the
  user's hand-written uhf_com_functions.py: all 34 single-ordering functions that are
  plain commutators ([Phi,T1], [Phi,T2], [[Phi,T1],T2] for singles and doubles, all
  spin blocks) match ALQEMI exactly; all 63 driver cases (2-, 3- and 4-fold nested
  commutators, TT/TR, TTT/TTR/RTT/TAR, TTTT/TTTR/TTRR/TTBR/TAAR/TARR/TABR) equal the
  polarized BCH term exactly (factors 1/2, 1/6, 1/24, ...). No mathematical error
  found. Reported to the user: build_doubles_2com_T1_drv and
  build_doubles_4com_T1_drv call functions that are not defined in the file.

## Step 13 (done 2026-11-24): direct LaTeX printing
- Spec + tests: `alqemi/tests/test_step13_latex.py` (51 checks, spec in its
  docstring). Validated against a prototype: 1121 tests pass; 13 of 13 planted bugs
  caught; the real code gives the same (1121 pass, 13 of 13). The CCSD energy, CCD
  residual and UHF spin blocks also compile with
  pdflatex (checked locally, not in CI).
- `latex(x, *, double_bar=("v",), lhs=None, terms_per_line=None)` in
  `alqemi/printing/latex.py` prints every ALQEMI object directly (decision 13).
  Settled with the user: <pq||rs> as \langle pq \| rs \rangle (\| is the double
  bar), f^{a}_{c} and t^{ab}_{ij} as tensors, explicit \sum_{ck}, alpha/beta by
  upper/lower case, coefficients as \frac{p}{q} (so BCH factors always show when
  an expression holds them), `lhs` + `terms_per_line` give an align* block.
  `Expr._repr_latex_` for Jupyter.
- Printed term by term. P(ij)/P(ab) grouping is planned (below).

## Step 14 (done 2026-11-24): permutation operators (1 - P_ij), (1 - P_ab)
- Spec + tests: `alqemi/tests/test_step14_permutations.py` (55 checks, spec in its
  docstring). Validated against a prototype: 1179 tests pass; 13 of 14 planted bugs
  caught. The miss, grouping free indices by space only (ignoring spin), is
  equivalent: an alpha<->beta exchange of free indices maps a term to +-itself
  (never used) or to a spin-violating form no term has.
- Settled with the user: doubles forms (1 - P_{ij}), (1 - P_{ab}) and
  (1 - P_{ij})(1 - P_{ab}) only (triples forms P(i/jk) later), printed as
  (1 - P_{ij}) like the user's code comments, shared by LaTeX and code generation,
  opt-in (group=True).
- `group_permutations(expr)` -> PermutedTerm(term, swaps): partners found with
  Simplify's like-term key (exact up to dummy renaming and tensor symmetries);
  greedy in expression order, product first, then occupied pair, then virtual
  pair; a swap mapping a term to +-itself is never used for it.
  `expand_permutations` writes groups out; expanding always gives back the input.
- Results: CCD residual 18 terms -> 10 groups (textbook form); UHF same-spin
  blocks shrink (CCD OOVV 32 -> 15), mixed OoVv blocks have no swaps. Generated
  code: one einsum per group, then tmp - tmp.swapaxes(...), the user's style;
  identical numbers to term-by-term code (also checked on the UHF blocks in the
  scratchpad).
- Found while validating: unsimplified input with a duplicated term could pair a
  term with its own copy; fixed by the self-map rule above, with tests.

## Step 15 (spec 2026-11-25): user documentation
- Spec + tests: `alqemi/tests/test_step15_docs.py` (spec in its docstring). Validated
  against a prototype of the full docs with Psi4 1.11, PySCF 2.15 and Sphinx 9.1:
  1280 tests pass, none skipped; 15 of 16 planted bugs caught. The miss, a script
  that compares its CCSD energy with itself, is equivalent while ALQEMI is right
  (its energy matches Psi4 to 6e-12, inside the 1e-9 recheck); combined with wrong
  equations (BCH cut to order 2) it is caught by the independent recomputation.
- Settled with the user: reStructuredText (Sphinx, the existing setup); a one-page
  cheatsheet, a user guide (one page per concept) and tutorials; real numbers from
  Psi4 first, PySCF as a second program; the six planned tutorials plus MP2 (seven:
  commutator, MP2, CCD, CCSD, UHF-CCSD, Lambda-CCD, LaTeX); hosted on ReadTheDocs.
- Every example is tested: `>>>` examples on the guide pages run as doctests;
  tutorials are plain scripts (`docs/tutorials/scripts`) that pages include verbatim
  and that record CHECKS (label, ALQEMI value, reference); the test compares every
  check itself and recomputes every SCF-program reference independently, so a script
  cannot pass by comparing with itself. Tutorials 2-6 run once per program and skip
  when it is not installed; a CI `docs` job installs both and runs sphinx-build -W.
- Program-independent helpers: `integrals.py` builds the `moeris` blocks generated
  code reads (spin-orbital or UHF, on demand, from AO integrals + MO coefficients);
  `scf.py` runs the SCF with Psi4 or PySCF and returns numpy arrays.
- Checks used: matrices (commutator); Psi4/PySCF MP2, CCSD, UHF-CCSD; CCD (Psi4 has no
  conventional CCD) via its first Jacobi step = MP2; Lambda-CCD via a fixed-orbital
  finite-field derivative of the CCD energy equal to the Lambda dipole expectation.
- Found while writing the tutorials (recorded, not fixed in this step):
  - `bch(H, T1 + T2, order=4)` takes about 19 minutes; the CCSD projections need only
    order 3 plus the T1^4 term (order 4 contributes T1^4 to the doubles only), which
    takes seconds and gives the identical result (checked once against order 4).
  - No public product of two expressions, and brackets do not freshen dummy names
    that clash with the other operand's free indices (`comm(bch(...), excitation(2))`
    raises). The Lambda tutorial works around both with Term(...) and Term.rename.
  - Spin-integrated output prints some amplitudes with non-canonical index order
    (t^{aA}_{Ij}): correct by antisymmetry, but not the stored OoVv order.

## Planned: triples permutation operators
P(i/jk) = 1 - P_ij - P_ik, P(ij/k) and products, for triples-level equations; add
when ALQEMI derives triples.

## Spin integration: background (recorded before Step 12)
ALQEMI derives spin-orbital equations. The user's UHF codes store spin blocks
(t2_OOVV / t2_OoVv / t2_oovv for aa / ab / bb). Turning one spin-orbital equation
into its spin-block equations is a separate derivation step, reserved for UHF
implementations (settled 2026-11-24). Code generation then names spin blocks
(O/o, V/v) instead of spin-orbital ones.
Target: the user's `eris` class (non-redundant ERIs for real spin-orbitals, built
from chemist's (pq|rs) as physicist's blocks):
- One letter per index position: O/V = alpha occ/virt, o/v = beta occ/virt.
- Same-spin blocks are antisymmetrized <pq||rs>, e.g. g_OOVV[i,j,a,b] =
  (ia|jb) - (ib|ja); mixed-spin blocks are plain <pq|rs> (no exchange), e.g.
  g_OoVv[I,j,A,b] = <Ij|Ab>.
- 21 blocks (6 + 6 + 9): aa and bb each OOOO, OOOV, OOVV, OVOV, OVVV, VVVV; ab: OoOo, OoOv,
  oOoV, OoVv, OvOv, oVoV, OvVv, oVvV, VvVv. Occupied-first outranks alpha-first
  (OvOv and oVoV are both kept; never vOvO).
- In this convention g_OOVV is the alpha-alpha block, not the spin-orbital one, so
  Step 11's spin-orbital code is for spin-orbital (GHF-style) tests only.
Settled with the user (2026-11-24), from their hand-written UHF CC/CP code:
- Amplitudes and residuals: alpha at positions 1 and 3, t2_OoVv[i,J,a,B],
  t1_OV[i,a], t1_ov[i,a]; only alpha-beta mixed blocks, never beta-alpha (the
  beta-alpha block is the alpha-beta one transposed, no sign: exact by antisymmetry).
  Residual blocks use the same names (r2_OOVV, r2_OoVv, r2_oovv, r1_OV, r1_ov).
- One generated function per spin block, taking only the blocks it uses.
- Generated functions are the "build" functions for ONE commutator ordering with
  distinct amplitude arguments (e.g. <mu|[[Phi, T1], R1]|HF> with t1_*, r1_*). The
  user's hand-written drivers stay: they dispatch on object identity because in
  cluster perturbation theory different perturbative orders of the amplitudes are
  different arrays. ALQEMI does not generate drivers.
- No BCH factors are added to generated code: a function computes exactly the
  expression it is given (bch includes 1/k!, a bare commutator does not). LaTeX
  printing (later) must show the factors whenever they are present.
- The user's hand-written functions may be used as a numerical cross-check in the
  PRIVATE scratchpad only; never committed. Any discrepancy is reported to the user.

## Candidates (not scheduled)
- `adjoint(x)` for unitary CC: reverse and adjoint every operator string, keep real
  tensors as they are, so T2^dagger = 1/4 sum t^{ab}_{ij} a+_i a+_j a_b a_a uses the
  SAME tensor as T2. (`tensor_operator` cannot express this: it pairs upper tensor
  indices with creators, so it would make a different tensor t^{ij}_{ab}.) Then the
  UCC generator sigma = T - T^dagger works with the existing `bch`; its series never
  terminates, so orders 5, 6, ... are real truncation choices (`bch` accepts any order).
- Iterative fingerprint refinement (Weisfeiler-Lehman), measured 2026-11-24 in a
  prototype: each round extends a dummy's fingerprint with its tensor neighbours' and
  delta partners'. For the 4th commutator with T2 it cut the renamings from 8.8 million
  to 41,000 (worst term 2 million -> 9,216) and Simplify from >120 s to 5.4 s. Needed
  only for order 3-4 BCH with T2 (not on the roadmap); identical output not yet
  checked on order-3 terms.
- Projection-aware truncation: excitation level is additive and every T raises it,
  so a BCH term whose LOWEST possible level (general indices counted as de-exciting)
  already exceeds the projection rank n can be dropped while building the series.
  Exact. bch(H, T2, order=4) builds C3 and C4, which no CCD/CCSD/CCSDT projection with
  T2 uses; CCSD doubles needs order 4 only for T1 terms.
- Timings without these: bch(H, T2, order=3) 3.9 s (45 terms); bch(H, T2, order=4)
  and bch(H, T1+T2, order=4) did not finish in 10 min.

## OPEN
- None.
