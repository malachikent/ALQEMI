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

## Candidates: unitary CC and high-order BCH (not scheduled)
- `adjoint(x)`: reverse and adjoint every operator string, keep real tensors as they
  are, so T2^dagger = 1/4 sum t^{ab}_{ij} a+_i a+_j a_b a_a uses the SAME tensor as T2.
  (`tensor_operator` cannot express this: it pairs upper tensor indices with
  creators, so it would make a different tensor t^{ij}_{ab}.) Then the UCC generator
  sigma = T - T^dagger works with the existing `bch`. Its series never terminates
  (sigma contains de-excitations, which contract with each other), so orders 5, 6, ...
  are real truncation choices; `bch` already accepts any order.
- Performance of high-order BCH: [[H, T2], T2] takes 3.6 s; order 3-4 with T2 (and
  UCC orders 5-6) has not been timed. Simplify's dummy-renaming search is factorial
  in the number of dummies per space and is the likely bottleneck. Also possible:
  projection-aware truncation (only build BCH terms that survive projection onto
  rank n).

## OPEN
- None.
