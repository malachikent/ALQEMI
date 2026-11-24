# ALQEMI v3 progress

## Done (2026-10-22)
- Step 1 symbol layer: `core/indices.py` (`Space`, `Index`) and `core/operators.py` (`Kind`, `Op`, `cre`, `ann`, `OpString`), as frozen dataclasses with no sympy (`8cfa1cb`).
- Step 1 tests now cite HJO equation numbers and include golden (1S.2.4), whose result is not normal ordered (`edf25b9`). Docs updated (`c047037`). Removed the empty `alqemi/operators/` package.
- Pushed `claude_alqemi` to origin.

## Done (2026-11-23)
- Settled both open design questions (DESIGN_DECISIONS.md items 10 and 11, copied into CLAUDE.md):
  - ElementaryBracket [a_p, a†_q]_+ stays an unevaluated scalar until stage 3.
  - Default route splits the left string with (1.8.4), following HJO.
- Renamed the GitHub repo from `old_ALQEMI` to `ALQEMI` (public), and set `origin` to `git@github.com:malachikent/ALQEMI.git`.

- Step 2 spec and tests written: `alqemi/tests/test_step2_expr.py` (110 checks, the spec is in its docstring) and the oracle evaluator `alqemi/tests/evaluate.py`. Summary in DESIGN_DECISIONS.md, "Step 2".

- Step 2 implemented: `core/pool.py` (`IndexPool`) and `core/expr.py` (`Delta`, `ElementaryBracket`, `Bracket`, `Term`, `Expr`, `as_expr`, `multiply_terms`, `multiply`). No sympy. Each class has a readable `__str__` for traces.

- Settled decision 12: IndexPool names stay numbered, and dummies are relabelled to letters only in the final output.
- Step 3 (Linearize) spec and tests written: `alqemi/tests/test_step3_linearize.py` (65 checks, spec in its docstring). Summary in DESIGN_DECISIONS.md, "Step 3".

- Step 3 implemented: `pipeline/linearize.py` (`linearize`, `is_linear`). `core/expr.py` gained `Term.all_names` and `freshen_dummies`. Added 2 tests: fresh dummy names must avoid names in later factors and deeper brackets (a capture bug the reference implementation had).

- Step 4 (rule library + IdentityExpand) spec and tests written: `alqemi/tests/test_step4_identity_expand.py` (223 checks, spec in its docstring). Summary in DESIGN_DECISIONS.md, "Step 4".
- Checked the spec against the HJO text. Decision 11 revised to right-first, because it reproduces HJO's printed (1.8.14)-(1.8.16) and (1S.2.4). Decision 6 (strict rank reduction) kept; partial expansion recorded under "Deferred" in DESIGN_DECISIONS.md.

- Step 4 implemented: `rules/hjo.py` (`Rule`, `Side`, `RULES`, `is_elementary`, `is_rank_reducing`, `select_rule`, `UnsupportedBracket`) and `pipeline/identity_expand.py` (`identity_expand`, `TraceStep`). With elementary brackets read as δ or 0, the output reproduces HJO's printed (1.8.14), (1.8.15), (1.8.16) term for term, and the operator strings of (1S.2.4).

- Step 5 (DeltaEvaluate) spec and tests written: `alqemi/tests/test_step5_delta_evaluate.py` (66 checks, spec in its docstring). Summary in DESIGN_DECISIONS.md, "Step 5". A sum that no longer occurs (Σ_k δ_kk) raises `DanglingSum`; recorded under "Deferred: space dimensions".

- Step 5 implemented: `pipeline/delta_evaluate.py` (`delta_evaluate`, `DanglingSum`) and `Space.contains`. Through stage 3 the pipeline prints HJO's (1.8.11)-(1.8.16) as in the book, except for the order of deltas within a term (scalars are stored sorted: δ_QR δ_SM rather than the book's δ_SM δ_QR). Like terms are not merged yet: [E_PQ, Σ_p E_pp] gives a†_P a_Q - a†_P a_Q.

- Step 6 (Simplify + `run`) spec and tests written: `alqemi/tests/test_step6_simplify.py` (55 checks, spec in its docstring). Summary in DESIGN_DECISIONS.md, "Step 6".

- Step 6 implemented: `pipeline/simplify.py` (`simplify`, `relabel_dummies`) and `pipeline/run.py` (`run`). The four-stage pipeline is complete. `run` reproduces HJO's (1.8.11)-(1.8.16) and (1S.2.4); [E_PQ, Σ_k E_kk], [E_PQ, E_PQ] and the Jacobi identity give exactly 0; sums like [Σ_k E_kP, Σ_k E_kQ] come out as Σ_p a†_p a_Q - Σ_p a†_p a_P.

## In progress
- Step 7 (public API, legacy cleanup, packaging) spec and tests written:
  `alqemi/tests/test_step7_package.py` (53 checks, spec in its docstring). Not implemented yet.
- CLAUDE.md and DESIGN_DECISIONS.md: no SymPy anywhere in v3 (decision 13).
- Step 8 planned (DESIGN_DECISIONS.md, "Step 8"): `Tensor`, operator definitions,
  `hamiltonian()`, `cluster(n)`, `FluctuationPotential()`.

## Tests
- `pytest alqemi`: 602 pass + the new Step 7 file, which fails until Step 7 is implemented
  (8 of its 53 checks already pass).

## Next step
Implement Step 7. Then Step 8 spec and tests. Later: Fermi-vacuum projections and
einsum code generation, direct LaTeX printing (decision 13).
