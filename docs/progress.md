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

## In progress
- Nothing is half-done. Step 5 has no spec or tests yet.

## Tests
- `pytest alqemi`: 481 pass (70 in step 1, 110 in step 2, 67 in step 3, 223 in step 4, 11 legacy), 0 failing. About 20 s.

## Open questions
- None.

## Next step
Write the Step 5 spec and tests before any Step 5 code. Step 5 is stage 3 of the pipeline, DeltaEvaluate: turn ElementaryBrackets into deltas or zero by (1.2.27)-(1.2.29) (creator-first [a†_P, a_Q]_+ = δ_PQ, and the reverse order too), then evaluate deltas (δ_PP = 1, δ_ia = 0) and collapse sums over dummies.
