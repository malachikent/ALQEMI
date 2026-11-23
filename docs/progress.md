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

## In progress
- Nothing is half-done. Step 4 has no spec or tests yet.

## Tests
- `pytest alqemi`: 258 pass (70 in step 1, 110 in step 2, 67 in step 3, 11 legacy), 0 failing.

## Open questions
- None.

## Next step
Write the Step 4 spec and tests before any Step 4 code. Step 4 is stage 2 of the pipeline, IdentityExpand: the Rule objects for (1.8.4)-(1.8.9) and (1.8.1)/(1.8.2), selected by rank reduction, applied until only ElementaryBrackets remain. Unsupported brackets raise `UnsupportedBracket`.
