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

## In progress
- Step 3 code (`alqemi/pipeline/__init__.py`, `alqemi/pipeline/linearize.py`) is not written yet.

## Tests
- `pytest alqemi` currently stops at collection: `test_step3_linearize.py` cannot import `alqemi.pipeline`. This is expected until Step 3 code exists.
- Against a throwaway reference implementation: 256 pass (191 earlier, 65 in step 3).

## Open questions
- None.

## Next step
Implement Step 3 (`pipeline/linearize.py`) to make `test_step3_linearize.py` pass.
