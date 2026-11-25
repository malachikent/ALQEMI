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

## Done (2026-11-24)
- No SymPy anywhere in v3 (decision 13; CLAUDE.md updated).
- Step 7 spec and tests: `alqemi/tests/test_step7_package.py` (53 checks, spec in its docstring).
- Step 7 implemented:
  - `alqemi/__init__.py` exports the 25 public names via `__all__`; `__version__` falls back to "0+unknown".
  - Deleted the legacy SymPy code (`algebra/`, `sample.py`, legacy tests) and cookiecutter leftovers (`alqemi.py`, `data/`).
  - `core/cluster.py` and `core/tools.py` are kept as reference for Step 8.
  - `codegen/` is kept as the home of einsum code generation.
  - `pyproject.toml`: no run-time dependencies, Python >=3.11, packages found as `alqemi*`, Source URL.
  - `docs/api.rst` lists the public API; README badges point at malachikent/ALQEMI; CI runs on pushes to every branch.
  - A wheel built from the repo installs and runs in a clean virtualenv.

- Step 8 spec and tests: `alqemi/tests/test_step8_tensors.py` (163 checks), `TensorValues` in `tests/evaluate.py`. Decision 14 (Simplify matches like terms up to same-kind operator swaps); Step 6's two-dummy merge test updated for it.
- Step 8 implemented:
  - `core/tensor.py` (`Tensor`, `Swap`); `Term` accepts tensors as scalars.
  - DeltaEvaluate substitutes into tensors; Simplify handles tensor symmetries, same-kind operator swaps and terms equal to minus themselves.
  - `builders.py`: `tensor_operator`, `fock_operator`, `fluctuation_potential`, `hamiltonian(form=...)`, `cluster(n)`; plain by default, `normal_ordered=True` for the block form.
  - Public API and `docs/api.rst` extended; deleted `core/cluster.py` and `core/tools.py`; removed the temporary guard in `tests/evaluate.py`.
  - [H, T2] gives 8 terms in 0.02 s; [[H, T2], T2] gives 14 terms in 3.6 s.

## Done (2026-11-24, after 0.1.0)
- Released 0.1.0 (Steps 1-8): `main` fast-forwarded, tag `v0.1.0`, GitHub release. macOS/3.13 checked locally because GitHub's runner never started.

- Step 9 spec and tests: `alqemi/tests/test_step9_projection.py` (73 checks). Free general indices in projections recorded as deferred (EPT/CPT).
- Step 9 implemented: `excitation`, `deexcitation` (`builders.py`); `vacuum_expectation`, `projection` (`pipeline/project.py`); `bch` (`pipeline/bch.py`); public API, `docs/api.rst`, CHANGELOG. Derives the CCSD energy (5 terms) and the CCD doubles residual (18 terms).

- BCH performance investigated (DESIGN_DECISIONS.md, "Step 10" and "Candidates"). Kept as Step 10: fingerprint-restricted Simplify search and occupation-count pruning in projections; recorded the rest.

- Step 10 spec and tests: `alqemi/tests/test_step10_performance.py` (36 checks), brute-force referee `alqemi/tests/reference_simplify.py`.
- Step 10 implemented: fingerprint-restricted like-term search in `pipeline/simplify.py`; occupation-count pruning in `pipeline/project.py`. Output identical to the brute-force Simplify; CCD-residual budget tightened from 4 s to 2 s so it detects a missing pruning.

- Step 11 spec and tests: `alqemi/tests/test_step11_codegen.py` (76 checks), einsum code
  generation with occupied-first (OOVV) axes and moeris-style integral blocks. Step 7's
  PUBLIC_API gains `einsum_code`. UHF spin integration recorded as a later step.

- Step 11 implemented: `alqemi/codegen/einsum.py` (`einsum_code`); public API,
  `docs/api.rst`, `opt_einsum` in the test extra, CHANGELOG. The generated CCSD energy and
  CCD doubles residual match the textbook spin-orbital formulas.

- CI fix before 0.2.0: the Step 10 CCD-residual budget is relative (pruned vs
  unpruned on the same machine, at least 3x), since the fixed 2 s failed on slow
  runners; `opt_einsum` added to the CI conda environment (`devtools/conda-envs/test_env.yaml`).

- All three Step 10 budgets are now relative (Simplify vs the brute-force reference;
  CCSD energy vs the brute-force Simplify); no fixed time limits remain.

- Step 12 spec and tests: `alqemi/tests/test_step12_spin.py` (61 checks), UHF spin
  integration (`Spin`, `spin_integrate`) and spin-block code generation matching the
  user's 21-block `eris`. Step 7's PUBLIC_API gains `Spin` and `spin_integrate`.
  Private cross-check against the user's hand-written UHF functions: all agree.

- Step 12 implemented: `Spin` and `Index.spin` (`core/indices.py`), spin-aware keys in
  `core/expr.py`, `core/tensor.py` and Simplify, `pipeline/spin.py` (`spin_integrate`),
  spin blocks in `codegen/einsum.py`; public API, `docs/api.rst`, CHANGELOG. The private
  cross-check against the user's hand-written UHF functions still agrees.

- Step 13 spec and tests: `alqemi/tests/test_step13_latex.py` (51 checks), direct
  LaTeX printing (`latex`, `Expr._repr_latex_`). Step 7's PUBLIC_API gains `latex`.
  P(ij)/P(ab) grouping recorded as planned.

- Step 13 implemented: `alqemi/printing/latex.py` (`latex`), `Expr._repr_latex_`; public
  API, `docs/api.rst`, CHANGELOG. Output compiles with pdflatex.

- Step 14 spec and tests: `alqemi/tests/test_step14_permutations.py` (55 checks),
  (1 - P_ij)/(1 - P_ab) grouping for LaTeX and code generation (opt-in). Step 7's
  PUBLIC_API gains PermutedTerm, group_permutations, expand_permutations.

- Step 14 implemented: `alqemi/pipeline/permute.py` (`PermutedTerm`, `group_permutations`,
  `expand_permutations`), `group=` in `latex` and `einsum_code`; public API,
  `docs/api.rst`, CHANGELOG.

- Step 15 spec and tests: `alqemi/tests/test_step15_docs.py`, user documentation
  (cheatsheet, user guide, seven tutorials checked against Psi4 and PySCF,
  ReadTheDocs, CI docs job).

- Step 15 implemented: `docs/cheatsheet.rst`, `docs/user_guide/` (ten pages),
  `docs/tutorials/` (seven pages and their scripts, with `integrals.py` and `scf.py`),
  `.readthedocs.yaml`, `devtools/conda-envs/docs_env.yaml`, CI `docs` job; README,
  `pyproject.toml` and CHANGELOG point to alqemi.readthedocs.io.

## In progress
- Nothing.

## Tests
- `pytest alqemi`: 1280 pass (70 in step 1, 110 in step 2, 67 in step 3, 223 in step 4, 66 in step 5, 56 in step 6, 72 in step 7, 163 in step 8, 73 in step 9, 36 in step 10, 76 in step 11, 61 in step 12, 51 in step 13, 55 in step 14, 101 in step 15), 0 failing, with Psi4, PySCF and Sphinx installed; without them the tutorial and build tests skip.

## Next step
Import the repo on readthedocs.org (user). Then choose Step 16. Candidates found while
writing the tutorials (details in DESIGN_DECISIONS.md, Step 15):
- `bch(H, T1 + T2, order=4)` takes about 19 minutes (tutorials use order 3 + T1^4).
- No public product of two expressions; brackets don't freshen dummies that clash
  with the other operand's free indices (`comm(bch(...), excitation(2))` raises).
- Spin-integrated output prints some amplitudes in non-canonical index order (t^{aA}_{Ij}).
Other candidates: deltas in code generation, triples permutation operators. Also: fix the pytest deprecation warning
about a class-scoped fixture.
