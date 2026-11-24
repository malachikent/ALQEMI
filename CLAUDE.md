## Core design (v3)
- No Wick's theorem. No SymPy anywhere in v3 (not even for printing); LaTeX/text output is
  printed directly from ALQEMI's own objects.
- Own immutable core (frozen dataclasses); exact Fraction coefficients.
- Orbital space lives on Index(name, space) with Space.GENERAL|OCC|VIRT; never parsed from names.
- Dummies are Term.summed (frozenset[Index]); products freshen dummies via an explicit IndexPool. No globals.
- Each HJO identity is a Rule (name / matches / apply); the only condition on 1.8.5/1.8.7/1.8.9 is string parity.
  Unsupported cases raise, never quit().
- Pipeline: Linearize -> IdentityExpand -> DeltaEvaluate -> Simplify. Deltas exist only after stage 3.
- Tests check results against tests/oracle.py (matrix fermions), never expression-tree equality.
- Never edit tests/oracle.py or golden tests to make code pass.
- Math source of truth: MATHEMATICAL_CONTRACT.md. If code or tests disagree with it, stop and ask.
- Rationale and open questions: DESIGN_DECISIONS.md.
- Rule library: HJO (1.2.27)-(1.2.29), (1.8.4)-(1.8.9), plus antisymmetry from (1.8.1)/(1.8.2).
  Rule names cite these equation numbers. Jacobi (1.8.17) is a test, never a rule.
- Rule selection: HJO principle of rank reduction. Anticommute two odd-length strings;
  commute all other pairs. A rule fires only if all brackets it creates are rank-reducing.
- Never reorder results toward normal order.
- Elementary anticommutators [a_p, a†_q]_+ are unevaluated ElementaryBracket scalars through
  stages 1-2: no delta, delta_pp = 1, delta_ia = 0, or summation collapse before DeltaEvaluate.
- Default route: split the RIGHT string first; split the left only when the right can't be
  split rank-reducingly. This reproduces HJO's printed (1.8.14)-(1.8.16) and (1S.2.4).
- Strict rank reduction stays: no partial expansions (see DESIGN_DECISIONS.md, "Deferred").
- IndexPool names stay numbered (p1, i1, a1); relabel dummies to letters only in the final output.
