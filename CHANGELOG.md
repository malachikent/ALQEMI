# Changelog

All notable changes to ALQEMI are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) written as PEP 440. While the version is
0.x, the public API may change in any minor release.

Releases are git tags (`v0.1.0`, ...); versioningit reads the version from the tag.
To release: rename "Unreleased" to the new version and date, then tag that commit on `main`.

## [Unreleased]

### Added
- `group_permutations(expr)`: folds terms related by swapping two free indices into
  `PermutedTerm` groups under (1 - P_ij), (1 - P_ab) and their product;
  `expand_permutations` is its exact inverse.
- `latex(..., group=True)` prints (1 - P_{ij}) factors; `einsum_code(..., group=True)`
  evaluates each group once and antisymmetrizes with `swapaxes`. Both are opt-in;
  default output is unchanged.

## [0.4.0] - 2026-11-24

### Added
- `latex(x)`: LaTeX printing straight from ALQEMI's objects (no SymPy). Double-bar
  integrals `\langle pq \| rs \rangle`, tensors `t^{ab}_{ij}`, explicit sums,
  fractions (BCH factors always shown), alpha/beta by upper/lower case, and an
  `align*` block via `lhs=` and `terms_per_line=`. Expressions also render in Jupyter.

## [0.3.0] - 2026-11-24

### Added
- Spin integration for UHF: `Spin` and `Index(..., spin=...)`; `spin_integrate(expr)`
  splits a projected spin-orbital equation into its canonical alpha/beta blocks
  (OOVV, OoVv, oovv; OV, ov; one block for an energy). Exact by spin conservation;
  no Wick's theorem.
- `einsum_code` writes spin-block code: amplitudes `t2_OoVv[i,J,a,B]`, `t1_ov`, ...;
  integrals matched to the 21 UHF ERI blocks (`moeris.g_OoVv`, ...) and the 6 Fock
  blocks, assuming real orbitals. Spin-orbital output is unchanged.

### Changed
- Simplify keeps alpha and beta dummies apart and names alpha dummies in upper case.
- The performance tests compare each speed-up against the same work without it on
  the same machine, so they no longer depend on the speed of the CI runner.

## [0.2.0] - 2026-11-24

### Added
- `excitation(n)` (tau_mu with free indices) and `deexcitation(n)` (Lambda_n).
- `vacuum_expectation(x)` = <HF| x |HF> and `projection(n, x)` = <mu_n| x |HF>,
  evaluated without Wick's theorem: summed general indices are split into occupied
  and virtual copies, and each |HF>-killer is absorbed into a rank-reducing bracket.
- `bch(h, t, order=4)`: the BCH expansion of e^{-T} H e^{T}.
- Together these derive coupled-cluster equations, e.g. the CCSD energy and the
  18-term CCD doubles residual.
- `einsum_code(expr, name)`: writes a derived expression as a Python function, one
  `np.einsum` (or `opt_einsum.contract`) line per term. Occupied-first axes
  (t2[i,j,a,b], r[i,j,a,b]); integrals as occupied-first blocks of one integrals
  object (moeris.g_OOVV, moeris.f_OV, ...), rewritten exactly with their sign.
  Spin-orbital only.
- Test extra: `opt_einsum`.

### Changed
- Faster, with identical results: Simplify only exchanges dummies with equal
  name-free fingerprints ([[H, T2], T2]: 3.6 s -> 0.13 s), and projections drop
  terms that change the occupied or virtual electron count (CCD residual:
  3.5 s -> 0.4 s; CCSD energy derivation: ~9 s -> 0.3 s).

## [0.1.0] - 2026-11-24

First release of ALQEMI v3, a rewrite of the earlier SymPy-based prototype:
commutator algebra by HJO's identities (no Wick's theorem), with tensors and the
standard coupled-cluster operators.

### Added
- Symbol layer: `Space`, `Index`, `Kind`, `Op`, `OpString`, `cre`, `ann`.
- Expression layer: `Delta`, `ElementaryBracket`, `Bracket`, `comm`, `acomm`, `Term`,
  `Expr`, `IndexPool`, with exact `Fraction` coefficients.
- Pipeline: `linearize` -> `identity_expand` -> `delta_evaluate` -> `simplify`, and
  `run` for all four stages. Brackets are evaluated with HJO's identities
  (1.2.27)-(1.2.29) and (1.8.4)-(1.8.9), chosen by the principle of rank reduction;
  no Wick's theorem. `run` reproduces HJO (1.8.11)-(1.8.16) and (1S.2.4).
- Errors: `UnsupportedBracket`, `DanglingSum`.
- Tensors: `Tensor` and `Swap`. A tensor is a commuting scalar, antisymmetric within
  its upper and its lower indices, with a per-tensor upper/lower swap symmetry.
- Builders: `tensor_operator` (any tensor x operator-string object), `fock_operator`,
  `fluctuation_potential`, `hamiltonian(form="fock" | "core")`, `cluster(n)`. Plain
  forms by default; `normal_ordered=True` for the Fermi-vacuum block forms.
- Simplify merges terms equal up to tensor symmetries and up to swaps of adjacent
  same-kind operators, and drops terms equal to minus themselves.
- Test oracle: every result is checked against Jordan-Wigner matrices.

### Changed
- No run-time dependencies; Python 3.11 or newer.

### Removed
- The SymPy-based prototype (`alqemi.algebra`, `alqemi.sample`, `alqemi.core.cluster`,
  `alqemi.core.tools`) and the
  cookiecutter placeholders (`alqemi.canvas`, `alqemi/data`).
