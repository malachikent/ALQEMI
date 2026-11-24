# Changelog

All notable changes to ALQEMI are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) written as PEP 440. While the version is
0.x, the public API may change in any minor release.

Releases are git tags (`v0.1.0`, ...); versioningit reads the version from the tag.
To release: rename "Unreleased" to the new version and date, then tag that commit on `main`.

## [Unreleased]

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
