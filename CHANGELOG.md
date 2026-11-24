# Changelog

All notable changes to ALQEMI are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) written as PEP 440. While the version is
0.x, the public API may change in any minor release.

Releases are git tags (`v0.1.0`, ...); versioningit reads the version from the tag.
To release: rename "Unreleased" to the new version and date, then tag that commit on `main`.

## [Unreleased]

To be released as 0.1.0. First release of ALQEMI v3, a rewrite of the earlier
SymPy-based prototype.

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
