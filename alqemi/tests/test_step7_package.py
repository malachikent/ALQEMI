"""Step 7 tests: public API, legacy cleanup, and packaging.

These tests DEFINE Step 7. No new mathematics: the step turns the v3 code into a
package that installs and imports cleanly, with no SymPy (DESIGN_DECISIONS 13)
and no runtime dependencies.

  Public API: alqemi/__init__.py
    ``from alqemi import *`` gives exactly PUBLIC_API (below), via ``__all__``.
    (Steps 8, 9 and 11-14 added their names to PUBLIC_API.)
    Everything else stays importable from its submodule (alqemi.rules.hjo.RULES,
    alqemi.core.expr.as_expr, ...) but is not re-exported.
    ``alqemi.__version__`` is a str. It comes from alqemi/_version.py (written by
    versioningit at install), and falls back to "0+unknown" when that file is
    missing, so a fresh clone still imports.
    ``import alqemi`` imports neither sympy nor numpy.

  Legacy cleanup
    Removed: alqemi/alqemi.py (canvas placeholder), alqemi/sample.py,
    alqemi/algebra/, alqemi/data/, alqemi/tests/legacy/.
    Kept as reference until Step 8 rebuilds them, then deleted:
    alqemi/core/cluster.py (FluctuationPotential) and alqemi/core/tools.py
    (DoubleBar). They are the only files that may mention sympy, and nothing
    imports them.
    Kept: alqemi/codegen/, the future home of einsum code generation.

  Packaging: pyproject.toml
    requires-python ">=3.11" (CI tests 3.11-3.13).
    [project] dependencies: none. The test extra holds pytest and numpy (numpy
    is used only by the oracle).
    Package discovery: where = ["."], include = ["alqemi*"], so the installed
    packages are alqemi, alqemi.core, ..., never a top-level ``core``.

  Docs and metadata
    docs/api.rst lists every public name (alqemi.<name>), and every entry
    resolves. No cookiecutter placeholders (REPLACE_WITH_...) in README.md or
    docs/.

  CI (.github/workflows/CI.yaml, not tested here): runs on pushes to every
  branch and on pull requests to main.
"""
import ast
import importlib
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

import alqemi
from alqemi.tests.evaluate import mismatch
from alqemi.tests.oracle import FockSpace, delta

PUBLIC_API = {
    # symbols (Step 1)
    "Space", "Index", "Kind", "Op", "OpString", "cre", "ann",
    # expressions (Step 2)
    "Delta", "ElementaryBracket", "BracketKind", "Bracket", "comm", "acomm",
    "Term", "Expr", "IndexPool",
    # pipeline (Steps 3-6)
    "linearize", "identity_expand", "TraceStep", "delta_evaluate", "simplify",
    "relabel_dummies", "run",
    # errors
    "UnsupportedBracket", "DanglingSum",
    # tensors and builders (added by Step 8)
    "Tensor", "Swap", "tensor_operator", "fock_operator", "fluctuation_potential",
    "hamiltonian", "cluster",
    # projections (added by Step 9)
    "excitation", "deexcitation", "vacuum_expectation", "projection", "bch",
    # code generation (added by Step 11)
    "einsum_code",
    # spin integration (added by Step 12)
    "Spin", "spin_integrate",
    # printing (added by Step 13)
    "latex",
    # permutation operators (added by Step 14)
    "PermutedTerm", "group_permutations", "expand_permutations",
}

PACKAGE_DIR = Path(alqemi.__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
REFERENCE_ONLY = {"core/cluster.py", "core/tools.py"}  # deleted in Step 8


def _in_checkout():
    return (REPO_ROOT / "pyproject.toml").is_file()


needs_checkout = pytest.mark.skipif(not _in_checkout(), reason="needs the source checkout")


def _run_python(code):
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=60
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


# =========================================================================
# Public API
def test_all_is_exactly_the_public_api():
    assert set(alqemi.__all__) == PUBLIC_API
    assert len(alqemi.__all__) == len(PUBLIC_API)  # no duplicates


@pytest.mark.parametrize("name", sorted(PUBLIC_API))
def test_public_name_is_the_submodule_object(name):
    homes = {
        "Space": "alqemi.core.indices", "Index": "alqemi.core.indices",
        "Kind": "alqemi.core.operators", "Op": "alqemi.core.operators",
        "OpString": "alqemi.core.operators", "cre": "alqemi.core.operators",
        "ann": "alqemi.core.operators",
        "Delta": "alqemi.core.expr", "ElementaryBracket": "alqemi.core.expr",
        "BracketKind": "alqemi.core.expr", "Bracket": "alqemi.core.expr",
        "comm": "alqemi.core.expr", "acomm": "alqemi.core.expr",
        "Term": "alqemi.core.expr", "Expr": "alqemi.core.expr",
        "IndexPool": "alqemi.core.pool",
        "linearize": "alqemi.pipeline.linearize",
        "identity_expand": "alqemi.pipeline.identity_expand",
        "TraceStep": "alqemi.pipeline.identity_expand",
        "delta_evaluate": "alqemi.pipeline.delta_evaluate",
        "DanglingSum": "alqemi.pipeline.delta_evaluate",
        "simplify": "alqemi.pipeline.simplify",
        "relabel_dummies": "alqemi.pipeline.simplify",
        "run": "alqemi.pipeline.run",
        "UnsupportedBracket": "alqemi.rules.hjo",
        "Tensor": "alqemi.core.tensor", "Swap": "alqemi.core.tensor",
        "tensor_operator": "alqemi.builders", "fock_operator": "alqemi.builders",
        "fluctuation_potential": "alqemi.builders", "hamiltonian": "alqemi.builders",
        "cluster": "alqemi.builders",
        "excitation": "alqemi.builders", "deexcitation": "alqemi.builders",
        "vacuum_expectation": "alqemi.pipeline.project", "projection": "alqemi.pipeline.project",
        "bch": "alqemi.pipeline.bch",
        "einsum_code": "alqemi.codegen.einsum",
        "Spin": "alqemi.core.indices", "spin_integrate": "alqemi.pipeline.spin",
        "latex": "alqemi.printing.latex",
        "PermutedTerm": "alqemi.pipeline.permute", "group_permutations": "alqemi.pipeline.permute",
        "expand_permutations": "alqemi.pipeline.permute",
    }
    assert getattr(alqemi, name) is getattr(importlib.import_module(homes[name]), name)


def test_star_import_gives_the_public_api():
    names = _run_python(
        "ns = {}; exec('from alqemi import *', ns); "
        "print(' '.join(sorted(k for k in ns if not k.startswith('__'))))"
    ).split()
    assert set(names) == PUBLIC_API


@pytest.mark.parametrize("name", ["as_expr", "multiply", "RULES", "select_rule", "Rule", "canvas"])
def test_internal_or_removed_names_are_not_exported(name):
    assert not hasattr(alqemi, name)


def test_version_is_a_string():
    assert isinstance(alqemi.__version__, str) and alqemi.__version__


def test_version_falls_back_without_version_file():
    # sys.modules[name] = None makes ``import name`` raise ImportError.
    out = _run_python(
        "import sys; sys.modules['alqemi._version'] = None; "
        "import alqemi; print(alqemi.__version__)"
    )
    assert out == "0+unknown"


@pytest.mark.parametrize("module", ["sympy", "numpy"])
def test_import_alqemi_does_not_import(module):
    out = _run_python(f"import sys, alqemi; print({module!r} in sys.modules)")
    assert out == "False"


def test_top_level_api_end_to_end_hjo_1_8_14():
    """[E_PQ, E_RS] = delta_QR a+_P a_S - delta_PS a+_R a_Q, using only ``alqemi.*``."""
    from alqemi import Index, ann, comm, cre, run

    P, Q, R, S = (Index(x) for x in "PQRS")
    result = run(comm(cre(P) * ann(Q), cre(R) * ann(S)))
    fock = FockSpace(n_occ=1, n_virt=2)

    def expected(g):
        return delta(Q, R, g) * fock.string(cre(P) * ann(S), g) - delta(P, S, g) * fock.string(
            cre(R) * ann(Q), g
        )

    assert mismatch(fock, result, expected, extra=[P, Q, R, S]) is None


# =========================================================================
# Legacy cleanup
@pytest.mark.parametrize("module", ["alqemi.alqemi", "alqemi.sample", "alqemi.algebra"])
def test_legacy_module_removed(module):
    assert importlib.util.find_spec(module) is None


@needs_checkout
@pytest.mark.parametrize("path", ["alqemi/data", "alqemi/tests/legacy"])
def test_legacy_directory_removed(path):
    assert not (REPO_ROOT / path).exists()


def test_codegen_package_kept():
    spec = importlib.util.find_spec("alqemi.codegen")
    assert spec is not None
    doc = importlib.import_module("alqemi.codegen").__doc__ or ""
    assert "einsum" in doc  # states what the package is for


def _python_files():
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        rel = path.relative_to(PACKAGE_DIR).as_posix()
        if "__pycache__" not in rel:
            yield rel, path


def _imported_modules(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                yield node.module
            elif node.level > 0:
                yield "." * node.level + (node.module or "")


def test_no_sympy_outside_reference_files():
    offenders = [
        rel
        for rel, path in _python_files()
        if rel not in REFERENCE_ONLY
        and any(m == "sympy" or m.startswith("sympy.") for m in _imported_modules(path))
    ]
    assert offenders == []


def test_nothing_imports_the_reference_files():
    targets = {"alqemi.core.cluster", "alqemi.core.tools", "core.cluster", "core.tools", ".cluster", ".tools"}
    offenders = [
        rel
        for rel, path in _python_files()
        if rel not in REFERENCE_ONLY and targets & set(_imported_modules(path))
    ]
    assert offenders == []


# =========================================================================
# Packaging
@pytest.fixture(scope="module")
def pyproject():
    if not _in_checkout():
        pytest.skip("needs the source checkout")
    import tomllib

    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())


def _requirement_names(reqs):
    return {re.split(r"[<>=!~ ;\[]", r, maxsplit=1)[0].strip().lower() for r in reqs}


def test_no_runtime_dependencies(pyproject):
    assert pyproject["project"].get("dependencies", []) == []


def test_test_extra_has_pytest_and_numpy(pyproject):
    names = _requirement_names(pyproject["project"]["optional-dependencies"]["test"])
    assert {"pytest", "numpy"} <= names
    assert "sympy" not in names


def test_requires_python(pyproject):
    assert pyproject["project"]["requires-python"] == ">=3.11"


def test_package_discovery_config(pyproject):
    find = pyproject["tool"]["setuptools"]["packages"]["find"]
    assert find["where"] == ["."]
    assert find["include"] == ["alqemi*"]


def test_discovered_packages(pyproject):
    setuptools = pytest.importorskip("setuptools")
    find = pyproject["tool"]["setuptools"]["packages"]["find"]
    found = set(setuptools.find_packages(where=str(REPO_ROOT / find["where"][0]), include=find["include"]))
    assert all(name == "alqemi" or name.startswith("alqemi.") for name in found)
    assert {"alqemi", "alqemi.core", "alqemi.pipeline", "alqemi.rules", "alqemi.codegen", "alqemi.tests"} <= found
    assert "alqemi.algebra" not in found


# =========================================================================
# Docs and metadata
@needs_checkout
def test_api_docs_list_exactly_the_public_api():
    text = (REPO_ROOT / "docs" / "api.rst").read_text()
    listed = re.findall(r"^\s+alqemi\.(\w+)\s*$", text, flags=re.MULTILINE)
    assert set(listed) == PUBLIC_API
    for name in listed:
        assert hasattr(alqemi, name)


@needs_checkout
def test_no_cookiecutter_placeholders():
    files = [REPO_ROOT / "README.md", *sorted((REPO_ROOT / "docs").rglob("*.rst"))]
    offenders = [str(f.relative_to(REPO_ROOT)) for f in files if "REPLACE_WITH" in f.read_text()]
    assert offenders == []
